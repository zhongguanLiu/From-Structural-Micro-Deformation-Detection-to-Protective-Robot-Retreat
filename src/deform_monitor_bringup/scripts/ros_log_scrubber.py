#!/usr/bin/env python3
"""Deterministically redact credentials from one experiment's ROS logs.

The scrubber is intentionally run *after* roslaunch exits.  Replacing a log
that is still held open by roslaunch would detach the writer from the visible
path and lose the shutdown tail.  Scientific CSV/JSON finalization therefore
keeps its existing ROS handshake, while the run wrapper applies this final
post-shutdown output-contract step.
"""

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile


POLICY_VERSION = "ros-log-credential-redaction-v1"
REDACTION = "__REDACTED__"
REPORT_RELATIVE_PATH = "ros_log_scrub_report.json"

_SENSITIVE_NAME_PARTS = {
    "APIKEY", "API_KEY", "AUTH", "AUTHENTICATION", "AUTHORIZATION", "BEARER",
    "CLIENT_SECRET", "CREDENTIAL", "CREDENTIALS", "KEY", "OAUTH",
    "PASS", "PASSCODE", "PASSWD", "PASSWORD", "PRIVATE_KEY", "PWD",
    "SECRET", "TOKEN",
}
_SENSITIVE_EXACT_NAMES = {
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS", "GITHUB_TOKEN", "GITLAB_TOKEN",
    "HF_TOKEN", "OPENAI_API_KEY", "PROXY_AUTHORIZATION",
}
_PROXY_NAMES = {
    "ALL_PROXY", "FTP_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
    "RSYNC_PROXY",
}
_COMMON_NON_SECRETS = {
    "0", "1", ".", "/", "false", "localhost", "none", "null", "true",
}

_DOUBLE_QUOTED_MAPPING = re.compile(
    r'(?P<prefix>"(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)"\s*:\s*")'
    r'(?P<value>(?:\\.|[^"\\\r\n])*)(?P<suffix>")')
_SINGLE_QUOTED_MAPPING = re.compile(
    r"(?P<prefix>'(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)'\s*:\s*')"
    r"(?P<value>(?:\\.|[^'\\\r\n])*)(?P<suffix>')")
_DOUBLE_QUOTED_ASSIGNMENT = re.compile(
    r'(?<![A-Za-z0-9_.-])(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)'
    r'(?P<separator>\s*=\s*")(?P<value>(?:\\.|[^"\\\r\n])*)'
    r'(?P<suffix>")')
_SINGLE_QUOTED_ASSIGNMENT = re.compile(
    r"(?<![A-Za-z0-9_.-])(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)"
    r"(?P<separator>\s*=\s*')(?P<value>(?:\\.|[^'\\\r\n])*)"
    r"(?P<suffix>')")
_PLAIN_ASSIGNMENT = re.compile(
    r'(?<![A-Za-z0-9_.-])(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)'
    r'(?P<separator>\s*[=:]\s*)(?P<value>[^\s,;\]&}\)"\'][^\s,;\]&}\)]*)')
_DOUBLE_QUOTED_FLAG = re.compile(
    r'(?P<prefix>(?<![A-Za-z0-9_-])--?(?P<key>[A-Za-z][A-Za-z0-9_-]*)'
    r'(?:=|\s+)")(?P<value>(?:\\.|[^"\\\r\n])*)(?P<suffix>")')
_SINGLE_QUOTED_FLAG = re.compile(
    r"(?P<prefix>(?<![A-Za-z0-9_-])--?(?P<key>[A-Za-z][A-Za-z0-9_-]*)"
    r"(?:=|\s+)')(?P<value>(?:\\.|[^'\\\r\n])*)(?P<suffix>')")
_PLAIN_FLAG = re.compile(
    r'(?P<prefix>(?<![A-Za-z0-9_-])--?(?P<key>[A-Za-z][A-Za-z0-9_-]*)'
    r'(?:=|\s+))(?P<value>[^\s,;\]&}\)"\'][^\s,;\]&}\)]*)')
_XML_ENV = re.compile(
    r'(?P<prefix><env\b[^>]*\bname=["\'](?P<key>[A-Za-z_]'
    r'[A-Za-z0-9_.-]*)["\'][^>]*\bvalue=["\'])'
    r'(?P<value>[^"\']*)(?P<suffix>["\'])', re.IGNORECASE)
_AUTHORIZATION = re.compile(
    r'(?P<prefix>\b(?:proxy-authorization|authorization)\s*[:=]\s*'
    r'(?:(?:basic|bearer)\s+)?)'
    r'(?P<value>[^\s,;\]\}]+)', re.IGNORECASE)
_URI_USERINFO = re.compile(
    r'(?P<scheme>\b[A-Za-z][A-Za-z0-9+.-]*://)'
    r'(?P<userinfo>[^/@\s]+:[^/@\s]+)@')
_QUERY_PARAMETER = re.compile(
    r'(?P<prefix>[?&](?P<key>[A-Za-z_][A-Za-z0-9_.-]*)=)'
    r'(?P<value>[^&\s#]+)')


def is_sensitive_environment_name(name):
    """Return whether an environment/argument name conventionally holds a secret."""
    normalized = str(name).strip().upper().replace("-", "_").replace(".", "_")
    if normalized in _SENSITIVE_EXACT_NAMES or normalized in _PROXY_NAMES:
        return True
    parts = tuple(part for part in normalized.split("_") if part)
    if any(part in _SENSITIVE_NAME_PARTS for part in parts):
        return True
    # Compound spellings such as APIKEY and CLIENTSECRET have no separator.
    return ("APIKEY" in normalized or "CLIENTSECRET" in normalized
            or "PRIVATEKEY" in normalized or "ACCESSTOKEN" in normalized
            or normalized.endswith(("AUTHTOKEN", "PASSWORD", "SECRET",
                                    "CREDENTIAL")))


def sensitive_environment_values(environment):
    """Return non-trivial secret values without exposing their names."""
    values = set()
    for name, raw_value in dict(environment or {}).items():
        value = str(raw_value)
        if (is_sensitive_environment_name(name)
                and len(value) >= 4
                and value.strip().lower() not in _COMMON_NON_SECRETS
                and value != REDACTION):
            values.add(value)
    return tuple(sorted(values, key=lambda item: (-len(item), item)))


def _conditional_replacement(pattern, text, prefix_group="prefix"):
    count = 0

    def replace(match):
        nonlocal count
        if not is_sensitive_environment_name(match.group("key")):
            return match.group(0)
        value = match.group("value")
        if not value or value == REDACTION:
            return match.group(0)
        count += 1
        if prefix_group in match.re.groupindex:
            prefix = match.group(prefix_group)
        else:
            prefix = match.group("key") + match.group("separator")
        suffix = match.groupdict().get("suffix", "") or ""
        return prefix + REDACTION + suffix

    return pattern.sub(replace, text), count


def redact_log_text(text, environment=None):
    """Return ``(redacted_text, replacement_count)`` for one text payload."""
    output = str(text)
    replacements = 0

    # Exact environment values cover the full ``os.environ`` repr emitted by
    # roslaunch even when a value contains whitespace or nested punctuation.
    for value in sensitive_environment_values(environment):
        occurrences = output.count(value)
        if occurrences:
            output = output.replace(value, REDACTION)
            replacements += occurrences

    for pattern in (
            _DOUBLE_QUOTED_MAPPING, _SINGLE_QUOTED_MAPPING,
            _DOUBLE_QUOTED_ASSIGNMENT, _SINGLE_QUOTED_ASSIGNMENT,
            _DOUBLE_QUOTED_FLAG, _SINGLE_QUOTED_FLAG,
            _XML_ENV):
        output, count = _conditional_replacement(pattern, output)
        replacements += count

    def redact_authorization(match):
        nonlocal replacements
        if match.group("value") == REDACTION:
            return match.group(0)
        replacements += 1
        return match.group("prefix") + REDACTION

    output = _AUTHORIZATION.sub(redact_authorization, output)

    def redact_userinfo(match):
        nonlocal replacements
        if match.group("userinfo") == REDACTION:
            return match.group(0)
        replacements += 1
        return match.group("scheme") + REDACTION + "@"

    output = _URI_USERINFO.sub(redact_userinfo, output)
    output, count = _conditional_replacement(_QUERY_PARAMETER, output)
    replacements += count
    for pattern in (_PLAIN_FLAG, _PLAIN_ASSIGNMENT):
        output, count = _conditional_replacement(pattern, output)
        replacements += count
    return output, replacements


def _is_binary(payload):
    if b"\0" in payload:
        return True
    sample = payload[:8192]
    if not sample:
        return False
    controls = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return controls / float(len(sample)) > 0.05


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path, payload, mode=0o600):
    directory = os.path.dirname(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + os.path.basename(path) + ".", suffix=".tmp",
        dir=directory)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


def scrub_ros_log_directory(run_directory, environment=None):
    """Scrub regular text files under ``<run>/ros_logs`` and return a report."""
    run_root = os.path.realpath(os.path.abspath(run_directory))
    log_root = os.path.join(run_root, "ros_logs")
    if not os.path.isdir(log_root) or os.path.islink(log_root):
        raise ValueError("run ros_logs directory is missing or is a symlink")

    entries = []
    skipped_symlinks = []
    errors = []
    total_replacements = 0
    files_modified = 0
    binary_files_skipped = 0
    text_files_scanned = 0
    discovered = 0
    environment = dict(os.environ if environment is None else environment)

    for directory, directory_names, file_names in os.walk(
            log_root, topdown=True, followlinks=False):
        directory_names.sort()
        file_names.sort()
        for name in list(directory_names):
            path = os.path.join(directory, name)
            if os.path.islink(path):
                skipped_symlinks.append(os.path.relpath(path, log_root))
                directory_names.remove(name)
        for name in file_names:
            path = os.path.join(directory, name)
            relative = os.path.relpath(path, log_root)
            discovered += 1
            try:
                metadata = os.lstat(path)
                if stat.S_ISLNK(metadata.st_mode):
                    skipped_symlinks.append(relative)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                with open(path, "rb") as stream:
                    payload = stream.read()
                if _is_binary(payload):
                    binary_files_skipped += 1
                    entries.append({
                        "path": relative, "status": "SKIPPED_BINARY",
                        "size_bytes": len(payload), "replacement_count": 0,
                    })
                    continue
                text_files_scanned += 1
                text = payload.decode("utf-8", errors="surrogateescape")
                redacted, count = redact_log_text(text, environment)
                redacted_payload = redacted.encode(
                    "utf-8", errors="surrogateescape")
                if count:
                    _atomic_write(path, redacted_payload,
                                  stat.S_IMODE(metadata.st_mode))
                    files_modified += 1
                total_replacements += count
                entries.append({
                    "path": relative,
                    "status": "REDACTED" if count else "UNCHANGED",
                    "size_bytes": len(redacted_payload),
                    "replacement_count": count,
                    "sha256_after": _sha256_bytes(redacted_payload),
                })
            except (OSError, UnicodeError, ValueError) as error:
                safe_error, _ = redact_log_text(str(error), environment)
                errors.append({"path": relative, "error": safe_error})

    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "status": "FAILED" if errors else "COMPLETED",
        "log_root_relative_path": "ros_logs",
        "files_discovered": discovered,
        "text_files_scanned": text_files_scanned,
        "binary_files_skipped": binary_files_skipped,
        "symlinks_skipped": sorted(skipped_symlinks),
        "files_modified": files_modified,
        "replacement_count": total_replacements,
        "file_results": sorted(entries, key=lambda item: item["path"]),
        "errors": sorted(errors, key=lambda item: item["path"]),
    }


def write_scrub_report(run_directory, report):
    path = os.path.join(os.path.abspath(run_directory), REPORT_RELATIVE_PATH)
    payload = (json.dumps(report, indent=2, sort_keys=True,
                          ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write(path, payload, 0o600)
    return path, _sha256_bytes(payload)


def update_run_manifest(run_directory, report, report_sha256,
                        require_manifest=True):
    """Commit the post-roslaunch contract without changing recorder outcome."""
    manifest_path = os.path.join(os.path.abspath(run_directory),
                                 "run_manifest.json")
    if not os.path.isfile(manifest_path):
        if require_manifest:
            raise ValueError("run_manifest.json is missing")
        return False
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or int(manifest.get("schema_version", 0)) < 3:
        raise ValueError("run_manifest.json has an unsupported schema")
    contract = manifest.get("post_shutdown_artifact_contract", {})
    expected_post_paths = {REPORT_RELATIVE_PATH, "mission_summary.json"}
    declared_post_paths = set(contract.get("required_paths", [])) \
        if isinstance(contract, dict) else set()
    if not expected_post_paths.issubset(declared_post_paths):
        raise ValueError("run_manifest.json lacks the ROS-log scrub contract")
    errors = []
    if report.get("status") != "COMPLETED":
        errors.append("ROS log scrub reported errors")
    if manifest.get("required_artifact_errors"):
        errors.append("recorder required artifacts were incomplete")
    if manifest.get("finalization_outcome") != "COMPLETED":
        errors.append("recorder finalization outcome was not COMPLETED")
    for relative in sorted(expected_post_paths):
        if not os.path.isfile(os.path.join(
                os.path.abspath(run_directory), relative)):
            errors.append("missing post-shutdown artifact: " + relative)
    manifest["ros_log_sanitization"] = {
        "status": report.get("status"),
        "policy_version": report.get("policy_version"),
        "report_relative_path": REPORT_RELATIVE_PATH,
        "report_sha256": report_sha256,
        "text_files_scanned": report.get("text_files_scanned", 0),
        "files_modified": report.get("files_modified", 0),
        "replacement_count": report.get("replacement_count", 0),
    }
    manifest["post_shutdown_artifact_contract"] = {
        "phase": "POST_ROSLAUNCH",
        "required_paths": [REPORT_RELATIVE_PATH, "mission_summary.json"],
        "status": "COMPLETED" if not errors else "FAILED",
        "errors": errors,
    }
    manifest["final_output_contract_status"] = (
        "COMPLETED" if not errors else "FAILED")
    payload = (json.dumps(manifest, indent=2, sort_keys=True,
                          ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write(manifest_path, payload, 0o600)
    if errors:
        raise RuntimeError("; ".join(errors))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Redact credentials from a completed ICRA2027 ROS log tree")
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--allow-missing-manifest", action="store_true")
    arguments = parser.parse_args(argv)
    report = scrub_ros_log_directory(arguments.run_directory)
    _, report_sha256 = write_scrub_report(arguments.run_directory, report)
    update_run_manifest(
        arguments.run_directory, report, report_sha256,
        require_manifest=not arguments.allow_missing_manifest)
    if report["status"] != "COMPLETED":
        raise SystemExit(2)
    print("ROS log credential scrub complete: {} files, {} replacements".format(
        report["text_files_scanned"], report["replacement_count"]))
    return 0


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest


SCRIPT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts", "ros_log_scrubber.py"))
SPEC = importlib.util.spec_from_file_location("ros_log_scrubber", SCRIPT)
SCRUBBER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRUBBER)


class RosLogScrubberTests(unittest.TestCase):
    def test_both_run_wrappers_scrub_only_after_roslaunch_returns(self):
        package = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for name in (
                "run_icra2027_closed_loop.sh",
                "run_icra2027_stationary_full_process.sh"):
            with open(os.path.join(package, "scripts", name),
                      encoding="utf-8") as stream:
                source = stream.read()
            launch_index = source.index("roslaunch --sigint-timeout=180 --sigterm-timeout=30")
            self.assertIn('if [[ -f "${RUN_DIRECTORY}/run_manifest.json" ]] && (( SCRUB_STATUS == 0 )); then', source)
            master_cleanup_index = source.index(
                "cleanup_ros_master", launch_index)
            scrub_index = source.index("scripts/ros_log_scrubber.py")
            self.assertLess(launch_index, scrub_index)
            self.assertLess(launch_index, master_cleanup_index)
            self.assertLess(master_cleanup_index, scrub_index)
            self.assertNotIn("exec roslaunch", source)
            self.assertIn(
                'export ROS_MASTER_URI="http://127.0.0.1:${ROS_MASTER_PORT}"',
                source)
            self.assertIn('roscore -p "${ROS_MASTER_PORT}"', source)
            self.assertIn("trap cleanup_ros_master EXIT INT TERM", source)
            self.assertIn("unset ROS_IP || true", source)
            self.assertIn(
                'SCRUB_ARGUMENTS=(--run-directory "${RUN_DIRECTORY}")',
                source)
            self.assertIn("record_experiment:=false", source)
            self.assertIn("--allow-missing-manifest", source)
            self.assertIn('"${SCRUB_ARGUMENTS[@]}"', source)
            self.assertIn('ros_logs/roslaunch_console.log', source)
            self.assertIn("awk '/\\[closed_loop\\]/", source)
            self.assertIn("LAUNCH_STATUS=${PIPESTATUS[0]}", source)
            self.assertIn("RUN_INCOMPLETE: missing or failed final manifest", source)

        with open(os.path.join(
                package, "scripts", "icra2027_experiment_recorder.py"),
                encoding="utf-8") as stream:
            recorder = stream.read()
        self.assertIn('"required_artifact_contract": {', recorder)
        self.assertIn('"post_shutdown_artifact_contract": {', recorder)
        self.assertIn('"ros_log_scrub_report.json", "mission_summary.json"',
                      recorder)
        self.assertIn('"PENDING_POST_SHUTDOWN"', recorder)

    def test_redacts_roslaunch_environment_and_common_credential_forms(self):
        source = "\n".join([
            "roslaunch env is environ({'OPENAI_API_KEY': 'sk-live-value', "
            "'HTTP_PROXY': 'http://alice:proxy-pass@proxy:8080', "
            "'ROS_MASTER_URI': 'http://localhost:11311', "
            "'MONKEY': 'banana'})",
            "TOKEN=plain-token PASSWORD: plain-password",
            "node --client-secret cli-secret --api-key='quoted-secret'",
            "Authorization: Bearer bearer-secret",
            '<env name="AWS_SECRET_ACCESS_KEY" value="xml-secret" />',
            "https://bob:uri-secret@example.test/path?access_token=query-secret&x=1",
            "useful ROS node output remains visible",
        ])
        redacted, count = SCRUBBER.redact_log_text(source, {
            "OPENAI_API_KEY": "sk-live-value",
            "HTTP_PROXY": "http://alice:proxy-pass@proxy:8080",
        })
        for secret in (
                "sk-live-value", "proxy-pass", "plain-token",
                "plain-password", "cli-secret", "quoted-secret",
                "bearer-secret", "xml-secret", "uri-secret",
                "query-secret"):
            self.assertNotIn(secret, redacted)
        self.assertIn("ROS_MASTER_URI", redacted)
        self.assertIn("http://localhost:11311", redacted)
        self.assertIn("'MONKEY': 'banana'", redacted)
        self.assertIn("useful ROS node output remains visible", redacted)
        self.assertGreaterEqual(count, 10)
        repeated, repeated_count = SCRUBBER.redact_log_text(redacted, {})
        self.assertEqual(repeated, redacted)
        self.assertEqual(repeated_count, 0)

    def test_sensitive_name_matching_avoids_key_substrings(self):
        for name in (
                "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN",
                "CLIENTSECRET", "PROXY_AUTHORIZATION", "HTTPS_PROXY"):
            self.assertTrue(SCRUBBER.is_sensitive_environment_name(name), name)
        for name in (
                "MONKEY", "KEYBOARD_LAYOUT", "ROS_MASTER_URI", "NO_PROXY",
                "ROS_PACKAGE_PATH"):
            self.assertFalse(SCRUBBER.is_sensitive_environment_name(name), name)

    def test_directory_scrub_is_sorted_atomic_binary_safe_and_idempotent(self):
        with tempfile.TemporaryDirectory() as run:
            logs = os.path.join(run, "ros_logs")
            nested = os.path.join(logs, "run-id")
            os.makedirs(nested)
            first = os.path.join(logs, "roslaunch.log")
            second = os.path.join(nested, "node-1.log")
            binary = os.path.join(nested, "payload.bin")
            with open(first, "w", encoding="utf-8") as stream:
                stream.write("OPENAI_API_KEY=first-secret\nnode is healthy\n")
            with open(second, "w", encoding="utf-8") as stream:
                stream.write("Authorization: Basic second-secret\n")
            binary_payload = b"\x00OPENAI_API_KEY=binary-secret\xff"
            with open(binary, "wb") as stream:
                stream.write(binary_payload)
            link_supported = True
            try:
                os.symlink(first, os.path.join(nested, "do-not-follow.log"))
            except OSError:
                link_supported = False

            report = SCRUBBER.scrub_ros_log_directory(run, {})
            self.assertEqual(report["status"], "COMPLETED")
            self.assertEqual(report["text_files_scanned"], 2)
            self.assertEqual(report["files_modified"], 2)
            self.assertEqual(report["binary_files_skipped"], 1)
            self.assertEqual(
                [entry["path"] for entry in report["file_results"]],
                sorted(entry["path"] for entry in report["file_results"]))
            if link_supported:
                self.assertEqual(report["symlinks_skipped"],
                                 ["run-id/do-not-follow.log"])
            with open(first, encoding="utf-8") as stream:
                scrubbed = stream.read()
            self.assertNotIn("first-secret", scrubbed)
            self.assertIn("node is healthy", scrubbed)
            with open(binary, "rb") as stream:
                self.assertEqual(stream.read(), binary_payload)
            serialized_report = json.dumps(report, sort_keys=True)
            self.assertNotIn("first-secret", serialized_report)
            self.assertNotIn("second-secret", serialized_report)

            second_report = SCRUBBER.scrub_ros_log_directory(run, {})
            self.assertEqual(second_report["replacement_count"], 0)
            self.assertEqual(second_report["files_modified"], 0)

    def test_report_updates_only_declared_post_shutdown_manifest_contract(self):
        with tempfile.TemporaryDirectory() as run:
            os.makedirs(os.path.join(run, "ros_logs"))
            with open(os.path.join(run, "ros_logs", "node.log"), "w",
                      encoding="utf-8") as stream:
                stream.write("TOKEN=manifest-secret\n")
            with open(os.path.join(run, "mission_summary.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({"outcome": "COMPLETED"}, stream)
            manifest_path = os.path.join(run, "run_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "schema_version": 3,
                    "finalization_outcome": "COMPLETED",
                    "required_artifact_errors": [],
                    "required_artifact_contract": {
                        "phase": "RECORDER_FINALIZATION",
                        "required_paths": ["mission_decision_summary.json"],
                        "status": "COMPLETED",
                    },
                    "post_shutdown_artifact_contract": {
                        "phase": "POST_ROSLAUNCH",
                        "required_paths": [
                            "ros_log_scrub_report.json",
                            "mission_summary.json",
                        ],
                        "status": "PENDING",
                        "errors": [],
                    },
                    "final_output_contract_status": "PENDING_POST_SHUTDOWN",
                }, stream)

            report = SCRUBBER.scrub_ros_log_directory(run, {})
            report_path, digest = SCRUBBER.write_scrub_report(run, report)
            with open(report_path, "rb") as stream:
                self.assertEqual(
                    hashlib.sha256(stream.read()).hexdigest(), digest)
            self.assertTrue(SCRUBBER.update_run_manifest(
                run, report, digest, require_manifest=True))
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["final_output_contract_status"],
                             "COMPLETED")
            self.assertEqual(
                manifest["post_shutdown_artifact_contract"]["status"],
                "COMPLETED")
            self.assertEqual(
                manifest["ros_log_sanitization"]["report_sha256"], digest)
            self.assertEqual(manifest["required_artifact_errors"], [])

    def test_manifest_update_rejects_an_undeclared_or_incomplete_contract(self):
        report = {
            "status": "COMPLETED", "policy_version": "test",
            "text_files_scanned": 0, "files_modified": 0,
            "replacement_count": 0,
        }
        with tempfile.TemporaryDirectory() as run:
            path = os.path.join(run, "run_manifest.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({"schema_version": 3,
                           "finalization_outcome": "COMPLETED",
                           "required_artifact_errors": []}, stream)
            with self.assertRaises(ValueError):
                SCRUBBER.update_run_manifest(run, report, "digest")

    def test_manifest_does_not_hide_recorder_artifact_failure(self):
        report = {
            "status": "COMPLETED", "policy_version": "test",
            "text_files_scanned": 0, "files_modified": 0,
            "replacement_count": 0,
        }
        with tempfile.TemporaryDirectory() as run:
            for relative in ("ros_log_scrub_report.json",
                             "mission_summary.json"):
                with open(os.path.join(run, relative), "w",
                          encoding="utf-8") as stream:
                    stream.write("{}\n")
            path = os.path.join(run, "run_manifest.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({
                    "schema_version": 3,
                    "finalization_outcome": "FAILED",
                    "required_artifact_errors": ["missing: analysis/table.csv"],
                    "post_shutdown_artifact_contract": {
                        "required_paths": [
                            "ros_log_scrub_report.json",
                            "mission_summary.json"],
                    },
                }, stream)
            with self.assertRaises(RuntimeError):
                SCRUBBER.update_run_manifest(run, report, "digest")
            with open(path, encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["final_output_contract_status"],
                             "FAILED")
            self.assertEqual(manifest["required_artifact_errors"],
                             ["missing: analysis/table.csv"])


if __name__ == "__main__":
    unittest.main()

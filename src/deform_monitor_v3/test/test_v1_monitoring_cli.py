import importlib.util
import io
import pathlib
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


path = pathlib.Path(__file__).resolve().parents[1] / "scripts/v1_monitoring_cli.py"
spec = importlib.util.spec_from_file_location("monitor_cli", path)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class MonitoringCliTest(unittest.TestCase):
    def test_phase_call_runs_once_and_reports_completion(self):
        calls = []
        def service(request):
            calls.append(request)
            return "accepted"
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.call_phase_with_progress(service, 17, "Reference"), "accepted")
        self.assertEqual(calls, [17])
        self.assertIn("completed in", output.getvalue())

    def test_phase_call_propagates_failure_without_retry(self):
        def service(request):
            raise RuntimeError("bad reference")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "bad reference"):
            cli.call_phase_with_progress(service, None, "Reference")

    def test_reference_diagnostic_reads_detector_namespace(self):
        with patch.object(cli.rospy, "get_param", return_value="WAITING_DATA") as get:
            self.assertEqual(cli.reference_detail(), "WAITING_DATA")
        get.assert_called_once_with("/deform_monitor_v1/reference_diagnostics", "")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

import collections
import csv
import io
import pathlib
import subprocess
import unittest
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parent
SCRIPT = ROOT / "run_icra2027_stationary_speed_sweep.sh"
LAUNCH = (ROOT.parent / "deform_monitor_bringup" / "launch" /
          "icra2027_stationary_full_process.launch")
RUNNER = (ROOT.parent / "deform_monitor_bringup" / "scripts" /
          "run_icra2027_stationary_full_process.sh")


class StationarySpeedSweepTest(unittest.TestCase):
    def test_matrix_and_speed_forwarding(self):
        output = subprocess.check_output(
            ["bash", str(SCRIPT), "--print-matrix"], text=True)
        rows = list(csv.DictReader(io.StringIO(output)))
        self.assertEqual(len(rows), 40)
        counts = collections.Counter(row["speed_mm_s"] for row in rows)
        self.assertEqual(counts, {
            "1.0": 10, "3.0": 10, "5.0": 10, "10.0": 10})
        self.assertEqual(len({row["scenario_id"] for row in rows}), 40)

        launch = ET.parse(LAUNCH).getroot()
        defaults = {arg.get("name"): arg.get("default")
                    for arg in launch.findall("arg")}
        forwarded = {arg.get("name"): arg.get("value")
                     for arg in launch.find("include").findall("arg")}
        self.assertEqual(defaults["linear_speed_mm_s"], "1.0")
        self.assertEqual(defaults["angular_speed_rad_s"], "0.0015")
        self.assertEqual(forwarded["linear_speed_mm_s"],
                         "$(arg linear_speed_mm_s)")
        self.assertEqual(forwarded["angular_speed_rad_s"],
                         "$(arg angular_speed_rad_s)")

        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("DEFORM_MONITOR_RUN_DIRECTORY", runner)
        self.assertIn("DEFORM_MONITOR_SKIP_BUILD", runner)

        batch = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("validate_pure_translation_config", batch)
        self.assertIn("detection_outcome_used_as_success_gate", batch)
        self.assertIn("while true", batch)
        self.assertIn("reset_failed_run", batch)
        self.assertNotIn("--resume-batch-dir", batch)
        self.assertNotIn("MAX_RETRIES", batch)

        preview = subprocess.check_output(
            ["bash", str(SCRIPT), "--print-launch-command", "6"],
            text=True)
        self.assertIn("linear_speed_mm_s:=5.0", preview)
        self.assertIn("gui:=false", preview)
        self.assertIn("rviz:=false", preview)


if __name__ == "__main__":
    unittest.main()

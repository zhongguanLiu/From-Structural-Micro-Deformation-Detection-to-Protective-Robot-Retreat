import importlib.util
import io
import json
import pathlib
import unittest
from collections import Counter

from deform_monitor_v3.msg import V1DetectionTile, V1DeformationSample, V1ReferenceSupportSample

path = pathlib.Path(__file__).resolve().parents[1] / "scripts/v1_real_recorder.py"
spec = importlib.util.spec_from_file_location("v1_real_recorder", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecorderTest(unittest.TestCase):
    def test_raw_invalid_estimates_and_nonfinite_values_are_preserved(self):
        sample = V1DeformationSample()
        sample.displacement_mean.x = 0.00001
        sample.validity = 3
        sample.minimum_detectable_effect_m = float("inf")
        encoded = module.plain(sample)
        self.assertEqual(encoded["displacement_mean"]["x"], .00001)
        self.assertEqual(encoded["validity"], 3)
        self.assertEqual(encoded["minimum_detectable_effect_m"], "inf")
        self.assertNotIn("components", encoded)
        json.dumps(encoded, allow_nan=False)

    def test_references_saved_once_and_snapshots_deduplicated_by_id(self):
        recorder = module.Recorder.__new__(module.Recorder)
        recorder.snapshot_ids, recorder.reference_epochs = set(), set()
        recorder.written = Counter()
        recorder.streams = {name: io.StringIO() for name in ("references", "snapshots")}
        tile = V1DetectionTile()
        tile.common.session_id = "test"
        tile.common.reference_epoch = 1
        tile.common.snapshot_id.low = 1
        tile.reference_support = [V1ReferenceSupportSample()]
        tile.physical_deformation = [V1DeformationSample()]
        receipt = {"receive_wall_elapsed_s": 1.0}
        recorder.process("snapshots", tile, receipt)
        recorder.process("snapshots", tile, receipt)
        tile.common.snapshot_id.low = 2
        recorder.process("snapshots", tile, receipt)
        self.assertEqual(recorder.written["references"], 1)
        self.assertEqual(recorder.written["snapshots"], 2)
        self.assertEqual(recorder.written["duplicate_snapshots"], 1)
        records = [json.loads(line) for line in recorder.streams["snapshots"].getvalue().splitlines()]
        self.assertEqual(len(records[0]["physical_deformation"]), 1)
        self.assertNotIn("reference_support", records[0])


if __name__ == "__main__":
    unittest.main()

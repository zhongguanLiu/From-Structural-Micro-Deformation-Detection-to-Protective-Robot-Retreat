import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from analyze_v1_real import frame_summary, sample_response, response_intervals, reference_index, anchor_counts, recording_window

CONFIG=dict(minimum_arrow_active_probability=.95,displacement_display_scale=50.,minimum_display_arrow_m=.05)


def sample(mm,prob=.95,gdu=1):
    return dict(displacement_mean=dict(x=mm/1000,y=0,z=0),active_probability_lower=prob,
                gdu_id=dict(high=0,low=gdu))


class ResponseTest(unittest.TestCase):
    def test_executed_window_not_full_bag_or_silent_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);(run/'metadata').mkdir()
            bag=dict(start=100.,end=167.88)
            self.assertEqual(recording_window(run,bag,[13.,60.]),(0.,67.88))
            (run/'metadata/execution.json').write_text(json.dumps(dict(playback_start_s=0,playback_duration_s=60)))
            self.assertEqual(recording_window(run,bag,[13.,59.9]),(0.,60.))
            with self.assertRaises(ValueError): recording_window(run,bag,[13.,60.1])
            with self.assertRaises(ValueError): recording_window(run,bag,[-1.,59.9])

    def test_anchor_population_matching_and_missing_not_unchanged(self):
        a=sample(2);a['position']=dict(x=1,y=0,z=0)
        b=sample(.5);b['position']=dict(x=2,y=0,z=0)
        support=[dict(position=dict(x=x,y=0,z=0),gdu_id=dict(high=0,low=1),
                      surfel_id=dict(high=0,low=x)) for x in (1,2,3)]
        index=reference_index(support)
        self.assertEqual(anchor_counts([a,b],index,CONFIG),dict(reference_anchors=3,
            anchors_with_estimates=2,anchors_without_estimates=1,responding_anchors=1))
        with self.assertRaises(ValueError):anchor_counts([a,a],index,CONFIG)
        b['position']['x']=99
        with self.assertRaises(ValueError):anchor_counts([b],index,CONFIG)
        with self.assertRaises(ValueError):reference_index(support+support[:1])

    def test_continuity_splits_zero_missing_time_and_reference_changes(self):
        rows=[dict(time_s=t,responding_units=n,session='a',epoch=e)
              for t,n,e in [(0,1,1),(1,1,1),(2,0,1),(3,1,1),(10,1,1),(11,1,2)]]
        groups=response_intervals(rows,2)
        self.assertEqual([(g['first_sample_s'],g['last_sample_s'],g['samples']) for g in groups],
                         [(0,1,2),(3,3,1),(10,10,1),(11,11,1)])

    def test_original_display_gates_and_unclipped_physical_units(self):
        self.assertTrue(sample_response(sample(1),CONFIG)[1])
        self.assertFalse(sample_response(sample(.999),CONFIG)[1])
        self.assertFalse(sample_response(sample(100,.949),CONFIG)[1])
        self.assertFalse(sample_response(sample(100,float('nan')),CONFIG)[1])
        self.assertEqual(sample_response(sample(200),CONFIG),(200.,True))

    def test_population_statistics_unique_units_and_empty_not_zero_displacement(self):
        row=frame_summary([sample(2),sample(4),sample(6,gdu=2)],CONFIG)
        self.assertEqual(row['median_mm'],4)
        self.assertEqual(row['maximum_mm'],6)
        self.assertEqual(row['responding_units'],2)
        self.assertEqual(row['responding_queries'],3)
        empty=frame_summary([sample(.1)],CONFIG)
        self.assertEqual(empty['responding_units'],0)
        self.assertTrue(np.isnan(empty['median_mm']))


if __name__=='__main__':
    unittest.main()

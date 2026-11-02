import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from analyze_evacuation import plot_keyframes, save_csv, paper_metrics
from analyze_evacuation import GridGeometry, footprint_cost, sustained_exit, route_metrics, require_completed_run, navigation_background, imu_yaw_check, available_at, phase_trajectories
from evacuation_timeline import event_chain, path_following, plot_chain, DEFORMATION_COLOR


class EvacuationAnalysisTests(unittest.TestCase):
    def test_paper_index_preserves_missing_and_distinguishes_clocks(self):
        result = dict(events={'NAVIGATING_OUTBOUND': 10.},
                      event_lineage=[dict(event='strict_detection', source_time_s=None, source_file='raw.csv'),
                                     dict(event='arrival', source_time_s=25., source_file='raw.csv')],
                      response_intervals=[dict(start_event='a', end_event='b', duration_sim_s=None,
                                               interpretation='unobserved', clock='source_time_s')],
                      executed=dict(length_m=4., exposure_m=1., inside_cost_distance_m=2.),
                      arrival_check=dict(slam_home_distance_m=.5, gazebo_home_distance_m=.6),
                      return_published_plan_count=3)
        records = paper_metrics(result)
        self.assertIsNone(records[0]['value'])
        self.assertEqual(records[1]['value'], 15.)
        self.assertEqual(records[1]['unit'], 'simulation s')
        self.assertIn('NOT', records[-1]['definition'])

    def test_companion_has_two_panels_same_trigger_origin_and_real_capture_cutoff(self):
        states = dict(NAVIGATING_OUTBOUND=10., MONITORING=12., DEFORMATION_DETECTED=16.,
                      RETURNING_HOME=17., COMPLETED=25.)
        stamps = dict(truth_running=13., first_active=14., first_retained=14.5,
                      first_trigger_map=14.5, persistence_source=15.5, persistence_decision=16.,
                      return_goal=20., first_return_path=20.1, arrival=25.)
        ledger = [dict(event=k, source_time_s=v, receive_time_s=v+.01) for k, v in stamps.items()]
        series = [dict(message_time_s=t, displacement_rms_m=.001*(t-12)) for t in [12., 14., 14.5, 16., 18.]]
        with tempfile.TemporaryDirectory() as directory:
            with patch('matplotlib.figure.Figure.savefig') as save, patch('matplotlib.pyplot.close'):
                plot_chain(Path(directory), ledger, series,
                           [16., 17., 20., 25., 26.], [90., 99., 50., 0., 88.], states)
                figure = plt.gcf()
                self.assertEqual(len(figure.axes), 2)
                self.assertEqual(figure.get_size_inches()[0], 3.5)
                np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [-4., -2., -1.5, 0.])
                np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [0., 2., 2.5, 4.])
                np.testing.assert_allclose(figure.axes[0].lines[1].get_xdata(), [-1.5, -.5])
                np.testing.assert_allclose(figure.axes[1].lines[0].get_xdata(), [1., np.nan, 4., np.nan, 9.])
                np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), [99., np.nan, 50., np.nan, 0.])
                self.assertEqual(figure.axes[1].lines[0].get_color(), DEFORMATION_COLOR)
                self.assertEqual([t.get_text() for ax in figure.axes for t in ax.texts], ['1.00 s'])
                self.assertTrue(all(ax.get_legend() is None for ax in figure.axes))
                self.assertEqual([Path(c.args[0]).name for c in save.call_args_list],
                                 ['fig_evacuation_response.pdf', 'fig_evacuation_response.png'])
            plt.close(figure)

    def test_companion_rejects_nonfinite_evidence_and_invalid_costs(self):
        states = dict(MONITORING=1., DEFORMATION_DETECTED=3., RETURNING_HOME=4., COMPLETED=6.)
        series = [dict(message_time_s=t, displacement_rms_m=.001) for t in [1., 2.]]
        with tempfile.TemporaryDirectory() as directory:
            for costs in ([0., 100.], [-1., 0.], [0., float('inf')], [float('nan')]*2):
                with self.assertRaises(ValueError):
                    plot_chain(Path(directory), [], series, [4., 5.], costs, states)
            series[0]['displacement_rms_m'] = float('nan')
            with self.assertRaises(ValueError):
                plot_chain(Path(directory), [], series, [4., 5.], [0., 0.], states)

    def test_return_panel_labels_trigger_but_keeps_complete_recorded_return(self):
        states = dict(NAVIGATING_OUTBOUND=1., STABILIZING_AT_TARGET=3., MONITORING=4.,
                      DEFORMATION_DETECTED=6., RETURNING_HOME=7., COMPLETED=9.)
        poses = [dict(message_time_s=t, receive_time_s=t, ros_time_s=t,
                      frame_id='map', x_m=t, y_m=0, yaw_rad=0) for t in range(1, 10)]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            (run / 'navigation').mkdir()
            grid = np.zeros((12, 12), dtype=np.int8)
            metadata = dict(frame_id='map', resolution=1., origin=[0., 0.], orientation=[0, 0, 0, 1])
            digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()+grid.tobytes()).hexdigest()
            np.savez(run / 'map.npz', data=grid, metadata=json.dumps(metadata))
            save_csv(run / 'navigation/global_costmaps.csv', [dict(
                message_time_s=1., ros_time_s=1., relative_path='map.npz', map_id=digest)])
            with patch('matplotlib.figure.Figure.savefig') as save, patch('matplotlib.pyplot.close'):
                records, _ = plot_keyframes(run, run, states,
                    [dict(event='truth_running', source_time_s=5.)], poses, [1., 0.])
                figure = plt.gcf()
                labels = [text.get_text() for text in figure.axes[3].texts]
                self.assertIn('$t=5.00$ s', labels)
                self.assertFalse(any('8.00' in text or 'Triggered at' in text for text in labels))
                self.assertEqual(save.call_count, 2)
            plt.close(figure)
            self.assertEqual(records[3]['event_sim_time_s'], 6.)
            self.assertEqual(records[3]['data_cutoff_sim_time_s'], 9.)
            self.assertEqual(records[3]['return_point_count'], 3)
            self.assertEqual(records[2]['return_point_count'], 0)
            # The paper's trigger panel must not borrow the completed return.
            stages = [('Start', 1., 1.), ('Observe', 3., 3.),
                      ('Retreat trigger', 6., 6.), ('Returned', 9., 9.)]
            with patch('matplotlib.figure.Figure.savefig'), patch('matplotlib.pyplot.close'):
                records, _ = plot_keyframes(run, run, states,
                    [dict(event='truth_running', source_time_s=5.)], poses, [1., 0.], stages=stages)
                figure = plt.gcf()
                self.assertIn('$t=5.00$ s', [t.get_text() for t in figure.axes[2].texts])
                self.assertIn('$t=8.00$ s', [t.get_text() for t in figure.axes[3].texts])
                for axis in figure.axes[:4]:
                    stamp, = [t for t in axis.texts if t.get_text().startswith('$t=')]
                    self.assertIsNone(stamp.get_bbox_patch())
            plt.close(figure)
            self.assertEqual([r['data_cutoff_sim_time_s'] for r in records], [1., 3., 6., 9.])
            self.assertEqual(records[2]['return_point_count'], 0)
            self.assertEqual(records[3]['return_point_count'], 3)
            self.assertTrue(all(r['map_receive_time_s'] <= r['data_cutoff_sim_time_s'] for r in records))

    def test_process_panels_preserve_outbound_without_future_return(self):
        states = dict(NAVIGATING_OUTBOUND=1, STABILIZING_AT_TARGET=3, RETURNING_HOME=7)
        rows = [dict(message_time_s=t, ros_time_s=t+.1) for t in [1, 2, 3, 4, 7, 8, 9]]
        before = phase_trajectories(rows, 1, states)
        self.assertEqual(before, ([], []))
        at_goal = phase_trajectories(rows, 3.2, states)
        during_motion = phase_trajectories(rows, 6, states)
        complete = phase_trajectories(rows, 9.2, states)
        self.assertEqual(at_goal, during_motion)
        self.assertEqual(at_goal[0], complete[0])
        self.assertEqual([r['message_time_s'] for r in complete[1]], [7, 8, 9])
        self.assertEqual(at_goal[1], [])

    def test_keyframe_rejects_future_source_and_delayed_receipt(self):
        old = dict(message_time_s='1', ros_time_s='2')
        delayed = dict(message_time_s='1', ros_time_s='2', receive_time_s='4')
        future = dict(message_time_s='4', ros_time_s='2')
        boundary = dict(message_time_s='3', ros_time_s='3', point_index='-1')
        self.assertEqual(available_at([old, delayed, future, boundary], 3), [old, boundary])
        self.assertEqual(available_at([old], .5), [])

    def test_path_following_uses_past_received_segments_not_future_vertices(self):
        plans = [dict(plan_id='P', point_index=i, x_m=x, y_m=0) for i, x in enumerate([0, 2, 2])]
        pubs = [dict(plan_id='P', receive_time_s=2., point_count=3),
                dict(plan_id='EMPTY', receive_time_s=4., point_count=0)]
        data = path_following(plans, pubs, [1., 2., 3., 4.],
                              np.array([[1, 1, 0]]*4), 0)
        self.assertIsNone(data[0]['distance_to_last_published_path_m'])
        self.assertEqual(data[1]['distance_to_last_published_path_m'], 1.)
        self.assertEqual(data[2]['path_receive_age_s'], 1.)
        self.assertIsNone(data[3]['distance_to_last_published_path_m'])

    def test_event_clock_lineage_and_publications_not_solver_count(self):
        states = dict(MONITORING=10., DEFORMATION_DETECTED=23., RETURNING_HOME=30., COMPLETED=40.)
        sample = dict(gdu_id='G1', message_time_s='12', receive_time_s='13',
                      source_snapshot_id='S1', source_accepted_frame_index='3',
                      active_now='True', retained_for_avoidance='True')
        maps = [dict(sample, active_source_ids='G10'), dict(sample, active_source_ids='G1|G2')]
        goal = dict(event='goal', message_time_s='30.1', ros_time_s='30.2', goal_id='goal')
        path = dict(message_time_s='30.3', ros_time_s='30.4', plan_id='P1',
                    point_index='0', point_count='1', frame_id='map', x_m='0', y_m='0')
        empty = dict(path, plan_id='P2', point_index='-1', point_count='0')
        stale = dict(path, message_time_s='29', plan_id='OLD')
        command = dict(ros_time_s='30.5', vx_mps='.1', vy_mps='0', wz_rad_s='0')
        turn_command = dict(ros_time_s='30.05', vx_mps='0', vy_mps='0', wz_rad_s='.15')
        args = (dict(return_trigger_algorithm_source_id='G1'), states,
                dict(freeze_time_s=24., frame_id='map'), [sample], [], maps, [goal],
                [stale, path, empty], [turn_command, command], [], 31., None)
        ledger, intervals, publications, _ = event_chain(*args, state_events=[
            dict(message_time_s='23', receive_time_s='23.2'),
            dict(message_time_s='40', receive_time_s='40.1')])
        events = {r['event']: r for r in ledger}
        self.assertEqual(events['persistence_decision']['source_time_s'], 23.)
        self.assertEqual(events['persistence_decision']['receive_time_s'], 23.2)
        self.assertEqual(events['arrival']['receive_time_s'], 40.1)
        self.assertIsNone(events['strict_detection']['source_time_s'])
        self.assertIsNone(events['first_motion_command']['source_time_s'])
        self.assertEqual(events['first_motion_command']['receive_time_s'], 30.5)
        self.assertEqual(events['first_return_control']['receive_time_s'], 30.05)
        self.assertEqual(events['first_trigger_map']['note'], 'Exact retained snapshot join: True')
        self.assertEqual([p['point_count'] for p in publications], [1, 0])
        self.assertEqual(events['first_return_path']['source_time_s'], 30.3)
        self.assertEqual(events['first_return_path']['relative_to_monitoring_s'], 20.3)
        self.assertIsNone(events['first_sustained_exit']['source_time_s'])
        self.assertEqual(next(r for r in intervals if r['start_event']=='persistence_decision')['duration_sim_s'], 7)
        path['point_count'] = '2'
        with self.assertRaises(ValueError):
            event_chain(*args)

    def test_gyro_integration_detects_static_imu_during_rotation(self):
        imu = [dict(message_time_s=t, wz_rad_s=1.) for t in (0, .5, 1.)]
        truth = [dict(message_time_s=t, model_yaw_rad=t) for t in (0, .5, 1.)]
        self.assertAlmostEqual(imu_yaw_check(imu, truth, 0, 1)["yaw_change_rmse_rad"], 0)
        for row in imu:
            row["wz_rad_s"] = 0
        self.assertGreater(imu_yaw_check(imu, truth, 0, 1)["yaw_change_rmse_rad"], .5)
        imu[1]["message_time_s"] = 0
        with self.assertRaises(ValueError):
            imu_yaw_check(imu, truth, 0, 1)

    def test_live_map_growth_preserves_common_domain_and_unknown(self):
        source = np.zeros((8, 10), dtype=int)
        source[3, 4] = -1
        source[4, 5] = 100
        metadata = dict(frame_id="map", resolution=.15, origin=[-.3, -.15])
        target = GridGeometry(.15, 4, 4, 0, 0)
        np.testing.assert_array_equal(
            navigation_background(source, metadata, target, "map"), source[1:5, 2:6])
        with self.assertRaises(ValueError):
            navigation_background(source, metadata, target, "other")
        with self.assertRaises(ValueError):
            navigation_background(source[:2, :2], metadata, target, "map")
        metadata["origin"] = [-.31, -.15]
        with self.assertRaises(ValueError):
            navigation_background(source, metadata, target, "map")

    def test_refuse_failed_incomplete_or_intervened_runs(self):
        for outcome, contract, intervened in (("FAILED", "COMPLETED", False),
                                             ("COMPLETED", "FAILED", False),
                                             ("COMPLETED", "COMPLETED", True)):
            with self.assertRaises(ValueError):
                require_completed_run({"outcome": outcome},
                                      {"final_output_contract_status": contract}, intervened)
        require_completed_run({"outcome": "COMPLETED"},
                              {"final_output_contract_status": "COMPLETED"}, False)

    def test_whole_footprint_not_center_and_unknown_bounds(self):
        grid = np.zeros((10, 10), dtype=int)
        grid[5, 6] = 99
        geometry = GridGeometry(1, 10, 10, 0, 0)
        self.assertEqual(footprint_cost(grid, geometry, (5.5, 5.5, 0), 0.6, 0.3), 99)
        self.assertEqual(footprint_cost(grid, geometry, (5, 5, 0), 0.3, 0.3), 0)
        self.assertTrue(np.isnan(footprint_cost(grid, geometry, (0, 0, 0), 0.3, 0.3)))

    def test_exit_requires_duration_and_no_gap(self):
        times = np.arange(0, 2.1, 0.25)
        self.assertEqual(sustained_exit(times, [99, 99, 0, 0, 0, 0, 0, 0, 0], 1), 0.5)
        self.assertIsNone(sustained_exit([0, 0.1, 2, 2.1], [0, 0, 0, 0], 1))
        self.assertIsNone(sustained_exit([0, 0.5, 1], [99, 99, 99], 99))

    def test_identical_exposure_scale(self):
        record = route_metrics(np.array([[0, 0], [3, 4]]), np.array([99., 99.]))
        self.assertEqual(record["length_m"], 5)
        self.assertEqual(record["exposure_m"], 5)
        self.assertEqual(record["inside_cost_distance_m"], 5)


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scene_check', ROOT/'scripts/icra2027_scene_check.py')
scene_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene_check)


class SceneCheckTest(unittest.TestCase):
    def test_world_plugin_ready_precedes_runtime_pose_queries(self):
        source = (ROOT/'scripts/icra2027_scene_check.py').read_text()
        self.assertLess(source.index('rospy.wait_for_service(ready_service'),
                        source.index("services = {}"))

    def test_saved_state_wins_without_rewriting_world(self):
        world = b'''<sdf><world><model name="m06"><pose>1 2 3 0 0 0</pose><static>0</static>
        <link name="link"><pose>0 0 1 0 0 0</pose></link></model>
        <model name="table_marble_clone"><static>1</static><pose>4 5 6 0 0 0</pose></model>
        <state><model name="m06"><pose>7 8 9 0 0 0</pose>
        <link name="link"><pose>7 8 10 0 0 0</pose></link></model></state></world></sdf>'''
        models = scene_check.saved_scene(world)
        self.assertEqual(models['m06']['pose'], [7, 8, 9, 0, 0, 0])
        np.testing.assert_allclose(models['m06']['links']['link'][:3, 3], [7, 8, 10])
        self.assertTrue(models['table_marble_clone']['static'])
        with self.assertRaises(ValueError):
            scene_check.saved_scene(world.replace(b'<static>1</static>', b'<static>0</static>'))

    def test_runtime_pose_tolerance_and_quaternion_sign(self):
        pose = SimpleNamespace(position=SimpleNamespace(x=1., y=2., z=3.),
                               orientation=SimpleNamespace(x=0., y=0., z=0., w=-1.))
        expected = scene_check.pose_matrix([1, 2, 3, 0, 0, 0])
        self.assertEqual(scene_check.check_pose(expected, pose)['position_error_m'], 0.)
        pose.position.x += .001
        with self.assertRaises(ValueError):
            scene_check.check_pose(expected, pose)

    def test_current_scene_contains_static_new_prop_and_saved_m06(self):
        models = scene_check.saved_scene((ROOT/'worlds/ICRA2027.world').read_bytes())
        self.assertEqual(models['m06']['pose_source'], 'saved_state')
        self.assertTrue(models['table_marble_clone']['static'])


if __name__ == '__main__':
    unittest.main()

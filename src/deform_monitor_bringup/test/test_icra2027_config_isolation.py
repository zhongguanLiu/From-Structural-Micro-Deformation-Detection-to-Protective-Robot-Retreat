#!/usr/bin/env python3

import os
import unittest
import xml.etree.ElementTree as ET

import yaml


PACKAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Icra2027ConfigIsolationTest(unittest.TestCase):
    def test_global_local_observations_and_nonholonomic_window_agree(self):
        layers = []
        maps = []
        for name in ("global", "local"):
            with open(os.path.join(PACKAGE, "config", "icra2027_" + name + "_costmap.yaml")) as stream:
                layer = yaml.safe_load(stream)[name + "_costmap"]
            self.assertIn("obstacle_layer", [p["name"] for p in layer["plugins"]])
            self.assertFalse(layer["static_layer"]["trinary_costmap"])
            self.assertEqual(layer["static_layer"]["lethal_cost_threshold"], 100)
            layers.append(layer["obstacle_layer"])
            maps.append(layer)
        self.assertEqual(*layers)
        self.assertEqual(maps[0]["footprint_padding"], maps[1]["footprint_padding"])
        self.assertEqual(maps[0]["inflation_layer"], maps[1]["inflation_layer"])
        self.assertEqual(maps[1]["resolution"], 0.15)
        with open(os.path.join(PACKAGE, "config", "icra2027_move_base.yaml")) as stream:
            config = yaml.safe_load(stream)
        dwa = config["DWAPlannerROS"]
        self.assertEqual((dwa["min_vel_y"], dwa["max_vel_y"], dwa["vy_samples"]), (0, 0, 1))
        self.assertEqual(dwa["min_vel_x"], -0.30)
        self.assertGreater(dwa["twirling_scale"], 0)
        self.assertLess(dwa["min_vel_trans"], dwa["acc_lim_x"] / config["controller_frequency"])
        self.assertLess(dwa["min_vel_theta"], dwa["acc_lim_theta"] / config["controller_frequency"])
        self.assertGreater(dwa["sim_time"], dwa["max_vel_trans"] / dwa["acc_lim_x"]
                           + dwa["stop_time_buffer"])

    def test_truth_metadata_is_recorder_only(self):
        with open(os.path.join(
                PACKAGE, "config", "icra2027_closed_loop.yaml"),
                encoding="utf-8") as stream:
            mission = yaml.safe_load(stream)["mission"]
        with open(os.path.join(
                PACKAGE, "config", "icra2027_evaluation.yaml"),
                encoding="utf-8") as stream:
            evaluation = yaml.safe_load(stream)["mission"]

        forbidden_online_keys = {
            "return_trigger_model",
            "monitor_regions_world",
            "monitor_region_labels",
            "stable_support_regions_world",
            "monitor_roi_center_world_xy",
            "target_model",
            "required_detection_models",
            "model_association_tolerance_m",
            "truth_model_to_fastlio_body_xyzyaw",
            "truth_model_to_navigation_base_xyzyaw",
            "motion_truth_topic",
            "approach_world_xy",
        }
        self.assertFalse(forbidden_online_keys.intersection(mission))
        self.assertEqual(set(evaluation), {
            "required_detection_models",
            "model_association_tolerance_m",
            "truth_model_to_fastlio_body_xyzyaw",
        })
        self.assertEqual(evaluation["required_detection_models"],
                         ["m01", "m02", "m03", "m05", "m06"])

        launch = ET.parse(os.path.join(
            PACKAGE, "launch", "icra2027_closed_loop.launch")).getroot()
        mission_node = next(node for node in launch.findall("node")
                            if node.get("name") ==
                            "icra2027_closed_loop_mission")
        recorder_node = next(node for node in launch.findall("node")
                             if node.get("name") ==
                             "icra2027_experiment_recorder")
        mission_files = [item.get("file") for item in
                         mission_node.findall("rosparam")]
        recorder_files = [item.get("file") for item in
                          recorder_node.findall("rosparam")]
        self.assertEqual(mission_files, ["$(arg mission_config)"])
        self.assertEqual(recorder_files, [
            "$(arg mission_config)", "$(arg evaluation_config)"])
        for node in launch.findall("node"):
            if node is recorder_node:
                continue
            self.assertNotIn("$(arg evaluation_config)", [
                item.get("file") for item in node.findall("rosparam")])

        with open(os.path.join(
                PACKAGE, "scripts", "icra2027_closed_loop_mission.py"),
                encoding="utf-8") as stream:
            source = stream.read()
        for forbidden in (
                "self.motion_truth_topic",
                "self.truth_models",
                "self.return_min_model_distance_m",
                "def _motion_truth_callback",
                'Subscriber(self.motion_truth_topic',
                'Subscriber("/ground_truth/odom"',
                "self.truth_odom",
                "truth_model_to_navigation_base_xyzyaw",
                "required_robot_truth_stamp_s"):
            self.assertNotIn(forbidden, source)
        status_callback = source[
            source.index("    def _motion_status_callback(self, message):"):
            source.index("    def _evaluation_status_callback", source.index(
                "    def _motion_status_callback(self, message):"))]
        self.assertIn('"experiment_elapsed", float("nan")', status_callback)
        self.assertIn("self.motion_experiment_elapsed_s = experiment_elapsed",
                      status_callback)

        recorder_callback = source[
            source.index("    def _recorder_status_callback(self, message):"):
            source.index("    def _map_callback", source.index(
                "    def _recorder_status_callback(self, message):"))]
        for required in (
                '"finalized"', '"finalization_error"',
                '"required_artifact_errors"', '"finalization_request_id"',
                '"run_manifest"'):
            self.assertIn(required, recorder_callback)
        for forbidden in (
                '"truth_row_count"', '"latest_truth_sim_time_s"',
                '"recorded_high_water"', '"robot_truth_stamp_s"'):
            self.assertNotIn(forbidden, recorder_callback)

        completion = source[
            source.index("    def _full_process_completion_ready(self):"):
            source.index("    def _wait_for_recorder_finalization", source.index(
                "    def _full_process_completion_ready(self):"))]
        self.assertIn('motion_state == "COMPLETED"', completion)
        self.assertNotIn("motion_truth_state", completion)
        self.assertNotIn("truth_models", completion)
        self.assertNotIn("recorder_timeline_status", completion)
        self.assertNotIn("completed_checkpoints_s", completion)

    def test_stationary_wrapper_forwards_evaluation_config(self):
        launch = ET.parse(os.path.join(
            PACKAGE, "launch",
            "icra2027_stationary_full_process.launch")).getroot()
        include = launch.find("include")
        self.assertIsNotNone(include)
        forwarded = {item.get("name"): item.get("value")
                     for item in include.findall("arg")}
        self.assertEqual(forwarded.get("evaluation_config"),
                         "$(arg evaluation_config)")

    def test_object_truth_topic_has_only_the_recorder_consumer(self):
        consumers = []
        scripts = os.path.join(PACKAGE, "scripts")
        for name in sorted(os.listdir(scripts)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(scripts, name)
            with open(path, encoding="utf-8") as stream:
                if 'Subscriber("/icra2027_motion/truth"' in stream.read():
                    consumers.append(name)
        self.assertEqual(consumers, ["icra2027_experiment_recorder.py"])

    def test_robot_truth_topic_has_only_the_recorder_consumer(self):
        consumers = []
        scripts = os.path.join(PACKAGE, "scripts")
        for name in sorted(os.listdir(scripts)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(scripts, name)
            with open(path, encoding="utf-8") as stream:
                if 'Subscriber("/ground_truth/odom"' in stream.read():
                    consumers.append(name)
        self.assertEqual(consumers, ["icra2027_experiment_recorder.py"])

    def test_online_chain_has_no_object_truth_interface(self):
        online_scripts = (
            "icra2027_closed_loop_mission.py",
            "closed_loop_core.py",
            "planning_core.py",
            "lio_twist_odometry.py",
            "stationary_monitor_cloud_adapter.py",
            "static_planning_evaluation.py",
            "mission_map_proxy.py",
        )
        forbidden_interfaces = (
            '"m01"', '"m02"', '"m03"', '"m04"', '"m05"', '"m06"',
            "laser_retro", "aabb_min", "aabb_max",
            "/icra2027_motion/truth", "/ground_truth/odom",
            "/gazebo/model_states", "/gazebo/link_states",
            "monitor_regions_world", "monitor_region_labels",
            "required_detection_models", "model_association_tolerance_m",
            "truth_model_to_fastlio_body_xyzyaw",
            "truth_model_to_navigation_base_xyzyaw",
        )
        for name in online_scripts:
            with open(os.path.join(PACKAGE, "scripts", name),
                      encoding="utf-8") as stream:
                source = stream.read().lower()
            for forbidden in forbidden_interfaces:
                self.assertNotIn(forbidden.lower(), source,
                                 msg="{} leaks {}".format(name, forbidden))

    def test_detector_evolution_and_avoidance_have_no_truth_interface(self):
        workspace = os.path.abspath(os.path.join(PACKAGE, ".."))
        sources = []
        roots = (
            os.path.join(workspace, "deform_monitor_v3", "src", "v1"),
            os.path.join(workspace, "deform_monitor_v3", "include",
                         "deform_monitor_v3", "v1"),
            os.path.join(workspace, "deform_evolution_map", "scripts"),
            os.path.join(workspace, "deform_evolution_map", "src",
                         "deform_evolution_map"),
            os.path.join(workspace, "deform_avoidance_map", "scripts"),
            os.path.join(workspace, "deform_avoidance_map", "src",
                         "deform_avoidance_map"),
        )
        for root in roots:
            for directory, _, names in os.walk(root):
                for name in names:
                    if name.endswith((".py", ".cpp", ".hpp")):
                        sources.append(os.path.join(directory, name))
        forbidden = (
            '"m01"', '"m02"', '"m03"', '"m04"', '"m05"', '"m06"',
            "/icra2027_motion/truth", "/ground_truth/odom",
            "/gazebo/model_states", "/gazebo/link_states",
            "monitor_regions_world", "monitor_region_labels",
            "required_detection_models", "model_association_tolerance_m",
            "truth_model_to_fastlio_body_xyzyaw",
            "truth_model_to_navigation_base_xyzyaw",
            "aabb_min", "aabb_max", "getlaserretro",
        )
        for path in sources:
            with open(path, encoding="utf-8") as stream:
                source = stream.read().lower()
            for token in forbidden:
                self.assertNotIn(token, source,
                                 msg="{} leaks {}".format(path, token))

        # XYZI is a transport contract only.  The sole laser_retro mention in
        # V1 documents the fail-closed normalization at the algorithm boundary.
        deterministic = os.path.join(
            workspace, "deform_monitor_v3", "src", "v1",
            "deterministic.cpp")
        with open(deterministic, encoding="utf-8") as stream:
            source = stream.read()
        self.assertEqual(source.lower().count("laser_retro"), 1)
        self.assertIn("point.intensity = 0.0F;", source)

    def test_motion_status_is_model_agnostic_and_truth_branch_is_conditional(self):
        workspace = os.path.abspath(os.path.join(PACKAGE, ".."))
        gazebo_package = os.path.join(
            workspace, "gazebo_test", "Mid360_simulation_plugin",
            "livox_laser_simulation")
        plugin_path = os.path.join(
            gazebo_package, "src", "icra2027_motion_world_plugin.cpp")
        with open(plugin_path, encoding="utf-8") as stream:
            plugin = stream.read()
        status = plugin[
            plugin.index("  std::string BuildStatusJson() const"):
            plugin.index("  void PublishStatus(", plugin.index(
                "  std::string BuildStatusJson() const"))]
        self.assertIn(r'\"experiment_elapsed\"', status)
        self.assertIn(r'\"motion_elapsed\"', status)
        for forbidden in (
                r'\"models\"', r'\"model_name\"', r'\"aabb_',
                r'\"laser_retro', r'\"truth_topic\"',
                r'\"sequence_order_ids\"'):
            self.assertNotIn(forbidden, status)

        simulation_launch = ET.parse(os.path.join(
            gazebo_package, "launch", "icra2027_simulation.launch")).getroot()
        arguments = {item.get("name"): item.get("default")
                     for item in simulation_launch.findall("arg")}
        self.assertEqual(arguments.get("record_truth"), "true")
        truth_recorders = [
            node for node in simulation_launch.findall("node")
            if node.get("type") == "icra2027_truth_recorder.py"]
        self.assertEqual(len(truth_recorders), 1)
        self.assertEqual(truth_recorders[0].get("if"), "$(arg record_truth)")

        closed_launch = ET.parse(os.path.join(
            PACKAGE, "launch", "icra2027_closed_loop.launch")).getroot()
        closed_arguments = {item.get("name"): item.get("default")
                            for item in closed_launch.findall("arg")}
        self.assertEqual(closed_arguments.get("record_truth"), "false")
        simulation_include = next(
            item for item in closed_launch.findall("include")
            if item.get("file").endswith("/icra2027_simulation.launch"))
        forwarded = {item.get("name"): item.get("value")
                     for item in simulation_include.findall("arg")}
        self.assertEqual(forwarded.get("record_truth"), "$(arg record_truth)")

        with open(os.path.join(
                PACKAGE, "scripts", "icra2027_closed_loop_mission.py"),
                encoding="utf-8") as stream:
            mission = stream.read()
        callback = mission[
            mission.index("    def _motion_status_callback(self, message):"):
            mission.index("    def _evaluation_status_callback", mission.index(
                "    def _motion_status_callback(self, message):"))]
        for forbidden in (
                '"models"', '"model_name"', '"aabb_', '"laser_retro',
                '"sequence_order_ids"', '"translation"'):
            self.assertNotIn(forbidden, callback)

    def test_stationary_wrapper_disables_only_move_base(self):
        closed_launch = ET.parse(os.path.join(
            PACKAGE, "launch", "icra2027_closed_loop.launch")).getroot()
        arguments = {item.get("name"): item.get("default")
                     for item in closed_launch.findall("arg")}
        self.assertEqual(arguments.get("enable_navigation_stack"), "true")

        move_base = next(node for node in closed_launch.findall("node")
                         if node.get("name") == "move_base")
        self.assertEqual(move_base.get("if"),
                         "$(arg enable_navigation_stack)")

        # These products are required by the stationary experiment even though
        # it never sends a navigation goal.
        for node_name in ("mission_octomap", "mission_map_proxy",
                          "lio_twist_odometry"):
            node = next(item for item in closed_launch.findall("node")
                        if item.get("name") == node_name)
            self.assertNotEqual(node.get("if"),
                                "$(arg enable_navigation_stack)")

        stationary_launch = ET.parse(os.path.join(
            PACKAGE, "launch",
            "icra2027_stationary_full_process.launch")).getroot()
        forwarded = {item.get("name"): item.get("value")
                     for item in stationary_launch.find("include").findall("arg")}
        self.assertEqual(forwarded.get("enable_navigation_stack"), "false")

    def test_interactive_rviz_defaults_are_bounded_without_thinning_detector(self):
        closed_launch = ET.parse(os.path.join(
            PACKAGE, "launch", "icra2027_closed_loop.launch")).getroot()
        fast_lio_include = next(
            item for item in closed_launch.findall("include")
            if item.get("file").endswith("/mapping_mid360.launch"))
        fast_lio_arguments = {
            item.get("name"): item.get("value")
            for item in fast_lio_include.findall("arg")}
        self.assertEqual(fast_lio_arguments.get("point_filter_num"), "1")
        self.assertEqual(fast_lio_arguments.get("cloud_output_point_stride"),
                         "1")

        with open(os.path.join(
                PACKAGE, "config", "rviz", "icra2027_closed_loop.rviz"),
                encoding="utf-8") as stream:
            rviz = yaml.safe_load(stream)
        manager = rviz["Visualization Manager"]
        self.assertEqual(manager["Global Options"]["Frame Rate"], 10)
        displays = manager["Displays"]
        point_cloud = next(item for item in displays
                           if item.get("Name") ==
                           "FAST-LIO registered cloud")
        self.assertEqual(point_cloud["Topic"], "/cloud_registered")
        self.assertEqual(point_cloud["Queue Size"], 1)
        self.assertEqual(point_cloud["Decay Time"], 0)
        # Rendering is bounded only at the UI.  The online detector continues
        # to consume the complete /monitor/cloud_registered stream.
        self.assertNotEqual(point_cloud["Topic"],
                            "/monitor/cloud_registered")
        paths = [item for item in displays
                 if item.get("Class") == "rviz/Path"]
        self.assertTrue(paths)
        self.assertTrue(all(item["Queue Size"] == 1 for item in paths))
        tf_display = next(item for item in displays
                          if item.get("Class") == "rviz/TF")
        self.assertFalse(tf_display["Enabled"])
        self.assertFalse(tf_display["Value"])
        robot_model = next(item for item in displays
                           if item.get("Class") == "rviz/RobotModel")
        self.assertEqual(robot_model["Update Interval"], 0.1)


if __name__ == "__main__":
    unittest.main()

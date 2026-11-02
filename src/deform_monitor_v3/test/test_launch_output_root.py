from __future__ import annotations

import pathlib
import unittest
import xml.etree.ElementTree as ET


LAUNCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "launch"
    / "legacy"
    / "deform_monitor_v3_sim.launch"
)
DYNAMIC_LAUNCH_PATH = LAUNCH_PATH.with_name("deform_monitor_v3_sim_dynamic.launch")
CANONICAL_LAUNCH_PATH = LAUNCH_PATH.parents[1] / "simulation.launch"


class DeformMonitorSimLaunchTests(unittest.TestCase):
    def test_real_launch_only_adds_monitor_and_detection_rviz(self):
        root = ET.parse(CANONICAL_LAUNCH_PATH.with_name("real.launch")).getroot()
        self.assertEqual([n.attrib["file"] for n in root.findall("include")],
                         ["$(find deform_monitor_v3)/launch/monitor_v1.launch"])
        self.assertEqual([(n.attrib["pkg"], n.attrib["type"])
                          for n in root.findall("node")],
                         [("deform_monitor_v3", "v1_real_recorder.py"), ("rviz", "rviz")])
        self.assertEqual(root.find("arg[@name='record']").attrib["default"], "false")
        self.assertEqual(root.find("node[@type='v1_real_recorder.py']").attrib["if"], "$(arg record)")
        self.assertIn("/config/rviz/v1_simulation.rviz", root.find("node[@pkg='rviz']").attrib["args"])
        self.assertEqual(root.find("arg[@name='use_sim_time']").attrib["default"], "true")
        self.assertEqual(root.find("param[@name='/use_sim_time']").attrib["value"],
                         "$(arg use_sim_time)")
        args = {n.attrib["name"]: n.attrib["value"] for n in root.find("include").findall("arg")}
        self.assertEqual(args["upstream_registered_cloud_deskew_verified"], "true")
        self.assertEqual(args["parameter_overrides"], "$(arg real_parameters)")
        self.assertEqual(args["use_fastlio_extrinsic"], "true")
        self.assertEqual(args["minimum_arrow_active_probability"],
                         "$(arg minimum_arrow_active_probability)")
        monitor = ET.parse(CANONICAL_LAUNCH_PATH.with_name("monitor_v1.launch")).getroot()
        self.assertEqual(monitor.find("arg[@name='parameter_overrides']").attrib["default"], "")
        self.assertEqual(monitor.find("arg[@name='use_fastlio_extrinsic']").attrib["default"], "false")
        self.assertEqual(monitor.find("arg[@name='minimum_arrow_active_probability']").attrib["default"], "0.0")
        loads = monitor.find("node").findall("rosparam")
        self.assertEqual(loads[-1].attrib["file"], "$(arg parameter_overrides)")

    def test_canonical_launch_composes_gazebo_fast_lio_and_v1(self):
        root = ET.parse(CANONICAL_LAUNCH_PATH).getroot()
        include_files = {
            include.attrib.get("file") for include in root.findall("include")
        }
        self.assertIn(
            "$(find livox_laser_simulation)/launch/synchronized_multi_model_motion.launch",
            include_files,
        )
        self.assertIn(
            "$(find fast_lio)/launch/mapping_mid360.launch",
            include_files,
        )
        self.assertIn(
            "$(find deform_monitor_v3)/launch/monitor_v1.launch",
            include_files,
        )

    def test_canonical_launch_never_starts_an_implicit_rosbag_recorder(self):
        root = ET.parse(CANONICAL_LAUNCH_PATH).getroot()
        arg_names = {element.attrib["name"] for element in root.findall("arg")}
        self.assertNotIn("record_bag", arg_names)
        self.assertNotIn("bag_prefix", arg_names)
        self.assertNotIn("output_root", arg_names)
        for node in root.findall("node"):
            self.assertNotEqual(node.attrib.get("pkg"), "rosbag")
            self.assertNotEqual(node.attrib.get("type"), "record")
            self.assertNotEqual(node.attrib.get("name"), "deform_v1_sim_bag_recorder")

    def test_canonical_launch_verifies_only_the_known_simulation_deskew_chain(self):
        root = ET.parse(CANONICAL_LAUNCH_PATH).getroot()
        monitor_include = next(
            include for include in root.findall("include")
            if include.attrib.get("file") ==
            "$(find deform_monitor_v3)/launch/monitor_v1.launch"
        )
        forwarded = {
            arg.attrib.get("name"): arg.attrib.get("value")
            for arg in monitor_include.findall("arg")
        }
        self.assertEqual(
            forwarded.get("upstream_registered_cloud_deskew_verified"), "true"
        )

    def test_launch_exposes_sim_experiment_output_root_arg_and_wires_recorder_param(self):
        tree = ET.parse(LAUNCH_PATH)
        root = tree.getroot()

        args = {elem.attrib.get("name"): elem.attrib for elem in root.findall("arg")}
        arg_names = list(args)
        self.assertIn("sim_experiment_output_root", arg_names)

        recorder_node = None
        for node in root.findall("node"):
            if node.attrib.get("name") == "sim_experiment_recorder":
                recorder_node = node
                break

        self.assertIsNotNone(recorder_node)
        param_pairs = {
            elem.attrib.get("name"): elem.attrib.get("value")
            for elem in recorder_node.findall("param")
        }
        self.assertEqual(
            param_pairs.get("output_root"),
            "$(arg sim_experiment_output_root)",
        )
        self.assertNotIn(
            "/src/output",
            args["sim_experiment_output_root"].get("default", ""),
        )

    def test_launch_exposes_final_alarm_risk_threshold_as_config_override(self):
        tree = ET.parse(LAUNCH_PATH)
        root = tree.getroot()

        args = {elem.attrib.get("name"): elem.attrib for elem in root.findall("arg")}
        self.assertIn("alarm_mean_risk_threshold", args)
        self.assertEqual(args["alarm_mean_risk_threshold"].get("default"), "0.55")

        monitor_node = next(
            node for node in root.findall("node")
            if node.attrib.get("name") == "deform_monitor_v3"
        )
        params = {
            elem.attrib.get("name"): elem.attrib.get("value")
            for elem in monitor_node.findall("param")
        }
        self.assertEqual(
            params.get("deform_monitor/persistent_risk/min_confirmed_mean_risk"),
            "$(arg alarm_mean_risk_threshold)",
        )

    def test_dynamic_launch_uses_the_same_final_alarm_threshold_override(self):
        tree = ET.parse(DYNAMIC_LAUNCH_PATH)
        root = tree.getroot()

        args = {elem.attrib.get("name"): elem.attrib for elem in root.findall("arg")}
        self.assertEqual(args["alarm_mean_risk_threshold"].get("default"), "0.55")
        monitor_node = next(
            node for node in root.findall("node")
            if node.attrib.get("name") == "deform_monitor_v3"
        )
        params = {
            elem.attrib.get("name"): elem.attrib.get("value")
            for elem in monitor_node.findall("param")
        }
        self.assertEqual(
            params.get("deform_monitor/persistent_risk/min_confirmed_mean_risk"),
            "$(arg alarm_mean_risk_threshold)",
        )


if __name__ == "__main__":
    unittest.main()

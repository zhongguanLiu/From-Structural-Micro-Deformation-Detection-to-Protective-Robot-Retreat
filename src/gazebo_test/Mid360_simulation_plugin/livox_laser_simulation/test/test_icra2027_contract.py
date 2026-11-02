#!/usr/bin/env python3

import csv
import importlib.util
import math
import pathlib
import unittest
import xml.etree.ElementTree as ET

import yaml


PACKAGE = pathlib.Path(__file__).resolve().parents[1]
WORLD_PATH = PACKAGE / "worlds" / "ICRA2027.world"
CONFIG_PATH = PACKAGE / "config" / "icra2027_motion.yaml"
STATIONARY_CONFIG_PATH = (
    PACKAGE / "config" / "icra2027_stationary_full_process_motion.yaml")
LAUNCH_PATH = PACKAGE / "launch" / "icra2027_simulation.launch"
GENERIC_LAUNCH_PATH = PACKAGE / "launch" / "mid360_fastlio.launch"
SIM_CLOCK_JOINT_PUBLISHER = (
    PACKAGE / "scripts" / "sim_clock_joint_state_publisher.py")
A2_PACKAGE = PACKAGE.parent / "a2_description"
A2_UPSTREAM_URDF = A2_PACKAGE / "urdf" / "a2.urdf"
A2_SIM_URDF = A2_PACKAGE / "urdf" / "a2_mid360.urdf.xacro"
MID360_PATTERN = PACKAGE / "scan_mode" / "mid360-real-centr.csv"
MOTION_PLUGIN = PACKAGE / "src" / "icra2027_motion_world_plugin.cpp"
FAST_LIO_CONFIG = PACKAGE.parents[2] / "FAST_LIO" / "config" / "mid360.yaml"


class Icra2027ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = ET.parse(str(WORLD_PATH)).getroot().find("world")
        cls.models = {model.get("name"): model for model in cls.world.findall("model")}
        with CONFIG_PATH.open("r", encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)

    def test_collision_retro_metadata_is_well_formed_but_not_a_scene_lock(self):
        """Do not turn saved-world metadata into a detector oracle.

        The user owns ICRA2027.world and may add, remove or rearrange props.
        ``laser_retro`` only has to be parseable and internally consistent for
        one model; global counts, fixed prop names and globally unique values
        are deliberately not an online sensing contract.  The deformation
        algorithm has intensity likelihood disabled and cannot consume these
        values as identities.
        """
        ids_by_model = {}
        for name, model in self.models.items():
            collisions = model.findall(".//collision")
            if not collisions:
                continue
            raw_values = [collision.findtext("laser_retro")
                          for collision in collisions]
            # New static props need no evaluation ID. Only the six controlled
            # fixtures require explicit metadata for the recorder branch.
            if name in {'m01', 'm02', 'm03', 'm04', 'm05', 'm06'}:
                self.assertTrue(all(value is not None for value in raw_values),
                                "{} has collision without laser_retro".format(name))
            values = {int(float(value or 0)) for value in raw_values}
            self.assertEqual(len(values), 1, "{} has inconsistent collision IDs".format(name))
            ids_by_model[name] = values.pop()
        self.assertTrue(ids_by_model)
        for index in range(1, 7):
            self.assertIn("m0{}".format(index), ids_by_model)

    def test_controlled_models_are_kinematic_and_plugin_owned(self):
        for index in range(1, 7):
            name = "m0{}".format(index)
            model = self.models[name]
            self.assertEqual(model.findtext("static"), "0")
            self.assertEqual(model.findtext("link/kinematic"), "1")
        plugin = self.world.find("plugin[@name='icra2027_synchronized_motion']")
        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.get("filename"), "libicra2027_motion_world_plugin.so")

        plugin_source = (PACKAGE / "src" / "icra2027_motion_world_plugin.cpp").read_text(
            encoding="utf-8")
        self.assertIn("handle.driveLink->SetWorldPose", plugin_source)
        self.assertIn("driveLink->WorldPose()", plugin_source)
        self.assertIn("if (!handle.driveLink->GetKinematic())\n        handle.driveLink->SetKinematic(true);", plugin_source)
        self.assertIn("PhysicsOwnershipValid()", plugin_source)

    def test_controlled_and_decorative_saved_poses_are_well_formed(self):
        # Validate the user's currently saved scene without restoring any old
        # hard-coded coordinates.  A saved Gazebo <state> is a runtime snapshot
        # and is allowed to differ from the model's authored spawn pose.
        for index in range(1, 7):
            name = "m0{}".format(index)
            authored = [float(value) for value in
                        self.models[name].findtext("pose").split()]
            saved = self.world.find("state/model[@name='{}']".format(name))
            self.assertIsNotNone(saved)
            saved_pose = [float(value) for value in saved.findtext("pose").split()]
            self.assertEqual(len(authored), 6)
            self.assertEqual(len(saved_pose), 6)
            self.assertTrue(all(math.isfinite(value) for value in authored))
            self.assertTrue(all(math.isfinite(value) for value in saved_pose))

        decorative_names = sorted(
            name for name in self.models if name.startswith("wooden_case_clone"))
        self.assertTrue(decorative_names)
        controlled_names = {
            entry["model_name"] for entry in self.config["models"]
        }
        for name in decorative_names:
            self.assertEqual(self.models[name].findtext("static"), "1")
            self.assertNotIn(name, controlled_names)
            saved = self.world.find("state/model[@name='{}']".format(name))
            self.assertIsNotNone(saved)
            self.assertEqual(len(saved.findtext("pose").split()), 6)

    def test_ground_uses_anquanshengchan_site_datum_without_moving_scene(self):
        site_floor_z = 2.163280 - 0.657382
        ground = self.models["ground_plane"]
        # Gazebo Save World formats this value as 1.5059; the 2 micrometre
        # difference from the derived datum is geometrically insignificant.
        self.assertAlmostEqual(
            float(ground.findtext("pose").split()[2]), site_floor_z, delta=5.0e-6
        )
        self.assertAlmostEqual(
            float(ground.find("link/collision[@name='collision']/pose").text.split()[2]),
            -0.002,
            places=6,
        )
        self.assertAlmostEqual(
            float(ground.find("link/visual[@name='visual']/pose").text.split()[2]),
            -0.003,
            places=6,
        )

        # Both the authored model pose and Gazebo's saved scene pose remain
        # untouched; only ground_plane is aligned to their derived floor.
        self.assertEqual(
            self.models["anquanshengchan"].findtext("pose"),
            "4.78101 3.79558 0 0 -0 0",
        )
        saved_model = self.world.find("state/model[@name='anquanshengchan']")
        self.assertEqual(saved_model.findtext("pose"), "-3.90613 3.87824 2.16328 0 -0 0")

        launch_text = LAUNCH_PATH.read_text(encoding="utf-8")
        self.assertIn('<arg name="robot_z" default="1.505898"/>', launch_text)

    def test_motion_axes_and_truth_points(self):
        expected = {
            1: ([-0.9000116564182479, -0.4274638382988337, 0.0851685696610569], [0, 0, 0]),
            2: ([0.9008828563623635, -0.4052922873611565, 0.1553970428224124], [0, 0, 0]),
            3: ([0.0966034850335850, -0.9902357233980940, 0.1005036257336987], [0, 0, 0]),
            4: ([0.6989996111759129, -0.7151220480281130, 0.0], [0, 0, 0]),
            5: ([-0.7603891569279329, -0.6493907397540762, -0.0099998574030501], [0, 0, 0]),
            6: ([-0.0730428778713612, -0.0822176285782220, -0.9939341022136442], [0, 0, 0]),
        }
        entries = {entry["id"]: entry for entry in self.config["models"]}
        self.assertEqual(set(entries), set(range(1, 7)))
        for retro_id, (linear, angular) in expected.items():
            self.assertEqual(entries[retro_id]["model_name"], "m0{}".format(retro_id))
            self.assertEqual(entries[retro_id]["linear_direction"], linear)
            self.assertEqual(entries[retro_id]["angular_axis"], angular)
            if any(angular):
                self.assertEqual(len(entries[retro_id]["surface_points"]), 8)
            else:
                self.assertEqual(entries[retro_id]["surface_points"], [])
            if any(linear):
                self.assertAlmostEqual(
                    math.sqrt(sum(value * value for value in linear)), 1.0,
                    places=9)

        with STATIONARY_CONFIG_PATH.open("r", encoding="utf-8") as stream:
            stationary_entries = {
                entry["id"]: entry for entry in yaml.safe_load(stream)["models"]}
        for retro_id, (linear, angular) in expected.items():
            self.assertEqual(stationary_entries[retro_id]["linear_direction"],
                             linear)
            self.assertEqual(stationary_entries[retro_id]["angular_axis"],
                             angular)

        plugin_source = MOTION_PLUGIN.read_text(encoding="utf-8")
        self.assertNotIn("ExpectedAxes", plugin_source)
        self.assertIn("exactly one unit translation or", plugin_source)

    def test_default_protocol_and_theoretical_displacements(self):
        self.assertEqual(self.config["start_delay"], 3.0)
        self.assertEqual(self.config["motion_mode"], "simultaneous")
        self.assertNotIn("sequence_order_ids", self.config)
        self.assertEqual(self.config["duration"], 15.0)
        self.assertEqual(self.config["end_hold"], 10.0)
        self.assertAlmostEqual(1.0 * 0.001 * 15.0, 0.015)
        self.assertAlmostEqual(10.0 * 0.001 * 15.0, 0.15)
        self.assertAlmostEqual(0.0015 * 15.0, 0.0225)
        self.assertAlmostEqual(0.0025 * 15.0, 0.0375)

    def test_online_motion_status_is_model_agnostic_progress_only(self):
        source = MOTION_PLUGIN.read_text(encoding="utf-8")
        status = source[
            source.index("  std::string BuildStatusJson() const"):
            source.index("  void PublishStatus(", source.index(
                "  std::string BuildStatusJson() const"))]
        self.assertIn(r'\"experiment_elapsed\"', status)
        self.assertIn(r'\"motion_elapsed\"', status)
        self.assertNotIn(r'\"sequence_order_ids\"', status)
        self.assertNotIn(r'\"models\"', status)
        self.assertNotIn(r'\"model_name\"', status)
        self.assertNotIn(r'\"aabb_', status)
        self.assertNotIn(r'\"truth_topic\"', status)

        publish_truth = source[
            source.index("  void PublishTruth("):
            source.index("  std::string BuildStatusJson() const")]
        self.assertIn("this->PublishStatus(false);", publish_truth)
        publish_status = source[
            source.index("  void PublishStatus("):
            source.index("  physics::WorldPtr world_", source.index(
                "  void PublishStatus("))]
        self.assertIn("kStatusIntervalSeconds = 0.1", publish_status)
        self.assertNotIn("truthRateHz_", publish_status)

    def test_launch_passes_real_mid360_argument_names(self):
        launch_text = LAUNCH_PATH.read_text(encoding="utf-8")
        for name in ("x", "y", "z", "yaw"):
            self.assertIn('<arg name="{}" value="$(arg robot_'.format(name), launch_text)
        self.assertIn('name="icra2027_motion/truth_rate_hz"', launch_text)
        self.assertNotIn('<arg name="robot_x" value=', launch_text)
        urdf_text = A2_SIM_URDF.read_text(encoding="utf-8")
        self.assertIn("libicra2027_planar_move_plugin.so", urdf_text)
        self.assertNotIn("libgazebo_ros_planar_move.so", urdf_text)
        self.assertIn("<kinematicMode>true</kinematicMode>", urdf_text)
        self.assertIn("package://a2_description/meshes/base_link.STL", urdf_text)
        self.assertIn('<arg name="enable_imu" default="true"/>', launch_text)
        self.assertIn('<arg name="enable_rear_lidar" default="false"/>', launch_text)
        self.assertIn('type="dual_livox_merge_node"', launch_text)
        self.assertIn(
            'value="a2-mid360-frontlevel-tail-rearup25deg-v3"', launch_text)
        self.assertIn('<arg name="rviz" default="true"/>', launch_text)
        self.assertIn('<arg name="visualize_lidar_rays" default="false"/>', launch_text)
        self.assertIn('<arg name="lidar_point_time_mode" default="snapshot"/>', launch_text)
        self.assertIn(
            '<arg name="publish_rviz_cloud" value="$(arg publish_raw_rviz_cloud)"/>',
            launch_text,
        )
        self.assertIn('name="robot_name" value="a2_mid360"', launch_text)
        self.assertNotIn("start_slam", launch_text)

    def test_formal_dual_lidar_ray_visualization_remains_disabled(self):
        launch = ET.parse(str(LAUNCH_PATH)).getroot()
        visualize_arg = launch.find("arg[@name='visualize_lidar_rays']")
        self.assertIsNotNone(visualize_arg)
        self.assertEqual(visualize_arg.get("default"), "false")

        urdf_text = A2_SIM_URDF.read_text(encoding="utf-8")
        self.assertEqual(
            urdf_text.count(
                "<visualize>$(arg visualize_lidar_rays)</visualize>"),
            4,
        )

    def test_fixed_visualization_joints_are_driven_only_by_sim_clock(self):
        generic_launch = ET.parse(str(GENERIC_LAUNCH_PATH)).getroot()
        publisher = generic_launch.find(
            "node[@name='a2_sim_clock_joint_state_publisher']")
        self.assertIsNotNone(publisher)
        self.assertEqual(publisher.get("pkg"), "livox_laser_simulation")
        self.assertEqual(
            publisher.get("type"), "sim_clock_joint_state_publisher.py")
        self.assertEqual(
            publisher.find("param[@name='publish_rate_hz']").get("value"),
            "2.0")
        launch_text = GENERIC_LAUNCH_PATH.read_text(encoding="utf-8")
        self.assertNotIn('pkg="joint_state_publisher"', launch_text)

        spec = importlib.util.spec_from_file_location(
            "sim_clock_joint_state_publisher",
            str(SIM_CLOCK_JOINT_PUBLISHER))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Stamp:
            def __init__(self, secs, nsecs):
                self.secs = secs
                self.nsecs = nsecs

        self.assertEqual(module.stamp_to_nanoseconds(Stamp(4, 7)),
                         4_000_000_007)
        self.assertFalse(module.publish_due(None, 0, 500_000_000))
        self.assertTrue(module.publish_due(None, 1, 500_000_000))
        self.assertFalse(module.publish_due(
            1_000_000_000, 1_499_999_999, 500_000_000))
        self.assertTrue(module.publish_due(
            1_000_000_000, 1_500_000_000, 500_000_000))
        self.assertTrue(module.publish_due(
            10_000_000_000, 1_000_000_000, 500_000_000))

    def test_merge_diagnostics_expose_auditable_effective_cadence(self):
        source = (PACKAGE / "src" / "dual_livox_merge_node.cpp").read_text(
            encoding="utf-8")
        self.assertIn('{\\"schema_version\\":5', source)
        for field in (
                "zero_no_return_compaction_enabled",
                "configured_sensor_update_rate_hz",
                "configured_min_range_m",
                "configured_max_range_m",
                "configured_ray_samples_per_sensor",
                "primary_scheduler_catch_up_suppressed_count",
                "first_accepted_measurement_time_ns",
                "last_accepted_measurement_time_ns",
                "accepted_measurement_interval_count",
                "nonpositive_accepted_measurement_interval_count",
                "minimum_accepted_measurement_interval_ns",
                "maximum_accepted_measurement_interval_ns",
                "mean_accepted_measurement_interval_ns",
                "realized_measurement_hz",
                "last_total_raw_input_point_count",
                "last_compacted_output_point_count",
                "last_removed_no_return_sentinel_count",
                "last_compacted_output_to_total_raw_input_ratio"):
            self.assertIn('\\"{}\\"'.format(field), source)
        implementation = (PACKAGE / "src" / "dual_livox_merge.cpp").read_text(
            encoding="utf-8")
        self.assertIn("interval_count * 1.0e9", implementation)
        self.assertIn("accepted_measurement_time_ns >", implementation)

    def test_a2_simulation_wrapper_preserves_the_complete_official_urdf(self):
        upstream = ET.parse(str(A2_UPSTREAM_URDF)).getroot()
        simulation = ET.parse(str(A2_SIM_URDF)).getroot()

        upstream_links = {link.get("name") for link in upstream.findall("link")}
        upstream_joints = {joint.get("name") for joint in upstream.findall("joint")}
        simulation_links = {link.get("name") for link in simulation.findall("link")}
        simulation_joints = {joint.get("name") for joint in simulation.findall("joint")}

        self.assertEqual(len(upstream_links), 20)
        self.assertEqual(len(upstream_joints), 19)
        self.assertTrue(upstream_links.issubset(simulation_links))
        self.assertTrue(upstream_joints.issubset(simulation_joints))
        self.assertEqual(len(simulation_links), 25)
        self.assertEqual(len(simulation_joints), 24)
        self.assertEqual(len(upstream.findall(".//visual/geometry/mesh")), 17)
        self.assertEqual(len(simulation.findall(".//visual/geometry/mesh")), 19)

    def test_kinematic_imu_pose_matches_fixed_joint_calibration(self):
        root = ET.parse(str(A2_SIM_URDF)).getroot()
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        offset = [0., 0., 0.]
        child = "livox_imu_link"
        while child != "base_footprint":
            joint = joints[child]
            origin = joint.find("origin")
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            self.assertEqual(rpy, [0., 0., 0.])
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            offset = [a+b for a, b in zip(offset, xyz)]
            child = joint.find("parent").get("link")
        plugin = root.find(".//plugin[@name='a2_planar_move']")
        configured = [float(v) for v in plugin.findtext("imuPose").split()]
        for measured, expected in zip(configured, offset+[0., 0., 0.]):
            self.assertAlmostEqual(measured, expected, places=9)
        self.assertEqual(plugin.findtext("imuFrame"), "livox_imu_link")
        disabled_physics = root.find(".//{http://www.ros.org/wiki/xacro}unless[@value='$(arg enable_planar_move)']")
        self.assertIsNotNone(disabled_physics.find(".//plugin[@filename='libgazebo_ros_imu_sensor.so']"))

    def test_rear_lidar_is_tail_mounted_with_one_rigid_calibration_and_one_imu(self):
        simulation = ET.parse(str(A2_SIM_URDF)).getroot()
        joint = simulation.find("joint[@name='livox_rear_joint']")
        self.assertIsNotNone(joint)
        self.assertEqual(joint.find("parent").get("link"), "base_link")
        self.assertEqual(joint.find("child").get("link"), "livox_rear_frame")
        self.assertEqual(joint.find("origin").get("xyz"), "-0.3000 0 0.1925")
        self.assertEqual(
            joint.find("origin").get("rpy"),
            "0 -0.4363323129985824 3.141592653589793")
        text = A2_SIM_URDF.read_text(encoding="utf-8")
        self.assertIn("<ros_topic>/livox/front/lidar</ros_topic>", text)
        self.assertIn("<ros_topic>/livox/rear/lidar</ros_topic>", text)
        self.assertEqual(text.count("libgazebo_ros_imu_sensor.so"), 1)
        self.assertEqual(
            text.count("<point_time_mode>$(arg lidar_point_time_mode)</point_time_mode>"),
            2,
        )
        self.assertIn(
            '<xacro:arg name="primary_lidar_downsample" default="1"/>', text)
        self.assertIn(
            '<xacro:arg name="secondary_lidar_downsample" default="1"/>', text)
        self.assertEqual(
            text.count(
                "<downsample>$(arg primary_lidar_downsample)</downsample>"),
            1)
        self.assertEqual(
            text.count(
                "<downsample>$(arg secondary_lidar_downsample)</downsample>"),
            1)
        self.assertIn(
            '<xacro:arg name="lidar_update_rate_hz" default="10.0"/>',
            text)
        self.assertEqual(
            text.count(
                "<update_rate>$(arg lidar_update_rate_hz)</update_rate>"),
            2)
        self.assertEqual(
            text.count(
                "<synchronization_group>icra2027_a2_dual_mid360"
                "</synchronization_group>"),
            2,
        )
        self.assertEqual(text.count("<synchronization_role>primary"), 1)
        self.assertEqual(text.count("<synchronization_role>secondary"), 1)

        generic_urdf = (
            PACKAGE / "urdf" / "mid360_fastlio.urdf.xacro"
        ).read_text(encoding="utf-8")
        generic_launch = (
            PACKAGE / "launch" / "mid360_fastlio.launch"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '<xacro:arg name="lidar_point_time_mode" default="legacy_rolling"',
            generic_urdf,
        )
        self.assertIn(
            '<arg name="lidar_point_time_mode" default="legacy_rolling"',
            generic_launch,
        )
        self.assertIn(
            '<xacro:arg name="primary_lidar_downsample" default="1"',
            generic_urdf,
        )
        self.assertIn(
            '<xacro:arg name="secondary_lidar_downsample" default="1"',
            generic_urdf,
        )
        self.assertIn(
            '<arg name="primary_lidar_downsample" default="1"',
            generic_launch,
        )
        self.assertIn(
            '<arg name="secondary_lidar_downsample" default="1"',
            generic_launch,
        )
        self.assertIn(
            '<xacro:arg name="lidar_update_rate_hz" default="10.0"',
            generic_urdf,
        )
        self.assertIn(
            '<arg name="lidar_update_rate_hz" default="10.0"',
            generic_launch,
        )
        self.assertIn(
            "lidar_update_rate_hz:=$(arg lidar_update_rate_hz)",
            generic_launch,
        )

        # Derived from the two base_link poses, not independently tuned:
        # t_P_S=(-0.3000,0,0.1925)-(0.3181,0,0.1925).
        launch_text = LAUNCH_PATH.read_text(encoding="utf-8")
        self.assertIn(
            '<arg name="primary_lidar_downsample" default="1"/>',
            launch_text)
        self.assertIn(
            '<arg name="secondary_lidar_downsample" default="2"/>',
            launch_text)
        self.assertIn(
            '<arg name="lidar_update_rate_hz" default="10.0"/>',
            launch_text)
        self.assertIn(
            '<arg name="lidar_update_rate_hz" '
            'value="$(arg lidar_update_rate_hz)"/>', launch_text)
        self.assertIn(
            '<param name="configured_sensor_update_rate_hz"\n'
            '           value="$(arg lidar_update_rate_hz)"/>', launch_text)
        self.assertIn(
            '<param name="primary_downsample" '
            'value="$(arg primary_lidar_downsample)"/>', launch_text)
        self.assertIn(
            '<param name="secondary_downsample" '
            'value="$(arg secondary_lidar_downsample)"/>', launch_text)
        self.assertIn(
            '<param name="overlap_voxel_size_m" value="0.20"/>',
            launch_text)
        self.assertIn(
            '<param name="overlap_evaluation_pair_stride" value="10"/>',
            launch_text)
        self.assertIn(
            '<rosparam param="translation_primary_secondary_m">'
            '[-0.6181, 0.0, 0.0]</rosparam>', launch_text)
        self.assertIn(
            '[-0.9063077870366499, 0.0, 0.4226182617406994,',
            launch_text)
        self.assertIn('0.0, -1.0, 0.0,', launch_text)
        self.assertIn(
            '0.4226182617406994, 0.0, 0.9063077870366499]', launch_text)
        self.assertIn(
            '<param name="merge_ready_topic" value="/livox/merge/ready"/>',
            launch_text,
        )
        self.assertIn(
            '<param name="readiness_topic" value="/livox/merge/ready"/>',
            launch_text,
        )
        self.assertIn(
            'value="/livox/front/lidar/'
            'scheduler_catch_up_suppressed_count"',
            launch_text,
        )

        with FAST_LIO_CONFIG.open("r", encoding="utf-8") as stream:
            fast_lio = yaml.safe_load(stream)
        self.assertEqual(fast_lio["common"]["lid_topic"], "/livox/lidar")
        self.assertEqual(fast_lio["common"]["imu_topic"], "/livox/imu")
        self.assertFalse(fast_lio["common"]["time_sync_en"])

        plugin_source = (PACKAGE / "src" / "livox_points_plugin.cpp").read_text(
            encoding="utf-8")
        self.assertIn(".TriggerPairWithMinimumInterval(", plugin_source)
        self.assertIn("raySensor->UpdateRate()", plugin_source)
        self.assertIn("std::ceil(interval_ns)", plugin_source)
        self.assertIn(
            '"/scheduler_catch_up_suppressed_count"', plugin_source)
        self.assertNotIn("raySensor->ResetLastUpdateTime()", plugin_source)
        group_gate = plugin_source.index(".GroupReady(\n            synchronizationGroup)")
        first_ray_cast = plugin_source.index("GenerateScanAt(measurementTime)")
        self.assertLess(group_gate, first_ray_cast)
        secondary_role = plugin_source.index(
            "DualLidarTriggerRole::SECONDARY) {",
            plugin_source.index("if (synchronizedAcquisition)"))
        secondary_deactivation = plugin_source.index(
            "raySensor->SetActive(false);", secondary_role)
        synchronization_registration = plugin_source.index(
            "GlobalDualLidarTriggerRegistry().Register(", secondary_role)
        self.assertLess(secondary_deactivation, synchronization_registration)

    def test_formal_physical_range_is_half_to_twenty_metres_per_sensor(self):
        launch_text = LAUNCH_PATH.read_text(encoding="utf-8")
        self.assertIn('<arg name="lidar_min_range_m" default="0.5"/>',
                      launch_text)
        self.assertIn('<arg name="lidar_max_range_m" default="20.0"/>',
                      launch_text)
        simulation = ET.parse(str(A2_SIM_URDF)).getroot()
        lidar_sensors = [
            sensor for sensor in simulation.findall(".//sensor")
            if sensor.get("name") in (
                "a2_mid360_lidar", "a2_mid360_rear_lidar")]
        self.assertEqual(len(lidar_sensors), 2)
        for sensor in lidar_sensors:
            range_element = sensor.find("plugin/ray/range")
            self.assertIsNotNone(range_element, sensor.get("name"))
            self.assertEqual(range_element.findtext("min"),
                             "$(arg lidar_min_range_m)")
            self.assertEqual(range_element.findtext("max"),
                             "$(arg lidar_max_range_m)")
        for field, argument in (
                ("configured_min_range_m", "lidar_min_range_m"),
                ("configured_max_range_m", "lidar_max_range_m")):
            self.assertIn(
                '<param name="{}" value="$(arg {})"/>'.format(
                    field, argument), launch_text)

    def test_launch_calibration_is_derived_from_the_urdf_tf_chain(self):
        simulation = ET.parse(str(A2_SIM_URDF)).getroot()
        camera_xyz = [float(value) for value in simulation.find(
            "joint[@name='camera_joint']/origin").get("xyz").split()]
        primary_local = [float(value) for value in simulation.find(
            "joint[@name='livox_frame_joint']/origin").get("xyz").split()]
        primary_xyz = [camera_xyz[index] + primary_local[index]
                       for index in range(3)]
        rear_joint = simulation.find("joint[@name='livox_rear_joint']")
        rear_xyz = [float(value) for value in rear_joint.find(
            "origin").get("xyz").split()]
        roll, pitch, yaw = [float(value) for value in rear_joint.find(
            "origin").get("rpy").split()]
        self.assertAlmostEqual(roll, 0.0, places=12)
        self.assertAlmostEqual(primary_xyz[0], 0.3181, places=12)
        self.assertAlmostEqual(primary_xyz[1], 0.0, places=12)
        self.assertAlmostEqual(primary_xyz[2], 0.1925, places=12)
        expected_translation = [rear_xyz[index] - primary_xyz[index]
                                for index in range(3)]
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        expected_rotation = [
            cy * cp, -sy, cy * sp,
            sy * cp, cy, sy * sp,
            -sp, 0.0, cp,
        ]
        launch = ET.parse(str(LAUNCH_PATH)).getroot()
        merge = launch.find("node[@name='dual_livox_merge']")
        translation = [float(value) for value in merge.find(
            "rosparam[@param='translation_primary_secondary_m']").text
                       .replace("[", "").replace("]", "")
                       .replace(",", " ").split()]
        rotation = [float(value) for value in merge.find(
            "rosparam[@param='rotation_primary_secondary_row_major']").text
                    .replace("[", "").replace("]", "").replace(",", " ")
                    .split()]
        for actual, expected in zip(translation, expected_translation):
            self.assertAlmostEqual(actual, expected, places=10)
        for actual, expected in zip(rotation, expected_rotation):
            self.assertAlmostEqual(actual, expected, places=10)

    def test_front_and_rear_nominal_scan_envelopes_have_large_overlap(self):
        # This is an angular observability contract, not a claim that every
        # direction returns from the same physical surface.  Sample the real
        # Mid360 pattern deterministically and express each front direction
        # in the calibrated rear frame.  More than 70% must remain inside the
        # rear Mid360's own zenith envelope.
        sampled = []
        with MID360_PATTERN.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            for index, row in enumerate(reader):
                if index % 100 == 0:
                    sampled.append((math.radians(float(row[1])),
                                    math.radians(float(row[2]))))
        self.assertGreater(len(sampled), 7000)
        lower = min(zenith for _, zenith in sampled)
        upper = max(zenith for _, zenith in sampled)
        rotation = (
            (-0.9063077870366499, 0.0, 0.4226182617406994),
            (0.0, -1.0, 0.0),
            (0.4226182617406994, 0.0, 0.9063077870366499),
        )
        overlap_count = 0
        for azimuth, zenith in sampled:
            primary = (math.sin(zenith) * math.cos(azimuth),
                       math.sin(zenith) * math.sin(azimuth),
                       math.cos(zenith))
            # v_secondary = R_primary_secondary^T v_primary.
            secondary_z = sum(rotation[row][2] * primary[row]
                              for row in range(3))
            secondary_zenith = math.acos(max(-1.0, min(1.0, secondary_z)))
            overlap_count += lower <= secondary_zenith <= upper
        overlap_fraction = overlap_count / float(len(sampled))
        self.assertGreater(overlap_fraction, 0.70)
        self.assertLess(overlap_fraction, 0.85)

    def test_a2_fixed_standing_pose_and_head_mounted_lidar(self):
        simulation = ET.parse(str(A2_SIM_URDF)).getroot()

        base_joint = simulation.find("joint[@name='base_footprint_to_base_link']")
        self.assertEqual(base_joint.find("origin").get("xyz"), "0 0 0.434428878")

        for prefix in ("FL", "FR", "RL", "RR"):
            thigh = simulation.find("joint[@name='{}_thigh_joint']".format(prefix))
            calf = simulation.find("joint[@name='{}_calf_joint']".format(prefix))
            self.assertEqual(thigh.find("origin").get("rpy"), "0 0.75 0")
            self.assertEqual(calf.find("origin").get("rpy"), "0 -1.5 0")

        self.assertIsNone(simulation.find("link[@name='livox_mount']"))
        self.assertIsNone(simulation.find("joint[@name='livox_mount_joint']"))
        lidar_joint = simulation.find("joint[@name='livox_frame_joint']")
        self.assertEqual(lidar_joint.find("parent").get("link"), "camera_link")
        self.assertEqual(lidar_joint.find("origin").get("xyz"), "-0.02 -0.0336 0.14")

    def test_truth_recorder_schema(self):
        module_path = PACKAGE / "scripts" / "icra2027_truth_recorder.py"
        spec = importlib.util.spec_from_file_location("icra2027_truth_recorder", str(module_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = {
            "schema_version": 2, "state": "COMPLETED", "sim_time": 2.0,
            "motion_elapsed": 2.0, "linear_speed_mm_s": 10.0,
            "angular_speed_rad_s": 0.0025,
            "models": [{
                "laser_retro_id": 2, "model_name": "m02",
                "position": [1, 2, 3], "orientation": [0, 0, 0, 1],
                "aabb_min": [0.5, 1.5, 2.5],
                "aabb_max": [1.5, 2.5, 3.5],
                "translation": [0, 0.02, 0], "translation_magnitude": 0.02,
                "translation_expected": 0.02, "translation_error": 0.0,
            }],
        }
        row = module.rows_from_truth(payload)[0]
        self.assertEqual(row["laser_retro_id"], 2)
        self.assertEqual(row["model_name"], "m02")
        self.assertAlmostEqual(row["translation_y_m"], 0.02)
        self.assertAlmostEqual(row["aabb_min_x_m"], 0.5)
        self.assertAlmostEqual(row["aabb_max_z_m"], 3.5)


if __name__ == "__main__":
    unittest.main()

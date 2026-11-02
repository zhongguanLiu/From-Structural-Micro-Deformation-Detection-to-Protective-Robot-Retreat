from __future__ import annotations

import pathlib
import unittest
import xml.etree.ElementTree as ET


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLD_PATH = (
    PACKAGE_ROOT
    / "worlds"
    / "tracked_mid360_fastlio_collapse_microdeform.world"
)
ALTERNATE_WORLD_PATH = (
    PACKAGE_ROOT
    / "worlds"
    / "tracked_mid360_fastlio_collapse_microdeform1.world"
)

FACES = ("px", "nx", "py", "ny", "pz", "nz")
TARGET_MODELS = {
    "model_01": {
        "half_size": (0.733785, 0.650005, 0.403245),
        "object_id": 1,
    },
    "model_02": {
        "half_size": (0.55, 0.816605, 0.576325),
        "object_id": 2,
    },
    "model_03": {
        "half_size": (0.0721255, 1.78669, 1.23306),
        "object_id": 3,
    },
    "model_04": {
        "half_size": (0.55, 0.65, 0.83847),
        "object_id": 4,
    },
}

EXPECTED_OBJECT_ID_CATALOG = {
    1: "model_01",
    2: "model_02",
    3: "model_03",
    4: "model_04",
    5: "debris_block_01",
    6: "debris_block_02",
    7: "debris_panel_01_clone",
    8: "debris_panel_01_clone_0",
    9: "debris_panel_01_clone_clone",
    10: "debris_panel_01_clone_clone_0",
    11: "debris_panel_02_clone_0",
    12: "debris_panel_04",
    13: "debris_panel_04_clone",
    14: "debris_panel_04_clone_0",
    15: "debris_panel_04_clone_1",
    16: "debris_panel_04_clone_1_clone",
    17: "debris_panel_04_clone_1_clone_clone",
    18: "east_wall",
    19: "ground_plane",
    20: "obstacle_block_left_clone",
    21: "obstacle_block_left_clone_0",
    22: "obstacle_block_left_clone_1",
    23: "obstacle_block_right_clone",
    24: "partition_right",
    25: "partition_right_clone",
    26: "support_column_01",
    27: "tilted_support_wall",
    28: "top_slab_01",
    29: "top_slab_02",
    30: "west_wall",
    31: "west_wall_clone",
    32: "west_wall_clone_clone",
}


def expected_points(half_size: tuple[float, float, float]):
    a, b, c = half_size
    face_points = {
        "px": ((a, -b, -c), (a, b, -c), (a, b, c), (a, -b, c)),
        "nx": ((-a, -b, -c), (-a, b, -c), (-a, b, c), (-a, -b, c)),
        "py": ((-a, b, -c), (a, b, -c), (a, b, c), (-a, b, c)),
        "ny": ((-a, -b, -c), (a, -b, -c), (a, -b, c), (-a, -b, c)),
        "pz": ((-a, -b, c), (a, -b, c), (a, b, c), (-a, b, c)),
        "nz": ((-a, -b, -c), (a, -b, -c), (a, b, -c), (-a, b, -c)),
    }
    return {
        f"ground_truth_{face}_{index}": position
        for face in FACES
        for index, position in enumerate(face_points[face])
    }


def load_world_models(world_path: pathlib.Path) -> dict[str, ET.Element]:
    root = ET.parse(world_path).getroot()
    world = root.find("world")
    if world is None:
        raise AssertionError(f"Missing <world> in {world_path}")
    return {model.get("name"): model for model in world.findall("model")}


def ground_truth_links(model: ET.Element) -> dict[str, ET.Element]:
    return {
        link.get("name"): link
        for link in model.findall("link")
        if (link.get("name") or "").startswith("ground_truth_")
    }


def numeric_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split())


class ModelSurfaceGroundTruthTests(unittest.TestCase):
    def setUp(self):
        self.models = load_world_models(WORLD_PATH)

    def test_all_targets_are_top_level_models(self):
        self.assertTrue(set(TARGET_MODELS).issubset(self.models))

    def test_box_collision_assumptions_match_declared_half_sizes(self):
        for model_name, specification in TARGET_MODELS.items():
            with self.subTest(model=model_name):
                collision = self.models[model_name].find(
                    "link[@name='link']/collision"
                )
                self.assertIsNotNone(collision)
                self.assertIsNotNone(collision.find("geometry/box"))

                collision_pose = numeric_tuple(
                    collision.findtext("pose", "0 0 0 0 0 0")
                )
                self.assertEqual(collision_pose, (0.0,) * 6)

                box_size = numeric_tuple(
                    collision.findtext("geometry/box/size", "")
                )
                expected_size = tuple(
                    2.0 * value for value in specification["half_size"]
                )
                self.assertEqual(len(box_size), 3)
                for actual, expected in zip(box_size, expected_size):
                    self.assertAlmostEqual(actual, expected, places=9)

    def test_each_completed_box_has_four_independent_points_per_face(self):
        for model_name, specification in TARGET_MODELS.items():
            with self.subTest(model=model_name):
                model = self.models[model_name]
                raw_links = [
                    link
                    for link in model.findall("link")
                    if (link.get("name") or "").startswith("ground_truth_")
                ]
                links = ground_truth_links(model)
                expected = expected_points(specification["half_size"])
                self.assertEqual(len(raw_links), 24)
                self.assertEqual(len(links), 24)
                self.assertEqual(set(links), set(expected))

                for face in FACES:
                    face_names = {
                        name
                        for name in links
                        if name.startswith(f"ground_truth_{face}_")
                    }
                    self.assertEqual(
                        face_names,
                        {
                            f"ground_truth_{face}_{index}"
                            for index in range(4)
                        },
                    )

    def test_points_are_at_local_face_corners_and_visual_only(self):
        for model_name, specification in TARGET_MODELS.items():
            links = ground_truth_links(self.models[model_name])
            expected = expected_points(specification["half_size"])

            for point_name, expected_position in expected.items():
                with self.subTest(model=model_name, point=point_name):
                    self.assertIn(point_name, links)
                    link = links[point_name]
                    pose = numeric_tuple(link.findtext("pose", ""))
                    self.assertEqual(len(pose), 6)
                    for actual, expected_value in zip(
                        pose[:3], expected_position
                    ):
                        self.assertAlmostEqual(
                            actual,
                            expected_value,
                            places=9,
                        )
                    self.assertEqual(pose[3:], (0.0, 0.0, 0.0))

                    visual = link.find(
                        "visual[@name='ground_truth_marker']"
                    )
                    self.assertIsNotNone(visual)
                    radius = visual.findtext("geometry/sphere/radius")
                    self.assertIsNotNone(radius)
                    self.assertAlmostEqual(float(radius), 0.005, places=9)
                    self.assertEqual(
                        numeric_tuple(visual.findtext("material/ambient", "")),
                        (1.0, 0.0, 0.0, 1.0),
                    )
                    self.assertEqual(
                        numeric_tuple(visual.findtext("material/diffuse", "")),
                        (1.0, 0.0, 0.0, 1.0),
                    )
                    self.assertEqual(visual.findtext("cast_shadows"), "0")
                    for forbidden_element in (
                        "collision",
                        "inertial",
                        "sensor",
                        "plugin",
                    ):
                        self.assertIsNone(link.find(forbidden_element))

    def test_each_point_has_one_fixed_joint_to_the_original_link(self):
        for model_name, specification in TARGET_MODELS.items():
            model = self.models[model_name]
            expected = expected_points(specification["half_size"])
            raw_joints = [
                joint
                for joint in model.findall("joint")
                if (joint.get("name") or "").startswith("ground_truth_")
            ]
            joints = {
                joint.get("name"): joint
                for joint in raw_joints
            }
            expected_joint_names = {
                f"{point_name}_fixed_joint" for point_name in expected
            }

            with self.subTest(model=model_name):
                self.assertEqual(len(raw_joints), 24)
                self.assertEqual(len(joints), 24)
                self.assertEqual(set(joints), expected_joint_names)

            children = []
            for point_name in expected:
                with self.subTest(model=model_name, point=point_name):
                    joint = joints[f"{point_name}_fixed_joint"]
                    self.assertEqual(joint.get("type"), "fixed")
                    self.assertEqual(joint.findtext("parent"), "link")
                    self.assertEqual(joint.findtext("child"), point_name)
                    children.append(joint.findtext("child"))
            self.assertEqual(len(children), len(set(children)))

    def test_target_collisions_have_unique_valid_object_ids(self):
        observed_ids = {}
        for model_name, specification in TARGET_MODELS.items():
            collisions = self.models[model_name].findall(".//collision")
            self.assertGreater(len(collisions), 0, msg=model_name)
            collision_ids = set()
            for collision in collisions:
                retro_text = collision.findtext("laser_retro")
                self.assertIsNotNone(retro_text, msg=model_name)
                retro_value = float(retro_text)
                self.assertTrue(retro_value.is_integer(), msg=model_name)
                object_id = int(retro_value)
                self.assertGreaterEqual(object_id, 1)
                self.assertLessEqual(object_id, 254)
                collision_ids.add(object_id)

            expected_id = specification["object_id"]
            self.assertEqual(collision_ids, {expected_id})
            observed_ids[model_name] = expected_id

        self.assertEqual(
            len(set(observed_ids.values())),
            len(observed_ids),
        )

    def test_every_environment_collision_has_the_catalog_object_id(self):
        collision_models = {
            model_name: model
            for model_name, model in self.models.items()
            if model.findall(".//collision")
        }
        self.assertEqual(
            set(collision_models),
            set(EXPECTED_OBJECT_ID_CATALOG.values()),
        )

        observed = {}
        for object_id, model_name in EXPECTED_OBJECT_ID_CATALOG.items():
            collisions = collision_models[model_name].findall(".//collision")
            self.assertGreater(len(collisions), 0, model_name)
            for collision in collisions:
                with self.subTest(model=model_name, collision=collision.get("name")):
                    retro = collision.findtext("laser_retro")
                    self.assertIsNotNone(retro)
                    self.assertEqual(float(retro), float(object_id))
            observed[object_id] = model_name

        self.assertEqual(observed, EXPECTED_OBJECT_ID_CATALOG)

    def test_world_uses_model_pose_as_the_only_initial_pose_source(self):
        root = ET.parse(WORLD_PATH).getroot()
        world = root.find("world")
        self.assertIsNotNone(world)
        self.assertIsNone(world.find("state"))

    def test_alternate_world_remains_without_surface_truth_changes(self):
        alternate_models = load_world_models(ALTERNATE_WORLD_PATH)
        alternate_model_02 = alternate_models["model_02"]
        self.assertEqual(ground_truth_links(alternate_model_02), {})


if __name__ == "__main__":
    unittest.main()

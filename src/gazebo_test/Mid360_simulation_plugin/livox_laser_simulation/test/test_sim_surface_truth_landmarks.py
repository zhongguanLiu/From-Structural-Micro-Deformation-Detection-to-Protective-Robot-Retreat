from __future__ import annotations

import itertools
import math
import pathlib
import unittest
import xml.etree.ElementTree as ET


WORLD_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "worlds"
    / "tracked_mid360_fastlio_collapse_microdeform_sim.world"
)

EXPECTED_COUNTS = {
    1: 16,
    2: 8,
    3: 8,
    4: 12,
    5: 8,
    6: 10,
    7: 12,
    8: 8,
    9: 8,
    10: 16,
}

EXPECTED_OBJECT_CATALOG = {
    1: "l_shape_target",
    2: "model_01",
    3: "model_02",
    4: "table",
    5: "table_marble",
    6: "person_walking",
    7: "table_clone",
    8: "cardboard_box",
    9: "lzg_bianwood",
    10: "bookshelf",
    11: "frc2016_drawbridge",
    12: "lzg_grey_wall",
    13: "wooden_case_metal_peg",
    14: "cardboard_box_0",
    15: "bookshelf_0",
    16: "lzg_three_zhuzi",
    17: "cardboard_box_clone",
    18: "cardboard_box_1",
    19: "lzg_bianwood_clone",
    20: "lzg_bianwood_clone_0",
    21: "lzg_bianwood_clone_1",
    22: "grey_wall",
    23: "grey_wall_0",
    24: "ground_plane",
    25: "lzg_wall",
}

BOX_MODELS = {
    # model_01 was resized to a taller, narrower box (constant_rotation
    # about world -Z, counter-rotating l_shape_target) -- these are its
    # current collision/visual box dimensions, not the original
    # translating-object footprint.
    2: (2.53211, 0.676971, 2.31086),
    3: (1.1, 1.63321, 2.74334),
    8: (0.5, 0.4, 0.3),
    9: (1.19964, 0.165479, 2.85787),
}

EXPECTED_PARENT_LINKS = {
    9: "link_0",
}

# Corner marker sphere radius is purely a debug-visualization aid, not a
# measurement primitive -- model_01's resize tool scaled its markers along
# with the model instead of keeping the fixed 0.02 default. This is
# intentionally kept (not reverted), so its expected radius is overridden
# here rather than treated as a defect.
MARKER_RADIUS_OVERRIDES = {
    2: 0.0259831,
}


def numeric_tuple(text):
    return tuple(float(value) for value in str(text or "").split())


def transform_point(pose, point):
    x, y, z, roll, pitch, yaw = pose
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return tuple(
        origin
        + sum(rotation[row][column] * point[column] for column in range(3))
        for row, origin in enumerate((x, y, z))
    )


def load_models():
    world = ET.parse(WORLD_PATH).getroot().find("world")
    if world is None:
        raise AssertionError("missing <world>")
    return list(world.findall("model"))


def load_world():
    world = ET.parse(WORLD_PATH).getroot().find("world")
    if world is None:
        raise AssertionError("missing <world>")
    return world


def model_id(model):
    values = {
        int(float(collision.findtext("laser_retro")))
        for collision in model.findall(".//collision")
    }
    if len(values) != 1:
        raise AssertionError(f"{model.get('name')}: IDs={values}")
    return values.pop()


def truth_links(model):
    return [
        link
        for link in model.findall("link")
        if (link.get("name") or "").startswith("ground_truth_")
    ]


def truth_joints(model):
    return [
        joint
        for joint in model.findall("joint")
        if (joint.get("name") or "").startswith("ground_truth_")
    ]


def truth_marker_visuals(model, object_id):
    parent_name = EXPECTED_PARENT_LINKS.get(object_id, "link")
    parent_link = model.find(f"link[@name='{parent_name}']")
    if parent_link is None:
        raise AssertionError(f"{model.get('name')}: missing parent link {parent_name}")
    return [
        visual
        for visual in parent_link.findall("visual")
        if (visual.get("name") or "").startswith("ground_truth_marker_v_")
    ]


class SimSurfaceTruthLandmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = load_models()
        cls.models_by_id = {model_id(model): model for model in cls.models}

    def test_only_evaluated_ids_have_expected_landmarks(self):
        self.assertEqual(set(self.models_by_id), set(range(1, 26)))
        self.assertEqual(
            {
                object_id: model.get("name")
                for object_id, model in self.models_by_id.items()
            },
            EXPECTED_OBJECT_CATALOG,
        )
        total = 0
        for object_id, model in self.models_by_id.items():
            links = truth_links(model)
            joints = truth_joints(model)
            markers = truth_marker_visuals(model, object_id) if object_id <= 10 else []
            expected = EXPECTED_COUNTS.get(object_id, 0)
            with self.subTest(object_id=object_id, model=model.get("name")):
                self.assertEqual(links, [])
                self.assertEqual(joints, [])
                self.assertEqual(len(markers), expected)
            total += len(markers)
        self.assertEqual(total, 106)

    def test_parent_link_marker_visuals_define_the_surface_catalog(self):
        for object_id in EXPECTED_COUNTS:
            model = self.models_by_id[object_id]
            expected_parent = EXPECTED_PARENT_LINKS.get(object_id, "link")
            parent_link = model.find(f"link[@name='{expected_parent}']")
            self.assertIsNotNone(parent_link)
            marker_visuals = sorted(
                truth_marker_visuals(model, object_id),
                key=lambda visual: visual.get("name"),
            )
            self.assertEqual(len(marker_visuals), EXPECTED_COUNTS[object_id])
            positions = []
            for index, visual in enumerate(marker_visuals):
                expected_marker_name = f"ground_truth_marker_v_{index:03d}"
                with self.subTest(object_id=object_id, landmark=expected_marker_name):
                    self.assertEqual(visual.get("name"), expected_marker_name)
                    marker_pose = numeric_tuple(visual.findtext("pose"))
                    self.assertEqual(len(marker_pose), 6)
                    self.assertEqual(marker_pose[3:], (0.0, 0.0, 0.0))
                    positions.append(marker_pose[:3])
                    self.assertEqual(
                        float(visual.findtext("geometry/sphere/radius")),
                        MARKER_RADIUS_OVERRIDES.get(object_id, 0.02),
                    )
                    self.assertEqual(
                        numeric_tuple(visual.findtext("material/emissive")),
                        (1.0, 0.0, 0.0, 1.0),
                    )
            self.assertEqual(len(positions), len(set(positions)))

    def test_all_catalog_models_are_lidar_addressable_and_joints_are_local(self):
        for object_id, model in self.models_by_id.items():
            declared_links = {link.get("name") for link in model.findall("link")}
            collisions = model.findall(".//collision")
            with self.subTest(object_id=object_id, model=model.get("name")):
                self.assertGreater(len(collisions), 0)
            for collision in collisions:
                with self.subTest(
                    object_id=object_id,
                    model=model.get("name"),
                    collision=collision.get("name"),
                ):
                    self.assertEqual(
                        float(collision.findtext("laser_retro")),
                        float(object_id),
                    )
                    geometry = collision.find("geometry")
                    self.assertIsNotNone(geometry)
                    self.assertEqual(len(geometry), 1)
                    shape = geometry[0]
                    self.assertIn(
                        shape.tag,
                        {"box", "cylinder", "mesh", "plane", "sphere"},
                    )
                    if shape.tag == "box":
                        self.assertTrue(
                            all(value > 0.0 for value in numeric_tuple(shape.findtext("size")))
                        )
                    elif shape.tag == "cylinder":
                        self.assertGreater(float(shape.findtext("radius")), 0.0)
                        self.assertGreater(float(shape.findtext("length")), 0.0)
                    elif shape.tag == "sphere":
                        self.assertGreater(float(shape.findtext("radius")), 0.0)
                    elif shape.tag == "plane":
                        self.assertTrue(
                            all(value > 0.0 for value in numeric_tuple(shape.findtext("size")))
                        )
                    elif shape.tag == "mesh":
                        self.assertTrue((shape.findtext("uri") or "").strip())

            for joint in model.findall("joint"):
                with self.subTest(
                    object_id=object_id,
                    model=model.get("name"),
                    joint=joint.get("name"),
                ):
                    self.assertIn(joint.findtext("parent"), declared_links)
                    self.assertIn(joint.findtext("child"), declared_links)

    def test_box_models_use_current_unique_collision_vertices(self):
        for object_id, size in BOX_MODELS.items():
            model = self.models_by_id[object_id]
            parent_link = model.find(
                f"link[@name='{EXPECTED_PARENT_LINKS.get(object_id, 'link')}']"
            )
            parent_pose = numeric_tuple(
                parent_link.findtext("pose", "0 0 0 0 0 0")
            )
            actual = {
                transform_point(
                    parent_pose,
                    numeric_tuple(visual.findtext("pose"))[:3],
                )
                for visual in truth_marker_visuals(model, object_id)
            }
            half = tuple(value * 0.5 for value in size)
            expected = {
                transform_point(
                    numeric_tuple(
                        model.find(
                            f"link[@name='{EXPECTED_PARENT_LINKS.get(object_id, 'link')}']"
                        ).findtext("pose", "0 0 0 0 0 0")
                    ),
                    tuple(sign[axis] * half[axis] for axis in range(3)),
                )
                for sign in itertools.product((-1.0, 1.0), repeat=3)
            }
            with self.subTest(object_id=object_id, model=model.get("name")):
                self.assertEqual(len(actual), len(expected))
                for expected_point in expected:
                    self.assertTrue(
                        any(
                            all(
                                abs(actual_value - expected_value) < 1.0e-5
                                for actual_value, expected_value in zip(
                                    actual_point, expected_point
                                )
                            )
                            for actual_point in actual
                        )
                    )

    def test_l_shape_is_one_rigid_physical_link(self):
        model = self.models_by_id[1]
        physical_links = [
            link
            for link in model.findall("link")
            if not (link.get("name") or "").startswith("ground_truth_")
        ]
        self.assertEqual([link.get("name") for link in physical_links], ["link"])
        link = physical_links[0]
        self.assertEqual(
            [collision.get("name") for collision in link.findall("collision")],
            ["collision_0", "collision_1"],
        )
        self.assertEqual(
            [
                visual.get("name")
                for visual in link.findall("visual")
                if not (visual.get("name") or "").startswith(
                    "ground_truth_marker_v_"
                )
            ],
            ["visual_0", "visual_1"],
        )
        self.assertAlmostEqual(float(link.findtext("inertial/mass")), 2.9863)
        inertia = {
            name: float(link.findtext(f"inertial/inertia/{name}"))
            for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        }
        self.assertGreater(inertia["ixx"], 0.0)
        self.assertGreater(
            inertia["ixx"] * inertia["iyy"] - inertia["ixy"] ** 2,
            0.0,
        )
        determinant = (
            inertia["ixx"] * inertia["iyy"] * inertia["izz"]
            + 2.0 * inertia["ixy"] * inertia["ixz"] * inertia["iyz"]
            - inertia["ixx"] * inertia["iyz"] ** 2
            - inertia["iyy"] * inertia["ixz"] ** 2
            - inertia["izz"] * inertia["ixy"] ** 2
        )
        self.assertGreater(determinant, 0.0)

    def test_world_has_no_saved_state_snapshot(self):
        world = load_world()
        self.assertIsNone(world.find("state"))


if __name__ == "__main__":
    unittest.main()

import pathlib
import unittest
import xml.etree.ElementTree as ET


WORLD_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "gazebo_test"
    / "Mid360_simulation_plugin"
    / "livox_laser_simulation"
    / "worlds"
    / "tracked_mid360_fastlio_collapse_microdeform.world"
)


TARGET_MODELS = [
    "debris_panel_01_clone",
    "debris_panel_01_clone_clone",
    "debris_panel_04_clone_0",
    "debris_panel_04_clone_1",
]


def parse_pose(text):
    return [float(part) for part in (text or "").split()]


def parse_vector6(text):
    values = [float(part) for part in (text or "").split()]
    return values if len(values) == 6 else None


class WorldStaticStateTests(unittest.TestCase):
    def test_all_inline_scene_models_are_static(self):
        root = ET.parse(WORLD_PATH).getroot()
        world = root.find("world")
        self.assertIsNotNone(world)

        models = world.findall("model")
        self.assertTrue(models)
        for model in models:
            model_name = model.attrib.get("name", "")
            self.assertEqual(
                (model.findtext("static") or "").strip(),
                "1",
                f"{model_name} must move only through an explicit controller",
            )

    def test_target_panels_have_static_model_definitions(self):
        root = ET.parse(WORLD_PATH).getroot()
        world = root.find("world")
        self.assertIsNotNone(world)

        for model_name in TARGET_MODELS:
            model = next(
                (item for item in world.findall("model") if item.attrib.get("name") == model_name),
                None,
            )
            self.assertIsNotNone(model, model_name)
            self.assertEqual((model.findtext("static") or "").strip(), "1", model_name)

    def test_saved_world_has_no_state_snapshot_that_can_override_model_poses(self):
        root = ET.parse(WORLD_PATH).getroot()
        world = root.find("world")
        self.assertIsNotNone(world)
        self.assertIsNone(world.find("state"))

    def test_model_03_uses_the_confirmed_stable_flat_pose(self):
        root = ET.parse(WORLD_PATH).getroot()
        world = root.find("world")
        model = next(
            item
            for item in world.findall("model")
            if item.attrib.get("name") == "model_03"
        )

        self.assertEqual(
            parse_pose(model.findtext("pose")),
            [
                -6.153825967,
                3.091319891,
                0.0721255,
                0.0,
                1.570796327,
                -0.16,
            ],
        )


if __name__ == "__main__":
    unittest.main()

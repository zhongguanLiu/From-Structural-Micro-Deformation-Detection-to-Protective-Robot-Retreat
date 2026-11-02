#!/usr/bin/env python3
"""Run after sourcing the workspace's devel/setup.bash."""

import io
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import icra2027_closed_loop_mission as mission


class OperatorStartTest(unittest.TestCase):
    def test_terminal_confirmation_and_closed_input(self):
        for answer in ("\n", "1\n", "invalid\n1\n"):
            output = io.StringIO()
            self.assertTrue(mission.confirm_video_start(io.StringIO(answer), output))
            self.assertIn("READY_TO_RECORD", output.getvalue())
        for answer in ("", "invalid\n"):
            self.assertFalse(mission.confirm_video_start(io.StringIO(answer), io.StringIO()))

    def test_gate_pauses_until_confirmation_and_does_not_resume_on_cancel(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                ready = threading.Event()
                shutdown = threading.Event()
                node = Mock()
                ros = Mock()
                ros.is_shutdown.side_effect = shutdown.is_set
                callback = []
                failures = []

                def advertise(_name, _type, handler):
                    callback.append(handler)
                    ready.set()
                    return Mock()

                def run():
                    try:
                        mission.ClosedLoopMission._wait_for_operator_start(node)
                    except mission.MissionFailure as error:
                        failures.append(error)

                ros.Service.side_effect = advertise
                with patch.object(mission, "rospy", ros):
                    worker = threading.Thread(target=run, daemon=True)
                    worker.start()
                    try:
                        self.assertTrue(ready.wait(2))
                        self.assertTrue(worker.is_alive())
                        self.assertEqual(ros.ServiceProxy.call_args_list[0].args[0],
                                         "/gazebo/pause_physics")
                        self.assertEqual(ros.ServiceProxy.call_count, 1)
                        node._set_state.assert_called_once()
                        self.assertEqual(node._set_state.call_args.args[0], "WAITING_FOR_OPERATOR")
                        if cancel:
                            shutdown.set()
                        else:
                            self.assertTrue(callback[0](None).success)
                        worker.join(2)
                        self.assertFalse(worker.is_alive())
                        self.assertEqual(bool(failures), cancel)
                        self.assertEqual(ros.ServiceProxy.call_count, 1 if cancel else 2)
                        if not cancel:
                            self.assertEqual(ros.ServiceProxy.call_args.args[0],
                                             "/gazebo/unpause_physics")
                            self.assertEqual(node._set_state.call_args.args[0], "OPERATOR_STARTED")
                    finally:
                        shutdown.set()
                        worker.join(2)


if __name__ == "__main__":
    unittest.main()

import threading
import time
import unittest
import queue
from collections import deque

from restim_sensor_bridge import Reader, RestimSensorBridge


class RestimSensorSemanticsTests(unittest.TestCase):
    def bridge(self):
        bridge = RestimSensorBridge.__new__(RestimSensorBridge)
        bridge.lock = threading.Lock()
        bridge.samples = {"IMU": deque(maxlen=1600), "AS5311": deque(maxlen=1400)}
        bridge.connected = {"IMU": True, "AS5311": True}
        bridge.errors = {}
        bridge.started = time.time() - 2.0
        bridge.last_observation = ""
        bridge.observation_since = time.time() - 5.0
        return bridge

    def test_quiet_relative_state(self):
        b = self.bridge(); now = time.time()
        for i in range(100):
            t = now - 1 + i / 100
            b.samples["IMU"].append((t, {"gyr_x": .003, "gyr_y": .008, "gyr_z": -.009}))
            b.samples["AS5311"].append((t, {"x": -.00095 + (i % 2) * .000002}))
        state = b.snapshot()
        self.assertEqual(state["movement"]["state"], "still")
        self.assertEqual(state["clenching"]["state"], "quiet")
        self.assertEqual(state["response_interpretation"]["default"], "quiet baseline")
        self.assertTrue(state["read_only"])

    def test_active_movement_and_clenching(self):
        b = self.bridge(); now = time.time()
        for i in range(100):
            t = now - 1 + i / 100
            sign = -1 if i % 2 else 1
            b.samples["IMU"].append((t, {"gyr_x": .2 * sign, "gyr_y": .25, "gyr_z": .1}))
            b.samples["AS5311"].append((t, {"x": -.0007 + sign * .0003}))
        state = b.snapshot()
        self.assertEqual(state["movement"]["state"], "vigorous")
        self.assertEqual(state["clenching"]["state"], "strong")
        self.assertEqual(state["response_interpretation"]["default"], "positive response")
        self.assertIn("hip thrust / arousal response", state["response_interpretation"]["cues"])
        self.assertIn("induced clenching", state["response_interpretation"]["cues"])

    def test_reader_retries_after_initial_connection_failure(self):
        output = queue.Queue()
        reader = Reader("IMU", "/sensors/imu", output)

        class ImmediateRetryStop:
            def __init__(self): self.stopped = False
            def is_set(self): return self.stopped
            def wait(self, _seconds): return self.stopped
            def set(self): self.stopped = True

        stop = ImmediateRetryStop()
        attempts = []
        def connect():
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("Restim not ready")
            stop.set()
        reader.stop_event = stop
        reader.connect = connect
        reader._close_socket = lambda: None
        reader.run()
        statuses = [output.get_nowait() for _ in range(output.qsize())]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(statuses[0][0:3], ("status", "IMU", False))
        self.assertEqual(statuses[1][0:3], ("status", "IMU", True))


if __name__ == "__main__": unittest.main()

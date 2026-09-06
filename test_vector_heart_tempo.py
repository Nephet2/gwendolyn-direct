import threading
import time
import unittest

import requests

import gwendolyn_direct as gd


class VectorHeartTempoTests(unittest.TestCase):
    def make_core(self):
        core = gd.GwendolynCore.__new__(gd.GwendolynCore)
        core.cfg = {"vector_heart_tempo_follow": True}
        core.vector_autonomous_active = True
        core.vector_autonomous_held = False
        core.vector_autonomous_generates_motion = lambda: True
        core.vector_motion_plan = {
            "pattern": "sine", "minimum": 0.2, "maximum": 0.7,
            "stroke_duration_ms": 1000, "transition_seconds": 8,
            "duration_seconds": 300, "reason": "test",
        }
        core.heart_tempo_lock = threading.Lock()
        core.heart_tempo_last_apply_at = 0.0
        core.heart_tempo_last_stroke_ms = 1000.0
        core.heart_tempo_server = None
        core.heart_tempo_thread = None
        return core

    def test_tick_changes_only_tempo_fields(self):
        core = self.make_core()
        core.heart_tempo = {
            "available": True, "bpm": 60.0, "ratio": 0.9,
            "target_cpm": 54.0, "confidence": 90.0, "received_at": time.time(),
        }
        calls = []
        core.vector_post = lambda path, payload, timeout=3.0: calls.append((path, dict(payload))) or {"ok": True}
        original = dict(core.vector_motion_plan)
        core.vector_heart_tempo_tick()
        self.assertEqual(len(calls), 1)
        path, applied = calls[0]
        self.assertEqual(path, "/v1/generated-motion/plan")
        self.assertEqual(applied["pattern"], original["pattern"])
        self.assertEqual(applied["minimum"], original["minimum"])
        self.assertEqual(applied["maximum"], original["maximum"])
        self.assertEqual(applied["duration_seconds"], original["duration_seconds"])
        self.assertEqual(applied["transition_seconds"], 3)
        self.assertEqual(applied["stroke_duration_ms"], 711)

    def test_double_heart_rate_paces_generated_motion_at_two_x(self):
        core = self.make_core()
        core.heart_tempo = {
            "available": True, "bpm": 60.0, "ratio": 2.0,
            "target_cpm": 120.0, "confidence": 95.0, "received_at": time.time(),
        }
        calls = []
        core.vector_post = lambda path, payload, timeout=3.0: calls.append((path, dict(payload))) or {"ok": True}
        core.vector_heart_tempo_tick()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["stroke_duration_ms"], 512)

    def test_stale_telemetry_is_ignored(self):
        core = self.make_core()
        core.heart_tempo = {
            "available": True, "bpm": 80.0, "ratio": 1.0,
            "target_cpm": 80.0, "confidence": 95.0, "received_at": time.time() - 4.0,
        }
        calls = []
        core.vector_post = lambda *args, **kwargs: calls.append(args) or {"ok": True}
        core.vector_heart_tempo_tick()
        self.assertEqual(calls, [])

    def test_snapshot_exposes_fresh_applied_telemetry(self):
        core = self.make_core()
        core.heart_tempo = {
            "available": True, "bpm": 72.0, "ratio": 0.9,
            "target_cpm": 64.8, "confidence": 88.0, "received_at": time.time(),
        }
        snapshot = core.vector_heart_tempo_snapshot()
        self.assertTrue(snapshot["fresh"])
        self.assertTrue(snapshot["tempo_following_active"])
        self.assertEqual(snapshot["bpm"], 72.0)
        self.assertEqual(snapshot["target_cycles_per_minute"], 64.8)

    def test_snapshot_hides_stale_measurement(self):
        core = self.make_core()
        core.heart_tempo = {
            "available": True, "bpm": 72.0, "ratio": 0.9,
            "target_cpm": 64.8, "confidence": 88.0, "received_at": time.time() - 4.0,
        }
        snapshot = core.vector_heart_tempo_snapshot()
        self.assertFalse(snapshot["fresh"])
        self.assertIsNone(snapshot["bpm"])
        self.assertFalse(snapshot["tempo_following_active"])

    def test_loopback_listener_validates_and_accepts_heart_data(self):
        core = self.make_core()
        core.heart_tempo = {"available": False, "received_at": 0.0}
        core.start_heart_tempo_listener()
        try:
            response = requests.post(
                "http://127.0.0.1:18767/v1/heart-tempo",
                json={"bpm": 72.0, "ratio": 0.9, "confidence": 88.0},
                headers={"Origin": "http://127.0.0.1:8766"}, timeout=2.0,
            )
            self.assertEqual(response.status_code, 200)
            self.assertAlmostEqual(response.json()["target_cpm"], 64.8)
            self.assertTrue(core.heart_tempo["available"])
        finally:
            core.close_heart_tempo_listener()
            core.heart_tempo_server.server_close()


if __name__ == "__main__":
    unittest.main()

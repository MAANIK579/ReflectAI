"""
tests/test_camera_motion.py — Unit tests for Motion-Activated Camera & Face Recognition.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from app.dashboard.dashboard import app, process_esp32_payload, _esp32_state
from app.database.repositories import SettingsRepository, UserRepository
from face_engine.face_service import FaceRecognitionService, match_face


class TestCameraMotion(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        # Seed test user
        UserRepository.create_or_update("maanik", "MAANIK")
        SettingsRepository.set("active_user", "guest")
        SettingsRepository.set("privacy_mode", "false")

    def test_face_recognition_service_init_and_motion(self):
        service = FaceRecognitionService(
            camera_index=99,  # dummy index
            motion_timeout=10.0,
            no_face_timeout=5.0,
        )
        self.assertEqual(service.current_user, "guest")
        self.assertTrue(service.is_motion_detected)
        
        # Test on_motion_detected updates timestamp and sets event
        old_time = service.last_motion_time
        time.sleep(0.01)
        service.on_motion_detected()
        self.assertGreaterEqual(service.last_motion_time, old_time)
        self.assertTrue(service._motion_event.is_set())

        status = service.get_status()
        self.assertIn("camera_status", status)
        self.assertIn("motion_active", status)
        self.assertEqual(status["active_user"], "guest")

    def test_match_face_cosine(self):
        # Mock recognizer
        recognizer = MagicMock()
        recognizer.match.side_effect = lambda query, ref, metric: 0.72 if ref is not None else 0.1

        registered = {
            "maanik": np.ones((1, 128), dtype=np.float32),
            "ayush": np.zeros((1, 128), dtype=np.float32),
        }
        query = np.ones((1, 128), dtype=np.float32)

        best_id, score = match_face(query, registered, recognizer, threshold=0.363)
        self.assertIsNotNone(best_id)
        self.assertGreaterEqual(score, 0.363)

    def test_esp32_motion_payload_wakes_camera_and_triggers_user(self):
        mock_camera_svc = MagicMock()
        with patch("app.dashboard.dashboard._face_camera_service", mock_camera_svc):
            payload = {
                "device": "esp32",
                "status": "online",
                "motion": True,
                "temperature": 23.4,
                "humidity": 45.0,
            }
            res = process_esp32_payload(payload)
            self.assertTrue(res["motion"])
            self.assertEqual(res["temperature"], 23.4)
            # Verify camera service was awakened
            mock_camera_svc.on_motion_detected.assert_called_once()

    def test_api_status_reports_camera_diagnostics(self):
        mock_camera_svc = MagicMock()
        mock_camera_svc.running = True
        mock_camera_svc.camera_status = "scanning for faces"
        mock_camera_svc.get_status.return_value = {
            "running": True,
            "camera_status": "scanning for faces",
            "active_user": "guest",
            "motion_active": True,
        }

        with patch("app.dashboard.dashboard._face_camera_service", mock_camera_svc):
            res = self.client.get("/api/status")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertIn("system", data)
            self.assertEqual(data["system"]["camera_status"], "scanning for faces")
            self.assertIsNotNone(data["system"]["camera_diag"])


if __name__ == "__main__":
    unittest.main()

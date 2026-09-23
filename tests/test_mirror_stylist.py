"""
tests/test_mirror_stylist.py — Tests for Live Camera Clothing Recognition and Mirror Stylist.

Run with: python -m unittest tests.test_mirror_stylist -v
"""

import io
import unittest
from datetime import date
from unittest.mock import patch

from PIL import Image

from app.config.settings import settings
from app.database.database import get_connection, init_db
from app.database.repositories import WardrobeRepository, OutfitLogRepository
from app.recommendations.mirror_stylist import MirrorStylist
from app.dashboard.dashboard import app


class TestMirrorStylist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.DATABASE_PATH = settings.DATA_DIR / "test_reflectai_stylist.db"
        if settings.DATABASE_PATH.exists():
            settings.DATABASE_PATH.unlink()
        init_db()

    @classmethod
    def tearDownClass(cls):
        if settings.DATABASE_PATH.exists():
            settings.DATABASE_PATH.unlink()

    def setUp(self):
        with get_connection() as conn:
            conn.execute("DELETE FROM wardrobe")
            conn.execute("DELETE FROM outfit_log")

    @patch('app.recommendations.mirror_stylist.WeatherService')
    @patch('app.recommendations.mirror_stylist.CalendarService')
    def test_mirror_stylist_cold_weather_layering(self, mock_cal, mock_weather):
        mock_weather.get_weather.return_value = {
            "temperature": 12,
            "icon": "rain",
            "condition": "Moderate rain",
        }
        mock_cal.get_events_for_user.return_value = []

        user_id = "stylist_user"
        WardrobeRepository.add_item(user_id, "Black Warm Coat", "outerwear", weather="cold")
        WardrobeRepository.add_item(user_id, "Blue Jeans", "bottom", color="Blue")

        # User standing in front of camera wearing a white t-shirt
        detected = {
            "category": "top",
            "color": "White",
            "sub_category": "Topwear",
            "suggested_name": "White Topwear",
        }

        result = MirrorStylist.assess_clothing(user_id, precomputed_detection=detected)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["needs_layer"])
        self.assertIn("12°", result["advice"])

        # Outerwear should be recommended from wardrobe
        pairing_names = [p["name"] for p in result["pairings"]]
        self.assertIn("Black Warm Coat", pairing_names)
        self.assertIn("Blue Jeans", pairing_names)

    @patch('app.recommendations.mirror_stylist.WeatherService')
    @patch('app.recommendations.mirror_stylist.CalendarService')
    def test_mirror_stylist_schedule_recognition(self, mock_cal, mock_weather):
        mock_weather.get_weather.return_value = {
            "temperature": 25,
            "icon": "clear",
            "condition": "Clear sky",
        }
        mock_cal.get_events_for_user.return_value = [
            {"time": "10:00 AM", "title": "Client Meeting & Review"}
        ]

        user_id = "schedule_user"
        WardrobeRepository.add_item(user_id, "Grey Trousers", "bottom", color="Grey")

        detected = {
            "category": "top",
            "color": "Black",
            "sub_category": "Topwear",
            "suggested_name": "Black Topwear",
        }

        result = MirrorStylist.assess_clothing(user_id, precomputed_detection=detected)
        self.assertEqual(result["status"], "ok")
        self.assertIn("Client Meeting & Review", result["advice"])

    def test_api_mirror_clothing_endpoints(self):
        client = app.test_client()

        # Create dummy torso crop image
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        # POST /api/mirror/clothing/live
        res = client.post(
            "/api/mirror/clothing/live",
            data={"userId": "test_cam_user", "image": (buf, "torso.jpg")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("assessment", data)

        # GET /api/mirror/clothing/status
        res_status = client.get("/api/mirror/clothing/status?user_id=test_cam_user")
        self.assertEqual(res_status.status_code, 200)
        status_data = res_status.get_json()
        self.assertEqual(status_data.get("status"), "ok")
        self.assertIn("advice", status_data)

    def test_extract_body_crops_logic(self):
        from face_engine.face_service import extract_body_crops
        import numpy as np

        # Create a mock 720x1280 frame
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Mock face bounding box: x=500, y=50, w=100, h=100
        mock_face = [500, 50, 100, 100]

        torso_crop, legs_crop, head_crop = extract_body_crops(mock_frame, mock_face)
        self.assertIsNotNone(torso_crop)
        self.assertIsNotNone(legs_crop)
        self.assertIsNotNone(head_crop)
        # Torso height ~245px, legs height ~430px, head crop > 0
        self.assertGreater(torso_crop.shape[0], 80)
        self.assertGreater(legs_crop.shape[0], 100)
        self.assertGreater(head_crop.shape[0], 50)

    def test_grooming_classifier_and_endpoints(self):
        from app.recommendations.grooming_classifier import GroomingClassifier

        img = Image.new("RGB", (224, 224), color=(180, 140, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        # Test classifier direct analysis
        result = GroomingClassifier.analyze(raw_bytes)
        self.assertIn("scores", result)
        self.assertIn("hair", result)
        self.assertIn("facial_hair", result)
        self.assertIn("skin", result)
        self.assertIn("well_groomed", result)

        # Test POST /api/mirror/grooming/live
        client = app.test_client()
        buf.seek(0)
        res = client.post(
            "/api/mirror/grooming/live",
            data={"userId": "test_grooming_user", "image_head": (buf, "head.jpg")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("grooming", data)

        # Test GET /api/mirror/grooming/status
        res_status = client.get("/api/mirror/grooming/status?user_id=test_grooming_user")
        self.assertEqual(res_status.status_code, 200)
        status_data = res_status.get_json()
        self.assertEqual(status_data.get("status"), "ok")
        self.assertIn("hair", status_data)
        self.assertIn("skin", status_data)

        # Test GET /api/status includes grooming
        res_overall = client.get("/api/status")
        self.assertEqual(res_overall.status_code, 200)
        overall_data = res_overall.get_json()
        self.assertIn("grooming", overall_data)

    def test_mirror_stylist_single_shirt_no_phantom_pants(self):
        """Ensure that when only a shirt is visible, MirrorStylist does not hallucinate pants."""
        user_id = "single_shirt_user"
        detected = {
            "category": "top",
            "color": "White",
            "sub_category": "Shirt",
            "suggested_name": "White Shirt",
            "confidence": 0.95,
        }
        result = MirrorStylist.assess_clothing(user_id, precomputed_detection=detected)
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["detected"]["bottom"])
        self.assertEqual(result["detected"]["name"], "White Shirt")
        self.assertNotIn("+", result["detected"]["name"])
        self.assertIn("wearing your White Shirt", result["voice_summary"])
        self.assertNotIn("with", result["voice_summary"])


if __name__ == "__main__":
    unittest.main()

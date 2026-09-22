import unittest
from datetime import date
from unittest.mock import patch

from app.database.database import get_connection, init_db
from app.database.repositories import WardrobeRepository, OutfitLogRepository
from app.recommendations.outfit_engine import OutfitEngine
from app.config.settings import settings


class TestWardrobeAndOutfitEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an in-memory test database
        settings.DATABASE_PATH = settings.DATA_DIR / "test_reflectai.db"
        
        # Clear it if exists
        if settings.DATABASE_PATH.exists():
            settings.DATABASE_PATH.unlink()
            
        init_db()

    @classmethod
    def tearDownClass(cls):
        if settings.DATABASE_PATH.exists():
            settings.DATABASE_PATH.unlink()

    def setUp(self):
        # Clear tables before each test
        with get_connection() as conn:
            conn.execute("DELETE FROM wardrobe")
            conn.execute("DELETE FROM outfit_log")

    def test_wardrobe_crud(self):
        # Add item
        item_id = WardrobeRepository.add_item(
            user_id="test_user",
            name="Blue Jeans",
            category="bottom",
            color="Blue",
            weather="any",
            occasion="casual",
            image_filename="jeans.jpg"
        )
        self.assertIsNotNone(item_id)

        # Get item
        item = WardrobeRepository.get_item(item_id)
        self.assertEqual(item["name"], "Blue Jeans")
        self.assertEqual(item["category"], "bottom")
        self.assertEqual(item["image_filename"], "jeans.jpg")
        self.assertEqual(item["times_worn"], 0)

        # Update item
        WardrobeRepository.update_item(item_id, color="Navy Blue")
        updated_item = WardrobeRepository.get_item(item_id)
        self.assertEqual(updated_item["color"], "Navy Blue")

        # Mark worn
        today_str = date.today().isoformat()
        WardrobeRepository.mark_worn(item_id, today_str)
        worn_item = WardrobeRepository.get_item(item_id)
        self.assertEqual(worn_item["times_worn"], 1)
        self.assertEqual(worn_item["last_worn"], today_str)

        # List items
        WardrobeRepository.add_item("test_user", "Red Shirt", "top")
        items = WardrobeRepository.list_for_user("test_user")
        self.assertEqual(len(items), 2)
        
        # Filter by category
        tops = WardrobeRepository.list_for_user("test_user", category="top")
        self.assertEqual(len(tops), 1)
        self.assertEqual(tops[0]["name"], "Red Shirt")

    def test_outfit_log(self):
        user_id = "test_user_log"
        today_str = date.today().isoformat()
        
        OutfitLogRepository.log_outfit(user_id, today_str, top_id=1, bottom_id=2, temperature=25, weather_condition="Clear")
        
        today_log = OutfitLogRepository.get_today(user_id)
        self.assertIsNotNone(today_log)
        self.assertEqual(today_log["top_id"], 1)
        self.assertEqual(today_log["bottom_id"], 2)
        
        # Logging again on same day replaces it
        OutfitLogRepository.log_outfit(user_id, today_str, top_id=3, bottom_id=4)
        today_log_updated = OutfitLogRepository.get_today(user_id)
        self.assertEqual(today_log_updated["top_id"], 3)
        self.assertEqual(today_log_updated["bottom_id"], 4)

    @patch('app.recommendations.outfit_engine.WeatherService')
    def test_outfit_engine_recommendation(self, mock_weather):
        mock_weather.get_weather.return_value = {
            "temperature": 25,
            "icon": "clear",
            "condition": "Clear sky"
        }
        
        user_id = "engine_test_user"
        
        # Should return empty
        res_empty = OutfitEngine.recommend(user_id)
        self.assertEqual(res_empty["status"], "empty_wardrobe")

        # Add items
        WardrobeRepository.add_item(user_id, "T-Shirt", "top", weather="hot")
        WardrobeRepository.add_item(user_id, "Jeans", "bottom", weather="any")
        WardrobeRepository.add_item(user_id, "Sneakers", "footwear", weather="any")
        
        res = OutfitEngine.recommend(user_id)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["items"]), 3)
        
        labels = [i["label"] for i in res["items"]]
        self.assertIn("Top", labels)
        self.assertIn("Bottom", labels)
        self.assertIn("Footwear", labels)
        
        # Verify it logged
        today_log = OutfitLogRepository.get_today(user_id)
        self.assertIsNotNone(today_log)

    @patch('app.recommendations.outfit_engine.WeatherService')
    def test_outfit_engine_weather_filtering(self, mock_weather):
        # Cold and rainy
        mock_weather.get_weather.return_value = {
            "temperature": 10,
            "icon": "rain",
            "condition": "Moderate rain"
        }
        
        user_id = "weather_user"
        
        WardrobeRepository.add_item(user_id, "Tank Top", "top", weather="hot")
        WardrobeRepository.add_item(user_id, "Sweater", "top", weather="cold")
        WardrobeRepository.add_item(user_id, "Shorts", "bottom", weather="hot")
        WardrobeRepository.add_item(user_id, "Jeans", "bottom", weather="any")
        WardrobeRepository.add_item(user_id, "Rain Coat", "outerwear", weather="rain")
        
        res = OutfitEngine.recommend(user_id)
        self.assertEqual(res["status"], "ok")
        
        names = [i["name"] for i in res["items"]]
        self.assertIn("Sweater", names)
        self.assertIn("Jeans", names)
        self.assertIn("Rain Coat", names)
        self.assertNotIn("Tank Top", names)
        self.assertNotIn("Shorts", names)

    def test_clothing_classifier_inference(self):
        from app.recommendations.clothing_classifier import ClothingClassifier
        from PIL import Image
        import io

        # Create a small blank image in memory
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        result = ClothingClassifier.classify(buf)
        self.assertIsNotNone(result)
        self.assertIn("category", result)
        self.assertIn("color", result)
        self.assertIn("suggested_name", result)

    def test_api_wardrobe_classify(self):
        from app.dashboard.dashboard import app
        from PIL import Image
        import io

        client = app.test_client()
        img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        res = client.post("/api/wardrobe/classify", data={"image": (buf, "test_blue.jpg")})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("prediction", data)
        self.assertIn("category", data["prediction"])


if __name__ == "__main__":
    unittest.main()

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

    def test_shirt_not_misclassified_as_pants(self):
        """Verify real user shirts are recognized as top/Shirt and never as bottom/Pants."""
        from app.recommendations.clothing_classifier import ClothingClassifier
        import os
        from pathlib import Path

        wardrobe_dir = Path("data/wardrobe/maanik")
        if not wardrobe_dir.exists():
            self.skipTest("Wardrobe test images not present")

        shirt_files = ["c12b2f650091.jpg", "e118ad72a047.jpg"]
        for fn in shirt_files:
            p = wardrobe_dir / fn
            if p.exists():
                res = ClothingClassifier.classify(p)
                self.assertIsNotNone(res)
                self.assertEqual(res["category"], "top", f"Shirt {fn} misclassified as {res['category']}")
                self.assertEqual(res["sub_category"], "Shirt", f"Shirt {fn} subcategory was {res['sub_category']}")
                self.assertNotEqual(res["category"], "bottom")

    def test_bottom_accurately_classified(self):
        """Verify real user bottomwear is recognized as bottomwear."""
        from app.recommendations.clothing_classifier import ClothingClassifier
        from pathlib import Path

        wardrobe_dir = Path("data/wardrobe/maanik")
        if not wardrobe_dir.exists():
            self.skipTest("Wardrobe test images not present")

        bottom_files = ["66027f813ef4.jpg", "8b05794d1f1b.jpg", "c54f285323ab.jpg"]
        for fn in bottom_files:
            p = wardrobe_dir / fn
            if p.exists():
                res = ClothingClassifier.classify(p)
                self.assertIsNotNone(res)
                self.assertEqual(res["category"], "bottom", f"Garment {fn} misclassified as {res['category']}")

    def test_portrait_shirt_does_not_default_to_bottom(self):
        """Verify that portrait vertical aspect ratio (> 1.25) does not bias towards pants."""
        from app.recommendations.clothing_classifier import ClothingClassifier
        from pathlib import Path

        # Both sample shirts have portrait aspect ratio ~1.78
        for fn in ["c12b2f650091.jpg", "e118ad72a047.jpg"]:
            p = Path("data/wardrobe/maanik") / fn
            if p.exists():
                res = ClothingClassifier.classify(p)
                self.assertEqual(res["category"], "top")


if __name__ == "__main__":
    unittest.main()

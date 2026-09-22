"""
app/recommendations/mirror_stylist.py — Live Camera Clothing Stylist Engine.

Analyzes what the user is currently wearing in front of the mirror and delivers
multi-aspect styling advice grounded in:
  1. Live weather conditions (Open-Meteo)
  2. Today's calendar schedule & occasion
  3. The user's actual photographed wardrobe (SQLite)
  4. Wear history / repetition check
"""

import logging
from typing import Any, Dict, List, Optional

from app.calendar import CalendarService
from app.database.repositories import OutfitLogRepository, WardrobeRepository
from app.recommendations.clothing_classifier import ClothingClassifier
from app.weather import WeatherService

logger = logging.getLogger("reflectai.stylist")


class MirrorStylist:
    @classmethod
    def assess_clothing(
        cls,
        user_id: str,
        image_bytes: Optional[bytes] = None,
        image_top_bytes: Optional[bytes] = None,
        image_bottom_bytes: Optional[bytes] = None,
        precomputed_detection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assess what the user is wearing (topwear and bottomwear) and provide closet recommendations.
        """
        top_img = image_top_bytes or image_bytes
        detected = precomputed_detection
        if detected is None and top_img is not None:
            detected = ClothingClassifier.classify(top_img, region="top")

        if not detected:
            detected = {
                "category": "top",
                "color": "Neutral",
                "sub_category": "Topwear",
                "suggested_name": "Topwear",
            }

        detected_cat = detected.get("category", "top")
        detected_color = (detected.get("color") or "Neutral").title()
        detected_name = detected.get("suggested_name") or f"{detected_color} {detected_cat.title()}"

        # Detect bottomwear if camera sees lower body
        detected_bottom = None
        if image_bottom_bytes is not None:
            detected_bottom = ClothingClassifier.classify(image_bottom_bytes, region="bottom")

        # 2. Fetch live environmental & user context
        weather = WeatherService.get_weather()
        temp = weather.get("temperature", 24)
        condition = weather.get("condition", "Clear")
        icon = weather.get("icon", "clear")

        events = CalendarService.get_events_for_user(user_id)
        wardrobe = WardrobeRepository.list_for_user(user_id, active_only=True)

        # 3. Assess Weather Suitability
        weather_notes = []
        needs_layer = False

        if temp < 18 or icon in ("rain", "drizzle", "showers", "snow", "thunderstorm"):
            if detected_cat != "outerwear":
                needs_layer = True
                weather_notes.append(f"It's {temp}° & {condition.lower()} — a light top isn't warm enough!")
            else:
                weather_notes.append(f"Good jacket for today's {temp}° & {condition.lower()}.")
        elif temp >= 30:
            if detected_cat == "outerwear" or "black" in detected_color.lower():
                weather_notes.append(f"It's hot ({temp}°) today — a lighter layer is recommended.")
            else:
                weather_notes.append(f"Great lightweight choice for the {temp}° heat.")
        else:
            weather_notes.append(f"Comfortable for {temp}°.")

        # 4. Assess Schedule & Formality
        schedule_notes = []
        is_formal_day = False
        formal_keywords = ["interview", "meeting", "presentation", "conference", "review", "exam", "client"]
        athletic_keywords = ["gym", "workout", "run", "training", "cricket", "football", "yoga"]

        for evt in events:
            title_lower = evt.get("title", "").lower()
            if any(k in title_lower for k in formal_keywords):
                is_formal_day = True
                schedule_notes.append(f"You have '{evt['title']}' today.")
                break
            elif any(k in title_lower for k in athletic_keywords):
                schedule_notes.append(f"Workout day: '{evt['title']}'.")
                break

        # 5. Check Repeat Wears
        repeat_warning = None
        recent_logs = OutfitLogRepository.get_recent(user_id, days=2)
        if recent_logs:
            yesterday_log = recent_logs[0]
            # Check if user had a top logged yesterday
            if yesterday_log.get("top_id"):
                yesterday_top = WardrobeRepository.get_item(yesterday_log["top_id"])
                if yesterday_top and yesterday_top.get("color", "").lower() == detected_color.lower():
                    repeat_warning = f"You wore a {detected_color} top yesterday too!"

        # 6. Select Matching Items from Personal Wardrobe
        pairings: List[Dict[str, Any]] = []

        # Find recommended Bottom from closet
        bottoms = [item for item in wardrobe if item.get("category") == "bottom"]
        if bottoms:
            # Contrast color heuristic: light top pairs with dark bottoms
            preferred_bottom = None
            if detected_color.lower() in ("white", "beige", "cream", "yellow", "light blue"):
                # prefer dark bottoms
                dark_bottoms = [b for b in bottoms if (b.get("color") or "").lower() in ("black", "blue", "navy blue", "charcoal", "dark grey")]
                if dark_bottoms:
                    preferred_bottom = dark_bottoms[0]
            if not preferred_bottom:
                preferred_bottom = bottoms[0]

            pairings.append({
                "id": preferred_bottom["id"],
                "label": "Pair with Bottom",
                "name": preferred_bottom["name"],
                "category": "bottom",
                "color": preferred_bottom.get("color"),
                "image": preferred_bottom.get("image_filename"),
                "user_id": user_id,
            })

        # If cold/rain, find recommended Outerwear from closet
        if needs_layer:
            outerwear_items = [item for item in wardrobe if item.get("category") == "outerwear"]
            if outerwear_items:
                best_outerwear = outerwear_items[0]
                pairings.append({
                    "id": best_outerwear["id"],
                    "label": "Grab Jacket",
                    "name": best_outerwear["name"],
                    "category": "outerwear",
                    "color": best_outerwear.get("color"),
                    "image": best_outerwear.get("image_filename"),
                    "user_id": user_id,
                })

        # Find Footwear
        footwear_items = [item for item in wardrobe if item.get("category") == "footwear"]
        if footwear_items:
            best_shoes = footwear_items[0]
            pairings.append({
                "id": best_shoes["id"],
                "label": "Shoes",
                "name": best_shoes["name"],
                "category": "footwear",
                "color": best_shoes.get("color"),
                "image": best_shoes.get("image_filename"),
                "user_id": user_id,
            })

        # 7. Generate Conversational Voice & Text Advice
        advice_parts = []
        if weather_notes:
            advice_parts.append(weather_notes[0])
        if schedule_notes:
            advice_parts.append(schedule_notes[0])
        if repeat_warning:
            advice_parts.append(repeat_warning)

        if pairings:
            closet_names = [p["name"] for p in pairings if p["category"] in ("bottom", "outerwear")]
            if closet_names:
                advice_parts.append(f"Pair with your {', '.join(closet_names)} from your closet.")

        display_name = detected_name
        if detected_bottom and detected_bottom.get("suggested_name"):
            display_name = f"{detected_name} + {detected_bottom['suggested_name']}"

        advice_text = " · ".join(advice_parts) if advice_parts else "Looking good and ready for the day!"

        # Voice Assistant script
        voice_summary = f"You're wearing your {detected_name}"
        if detected_bottom and detected_bottom.get("suggested_name"):
            voice_summary += f" with {detected_bottom['suggested_name']}."
        else:
            voice_summary += f". It's {temp} degrees and {condition.lower()} today."

        if needs_layer and any(p["category"] == "outerwear" for p in pairings):
            layer_name = next(p["name"] for p in pairings if p["category"] == "outerwear")
            voice_summary += f" Make sure to grab your {layer_name} before leaving."

        return {
            "status": "ok",
            "user_id": user_id,
            "detected": {
                "name": display_name,
                "category": detected_cat,
                "color": detected_color,
                "sub_category": detected.get("sub_category", ""),
                "top": detected,
                "bottom": detected_bottom,
            },
            "weather": {
                "temperature": temp,
                "condition": condition,
            },
            "advice": advice_text,
            "needs_layer": needs_layer,
            "pairings": pairings,
            "voice_summary": voice_summary.strip(),
        }

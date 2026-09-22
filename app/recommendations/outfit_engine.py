"""
app/recommendations/outfit_engine.py — Smart outfit recommendation engine.

Recommends daily outfits exclusively from the user's own photographed wardrobe,
factoring in:
  1. Current weather (temperature + condition from Open-Meteo)
  2. Wear history (penalizes recently worn items to avoid repeats)
  3. Category requirements (top + bottom + footwear, outerwear if cold/rain)
"""

import logging
import random
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from app.database.repositories import WardrobeRepository, OutfitLogRepository
from app.weather import WeatherService

logger = logging.getLogger("reflectai.outfit")

# Weather-to-suitability mapping: which weather tags are suitable for conditions
WEATHER_SUITABILITY = {
    "hot":  {"clear", "partly-cloudy"},
    "warm": {"clear", "partly-cloudy", "cloudy", "fog"},
    "cool": {"cloudy", "fog", "drizzle", "showers"},
    "cold": {"snow", "fog", "cloudy"},
    "rain": {"rain", "drizzle", "showers", "thunderstorm"},
    "any":  {"clear", "partly-cloudy", "cloudy", "fog", "drizzle", "rain",
             "showers", "thunderstorm", "snow"},
}

# Temperature thresholds for weather category inference
def infer_weather_tag(temp: float) -> str:
    if temp >= 32:
        return "hot"
    elif temp >= 24:
        return "warm"
    elif temp >= 15:
        return "cool"
    else:
        return "cold"


def _freshness_score(item: Dict[str, Any]) -> float:
    """
    Score an item based on how recently it was worn.
    Higher score = more desirable to wear today.
    """
    last_worn = item.get("last_worn")
    if not last_worn:
        return 1.1  # Never worn — slight bonus

    try:
        last_date = datetime.fromisoformat(last_worn).date()
    except (ValueError, TypeError):
        return 1.0

    days_since = (date.today() - last_date).days

    if days_since <= 0:
        return 0.0   # Worn today — excluded
    elif days_since == 1:
        return 0.2   # Yesterday — heavy penalty
    elif days_since == 2:
        return 0.5   # Two days ago — moderate penalty
    elif days_since == 3:
        return 0.8   # Three days ago — light penalty
    else:
        return 1.0   # 4+ days — no penalty


def _weather_match_score(item: Dict[str, Any], weather_icon: str, temp: float) -> float:
    """
    Score an item based on how well it matches the current weather.
    Returns 0.0 if the item is unsuitable, 1.0 if suitable, 1.2 if ideal.
    """
    item_weather = (item.get("weather") or "any").lower()

    # "any" always matches
    if item_weather == "any":
        return 1.0

    # Check if item's weather tag is suitable for current conditions
    suitable_conditions = WEATHER_SUITABILITY.get(item_weather, set())
    if weather_icon in suitable_conditions:
        return 1.2  # Ideal match

    # Check temperature range compatibility
    temp_min = item.get("temperature_min")
    temp_max = item.get("temperature_max")
    if temp_min is not None and temp < temp_min:
        return 0.0  # Too cold for this item
    if temp_max is not None and temp > temp_max:
        return 0.0  # Too hot for this item

    # Infer from temperature
    inferred_tag = infer_weather_tag(temp)
    if item_weather == inferred_tag:
        return 1.2

    # Opposite weather — unsuitable
    opposites = {"hot": "cold", "cold": "hot"}
    if opposites.get(item_weather) == inferred_tag:
        return 0.1

    return 0.6  # Marginal match


def _pick_best(items: List[Dict], weather_icon: str, temp: float) -> Optional[Dict]:
    """Pick the best item from a list based on weather and freshness scores."""
    if not items:
        return None

    scored = []
    for item in items:
        freshness = _freshness_score(item)
        if freshness <= 0.0:
            continue  # Skip items worn today
        weather = _weather_match_score(item, weather_icon, temp)
        if weather <= 0.0:
            continue  # Skip unsuitable items
        total = freshness * weather + random.uniform(0, 0.15)  # Small randomness
        scored.append((total, item))

    if not scored:
        # All items filtered out — fall back to least recently worn
        items_sorted = sorted(items, key=lambda x: x.get("last_worn") or "")
        return items_sorted[0] if items_sorted else None

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _needs_outerwear(temp: float, weather_icon: str) -> bool:
    """Determine if outerwear should be recommended."""
    return temp < 20 or weather_icon in ("rain", "drizzle", "showers", "thunderstorm", "snow")


class OutfitEngine:
    @classmethod
    def recommend(cls, user_id: str) -> Dict[str, Any]:
        """
        Generate a daily outfit recommendation for the given user.
        Returns a dict with items, reasoning, and metadata.
        """
        # 1. Fetch weather
        weather = WeatherService.get_weather()
        temp = weather.get("temperature", 25)
        weather_icon = weather.get("icon", "clear")
        condition = weather.get("condition", "Clear")

        # 2. Fetch user's wardrobe
        all_items = WardrobeRepository.list_for_user(user_id, active_only=True)

        if not all_items:
            return {
                "status": "empty_wardrobe",
                "message": f"Add your clothes at /wardrobe to get personalized outfit suggestions!",
                "items": [],
                "weather": {"temperature": temp, "condition": condition},
            }

        # 3. Group by category
        by_category = {}
        for item in all_items:
            cat = (item.get("category") or "top").lower()
            by_category.setdefault(cat, []).append(item)

        # 4. Pick best item per category
        top = _pick_best(by_category.get("top", []), weather_icon, temp)
        bottom = _pick_best(by_category.get("bottom", []), weather_icon, temp)
        footwear = _pick_best(by_category.get("footwear", []), weather_icon, temp)

        outerwear = None
        if _needs_outerwear(temp, weather_icon) and by_category.get("outerwear"):
            outerwear = _pick_best(by_category["outerwear"], weather_icon, temp)

        # 5. Build result items list
        items = []
        for label, pick in [("Top", top), ("Bottom", bottom),
                             ("Footwear", footwear), ("Outerwear", outerwear)]:
            if pick:
                items.append({
                    "id": pick["id"],
                    "label": label,
                    "name": pick["name"],
                    "category": pick["category"],
                    "color": pick.get("color"),
                    "image": pick.get("image_filename"),
                    "user_id": user_id,
                })

        # 6. Build reasoning
        reasons = []
        reasons.append(f"{temp}° & {condition.lower()}")
        if _needs_outerwear(temp, weather_icon):
            reasons.append("layer up recommended")
        if top and top.get("last_worn"):
            reasons.append("fresh rotation")

        # 7. Log the outfit
        today_str = date.today().isoformat()
        OutfitLogRepository.log_outfit(
            user_id=user_id,
            date=today_str,
            top_id=top["id"] if top else None,
            bottom_id=bottom["id"] if bottom else None,
            footwear_id=footwear["id"] if footwear else None,
            outerwear_id=outerwear["id"] if outerwear else None,
            temperature=temp,
            weather_condition=condition,
        )

        # 8. Mark items as worn today
        for pick in [top, bottom, footwear, outerwear]:
            if pick:
                WardrobeRepository.mark_worn(pick["id"], today_str)

        return {
            "status": "ok",
            "items": items,
            "reason": " · ".join(reasons),
            "weather": {"temperature": temp, "condition": condition},
        }

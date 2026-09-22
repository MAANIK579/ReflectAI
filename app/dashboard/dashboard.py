"""
app/dashboard/dashboard.py — Full-screen smart mirror dashboard.

Serves the mirror UI and JSON APIs for: status, weather, calendar,
voice assistant, wardrobe management, and outfit recommendations.
"""

import logging
import os
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory

from app.calendar import CalendarService
from app.config.settings import settings
from app.database.repositories import (
    SettingsRepository, UserRepository, WardrobeRepository, OutfitLogRepository,
)
from app.recommendations import OutfitEngine, ClothingClassifier, MirrorStylist, GroomingClassifier
from app.weather import WeatherService

logger = logging.getLogger("reflectai.dashboard")

WARDROBE_IMAGE_DIR = settings.DATA_DIR / "wardrobe"

_live_stylist_state = {}
_live_grooming_state = {}

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="assets",
    static_url_path="/assets",
)

_voice_state = {
    "state": "idle",
    "text": "",
    "reply": ""
}

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/wardrobe")
def wardrobe_page():
    return render_template("wardrobe.html")


@app.route("/api/voice/event", methods=["POST"])
def voice_event():
    data = request.get_json(force=True, silent=True) or {}
    _voice_state["state"] = data.get("state", "idle")
    _voice_state["text"] = data.get("text", "")
    _voice_state["reply"] = data.get("reply", "")
    return jsonify({"status": "ok"})


@app.route("/api/user/active", methods=["POST"])
def set_active_user():
    """
    Called by face_engine/face_service.py whenever it confirms someone's
    identity (or reverts to 'guest' after no face is seen for a while).

    If the recognized user_id doesn't exist in ReflectAI's database yet
    (e.g. they were just registered via face_engine/register_user.py),
    create a minimal user record for them here so the dashboard can show
    their name immediately — richer profile data (outfit/hair/calendar
    preferences) gets wired in properly during Phases 6-8.
    """
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("userId")
    name = data.get("name")

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    if UserRepository.get(user_id) is None:
        UserRepository.create_or_update(user_id, name or user_id.replace("_", " ").title())
        logger.info(f"Auto-created user record for newly recognized face: {user_id}")

    SettingsRepository.set("active_user", user_id)
    logger.info(f"Active user switched to: {user_id}")

    return jsonify({"status": "ok", "active_user": user_id})


@app.route("/api/status")
def status():
    # active_user is set by the face engine's POST to /api/user/active
    # (see face_engine/face_service.py). Defaults to "user1" if nothing
    # has set it yet (e.g. face engine not running).
    active_user_id = SettingsRepository.get("active_user", default="user1")
    user = UserRepository.get(active_user_id) or {"name": "Guest"}

    privacy_mode = SettingsRepository.get("privacy_mode", default="false") == "true"

    return jsonify({
        "user": {
            "id": active_user_id,
            "name": user["name"],
        },
        "privacy_mode": privacy_mode,
        "voice": _voice_state,
        "stylist": _live_stylist_state.get(active_user_id),
        "grooming": _live_grooming_state.get(active_user_id),
        "system": {
            "esp32_mock_mode": settings.ESP32_MOCK_MODE,
            "camera_mock_mode": settings.CAMERA_MOCK_MODE,
            # Real connection status arrives in Phase 3 (ESP32) and
            # Phase 4 (camera) — reported as "not yet connected" until
            # those phases exist, rather than faking a status.
            "esp32_status": "not yet connected" if not settings.ESP32_MOCK_MODE else "mock mode",
            "camera_status": "not yet connected" if not settings.CAMERA_MOCK_MODE else "mock mode",
        },
    })


@app.route("/api/weather")
def weather():
    force_refresh = request.args.get("refresh", "false").lower() in ("1", "true", "yes")
    data = WeatherService.get_weather(force_refresh=force_refresh)
    return jsonify(data)


@app.route("/api/calendar")
def calendar():
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = SettingsRepository.get("active_user", default="guest")
    events = CalendarService.get_events_for_user(user_id)
    return jsonify({
        "user_id": user_id,
        "events": events,
    })


# ── Wardrobe API ──────────────────────────────────────────────

@app.route("/api/wardrobe")
def api_wardrobe_list():
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = SettingsRepository.get("active_user", default="guest")
    items = WardrobeRepository.list_for_user(user_id)
    return jsonify({"user_id": user_id, "items": items})


@app.route("/api/wardrobe/classify", methods=["POST"])
def api_wardrobe_classify():
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return jsonify({"error": "No image provided"}), 400

    img_bytes = image_file.read()
    prediction = ClothingClassifier.classify(img_bytes)
    if not prediction:
        return jsonify({"error": "Could not classify image"}), 500

    return jsonify({"status": "ok", "prediction": prediction})


@app.route("/api/wardrobe", methods=["POST"])
def api_wardrobe_add():
    user_id = request.form.get("user_id")
    name = request.form.get("name", "").strip()
    category = request.form.get("category", "").strip().lower()
    color = request.form.get("color", "").strip() or None
    weather_tag = request.form.get("weather", "any").strip().lower()
    occasion = request.form.get("occasion", "casual").strip().lower()

    if not user_id:
        user_id = SettingsRepository.get("active_user", default="guest")

    # Handle image upload
    image_filename = None
    image_file = request.files.get("image")
    if image_file and image_file.filename:
        img_bytes = image_file.read()

        # If user didn't specify name, category, or color, run AI classification
        if not category or not color or not name:
            pred = ClothingClassifier.classify(img_bytes)
            if pred:
                category = category or pred.get("category", "top")
                color = color or pred.get("color")
                if not name:
                    name = pred.get("suggested_name", "Clothing Item")

        user_dir = WARDROBE_IMAGE_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        ext = os.path.splitext(image_file.filename)[1] or ".jpg"
        image_filename = f"{uuid.uuid4().hex[:12]}{ext}"
        with open(str(user_dir / image_filename), "wb") as f:
            f.write(img_bytes)

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not category:
        category = "top"
    if category not in ("top", "bottom", "footwear", "outerwear", "accessory"):
        return jsonify({"error": "Invalid category"}), 400

    item_id = WardrobeRepository.add_item(
        user_id=user_id,
        name=name,
        category=category,
        color=color,
        weather=weather_tag,
        occasion=occasion,
        image_filename=image_filename,
    )
    logger.info(f"Added wardrobe item '{name}' (id={item_id}) for user {user_id}")
    return jsonify({"status": "ok", "item_id": item_id})


@app.route("/api/wardrobe/<int:item_id>", methods=["DELETE"])
def api_wardrobe_delete(item_id):
    item = WardrobeRepository.get_item(item_id)
    if not item:
        return jsonify({"error": "Item not found"}), 404
    WardrobeRepository.delete_item(item_id)
    # Clean up image file
    if item.get("image_filename"):
        img_path = WARDROBE_IMAGE_DIR / item["user_id"] / item["image_filename"]
        if img_path.exists():
            img_path.unlink()
    return jsonify({"status": "ok"})


# ── Outfit Recommendation API ────────────────────────────────

@app.route("/api/outfit")
def api_outfit():
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = SettingsRepository.get("active_user", default="guest")
    result = OutfitEngine.recommend(user_id)
    return jsonify(result)


# ── Live Camera Clothing & AI Stylist API ─────────────────────

@app.route("/api/mirror/clothing/live", methods=["POST"])
def api_mirror_clothing_live():
    """
    Called by face_service.py when a user is recognized in front of the camera.
    Receives a torso crop JPEG from the webcam and runs MirrorStylist.
    """
    user_id = request.form.get("userId") or request.args.get("userId")
    if not user_id:
        user_id = SettingsRepository.get("active_user", default="guest")

    image_file = request.files.get("image") or request.files.get("image_top")
    img_top_bytes = image_file.read() if image_file else None

    image_bottom_file = request.files.get("image_bottom")
    img_bottom_bytes = image_bottom_file.read() if image_bottom_file else None

    image_head_file = request.files.get("image_head")
    if image_head_file:
        try:
            head_bytes = image_head_file.read()
            grooming_res = GroomingClassifier.analyze(head_bytes)
            if grooming_res:
                _live_grooming_state[user_id] = grooming_res
        except Exception as e:
            logger.warning(f"Grooming analysis failed: {e}")

    assessment = MirrorStylist.assess_clothing(
        user_id=user_id,
        image_top_bytes=img_top_bytes,
        image_bottom_bytes=img_bottom_bytes,
    )
    _live_stylist_state[user_id] = assessment
    logger.info(f"Live clothing assessment for {user_id}: {assessment.get('detected', {}).get('name')}")
    return jsonify({
        "status": "ok",
        "assessment": assessment,
        "grooming": _live_grooming_state.get(user_id),
    })


@app.route("/api/mirror/clothing/status")
def api_mirror_clothing_status():
    """Returns the latest clothing assessment & stylist advice for the active user."""
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = SettingsRepository.get("active_user", default="guest")

    state = _live_stylist_state.get(user_id)
    if not state:
        state = MirrorStylist.assess_clothing(user_id=user_id)
        _live_stylist_state[user_id] = state

    return jsonify(state)


# ── Live Camera Grooming & Wellness API ───────────────────────

@app.route("/api/mirror/grooming/live", methods=["POST"])
def api_mirror_grooming_live():
    """
    Called by face_service.py to send head/face crop for grooming and skin analysis.
    """
    user_id = request.form.get("userId") or request.args.get("userId")
    if not user_id:
        user_id = SettingsRepository.get("active_user", default="guest")

    image_file = request.files.get("image") or request.files.get("image_head")
    if not image_file:
        return jsonify({"error": "No head crop image provided"}), 400

    img_bytes = image_file.read()
    analysis = GroomingClassifier.analyze(img_bytes)
    if not analysis:
        return jsonify({"error": "Analysis failed"}), 500

    _live_grooming_state[user_id] = analysis
    return jsonify({"status": "ok", "grooming": analysis})


@app.route("/api/mirror/grooming/status")
def api_mirror_grooming_status():
    """Returns the latest grooming & appearance analysis for the active user."""
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = SettingsRepository.get("active_user", default="guest")

    state = _live_grooming_state.get(user_id)
    return jsonify(state or {"status": "idle"})


# ── Wardrobe Image Serving ───────────────────────────────────

@app.route("/wardrobe-images/<user_id>/<filename>")
def wardrobe_image(user_id, filename):
    return send_from_directory(str(WARDROBE_IMAGE_DIR / user_id), filename)


def run_dashboard(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    logger.info(f"Starting dashboard server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    from app.config.logging_config import setup_logging
    setup_logging()
    run_dashboard(debug=True)


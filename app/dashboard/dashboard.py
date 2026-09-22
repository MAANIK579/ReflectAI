"""
app/dashboard/dashboard.py — Full-screen smart mirror dashboard.

Serves the mirror UI and JSON APIs for: status, weather, calendar,
voice assistant, wardrobe management, and outfit recommendations.
"""

import json
import logging
import os
import re
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory, Response

from app.calendar import CalendarService
from app.config.settings import settings
from app.database.repositories import (
    SettingsRepository, UserRepository, WardrobeRepository, OutfitLogRepository, ReminderRepository,
)
from app.recommendations import OutfitEngine, ClothingClassifier, MirrorStylist, GroomingClassifier
from app.weather import WeatherService

logger = logging.getLogger("reflectai.dashboard")

WARDROBE_IMAGE_DIR = settings.DATA_DIR / "wardrobe"
EMBEDDINGS_DIR = settings.DATA_DIR / "embeddings"
PROFILES_FILE = settings.DATA_DIR / "profiles.json"
FACE_MODELS_DIR = settings.PROJECT_ROOT / "face_engine" / "models"
YUNET_MODEL = FACE_MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = FACE_MODELS_DIR / "face_recognition_sface_2021dec.onnx"

def _load_profiles_json():
    if PROFILES_FILE.exists():
        try:
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_profiles_json(profiles):
    PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

def _enroll_face(user_id: str, img_bytes: bytes) -> bool:
    """
    Detects face using YuNet and generates 128-D SFace embedding.
    Saves:
      - data/embeddings/<user_id>.npy
      - data/embeddings/<user_id>.jpg (cropped face avatar)
    """
    import cv2
    import numpy as np

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return False

    if not YUNET_MODEL.exists() or not SFACE_MODEL.exists():
        # Fallback: Save image as avatar even if face recognition models are not present
        cv2.imwrite(str(EMBEDDINGS_DIR / f"{user_id}.jpg"), img)
        return False

    try:
        h, w = img.shape[:2]
        detector = cv2.FaceDetectorYN.create(
            model=str(YUNET_MODEL),
            config="",
            input_size=(w, h),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        recognizer = cv2.FaceRecognizerSF.create(model=str(SFACE_MODEL), config="")

        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        if faces is not None and len(faces) > 0:
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            face = faces_sorted[0]
            aligned = recognizer.alignCrop(img, face)
            feat = recognizer.feature(aligned)
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            np.save(str(EMBEDDINGS_DIR / f"{user_id}.npy"), feat)
            cv2.imwrite(str(EMBEDDINGS_DIR / f"{user_id}.jpg"), aligned)
            return True
        else:
            # No face detected in frame; still save as avatar
            cv2.imwrite(str(EMBEDDINGS_DIR / f"{user_id}.jpg"), img)
            return False
    except Exception as e:
        logger.error(f"Error enrolling face for {user_id}: {e}")
        try:
            cv2.imwrite(str(EMBEDDINGS_DIR / f"{user_id}.jpg"), img)
        except Exception:
            pass
        return False

_live_stylist_state = {}
_live_grooming_state = {}

_esp32_state = {
    "device": "esp32",
    "status": "offline",
    "temperature": None,
    "humidity": None,
    "light": None,
    "motion": False,
    "last_seen": None,
}

_last_esp32_user = None
_last_esp32_privacy = None
_last_camera_seen = None

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


@app.route("/app")
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
    global _last_camera_seen
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("userId")
    name = data.get("name")

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    _last_camera_seen = time.time()

    if UserRepository.get(user_id) is None:
        UserRepository.create_or_update(user_id, name or user_id.replace("_", " ").title())
        logger.info(f"Auto-created user record for newly recognized face: {user_id}")

    SettingsRepository.set("active_user", user_id)
    logger.info(f"Active user switched to: {user_id}")

    return jsonify({"status": "ok", "active_user": user_id})


@app.route("/api/esp32/state", methods=["POST"])
def esp32_state():
    """
    Receives JSON telemetry and physical button interactions from ESP32:
    - User selection button (User 1, User 2, Guest)
    - Privacy mode toggle
    - Sensors: DHT22 (temp, humidity), LDR (light), PIR (motion)
    """
    global _last_esp32_user, _last_esp32_privacy
    data = request.get_json(force=True, silent=True) or {}

    # 1. Update active user ONLY when an ESP32 physical button is actually pressed
    # (i.e. on state transition, not on every background periodic sensor packet)
    raw_user = data.get("user")
    if raw_user:
        if _last_esp32_user is None:
            _last_esp32_user = raw_user
        elif raw_user != _last_esp32_user:
            _last_esp32_user = raw_user
            # map "User 1" -> "user1", "User 2" -> "user2", "Guest" -> "guest"
            user_id = raw_user.strip().lower().replace(" ", "")
            if UserRepository.get(user_id) is not None:
                SettingsRepository.set("active_user", user_id)
                logger.info(f"ESP32 physical button switched active user to: {user_id}")

    # 2. Update privacy mode ONLY when the privacy button state changes on ESP32
    if "privacy" in data:
        raw_privacy = bool(data.get("privacy"))
        if _last_esp32_privacy is None:
            _last_esp32_privacy = raw_privacy
        elif raw_privacy != _last_esp32_privacy:
            _last_esp32_privacy = raw_privacy
            privacy_val = "true" if raw_privacy else "false"
            SettingsRepository.set("privacy_mode", privacy_val)
            logger.info(f"ESP32 physical button changed privacy mode to: {privacy_val}")

    # 3. Save telemetry data
    _esp32_state["device"] = data.get("device", "esp32")
    _esp32_state["status"] = data.get("status", "online")
    _esp32_state["temperature"] = data.get("temperature")
    _esp32_state["humidity"] = data.get("humidity")
    _esp32_state["light"] = data.get("light")
    _esp32_state["motion"] = bool(data.get("motion", False))
    _esp32_state["last_seen"] = time.time()

    current_active = SettingsRepository.get("active_user", default="guest")
    return jsonify({
        "status": "ok",
        "active_user": current_active,
        "message": "ESP32 state updated",
        "data": _esp32_state
    })


@app.route("/api/status")
def status():
    # active_user is set by face engine or physical ESP32 buttons.
    active_user_id = SettingsRepository.get("active_user", default="user1")
    user = UserRepository.get(active_user_id) or {"name": "Guest"}

    privacy_mode = SettingsRepository.get("privacy_mode", default="false") == "true"

    now = time.time()
    esp32_online = (
        _esp32_state["last_seen"] is not None and
        (now - _esp32_state["last_seen"]) < 15
    )
    if esp32_online:
        esp32_status_str = "online"
    elif settings.ESP32_MOCK_MODE:
        esp32_status_str = "mock mode"
    else:
        esp32_status_str = "disconnected"

    camera_online = (
        _last_camera_seen is not None and
        (now - _last_camera_seen) < 25
    )
    if camera_online:
        camera_status_str = "online"
        camera_mock = False
    elif settings.CAMERA_MOCK_MODE:
        camera_status_str = "mock mode"
        camera_mock = True
    else:
        camera_status_str = "disconnected"
        camera_mock = False

    return jsonify({
        "user": {
            "id": active_user_id,
            "name": user["name"],
        },
        "privacy_mode": privacy_mode,
        "voice": _voice_state,
        "stylist": _live_stylist_state.get(active_user_id),
        "grooming": _live_grooming_state.get(active_user_id),
        "sensors": _esp32_state,
        "system": {
            "esp32_mock_mode": settings.ESP32_MOCK_MODE,
            "camera_mock_mode": camera_mock,
            "esp32_status": esp32_status_str,
            "camera_status": camera_status_str,
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


# ── User Profile & Facial Enrollment API ─────────────────────

@app.route("/api/users")
def api_users_list():
    active_user_id = SettingsRepository.get("active_user", default="guest")
    all_users = UserRepository.list_all()
    user_list = []
    for u in all_users:
        uid = u["id"]
        has_face = (EMBEDDINGS_DIR / f"{uid}.npy").exists()
        has_avatar = (EMBEDDINGS_DIR / f"{uid}.jpg").exists()
        items = WardrobeRepository.list_for_user(uid)
        user_list.append({
            "id": uid,
            "name": u["name"],
            "preferred_style": u.get("preferred_style") or "Casual",
            "hair_preference": u.get("hair_preference") or "Natural",
            "temperature_unit": u.get("temperature_unit") or "C",
            "avatar_url": f"/avatar/{uid}",
            "has_face": has_face,
            "has_avatar": has_avatar,
            "wardrobe_count": len(items),
            "is_active": (uid == active_user_id),
        })
    return jsonify({"active_user": active_user_id, "users": user_list})


@app.route("/api/user/register", methods=["POST"])
def api_user_register():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    preferred_style = request.form.get("preferred_style", "Casual").strip()
    hair_preference = request.form.get("hair_preference", "Natural").strip()
    switch_now = request.form.get("switch_now", "true").lower() in ("true", "1", "yes")

    # Clean ID: alphanumeric + underscore
    user_id = re.sub(r'[^a-z0-9_]', '', name.lower().replace(" ", "_"))
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:6]}"

    # Save to SQLite
    UserRepository.create_or_update(
        user_id=user_id,
        name=name,
        preferred_style=preferred_style,
        hair_preference=hair_preference,
    )

    # Face enrollment if photo uploaded
    photo_file = request.files.get("photo")
    has_face = False
    if photo_file and photo_file.filename:
        photo_bytes = photo_file.read()
        has_face = _enroll_face(user_id, photo_bytes)

    # Update profiles.json
    profiles = _load_profiles_json()
    profiles[user_id] = {
        "id": user_id,
        "name": name.upper(),
        "greeting_name": name.upper(),
        "outfit": ["Black T-Shirt", "Blue Jeans", "White Sneakers"],
        "hair": hair_preference,
        "calendar": [
            "10:30 AM · Project Review",
            "04:00 PM · Gym & Workout",
        ],
        "news": [
            "Tech updates & innovations",
            "Local weather outlook",
        ],
    }
    _save_profiles_json(profiles)

    if switch_now:
        SettingsRepository.set("active_user", user_id)
        logger.info(f"Active user switched to newly registered user: {user_id}")

    return jsonify({
        "status": "ok",
        "user_id": user_id,
        "name": name,
        "has_face": has_face,
        "active": switch_now,
        "message": f"User '{name}' registered successfully!"
    })


@app.route("/api/user/delete/<user_id>", methods=["POST", "DELETE"])
def api_user_delete(user_id):
    if user_id.lower() == "guest":
        return jsonify({"error": "Cannot delete default guest profile"}), 400

    UserRepository.delete(user_id)

    # Delete embeddings and avatar if present
    for ext in (".npy", ".jpg"):
        f = EMBEDDINGS_DIR / f"{user_id}{ext}"
        if f.exists():
            try:
                f.unlink()
            except Exception:
                pass

    # Delete wardrobe images directory
    user_wardrobe_dir = WARDROBE_IMAGE_DIR / user_id
    if user_wardrobe_dir.exists():
        try:
            import shutil
            shutil.rmtree(user_wardrobe_dir, ignore_errors=True)
        except Exception:
            pass

    # Remove from profiles.json
    profiles = _load_profiles_json()
    if user_id in profiles:
        del profiles[user_id]
        _save_profiles_json(profiles)

    # If active user was deleted, switch to guest
    current_active = SettingsRepository.get("active_user", default="guest")
    if current_active == user_id:
        SettingsRepository.set("active_user", "guest")

    return jsonify({"status": "ok", "message": f"User {user_id} deleted"})


@app.route("/avatar/<user_id>")
def user_avatar(user_id):
    avatar_file = EMBEDDINGS_DIR / f"{user_id}.jpg"
    if avatar_file.exists():
        return send_from_directory(str(EMBEDDINGS_DIR), f"{user_id}.jpg", mimetype="image/jpeg")

    user = UserRepository.get(user_id)
    initial = (user["name"][0] if user and user.get("name") else user_id[0]).upper()
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
      <defs>
        <linearGradient id="g_{user_id}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#e8b76d" />
          <stop offset="100%" stop-color="#b45309" />
        </linearGradient>
      </defs>
      <rect width="120" height="120" rx="60" fill="url(#g_{user_id})" />
      <text x="50%" y="54%" font-family="system-ui, -apple-system, sans-serif" font-size="52" font-weight="600" fill="#0a0a0a" text-anchor="middle" dominant-baseline="middle">{initial}</text>
    </svg>"""
    return Response(svg, mimetype="image/svg+xml")


# ── Mirror Remote Controls & Agenda API ───────────────────────

@app.route("/api/mirror/privacy/toggle", methods=["POST"])
def toggle_privacy():
    current = SettingsRepository.get("privacy_mode", default="false") == "true"
    new_val = "false" if current else "true"
    SettingsRepository.set("privacy_mode", new_val)
    logger.info(f"Privacy mode toggled from remote to: {new_val}")
    return jsonify({"status": "ok", "privacy_mode": new_val == "true"})


@app.route("/api/reminders", methods=["GET"])
def api_reminders_list():
    user_id = request.args.get("user_id") or SettingsRepository.get("active_user", default="guest")
    reminders = ReminderRepository.list_for_user(user_id)
    return jsonify({"user_id": user_id, "reminders": reminders})


@app.route("/api/reminders", methods=["POST"])
def api_reminders_add():
    data = request.get_json(force=True, silent=True) or request.form or {}
    user_id = data.get("user_id") or SettingsRepository.get("active_user", default="guest")
    title = data.get("title", "").strip()
    datetime_str = data.get("datetime", "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400
    if not datetime_str:
        import datetime
        datetime_str = datetime.datetime.now().strftime("%I:%M %p")
    ReminderRepository.create(user_id=user_id, title=title, datetime_str=datetime_str)
    return jsonify({"status": "ok", "message": "Reminder added"})


@app.route("/api/reminders/<int:reminder_id>", methods=["DELETE"])
def api_reminders_delete(reminder_id):
    ReminderRepository.delete(reminder_id)
    return jsonify({"status": "ok"})


def run_dashboard(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    logger.info(f"Starting dashboard server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    from app.config.logging_config import setup_logging
    setup_logging()
    run_dashboard(debug=True)


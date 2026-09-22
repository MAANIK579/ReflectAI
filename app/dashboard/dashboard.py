"""
app/dashboard/dashboard.py — Full-screen smart mirror dashboard.

Serves the mirror UI and JSON APIs for: status, weather, calendar,
voice assistant, wardrobe management, and outfit recommendations.
"""

import io
import json
import logging
import os
import re
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory, Response

from app.assistant import VoiceConciergeService
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

def _preprocess_and_detect_faces(img_bytes: bytes):
    """
    Robust face detection for smartphone cameras:
    1. Fixes EXIF orientation (portrait vs landscape) using PIL.
    2. Resizes giant phone photos to max 960px for optimal YuNet detection.
    3. If initial orientation detects a face where eyes are above mouth, returns it.
    4. Otherwise, tests 90° CCW, 90° CW, and 180° rotations.
    Returns (img_bgr, faces_array).
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    if not YUNET_MODEL.exists():
        return None, None

    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        w_orig, h_orig = pil_img.size

        # Multi-scale pyramid: 640px is YuNet's sweet spot for mobile selfies, followed by 480px and 800px
        for max_d in [640, 480, 800]:
            scale = max_d / float(max(w_orig, h_orig)) if max(w_orig, h_orig) > max_d else 1.0
            resized = pil_img.resize((int(w_orig * scale), int(h_orig * scale)), Image.Resampling.BILINEAR) if scale < 1.0 else pil_img
            base_bgr = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)

            rotations = [
                ("orig", base_bgr),
                ("90_cw", cv2.rotate(base_bgr, cv2.ROTATE_90_CLOCKWISE)),
                ("90_ccw", cv2.rotate(base_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                ("180", cv2.rotate(base_bgr, cv2.ROTATE_180)),
            ]

            best_upright = None
            best_upright_score = -1.0
            best_any = None
            best_any_score = -1.0

            for rot_label, candidate in rotations:
                ch, cw = candidate.shape[:2]
                det = cv2.FaceDetectorYN.create(
                    model=str(YUNET_MODEL),
                    config="",
                    input_size=(cw, ch),
                    score_threshold=0.20,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                _, faces = det.detect(candidate)
                if faces is not None and len(faces) > 0:
                    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    top_face = faces_sorted[0]
                    score = float(top_face[14])
                    eyes_y = (top_face[5] + top_face[7]) / 2.0
                    mouth_y = (top_face[11] + top_face[13]) / 2.0
                    if mouth_y > eyes_y and score > best_upright_score:
                        best_upright_score = score
                        best_upright = (candidate, faces_sorted, rot_label, score)
                    if score > best_any_score:
                        best_any_score = score
                        best_any = (candidate, faces_sorted, rot_label, score)

            if best_upright is not None:
                cand, f_list, rot, sc = best_upright
                logger.info(f"YuNet face detected: {len(f_list)} faces (top score={sc:.3f}, rot={rot}, scale={max_d})")
                return cand, f_list

        if best_any is not None:
            cand, f_list, rot, sc = best_any
            logger.info(f"YuNet face detected (fallback): {len(f_list)} faces (top score={sc:.3f}, rot={rot})")
            return cand, f_list

        logger.warning("YuNet face detection: no face found across any scale or rotation")
        return base_bgr, None
    except Exception as e:
        logger.error(f"Error in _preprocess_and_detect_faces: {e}")
        return None, None


def _enroll_face(user_id: str, img_bytes: bytes) -> bool:
    """
    Detects face using robust multi-orientation YuNet and generates 128-D SFace embedding.
    Saves:
      - data/embeddings/<user_id>.npy
      - data/embeddings/<user_id>.jpg (cropped face avatar)
    """
    import cv2
    import numpy as np

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    img_bgr, faces = _preprocess_and_detect_faces(img_bytes)

    if img_bgr is None:
        return False

    if not SFACE_MODEL.exists() or faces is None or len(faces) == 0:
        # Fallback: Save full image as avatar even if face recognition not available
        cv2.imwrite(str(EMBEDDINGS_DIR / f"{user_id}.jpg"), img_bgr)
        return False

    try:
        recognizer = cv2.FaceRecognizerSF.create(model=str(SFACE_MODEL), config="")
        face = faces[0]
        aligned = recognizer.alignCrop(img_bgr, face)
        feat = recognizer.feature(aligned)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        np.save(str(EMBEDDINGS_DIR / f"{user_id}.npy"), feat)
        cv2.imwrite(str(EMBEDDINGS_DIR / f"{user_id}.jpg"), aligned)
        return True
    except Exception as e:
        logger.error(f"Error enrolling face for {user_id}: {e}")
        try:
            cv2.imwrite(str(EMBEDDINGS_DIR / f"{user_id}.jpg"), img_bgr)
        except Exception:
            pass
        return False

def _match_face_embedding(feat) -> tuple:
    """
    Compares feat with all registered embeddings in EMBEDDINGS_DIR.
    Returns (best_user_id, best_score).
    """
    import cv2
    import numpy as np

    if not EMBEDDINGS_DIR.exists():
        return None, 0.0

    recognizer = None
    if SFACE_MODEL.exists():
        try:
            recognizer = cv2.FaceRecognizerSF.create(model=str(SFACE_MODEL), config="")
        except Exception:
            recognizer = None

    best_id = None
    best_score = 0.0
    COSINE_THRESHOLD = 0.363

    for fname in os.listdir(EMBEDDINGS_DIR):
        if fname.endswith(".npy"):
            uid = fname[:-4]
            try:
                ref_emb = np.load(str(EMBEDDINGS_DIR / fname))
                if recognizer is not None:
                    score = float(recognizer.match(feat, ref_emb, cv2.FaceRecognizerSF_FR_COSINE))
                else:
                    norm_q = np.linalg.norm(feat)
                    norm_r = np.linalg.norm(ref_emb)
                    if norm_q > 0 and norm_r > 0:
                        score = float(np.dot(feat, ref_emb) / (norm_q * norm_r))
                    else:
                        score = 0.0

                if score >= COSINE_THRESHOLD and score > best_score:
                    best_score = score
                    best_id = uid
            except Exception as e:
                logger.warning(f"Error matching embedding {fname}: {e}")

    return best_id, best_score

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

_last_motion_time = None
_last_user_greet_time = {}
GREETING_COOLDOWN_SECONDS = 600  # 10 minutes between automatic greetings for the same user


def _trigger_auto_greeting_if_eligible(user_id: str, trigger_source: str = "camera") -> bool:
    """
    Checks motion sensor presence and recognized face to automatically deliver a
    personalized voice greeting & morning briefing to that specific user on the Smart Mirror.
    """
    global _last_user_greet_time
    if not user_id or user_id.lower() == "guest":
        return False

    # 1. Privacy check
    privacy_mode = SettingsRepository.get("privacy_mode", default="false") == "true"
    if privacy_mode:
        logger.info(f"Auto-greeting skipped for {user_id}: Privacy mode is active.")
        return False

    # 2. Setting toggle check
    auto_enabled = SettingsRepository.get("auto_greeting_enabled", default="true") == "true"
    if not auto_enabled:
        logger.info(f"Auto-greeting skipped for {user_id}: Auto greeting disabled in settings.")
        return False

    # 3. Cooldown check
    now = time.time()
    last_greet = _last_user_greet_time.get(user_id, 0)
    if (now - last_greet) < GREETING_COOLDOWN_SECONDS:
        logger.info(f"Auto-greeting cooldown active for {user_id} ({int(now - last_greet)}s < {GREETING_COOLDOWN_SECONDS}s).")
        return False

    # 4. Motion sensor check:
    # If ESP32 is online, verify motion was detected recently (<90s) or is active now.
    is_esp32_online = (
        _esp32_state["last_seen"] is not None and
        (now - _esp32_state["last_seen"]) < 20
    )
    if is_esp32_online and not settings.ESP32_MOCK_MODE:
        motion_active = bool(_esp32_state.get("motion"))
        recent_motion = (_last_motion_time is not None and (now - _last_motion_time) < 90)
        if not (motion_active or recent_motion):
            logger.info(f"Auto-greeting for {user_id} waiting for motion sensor detection.")
            return False

    if _voice_state.get("state") == "speaking":
        return False

    _last_user_greet_time[user_id] = now
    logger.info(f"🎉 Triggering auto voice greeting for recognized user '{user_id}' (source: {trigger_source})")

    briefing = VoiceConciergeService.build_briefing(
        user_id=user_id,
        esp32_data=_esp32_state,
        stylist_data=_live_stylist_state.get(user_id),
        grooming_data=_live_grooming_state.get(user_id),
    )
    text = briefing.get("full_text", "")

    def on_start(t):
        _voice_state["state"] = "speaking"
        _voice_state["text"] = "Voice Concierge Welcome"
        _voice_state["reply"] = t

    def on_finish():
        _voice_state["state"] = "idle"
        _voice_state["text"] = ""
        _voice_state["reply"] = ""

    VoiceConciergeService.play_on_mirror_async(
        text, user_id=user_id, on_start=on_start, on_finish=on_finish
    )
    return True


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


@app.route("/api/voice/briefing")
def voice_briefing():
    """
    Generates a personalized daily briefing for the specified or active user.
    """
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = SettingsRepository.get("active_user", default="guest")

    briefing = VoiceConciergeService.build_briefing(
        user_id=user_id,
        esp32_data=_esp32_state,
        stylist_data=_live_stylist_state.get(user_id),
        grooming_data=_live_grooming_state.get(user_id),
    )
    return jsonify(briefing)


@app.route("/api/voice/briefing/play", methods=["POST"])
def voice_briefing_play():
    """
    Triggers briefing playback on the Smart Mirror speakers, or updates state for phone stream.
    """
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id") or SettingsRepository.get("active_user", default="guest")
    target = data.get("target", "mirror")

    briefing = VoiceConciergeService.build_briefing(
        user_id=user_id,
        esp32_data=_esp32_state,
        stylist_data=_live_stylist_state.get(user_id),
        grooming_data=_live_grooming_state.get(user_id),
    )
    text = briefing.get("full_text", "")

    if target == "mirror":
        def on_start(t):
            _voice_state["state"] = "speaking"
            _voice_state["text"] = "Voice Concierge Briefing"
            _voice_state["reply"] = t

        def on_finish():
            _voice_state["state"] = "idle"
            _voice_state["text"] = ""
            _voice_state["reply"] = ""

        VoiceConciergeService.play_on_mirror_async(
            text, user_id=user_id, on_start=on_start, on_finish=on_finish
        )
        return jsonify({"status": "ok", "target": "mirror", "briefing": briefing})
    else:
        return jsonify({"status": "ok", "target": "phone", "briefing": briefing})


@app.route("/api/voice/briefing/audio")
def voice_briefing_audio():
    """
    Synthesizes and streams WAV audio of the daily briefing for in-browser phone playback.
    """
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = SettingsRepository.get("active_user", default="guest")

    briefing = VoiceConciergeService.build_briefing(
        user_id=user_id,
        esp32_data=_esp32_state,
        stylist_data=_live_stylist_state.get(user_id),
        grooming_data=_live_grooming_state.get(user_id),
    )
    text = briefing.get("full_text", "Welcome to ReflectAI.")
    wav_bytes = VoiceConciergeService.synthesize_to_wav(text)

    return Response(
        wav_bytes,
        mimetype="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=briefing.wav",
            "Content-Length": str(len(wav_bytes)),
            "Cache-Control": "no-cache, no-store, must-revalidate",
        }
    )


@app.route("/api/voice/stop", methods=["POST"])
def voice_stop():
    """
    Halts any active audio playback on mirror speakers and resets voice state.
    """
    VoiceConciergeService.stop_playback()
    _voice_state["state"] = "idle"
    _voice_state["text"] = ""
    _voice_state["reply"] = ""
    return jsonify({"status": "ok", "message": "Voice playback stopped"})



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

    # Auto-greet recognized face if motion is confirmed
    greeted = _trigger_auto_greeting_if_eligible(user_id, trigger_source="face_recognition")

    return jsonify({"status": "ok", "active_user": user_id, "greeted": greeted})


@app.route("/api/esp32/state", methods=["POST"])
def esp32_state():
    """
    Receives JSON telemetry and physical button interactions from ESP32:
    - User selection button (User 1, User 2, Guest)
    - Privacy mode toggle
    - Sensors: DHT22 (temp, humidity), LDR (light), PIR (motion)
    """
    global _last_esp32_user, _last_esp32_privacy, _last_motion_time
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
    motion_val = bool(data.get("motion", False))
    _esp32_state["motion"] = motion_val
    _esp32_state["last_seen"] = time.time()

    if motion_val:
        _last_motion_time = time.time()
        # If camera has recently seen a registered user (within last 45s), greet them!
        active_user = SettingsRepository.get("active_user", default="guest")
        if active_user and active_user.lower() != "guest" and _last_camera_seen and (time.time() - _last_camera_seen) < 45:
            _trigger_auto_greeting_if_eligible(active_user, trigger_source="motion_sensor")

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

@app.route("/api/auth/face-login", methods=["POST"])
def api_face_login():
    """
    Authenticates a user via face scan from their phone.
    Returns:
      - status: 'ok', user: {...}, score: float (if recognized)
      - status: 'not_registered', message: '...' (if face detected but not recognized)
      - status: 'no_face', message: '...' (if no face detected in image)
    """
    import cv2
    import numpy as np

    image_file = request.files.get("photo") or request.files.get("image")
    if not image_file or not image_file.filename:
        return jsonify({"status": "no_face", "error": "No face image provided"}), 400

    img_bytes = image_file.read()
    img_bgr, faces = _preprocess_and_detect_faces(img_bytes)

    if img_bgr is None:
        return jsonify({"status": "no_face", "error": "Could not process image"}), 400

    if faces is None or len(faces) == 0:
        return jsonify({
            "status": "no_face",
            "message": "No face detected in photo. Please ensure good lighting, look directly into the camera, and try again."
        }), 200

    if not SFACE_MODEL.exists():
        return jsonify({
            "status": "error",
            "error": "Face recognition models not found on server"
        }), 500

    try:
        recognizer = cv2.FaceRecognizerSF.create(model=str(SFACE_MODEL), config="")
        face = faces[0]
        aligned = recognizer.alignCrop(img_bgr, face)
        feat = recognizer.feature(aligned)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        matched_user_id, match_score = _match_face_embedding(feat)

        if matched_user_id:
            user = UserRepository.get(matched_user_id) or {"name": matched_user_id.replace("_", " ").title()}
            SettingsRepository.set("active_user", matched_user_id)
            logger.info(f"Face ID login successful for: {matched_user_id} (score={match_score:.3f})")

            # Check and trigger auto greeting on mirror
            _trigger_auto_greeting_if_eligible(matched_user_id, trigger_source="face_login")

            return jsonify({
                "status": "ok",
                "user": {
                    "id": matched_user_id,
                    "name": user["name"],
                    "preferred_style": user.get("preferred_style") or "Casual",
                    "avatar_url": f"/avatar/{matched_user_id}",
                },
                "score": float(match_score),
                "message": f"Welcome back, {user['name']}!"
            }), 200
        else:
            logger.info(f"Face login: face detected but not recognized as any registered profile (best match score={match_score:.3f})")
            return jsonify({
                "status": "not_registered",
                "message": "Face detected, but you are not registered in ReflectAI yet. Please register your profile first."
            }), 200

    except Exception as e:
        logger.error(f"Face login error: {e}")
        return jsonify({"status": "error", "error": f"Face recognition failed: {str(e)}"}), 500


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


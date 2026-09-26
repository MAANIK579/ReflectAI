"""
face_engine/face_service.py — Automatic Face Recognition Service for ReflectAI.

Features:
  - Wake-on-Motion: Enters low-power standby (~0% CPU) when room is idle.
  - Automatically wakes up and starts scanning when ESP32 PIR motion sensor trips.
  - Detects faces with YuNet and recognizes registered profiles with SFace.
  - Automatically switches dashboard profile and loads personal data (calendar, reminders, styling).
  - Triggers personalized morning briefings and outfit recommendations.
  - Gracefully reverts to Guest mode when no face is present.
  - Supports both embedded dashboard operation and standalone CLI execution.
"""

import json
import logging
import os
import sys
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger("reflectai.face_service")

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")  # -> ReflectAI/data
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
PROFILES_FILE = os.path.join(DATA_DIR, "profiles.json")

YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

FALLBACK_USER_ID = "guest"
DEFAULT_COSINE_THRESHOLD = 0.363


def load_registered_embeddings(embeddings_dir: str = EMBEDDINGS_DIR) -> Dict[str, np.ndarray]:
    """Load all registered face embeddings from data/embeddings/*.npy"""
    embeddings = {}
    if not os.path.isdir(embeddings_dir):
        return embeddings

    for filename in os.listdir(embeddings_dir):
        if filename.endswith(".npy"):
            user_id = filename[:-4]
            path = os.path.join(embeddings_dir, filename)
            try:
                emb = np.load(path)
                embeddings[user_id] = emb
            except Exception as e:
                logger.warning(f"Failed to load embedding {filename}: {e}")

    return embeddings


def load_profiles(profiles_file: str = PROFILES_FILE) -> Dict[str, dict]:
    """Load user profiles from profiles.json"""
    if os.path.exists(profiles_file):
        try:
            with open(profiles_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def extract_body_crops(frame: np.ndarray, face_box: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract upper-body (topwear), lower-body (bottomwear), and head (grooming) regions."""
    try:
        frame_h, frame_w = frame.shape[:2]
        x, y, w_box, h_box = map(int, face_box[:4])

        # 1. Topwear (chin to waist)
        torso_y1 = int(min(frame_h, max(0, y + int(h_box * 0.85))))
        torso_y2 = int(min(frame_h, y + int(h_box * 3.3)))
        torso_cx = x + w_box / 2
        torso_half_w = (w_box * 2.4) / 2
        torso_x1 = int(max(0, torso_cx - torso_half_w))
        torso_x2 = int(min(frame_w, torso_cx + torso_half_w))

        torso_crop = None
        if torso_y2 > torso_y1 + 30 and torso_x2 > torso_x1 + 30:
            torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]

        # 2. Bottomwear (waist down towards legs/thighs)
        legs_y1 = int(min(frame_h, y + int(h_box * 3.2)))
        legs_y2 = int(min(frame_h, y + int(h_box * 7.5)))
        legs_cx = x + w_box / 2
        legs_half_w = (w_box * 2.2) / 2
        legs_x1 = int(max(0, legs_cx - legs_half_w))
        legs_x2 = int(min(frame_w, legs_cx + legs_half_w))

        legs_crop = None
        if legs_y2 > legs_y1 + 40 and legs_x2 > legs_x1 + 30:
            legs_crop = frame[legs_y1:legs_y2, legs_x1:legs_x2]

        # 3. Head & Face (for Grooming Classifier: hair, beard, skin)
        head_y1 = int(max(0, y - int(h_box * 0.45)))
        head_y2 = int(min(frame_h, y + int(h_box * 1.35)))
        head_cx = x + w_box / 2
        head_half_w = (w_box * 1.5) / 2
        head_x1 = int(max(0, head_cx - head_half_w))
        head_x2 = int(min(frame_w, head_cx + head_half_w))

        head_crop = None
        if head_y2 > head_y1 + 40 and head_x2 > head_x1 + 40:
            head_crop = frame[head_y1:head_y2, head_x1:head_x2]

        return torso_crop, legs_crop, head_crop
    except Exception:
        pass
    return None, None, None


def match_face(query_embedding: np.ndarray, registered_embeddings: Dict[str, np.ndarray], recognizer: cv2.FaceRecognizerSF, threshold: float = DEFAULT_COSINE_THRESHOLD) -> Tuple[Optional[str], float]:
    """
    Compare query embedding against all registered embeddings.
    Returns (best_user_id, best_score) or (None, 0.0) if no match.
    """
    best_id = None
    best_score = 0.0

    for user_id, ref_emb in registered_embeddings.items():
        score = recognizer.match(query_embedding, ref_emb, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= threshold and score > best_score:
            best_score = score
            best_id = user_id

    return best_id, best_score


class FaceRecognitionService:
    """
    Motion-Aware Face Recognition Service for ReflectAI.
    Automatically sleeps when room is idle and wakes up to scan and identify users on motion.
    """

    def __init__(
        self,
        camera_index: int = 0,
        detection_fps: int = 4,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
        confirm_frames: int = 2,
        no_face_timeout: float = 20.0,
        motion_timeout: float = 45.0,
        show_preview: bool = False,
        api_host: str = "localhost:5000",
        on_user_recognized: Optional[Callable[[str, str], None]] = None,
        on_user_lost: Optional[Callable[[], None]] = None,
    ):
        self.camera_index = camera_index
        self.detection_fps = detection_fps
        self.cosine_threshold = cosine_threshold
        self.confirm_frames = confirm_frames
        self.no_face_timeout = no_face_timeout
        self.motion_timeout = motion_timeout
        self.show_preview = show_preview
        self.api_host = api_host
        self.on_user_recognized = on_user_recognized
        self.on_user_lost = on_user_lost

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._motion_event = threading.Event()
        self._motion_event.set()  # Start awake initially for 45s

        self.last_motion_time = time.time()
        self.is_motion_detected = True
        self.last_face_time = time.time()
        self.current_user = FALLBACK_USER_ID
        self.camera_status = "stopped"
        self.last_error: Optional[str] = None
        self.registered = {}
        self.profiles = {}

    def start(self):
        if self.running:
            return
        self.running = True
        self.camera_status = "starting"
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="FaceRecognitionService")
        self.thread.start()
        logger.info("Face recognition camera service started.")

    def stop(self):
        self.running = False
        self._motion_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.camera_status = "stopped"
        logger.info("Face recognition camera service stopped.")

    def on_motion_detected(self):
        """Called whenever the ESP32 PIR motion sensor detects motion."""
        self.last_motion_time = time.time()
        self._motion_event.set()
        if "standby" in self.camera_status:
            self.camera_status = "scanning for faces"
            logger.info("Motion sensor triggered! Camera awakened from standby to scan for faces.")

    def get_status(self) -> dict:
        now = time.time()
        motion_recent = (now - self.last_motion_time) < self.motion_timeout
        return {
            "running": self.running,
            "camera_status": self.camera_status,
            "active_user": self.current_user,
            "motion_active": motion_recent,
            "registered_users": list(self.registered.keys()),
            "last_error": self.last_error,
        }

    def _open_camera(self):
        """Attempts to open USB webcam (OpenCV) with fallback to Picamera2."""
        logger.info(f"Opening camera (index {self.camera_index})...")
        cap = cv2.VideoCapture(self.camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()

        # Try device index 0 if another was requested
        if self.camera_index != 0:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                cap.release()

        # Fallback to Raspberry Pi Picamera2
        try:
            from picamera2 import Picamera2
            logger.info("Attempting connection via Raspberry Pi Picamera2...")

            class PiCameraStream:
                def __init__(self, width=640, height=480):
                    self.picam2 = Picamera2()
                    config = self.picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
                    self.picam2.configure(config)
                    self.picam2.start()
                    time.sleep(0.5)

                def read(self):
                    f = self.picam2.capture_array()
                    if f is not None:
                        return True, cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    return False, None

                def isOpened(self):
                    return True

                def release(self):
                    try:
                        self.picam2.stop()
                        self.picam2.close()
                    except Exception:
                        pass

            pi_cam = PiCameraStream(640, 480)
            ret, frame = pi_cam.read()
            if ret and frame is not None:
                logger.info("Successfully connected to Raspberry Pi Camera via Picamera2!")
                return pi_cam
            pi_cam.release()
        except Exception as e:
            logger.debug(f"Picamera2 init failed: {e}")

        return None

    def _notify_mirror(self, user_id: str, name: Optional[str] = None):
        """Notifies ReflectAI dashboard to switch active profile."""
        if self.on_user_recognized and user_id != FALLBACK_USER_ID:
            try:
                self.on_user_recognized(user_id, name or user_id)
            except Exception as e:
                logger.debug(f"Direct callback failed: {e}")

        if self.on_user_lost and user_id == FALLBACK_USER_ID:
            try:
                self.on_user_lost()
            except Exception as e:
                logger.debug(f"User lost callback failed: {e}")

        # Also post via HTTP for dashboard & mobile API sync
        def _post():
            try:
                url = f"http://{self.api_host}/api/user/active"
                payload = {"userId": user_id}
                if name:
                    payload["name"] = name
                requests.post(url, json=payload, timeout=2)
            except Exception:
                pass

        threading.Thread(target=_post, daemon=True).start()

    def _notify_clothing(self, user_id: str, torso_crop, legs_crop, head_crop):
        """Sends extracted body crops to MirrorStylist for live outfit classification."""
        if user_id == FALLBACK_USER_ID or (torso_crop is None and legs_crop is None and head_crop is None):
            return

        def _post():
            try:
                files = {}
                if head_crop is not None and head_crop.size > 0:
                    s0, b0 = cv2.imencode(".jpg", head_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if s0:
                        files["image_head"] = ("head.jpg", b0.tobytes(), "image/jpeg")
                if torso_crop is not None and torso_crop.size > 0:
                    s1, b1 = cv2.imencode(".jpg", torso_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if s1:
                        files["image_top"] = ("torso.jpg", b1.tobytes(), "image/jpeg")
                if legs_crop is not None and legs_crop.size > 0:
                    s2, b2 = cv2.imencode(".jpg", legs_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if s2:
                        files["image_bottom"] = ("legs.jpg", b2.tobytes(), "image/jpeg")

                if files:
                    url = f"http://{self.api_host}/api/mirror/clothing/live"
                    requests.post(url, data={"userId": user_id}, files=files, timeout=4)
            except Exception:
                pass

        threading.Thread(target=_post, daemon=True).start()

    def _run_loop(self):
        # Validate models
        if not os.path.exists(YUNET_MODEL) or not os.path.exists(SFACE_MODEL):
            self.camera_status = "model_missing"
            self.last_error = f"Models missing in {MODELS_DIR}. Run download_face_models.py."
            logger.error(self.last_error)
            return

        self.registered = load_registered_embeddings()
        self.profiles = load_profiles()
        logger.info(f"Loaded {len(self.registered)} registered faces: {list(self.registered.keys())}")

        cap = self._open_camera()
        if not cap:
            self.camera_status = "no_camera"
            self.last_error = "No camera hardware detected (check USB or CSI cable)."
            logger.warning(f"[FaceService] {self.last_error}")
            return

        # Get initial resolution
        ret, frame = cap.read()
        if not ret or frame is None:
            self.camera_status = "no_camera"
            cap.release()
            return
        h, w = frame.shape[:2]

        detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL,
            config="",
            input_size=(w, h),
            score_threshold=0.75,
            nms_threshold=0.3,
            top_k=5000,
        )
        recognizer = cv2.FaceRecognizerSF.create(model=SFACE_MODEL, config="")

        candidate_user = None
        candidate_count = 0
        last_clothing_check_time = 0.0
        last_embedding_reload = time.time()
        frame_interval = 1.0 / self.detection_fps

        self.camera_status = "scanning for faces"
        logger.info("✓ Camera face engine active and ready.")

        try:
            while self.running:
                now = time.time()

                # 1. Motion awareness check:
                # If no motion has been sensed recently and current user is guest, enter low-power standby
                motion_recent = (now - self.last_motion_time) < self.motion_timeout
                if not motion_recent and self.current_user == FALLBACK_USER_ID:
                    self.camera_status = "standby (waiting for motion)"
                    self._motion_event.clear()
                    # Wait up to 1.5s for motion trigger
                    self._motion_event.wait(timeout=1.5)
                    continue

                self.camera_status = "scanning for faces" if self.current_user == FALLBACK_USER_ID else f"active: {self.current_user}"

                # Periodic embedding reload for hot-registration
                if now - last_embedding_reload > 10.0:
                    last_embedding_reload = now
                    self.registered = load_registered_embeddings()
                    self.profiles = load_profiles()

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.4)
                    continue

                detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, faces = detector.detect(frame)

                detected_user = None
                best_score = 0.0

                if faces is not None and len(faces) > 0:
                    self.last_face_time = now
                    # Select largest face (closest to mirror)
                    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    face = faces_sorted[0]

                    try:
                        aligned = recognizer.alignCrop(frame, face)
                        feat = recognizer.feature(aligned)
                        detected_user, best_score = match_face(
                            feat, self.registered, recognizer, threshold=self.cosine_threshold
                        )
                    except Exception:
                        pass

                # Identity confirmation
                if detected_user is not None:
                    if detected_user == candidate_user:
                        candidate_count += 1
                    else:
                        candidate_user = detected_user
                        candidate_count = 1

                    user_switched = (self.current_user != candidate_user)
                    if candidate_count >= self.confirm_frames and user_switched:
                        self.current_user = candidate_user
                        profile_name = self.profiles.get(self.current_user, {}).get("name", self.current_user.title())
                        logger.info(f"✨ Face Recognized: {profile_name} (score={best_score:.3f}) — Loading user data!")
                        self.camera_status = f"recognized: {profile_name}"
                        self._notify_mirror(self.current_user, name=profile_name)

                    if candidate_count >= self.confirm_frames and self.current_user != FALLBACK_USER_ID:
                        if user_switched or (now - last_clothing_check_time > 30):
                            last_clothing_check_time = now
                            torso_crop, legs_crop, head_crop = extract_body_crops(frame, face)
                            self._notify_clothing(self.current_user, torso_crop, legs_crop, head_crop)
                else:
                    candidate_user = None
                    candidate_count = 0

                    # Revert to guest after timeout with no face
                    if (now - self.last_face_time) > self.no_face_timeout and self.current_user != FALLBACK_USER_ID:
                        logger.info(f"No face detected for {self.no_face_timeout}s — reverting mirror to Guest.")
                        self.current_user = FALLBACK_USER_ID
                        self._notify_mirror(FALLBACK_USER_ID)

                # Optional GUI Preview (CLI standalone mode)
                if self.show_preview:
                    display = frame.copy()
                    if faces is not None and len(faces) > 0:
                        face = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                        x, y, w_box, h_box = map(int, face[:4])
                        color = (0, 230, 115) if detected_user else (100, 100, 255)
                        label = f"{detected_user.upper()} ({best_score:.2f})" if detected_user else "Unknown Face"
                        cv2.rectangle(display, (x, y), (x + w_box, y + h_box), color, 2)
                        cv2.putText(display, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imshow("ReflectAI — Face Recognition", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break

                # FPS Throttle
                elapsed = time.time() - now
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in face recognition loop: {e}")
            self.last_error = str(e)
        finally:
            cap.release()
            if self.show_preview:
                cv2.destroyAllWindows()
            self._notify_mirror(FALLBACK_USER_ID)
            self.camera_status = "stopped"
            logger.info("Camera released cleanly.")


def main():
    print("=" * 60)
    print("    ReflectAI — Face Recognition Service (Motion-Aware)")
    print("=" * 60)
    service = FaceRecognitionService(
        camera_index=int(os.getenv("CAMERA_INDEX", "0")),
        detection_fps=4,
        show_preview=os.getenv("SHOW_PREVIEW", "1").lower() in ("1", "true", "yes"),
    )
    service.start()

    # Synchronize motion with ReflectAI dashboard in standalone mode
    def _motion_sync_worker():
        while service.running:
            try:
                r = requests.get("http://localhost:5000/api/status", timeout=1.5)
                if r.status_code == 200:
                    data = r.json()
                    sensors = data.get("sensors", {})
                    if sensors.get("motion"):
                        service.on_motion_detected()
            except Exception:
                pass
            time.sleep(1.0)

    sync_thread = threading.Thread(target=_motion_sync_worker, daemon=True)
    sync_thread.start()

    try:
        while service.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping Face Recognition Service...")
        service.stop()
        print("Done.")


if __name__ == "__main__":
    main()

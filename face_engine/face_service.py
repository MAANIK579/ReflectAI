# face_service.py
# Continuous background face recognition service for ReflectAI.
# Captures webcam frames at ~3-5 FPS, detects faces with YuNet,
# recognizes them with SFace, and notifies the dashboard to switch profiles.
#
# CHANGES from the original version:
#   - API_URL repointed to ReflectAI's Flask dashboard (port 5000, not 3000)
#   - 'default' fallback renamed to 'guest' to match ReflectAI's seeded
#     users (user1 / user2 / guest) so it shows correctly on the dashboard
#     instead of creating a stray "default" user

import json
import os
import sys
import time
import threading
import cv2
import numpy as np
import requests

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')  # -> ReflectAI/data
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
PROFILES_FILE = os.path.join(DATA_DIR, 'profiles.json')

YUNET_MODEL = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx')
SFACE_MODEL = os.path.join(MODELS_DIR, 'face_recognition_sface_2021dec.onnx')

DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', 'localhost:5000')
API_URL = f'http://{DASHBOARD_HOST}/api/user/active'
CLOTHING_API_URL = f'http://{DASHBOARD_HOST}/api/mirror/clothing/live'

# ── Configuration ──────────────────────────────────────────────
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0')) # Webcam device index
DETECTION_FPS = 4           # Target frames per second for face checking
COSINE_THRESHOLD = 0.363    # SFace cosine similarity threshold (higher = stricter)
CONFIRM_FRAMES = 3          # Consecutive frames needed to confirm identity switch
NO_FACE_TIMEOUT = 15        # Seconds with no face before reverting to guest

# Auto-detect headless mode (e.g. over SSH without an X11 screen)
SHOW_PREVIEW_ENV = os.getenv('SHOW_PREVIEW', '').lower()
if SHOW_PREVIEW_ENV in ('0', 'false', 'no'):
    SHOW_PREVIEW = False
elif SHOW_PREVIEW_ENV in ('1', 'true', 'yes'):
    SHOW_PREVIEW = True
else:
    SHOW_PREVIEW = ('DISPLAY' in os.environ) if os.name != 'nt' else True

FALLBACK_USER_ID = 'guest'  # matches ReflectAI's seeded default users


def notify_clothing_analysis(user_id, torso_crop, legs_crop=None, head_crop=None):
    """Non-blocking background thread to send head, torso, and legs crops to mirror stylist."""
    if (torso_crop is None) and (legs_crop is None) and (head_crop is None):
        return
    if user_id == FALLBACK_USER_ID:
        return

    def _post():
        try:
            files = {}
            if head_crop is not None and head_crop.size > 0:
                s0, b0 = cv2.imencode('.jpg', head_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if s0:
                    files['image_head'] = ('head.jpg', b0.tobytes(), 'image/jpeg')

            if torso_crop is not None and torso_crop.size > 0:
                s1, b1 = cv2.imencode('.jpg', torso_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if s1:
                    files['image_top'] = ('torso.jpg', b1.tobytes(), 'image/jpeg')

            if legs_crop is not None and legs_crop.size > 0:
                s2, b2 = cv2.imencode('.jpg', legs_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if s2:
                    files['image_bottom'] = ('legs.jpg', b2.tobytes(), 'image/jpeg')

            if not files:
                return

            data = {'userId': user_id}
            requests.post(CLOTHING_API_URL, data=data, files=files, timeout=4)
        except Exception:
            pass

    threading.Thread(target=_post, daemon=True).start()


def load_registered_embeddings():
    """Load all registered face embeddings from data/embeddings/*.npy"""
    embeddings = {}
    if not os.path.isdir(EMBEDDINGS_DIR):
        return embeddings

    for filename in os.listdir(EMBEDDINGS_DIR):
        if filename.endswith('.npy'):
            user_id = filename[:-4]  # strip .npy
            path = os.path.join(EMBEDDINGS_DIR, filename)
            try:
                emb = np.load(path)
                embeddings[user_id] = emb
            except Exception as e:
                print(f'[WARN] Failed to load embedding {filename}: {e}')

    return embeddings


def load_profiles():
    """Load user profiles from profiles.json"""
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def notify_mirror(user_id, name=None):
    """Non-blocking POST to ReflectAI's dashboard to switch active user."""
    def _post():
        try:
            payload = {'userId': user_id}
            if name:
                payload['name'] = name
            requests.post(API_URL, json=payload, timeout=2)
        except Exception:
            pass  # Dashboard server may not be running
    t = threading.Thread(target=_post, daemon=True)
    t.start()


def extract_body_crops(frame, face_box):
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


def match_face(query_embedding, registered_embeddings, recognizer):
    """
    Compare query embedding against all registered embeddings.
    Returns (best_user_id, best_score) or (None, 0.0) if no match.
    """
    best_id = None
    best_score = 0.0

    for user_id, ref_emb in registered_embeddings.items():
        score = recognizer.match(
            query_embedding, ref_emb, cv2.FaceRecognizerSF_FR_COSINE
        )
        if score >= COSINE_THRESHOLD and score > best_score:
            best_score = score
            best_id = user_id

    return best_id, best_score


def main():
    print('=' * 60)
    print('    ReflectAI — Face Recognition Service')
    print('=' * 60)

    # Validate models
    if not os.path.exists(YUNET_MODEL):
        print(f'[ERROR] YuNet model not found: {YUNET_MODEL}')
        print('Run: python download_face_models.py')
        return
    if not os.path.exists(SFACE_MODEL):
        print(f'[ERROR] SFace model not found: {SFACE_MODEL}')
        print('Run: python download_face_models.py')
        return

    # Load registered face embeddings
    registered = load_registered_embeddings()
    profiles = load_profiles()
    print(f'Loaded {len(registered)} registered face(s): {list(registered.keys())}')

    if not registered:
        print('[WARN] No registered faces. Run register_user.py first.')
        print('Service will continue and detect faces, but cannot identify anyone.')

    # Open camera
    print(f'\nOpening camera (index {CAMERA_INDEX})...')
    cap = cv2.VideoCapture(CAMERA_INDEX)
    is_opened = cap.isOpened()
    frame = None
    if is_opened:
        ret, frame = cap.read()
        if not ret or frame is None:
            is_opened = False
            cap.release()

    if not is_opened:
        # Fallback to native Raspberry Pi Picamera2
        try:
            from picamera2 import Picamera2
            print('[INFO] Attempting to connect via Raspberry Pi Picamera2...')
            class PiCameraStream:
                def __init__(self, width=640, height=480):
                    self.picam2 = Picamera2()
                    config = self.picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
                    self.picam2.configure(config)
                    self.picam2.start()
                    time.sleep(1)

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

            cap = PiCameraStream(640, 480)
            ret, frame = cap.read()
            if not ret or frame is None:
                print('[ERROR] Could not read frame from Picamera2.')
                cap.release()
                return
            print('[INFO] Successfully connected to Raspberry Pi Camera via Picamera2!')
        except Exception as e:
            print(f'[ERROR] Cannot open camera with OpenCV or Picamera2: {e}')
            return

    h, w = frame.shape[:2]
    print(f'Camera resolution: {w}x{h}')

    # Initialize OpenCV face detector and recognizer
    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL,
        config='',
        input_size=(w, h),
        score_threshold=0.75,
        nms_threshold=0.3,
        top_k=5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(model=SFACE_MODEL, config='')

    # ── State tracking ─────────────────────────────────────────
    current_user = FALLBACK_USER_ID
    candidate_user = None
    candidate_count = 0
    last_face_time = time.time()
    last_clothing_check_time = 0.0
    frame_interval = 1.0 / DETECTION_FPS

    print(f'\nFace recognition active (target ~{DETECTION_FPS} FPS)')
    print('Press Q or ESC to quit.\n')

    notify_mirror(FALLBACK_USER_ID)  # Start with guest profile

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print('[WARN] Frame read failed, retrying...')
                time.sleep(0.5)
                continue

            # Detect faces
            detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = detector.detect(frame)

            detected_user = None
            best_score = 0.0

            if faces is not None and len(faces) > 0:
                last_face_time = time.time()

                # Use largest face (closest to camera)
                faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                face = faces_sorted[0]

                # Get face embedding
                try:
                    aligned = recognizer.alignCrop(frame, face)
                    feat = recognizer.feature(aligned)
                    detected_user, best_score = match_face(
                        feat, registered, recognizer
                    )
                except Exception:
                    pass  # Alignment can fail on edge cases

            # ── Identity confirmation logic ────────────────────
            if detected_user is not None:
                if detected_user == candidate_user:
                    candidate_count += 1
                else:
                    candidate_user = detected_user
                    candidate_count = 1

                # Switch profile after enough consecutive confirmations
                user_switched = (current_user != candidate_user)
                if candidate_count >= CONFIRM_FRAMES and user_switched:
                    current_user = candidate_user
                    profile_name = profiles.get(current_user, {}).get('name', current_user)
                    print(f'[RECOGNIZED] {profile_name} (score={best_score:.3f}) — switching dashboard')
                    notify_mirror(current_user, name=profile_name)

                # Trigger clothing analysis in background (on identity switch or every 30s)
                if candidate_count >= CONFIRM_FRAMES and current_user != FALLBACK_USER_ID:
                    now = time.time()
                    if user_switched or (now - last_clothing_check_time > 30):
                        last_clothing_check_time = now
                        torso_crop, legs_crop, head_crop = extract_body_crops(frame, face)
                        notify_clothing_analysis(current_user, torso_crop, legs_crop, head_crop)
            else:
                candidate_user = None
                candidate_count = 0

                # Revert to guest after timeout with no face
                elapsed_no_face = time.time() - last_face_time
                if elapsed_no_face > NO_FACE_TIMEOUT and current_user != FALLBACK_USER_ID:
                    print(f'[TIMEOUT] No face for {NO_FACE_TIMEOUT}s — reverting to guest')
                    current_user = FALLBACK_USER_ID
                    notify_mirror(FALLBACK_USER_ID)

            # ── Optional preview window ────────────────────────
            if SHOW_PREVIEW:
                display = frame.copy()

                if faces is not None and len(faces) > 0:
                    face = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                    x, y, w_box, h_box = map(int, face[:4])

                    if detected_user and candidate_count >= CONFIRM_FRAMES:
                        color = (0, 230, 115)  # Green — recognized
                        label = f'{current_user.upper()} ({best_score:.2f})'
                    elif detected_user:
                        color = (0, 200, 255)  # Yellow — confirming
                        label = f'{detected_user}? ({candidate_count}/{CONFIRM_FRAMES})'
                    else:
                        color = (100, 100, 255)  # Red — unknown
                        label = 'Unknown'

                    cv2.rectangle(display, (x, y), (x + w_box, y + h_box), color, 2)
                    cv2.putText(display, label, (x, max(20, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(display, 'No face detected', (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

                # Status bar
                status = f'Active: {current_user.upper()} | Registered: {len(registered)}'
                cv2.putText(display, status, (20, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow('ReflectAI — Face Service', display)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    print('\n[EXIT] Face service stopped by user.')
                    break

                # Hot-reload embeddings on 'R' key
                if key in (ord('r'), ord('R')):
                    registered = load_registered_embeddings()
                    profiles = load_profiles()
                    print(f'[RELOAD] Reloaded {len(registered)} embeddings: {list(registered.keys())}')

            # ── FPS throttle ───────────────────────────────────
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print('\n[EXIT] Face service stopped (Ctrl+C).')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Revert to guest on exit
        notify_mirror(FALLBACK_USER_ID)
        print('Camera released. Goodbye.')


if __name__ == '__main__':
    main()

# register_user.py
# Register a user's face for ReflectAI.
# Uses OpenCV YuNet for face detection and SFace for 128-D feature embedding.
#
# CHANGES from the original version:
#   - API_ACTIVE_URL repointed to ReflectAI's Flask dashboard (port 5000)
#   - Now sends 'name' along with 'userId' so ReflectAI's database creates
#     the user record with the right display name immediately, instead of
#     falling back to the raw user_id
#   - Dropped the API_PROFILES_URL POST (ReflectAI doesn't have that
#     endpoint — profiles.json is still written locally as before, and
#     will be properly absorbed into ReflectAI's database in Phases 6-8
#     when outfit/hair/calendar features are actually built)

import json
import os
import sys
import time
import cv2
import numpy as np
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')  # -> ReflectAI/data
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
PROFILES_FILE = os.path.join(DATA_DIR, 'profiles.json')

YUNET_MODEL = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx')
SFACE_MODEL = os.path.join(MODELS_DIR, 'face_recognition_sface_2021dec.onnx')
API_ACTIVE_URL = 'http://localhost:5000/api/user/active'  # ReflectAI dashboard, not the old prototype's port 3000

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_profiles(profiles):
    with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2)

def main():
    print('=' * 60)
    print('    ReflectAI - Face Registration & Dashboard Setup')
    print('=' * 60)

    if not os.path.exists(YUNET_MODEL) or not os.path.exists(SFACE_MODEL):
        print('Face models not found. Please run download_face_models.py first.')
        return

    name = input('\nEnter user\'s display name (e.g. Rohan): ').strip()
    if not name:
        print('Name cannot be empty.')
        return

    user_id = name.lower().replace(' ', '_')

    print('\n[Optional] Personalize dashboard (press Enter to use defaults):')
    outfit_input = input('  Today\'s Outfit (comma-separated, e.g. Black T-Shirt, Blue Jeans, White Sneakers): ').strip()
    if outfit_input:
        outfit = [item.strip() for item in outfit_input.split(',') if item.strip()]
    else:
        outfit = ['Black T-Shirt', 'Blue Jeans', 'White Sneakers']

    hair = input('  Hair style note (e.g. Textured Crop): ').strip() or 'Textured Crop'

    cal_input = input('  Calendar events (comma-separated, e.g. 11:00 AM * Demo, 04:00 PM * Gym): ').strip()
    if cal_input:
        calendar = [c.strip() for c in cal_input.split(',') if c.strip()]
    else:
        calendar = ['10:30 AM · Project Review', '04:00 PM · Gym & Workout']

    news_input = input('  News topics / headlines (comma-separated): ').strip()
    if news_input:
        news = [n.strip() for n in news_input.split(',') if n.strip()]
    else:
        news = ['Tech updates & innovations', 'Local weather outlook']

    print('\nInitializing camera...')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open camera (index 0). Please ensure your webcam is connected.')
        return

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print('Error: Could not read frame from camera.')
        cap.release()
        return

    h, w = test_frame.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL,
        config='',
        input_size=(w, h),
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(model=SFACE_MODEL, config='')

    samples = []
    REQUIRED_SAMPLES = 5

    print('\n' + '-' * 60)
    print('Camera active!')
    print('Look directly at the camera.')
    print('Press SPACE to capture a face sample (need 5 samples).')
    print('Press ESC to abort.')
    print('-' * 60)

    best_face_crop = None

    while len(samples) < REQUIRED_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)

        display_frame = frame.copy()
        face_detected = faces is not None and len(faces) > 0

        if face_detected:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            face = faces[0]
            x, y, w_box, h_box = map(int, face[:4])

            cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), (0, 230, 115), 2)
            cv2.putText(
                display_frame,
                f'Face Ready ({len(samples)}/{REQUIRED_SAMPLES})',
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 230, 115),
                2,
            )
        else:
            cv2.putText(
                display_frame,
                'No face detected - step closer',
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            display_frame,
            f'Samples: {len(samples)}/{REQUIRED_SAMPLES} | [SPACE] Capture | [ESC] Cancel',
            (20, display_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
        )

        cv2.imshow('ReflectAI - Face Registration', display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print('Registration cancelled.')
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == 32:
            if face_detected:
                face = faces[0]
                aligned = recognizer.alignCrop(frame, face)
                feat = recognizer.feature(aligned)
                samples.append(feat)
                best_face_crop = aligned.copy()
                print(f'Captured sample {len(samples)}/{REQUIRED_SAMPLES}!')
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (255, 255, 255), 10)
                cv2.imshow('ReflectAI - Face Registration', display_frame)
                cv2.waitKey(80)
            else:
                print('Cannot capture: No clear face detected in frame.')

    cap.release()
    cv2.destroyAllWindows()

    print('\nProcessing facial feature embedding...')
    avg_feature = np.mean(samples, axis=0)
    avg_feature = avg_feature / np.linalg.norm(avg_feature)

    embedding_path = os.path.join(EMBEDDINGS_DIR, f'{user_id}.npy')
    np.save(embedding_path, avg_feature)
    print(f'Saved facial embedding to: {embedding_path}')

    if best_face_crop is not None:
        avatar_path = os.path.join(EMBEDDINGS_DIR, f'{user_id}.jpg')
        cv2.imwrite(avatar_path, best_face_crop)

    profiles = load_profiles()
    profiles[user_id] = {
        'id': user_id,
        'name': name.upper(),
        'greeting_name': name.upper(),
        'outfit': outfit,
        'hair': hair,
        'calendar': calendar,
        'news': news,
    }
    save_profiles(profiles)
    print(f'Profile for \'{name}\' saved to data/profiles.json')

    try:
        # Send name along with userId so ReflectAI's database creates the
        # user record with the correct display name right away.
        requests.post(API_ACTIVE_URL, json={'userId': user_id, 'name': name.upper()}, timeout=1.5)
        print(f'ReflectAI dashboard switched to {name}\'s profile!')
    except Exception:
        print('Note: ReflectAI dashboard is not currently running at http://localhost:5000.')

    print('\n' + '=' * 60)
    print(f'Registration complete for {name}! When you step in front of the mirror,')
    print('face_service.py will recognize you automatically.')
    print('=' * 60)

if __name__ == '__main__':
    main()

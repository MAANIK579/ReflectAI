# download_face_models.py
# Downloads OpenCV Zoo YuNet (face detection) and SFace (face recognition) ONNX models.

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    'face_detection_yunet_2023mar.onnx': {
        'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        'min_size': 200000,
    },
    'face_recognition_sface_2021dec.onnx': {
        'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
        'min_size': 30000000,
    },
}

def download_file(url, dest, min_size):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f'Already downloaded: {os.path.basename(dest)} ({os.path.getsize(dest) / 1024 / 1024:.2f} MB)')
        return

    print(f'Downloading {os.path.basename(dest)}...')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as resp, open(dest, 'wb') as out:
        total = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        chunk_size = 1024 * 64
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = (downloaded / total) * 100
                sys.stdout.write(f'\r  -> {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB)')
                sys.stdout.flush()
        print()

    if os.path.getsize(dest) < min_size:
        raise RuntimeError(f'Downloaded file {dest} is suspiciously small: {os.path.getsize(dest)} bytes')
    print(f'Successfully downloaded {os.path.basename(dest)} ({os.path.getsize(dest) / 1024 / 1024:.2f} MB)\n')

def main():
    print('Checking OpenCV Zoo Face Models...')
    for filename, info in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        download_file(info['url'], dest, info['min_size'])
    print('All face models are ready.')

if __name__ == '__main__':
    main()

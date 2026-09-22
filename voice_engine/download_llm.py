"""
voice_engine/download_llm.py — Download Qwen 2.5 - 1.5B Instruct GGUF model for ReflectAI.

Optimized for Raspberry Pi 4 (4GB RAM) and PC offline voice reasoning.
"""

import os
import sys
import time
import urllib.request

MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_PATH = os.path.join(SCRIPT_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")


def download_model(url: str = MODEL_URL, dest: str = DEST_PATH):
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        if size_mb > 1000:
            print(f"[OK] Model already exists at: {dest} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"[WARN] Incomplete file found ({size_mb:.1f} MB), redownloading...")

    print(f"Downloading Qwen 2.5 - 1.5B Instruct GGUF (~1.05 GB)...")
    print(f"Source: {url}")
    print(f"Target: {dest}")

    start_time = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "ReflectAI-Installer/1.0"})

    try:
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
            total_length = resp.headers.get("Content-Length")
            total_bytes = int(total_length) if total_length else None
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunk
            last_print = 0

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 2.0:  # print every 2 seconds
                    last_print = now
                    mb_done = downloaded / (1024 * 1024)
                    if total_bytes:
                        total_mb = total_bytes / (1024 * 1024)
                        pct = (downloaded / total_bytes) * 100
                        speed = mb_done / max(1, now - start_time)
                        print(f"  Progress: {mb_done:.1f} / {total_mb:.1f} MB ({pct:.1f}%) — {speed:.2f} MB/s")
                    else:
                        print(f"  Downloaded: {mb_done:.1f} MB")

        elapsed = time.time() - start_time
        final_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"\n[SUCCESS] Download completed in {elapsed:.1f}s! Size: {final_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        if os.path.exists(dest):
            try:
                os.remove(dest)
            except Exception:
                pass
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)

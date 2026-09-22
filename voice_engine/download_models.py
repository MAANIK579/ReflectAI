"""
download_models.py — run this ONCE before wakeword_test.py.

Downloads openWakeWord's pretrained model files (melspectrogram/embedding
models plus the wake word models themselves) into the package's local
resources folder.
"""

from openwakeword.utils import download_models

download_models()

print("Done. You can now run wakeword_test.py")

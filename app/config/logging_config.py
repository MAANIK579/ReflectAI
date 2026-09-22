"""
app/config/logging_config.py — one place to configure logging for the
whole app. Per the spec: log application lifecycle, connection status,
detection events, privacy changes, and errors — never raw images or
unnecessary personal data.
"""

import logging
import sys

from app.config.settings import settings


def setup_logging():
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = settings.LOGS_DIR / "reflectai.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
    root_logger.handlers = [file_handler, console_handler]

    return root_logger

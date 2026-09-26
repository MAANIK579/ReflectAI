"""
app/esp32/serial_bridge.py — USB Serial Bridge for ESP32.

Automatically detects and connects to the ESP32 when plugged in via a micro-USB cable
(on Raspberry Pi: /dev/ttyUSB*, /dev/ttyACM*, on Windows: COM*),
reads incoming JSON telemetry lines, and delivers them directly to ReflectAI.
Eliminates any requirement for Wi-Fi or router configuration!
"""

import json
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("reflectai.esp32_serial")

try:
    import serial
    import serial.tools.list_ports
    _HAS_PYSERIAL = True
except ImportError:
    _HAS_PYSERIAL = False


class ESP32SerialBridge:
    def __init__(self, callback: Callable[[dict], None], baud_rate: int = 115200):
        self.callback = callback
        self.baud_rate = baud_rate
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.active_port: Optional[str] = None

    def start(self):
        if not _HAS_PYSERIAL:
            logger.warning("pyserial is not installed. ESP32 USB Serial bridge disabled.")
            return
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="ESP32SerialBridge")
        self.thread.start()
        logger.info("ESP32 USB Serial bridge background worker started.")

    def stop(self):
        self.running = False

    def _find_esp32_port(self) -> Optional[str]:
        if not _HAS_PYSERIAL:
            return None
        try:
            ports = serial.tools.list_ports.comports()
            # 1. Look for known ESP32 / USB UART bridge chip signatures
            for p in ports:
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                dev = (p.device or "").lower()
                if any(k in desc or k in hwid for k in ["cp210", "ch340", "ch341", "ftdi", "uart", "esp32", "silicon labs", "usb-serial"]):
                    return p.device

            # 2. On Linux / Raspberry Pi, check standard USB serial devices
            for p in ports:
                dev = p.device.lower()
                if "ttyusb" in dev or "ttyacm" in dev:
                    return p.device
        except Exception as e:
            logger.debug(f"Error enumerating serial ports: {e}")
        return None

    def _run_loop(self):
        while self.running:
            port = self._find_esp32_port()
            if not port:
                time.sleep(3.0)
                continue

            try:
                logger.info(f"Connecting to ESP32 on USB port {port} at {self.baud_rate} baud...")
                with serial.Serial(port, self.baud_rate, timeout=2.0) as ser:
                    self.active_port = port
                    logger.info(f"✓ Connected to ESP32 on {port} via USB cable (No Wi-Fi required)!")
                    
                    while self.running:
                        try:
                            raw_line = ser.readline()
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if not line:
                                continue

                            # Parse JSON packet from ESP32
                            if line.startswith("{") and line.endswith("}"):
                                try:
                                    payload = json.loads(line)
                                    if isinstance(payload, dict) and payload.get("device") == "esp32":
                                        self.callback(payload)
                                except json.JSONDecodeError:
                                    pass
                        except (serial.SerialException, OSError) as read_err:
                            logger.warning(f"ESP32 Serial read error on {port}: {read_err}")
                            break
            except (serial.SerialException, OSError) as conn_err:
                logger.debug(f"Could not open serial port {port}: {conn_err}")
                self.active_port = None
                time.sleep(3.0)
            except Exception as e:
                logger.error(f"Unexpected error in ESP32 serial bridge: {e}")
                self.active_port = None
                time.sleep(3.0)
            finally:
                self.active_port = None

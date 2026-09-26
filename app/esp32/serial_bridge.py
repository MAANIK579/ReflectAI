"""
app/esp32/serial_bridge.py — USB Serial Bridge for ESP32.

Automatically detects and connects to the ESP32 when plugged in via a micro-USB cable
(on Raspberry Pi: /dev/ttyUSB*, /dev/ttyACM*, on Windows: COM*),
reads incoming JSON telemetry lines, and delivers them directly to ReflectAI.
Eliminates any requirement for Wi-Fi or router configuration!
"""

import glob
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Optional

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
        self.status: str = "stopped"
        self.last_error: Optional[str] = None
        self.last_packet_time: Optional[float] = None
        self.packets_received: int = 0

    def start(self):
        if not _HAS_PYSERIAL:
            self.status = "pyserial_missing"
            self.last_error = "pyserial is not installed in Python environment"
            logger.warning("pyserial is not installed. ESP32 USB Serial bridge disabled.")
            return
        if self.running:
            return
        self.running = True
        self.status = "starting"
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="ESP32SerialBridge")
        self.thread.start()
        logger.info("ESP32 USB Serial bridge background worker started.")

    def stop(self):
        self.running = False
        self.status = "stopped"

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "has_pyserial": _HAS_PYSERIAL,
            "running": self.running,
            "status": self.status,
            "active_port": self.active_port,
            "last_error": self.last_error,
            "packets_received": self.packets_received,
            "last_packet_time": self.last_packet_time,
        }

    def _find_esp32_port(self) -> Optional[str]:
        if not _HAS_PYSERIAL:
            return None
        try:
            ports = list(serial.tools.list_ports.comports())
            # 1. Match known ESP32 / USB UART bridge chip signatures
            for p in ports:
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                dev = (p.device or "").lower()
                if any(k in desc or k in hwid for k in ["cp210", "ch340", "ch341", "ch9102", "ftdi", "uart", "esp32", "silicon labs", "usb-serial"]):
                    return p.device

            # 2. Check standard USB serial devices from comports()
            for p in ports:
                dev = p.device.lower()
                if "ttyusb" in dev or "ttyacm" in dev:
                    return p.device
        except Exception as e:
            logger.debug(f"Error enumerating serial ports via comports(): {e}")

        # 3. Direct Linux character device fallback (/dev/ttyUSB*, /dev/ttyACM*)
        try:
            linux_devices = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
            if linux_devices:
                return linux_devices[0]
        except Exception as e:
            logger.debug(f"Error checking Linux /dev serial nodes: {e}")

        return None

    def _run_loop(self):
        while self.running:
            port = self._find_esp32_port()
            if not port:
                self.status = "no_port_found"
                self.active_port = None
                time.sleep(3.0)
                continue

            try:
                self.status = "connecting"
                logger.info(f"Connecting to ESP32 on USB port {port} at {self.baud_rate} baud...")
                with serial.Serial(port, self.baud_rate, timeout=2.0) as ser:
                    # Prevent ESP32 from getting stuck in reset or bootloader
                    try:
                        ser.dtr = False
                        ser.rts = False
                    except Exception:
                        pass

                    time.sleep(0.1)
                    try:
                        ser.reset_input_buffer()
                    except Exception:
                        pass

                    self.active_port = port
                    self.status = "connected"
                    self.last_error = None
                    logger.info(f"✓ Connected to ESP32 on {port} via USB cable (No Wi-Fi required)!")
                    
                    while self.running:
                        try:
                            raw_line = ser.readline()
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if not line:
                                continue

                            # 1. Parse JSON packet from ESP32 (resilient to surrounding text)
                            if "{" in line and "}" in line:
                                start_idx = line.find("{")
                                end_idx = line.rfind("}")
                                if end_idx > start_idx:
                                    json_str = line[start_idx : end_idx + 1]
                                    try:
                                        payload = json.loads(json_str)
                                        if isinstance(payload, dict) and payload.get("device") == "esp32":
                                            self.packets_received += 1
                                            self.last_packet_time = time.time()
                                            self.callback(payload)
                                            continue
                                    except json.JSONDecodeError:
                                        pass

                            # 2. Hybrid Fallback: Parse plain-text logs (if ESP32 still runs legacy firmware)
                            # Matches DHT format: "Temp: 24.5 C | Hum: 50.0 %"
                            dht_match = re.search(r"Temp:\s*([\d\.\-]+)\s*C\s*\|\s*Hum:\s*([\d\.\-]+)\s*%", line, re.IGNORECASE)
                            if dht_match:
                                try:
                                    t = float(dht_match.group(1))
                                    h = float(dht_match.group(2))
                                    self.packets_received += 1
                                    self.last_packet_time = time.time()
                                    self.callback({
                                        "device": "esp32",
                                        "status": "online",
                                        "temperature": t,
                                        "humidity": h,
                                    })
                                    continue
                                except ValueError:
                                    pass

                            # Matches physical button logs from legacy firmware
                            up_line = line.upper()
                            if "USER 1" in up_line and "SELECTED" in up_line:
                                self.packets_received += 1
                                self.last_packet_time = time.time()
                                self.callback({"device": "esp32", "status": "online", "user": "User 1"})
                            elif "USER 2" in up_line and "SELECTED" in up_line:
                                self.packets_received += 1
                                self.last_packet_time = time.time()
                                self.callback({"device": "esp32", "status": "online", "user": "User 2"})
                            elif "GUEST" in up_line and "SELECTED" in up_line:
                                self.packets_received += 1
                                self.last_packet_time = time.time()
                                self.callback({"device": "esp32", "status": "online", "user": "Guest"})
                            elif "PRIVACY MODE: ON" in up_line:
                                self.packets_received += 1
                                self.last_packet_time = time.time()
                                self.callback({"device": "esp32", "status": "online", "privacy": True})
                            elif "PRIVACY MODE: OFF" in up_line:
                                self.packets_received += 1
                                self.last_packet_time = time.time()
                                self.callback({"device": "esp32", "status": "online", "privacy": False})
                            elif any(k in line for k in ["REFLECTAI", "SYSTEM STATUS", "DHT22", "Entering main loop", "Wi-Fi Connected", "background reconnect"]):
                                # Heartbeat activity indicating ESP32 is powered and communicating on serial
                                self.packets_received += 1
                                self.last_packet_time = time.time()
                                self.callback({"device": "esp32", "status": "online"})
                        except (serial.SerialException, OSError) as read_err:
                            logger.warning(f"ESP32 Serial read error on {port}: {read_err}")
                            self.last_error = str(read_err)
                            break
            except (serial.SerialException, OSError) as conn_err:
                err_str = str(conn_err)
                self.last_error = err_str
                if "Permission denied" in err_str or "[Errno 13]" in err_str:
                    self.status = "permission_denied"
                    logger.error(
                        f"[ESP32 USB ERROR] Permission denied opening {port}! "
                        f"Please add your user to the dialout group by running: "
                        f"sudo usermod -a -G dialout $USER  (then restart terminal or reboot)"
                    )
                elif "Device or resource busy" in err_str or "[Errno 16]" in err_str:
                    self.status = "port_busy"
                    logger.warning(
                        f"[ESP32 USB WARNING] Port {port} is busy or locked by another process."
                    )
                else:
                    self.status = "connection_failed"
                    logger.warning(f"Could not open serial port {port}: {conn_err}")
                self.active_port = None
                time.sleep(3.0)
            except Exception as e:
                self.status = "error"
                self.last_error = str(e)
                logger.error(f"Unexpected error in ESP32 serial bridge: {e}")
                self.active_port = None
                time.sleep(3.0)
            finally:
                self.active_port = None


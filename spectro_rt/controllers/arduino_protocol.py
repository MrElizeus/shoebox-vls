# -*- coding: utf-8 -*-
# spectro_rt/controllers/arduino_protocol.py
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

try:
    import serial
except Exception as e:
    serial = None
    _ser_err = e


class ArduinoProtocol:
    """
    Lee líneas del Arduino y extrae temperatura y estado del heater.
    Acepta dos formatos:
      1) CSV: millis,tempC,heater,beacon
      2) Texto: '<tempC>    Heater Enabled|Disabled'
    Llama line_cb(ts, tempC, heater_bool)
    """

    def __init__(self, port: str = "COM3", baud: int = 115200,
                 line_cb: Optional[Callable[[float, float, bool], None]] = None):
        self.port = port
        self.baud = int(baud)
        self.line_cb = line_cb
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._ser = None

    def start(self):
        if serial is None:
            raise RuntimeError(f"pyserial no disponible: {_ser_err}")
        self._ser = serial.Serial(self.port, self.baud, timeout=1)
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)
        self._th = None
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def _run(self):
        while not self._stop.is_set():
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                ts = time.time()
                tempC, heater = self._parse(line)
                if tempC is not None and self.line_cb:
                    self.line_cb(ts, tempC, heater)
            except Exception:
                # ignora líneas corruptas
                pass

    @staticmethod
    def _parse(line: str):
        # CSV: millis,tempC,heater,beacon
        if "," in line:
            try:
                parts = line.split(",")
                temp = float(parts[1])
                heater = bool(int(parts[2])) if len(parts) > 2 else False
                return temp, heater
            except Exception:
                return None, False
        # Texto: "23.45    Heater Enabled"
        try:
            tokens = line.split()
            temp = float(tokens[0])
            heater = "Enabled" in line
            return temp, heater
        except Exception:
            return None, False

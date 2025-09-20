# -*- coding: utf-8 -*-
import time, threading
from typing import Callable, Optional, Dict
import numpy as np

try:
    import mss
    import cv2
except Exception as e:
    mss = None
    cv2 = None
    _screen_err = e

class ScreenReader:
    """
    Captura un rectángulo de pantalla con MSS y lo entrega como frame BGR.
    Usa el ROI de imaging como caja de captura.
    """
    def __init__(self, roi: Dict[str, int], fps: float = 8.0,
                 frame_cb: Optional[Callable[[float, np.ndarray], None]] = None,
                 monitor_index: int = 1):
        self.roi = {k: int(roi.get(k, 0)) for k in ("x", "y", "w", "h")}
        self.fps = max(0.5, float(fps))
        self.frame_cb = frame_cb
        self.monitor_index = monitor_index
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._mss = None

    def start(self):
        if mss is None or cv2 is None:
            raise RuntimeError(f"Captura de pantalla no disponible: {_screen_err}")
        self._mss = mss.mss()
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)
        self._th = None
        try:
            if self._mss:
                self._mss.close()
        except Exception:
            pass
        self._mss = None

    def _run(self):
        period = 1.0 / self.fps
        nxt = time.time()
        box = {
            "left": int(self.roi["x"]),
            "top":  int(self.roi["y"]),
            "width": int(self.roi["w"]),
            "height":int(self.roi["h"]),
        }
        while not self._stop.is_set():
            ts = time.time()
            shot = self._mss.grab(box)
            # MSS -> BGRA
            frame = np.array(shot)
            # BGRA -> BGR
            frame = frame[:, :, :3]
            if self.frame_cb:
                self.frame_cb(ts, frame)
            nxt += period
            sleep = nxt - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                nxt = time.time()

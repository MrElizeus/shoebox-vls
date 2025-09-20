# -*- coding: utf-8 -*-
# spectro_rt/controllers/image_sources/camera_reader.py
from __future__ import annotations

import time
import threading
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None
    _cv_err = e


class CameraReader:
    """
    Lector de cámara con OpenCV.
    Llama frame_cb(ts, frame_bgr) a ~fps. 'frame_bgr' es np.ndarray HxWx3 (BGR).
    """

    def __init__(
        self,
        cam_index: int = 0,
        fps: float = 8.0,
        frame_cb: Optional[Callable[[float, np.ndarray], None]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self.cam_index = cam_index
        self.fps = max(0.5, float(fps))
        self.frame_cb = frame_cb
        self.width = width
        self.height = height
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self.cap = None

    def start(self):
        if cv2 is None:
            raise RuntimeError(f"OpenCV no disponible: {_cv_err}")
        if self._th and self._th.is_alive():
            return
        backend = getattr(cv2, "CAP_DSHOW", 0)  # Windows abre más rápido
        self.cap = cv2.VideoCapture(self.cam_index, backend)
        if not self.cap or not self.cap.isOpened():
            # fallback sin backend
            self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara index={self.cam_index}")

        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)
        self._th = None
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    def _run(self):
        period = 1.0 / self.fps
        nxt = time.time()
        while not self._stop.is_set():
            ok, frame = self.cap.read() if self.cap else (False, None)
            ts = time.time()
            if ok and frame is not None and self.frame_cb:
                # ya es BGR
                self.frame_cb(ts, frame)
            nxt += period
            sleep = nxt - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                nxt = time.time()

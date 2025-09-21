# spectro_rt/controllers/image_sources/camera_reader.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import threading
from typing import Callable, Dict, Optional, Tuple, List

try:
    import cv2
except Exception as e:  # pragma: no cover
    cv2 = None
    _import_err = e
else:
    _import_err = None


class CameraReader:
    """
    Lector de cámara (OpenCV) robusto y sin auto-ajustes.
    - Auto-detecta backend (MSMF→DSHOW→ANY) e índice (0..9) si el configurado falla.
    - Desactiva auto-exposure/auto-wb/autofocus al iniciar (opcional).
    - Permite aplicar controles manuales en caliente.
    """

    def __init__(
        self,
        cam_index: int = -1,           # -1 = auto-detect
        fps: float = 12.0,
        frame_cb: Optional[Callable[[float, "np.ndarray"], None]] = None,
        initial_controls: Optional[Dict] = None,
        disable_auto_on_start: bool = True,
        max_scan_index: int = 9
    ):
        if cv2 is None:
            raise RuntimeError(f"OpenCV no disponible: {_import_err}")
        self.cam_index_cfg = int(cam_index)
        self.cam_index_real: Optional[int] = None
        self.backend_real: Optional[int] = None
        self.max_scan_index = int(max_scan_index)

        self.target_dt = 1.0 / max(1.0, float(fps))
        self.frame_cb = frame_cb
        self.controls = dict(initial_controls or {})
        self.disable_auto_on_start = bool(disable_auto_on_start)

        self._cap = None
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    # ---------- Enumeración ----------
    @staticmethod
    def _backend_list() -> List[int]:
        return [getattr(cv2, "CAP_MSMF", 0), getattr(cv2, "CAP_DSHOW", 700), getattr(cv2, "CAP_ANY", 0)]

    @staticmethod
    def backend_name(api: int) -> str:
        names = {getattr(cv2, "CAP_MSMF", -1): "MSMF",
                 getattr(cv2, "CAP_DSHOW", -1): "DSHOW",
                 getattr(cv2, "CAP_ANY", -1): "ANY"}
        return names.get(api, str(api))

    def _try_open(self, idx: int, api: int) -> bool:
        cap = cv2.VideoCapture(idx, api)
        ok = bool(cap and cap.isOpened())
        if ok:
            ok, _ = cap.read()
        if ok:
            self._cap = cap
            self.cam_index_real = idx
            self.backend_real = api
            return True
        if cap:
            cap.release()
        return False

    def _auto_open(self) -> None:
        # 1) Intenta el índice configurado con varios backends
        if self.cam_index_cfg >= 0:
            for api in self._backend_list():
                if self._try_open(self.cam_index_cfg, api):
                    return
        # 2) Explora índices 0..max_scan_index en varios backends
        for api in self._backend_list():
            for idx in range(0, self.max_scan_index + 1):
                if self._try_open(idx, api):
                    return
        # 3) Intento final con constructor por defecto
        cap = cv2.VideoCapture(self.cam_index_cfg if self.cam_index_cfg >= 0 else 0)
        if cap and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                self._cap = cap
                self.cam_index_real = self.cam_index_cfg if self.cam_index_cfg >= 0 else 0
                self.backend_real = None
                return
            cap.release()
        raise RuntimeError("No se encontró ninguna cámara disponible (probé varios backends e índices).")

    # ---------- Público ----------
    def start(self) -> None:
        if self._th and self._th.is_alive():
            return
        self._stop.clear()

        # Abrir cámara (auto si hace falta)
        self._auto_open()

        # FPS preferido (no todos los drivers lo aplican)
        try:
            self._cap.set(cv2.CAP_PROP_FPS, 1.0 / self.target_dt)
        except Exception:
            pass

        # Desactivar autos y aplicar controles manuales
        if self.disable_auto_on_start:
            self._disable_autos()
        if self.controls:
            self._apply_controls(self.controls)

        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)
        self._th = None
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def update_controls(self, controls: Dict, disable_auto: bool = True) -> None:
        self.controls = dict(controls or {})
        if self._cap is None:
            return
        if disable_auto:
            self._disable_autos()
        self._apply_controls(self.controls)

    def disable_autos_now(self) -> None:
        if self._cap is None:
            return
        self._disable_autos()

    def get_info(self) -> Tuple[Optional[int], Optional[str]]:
        return self.cam_index_real, (self.backend_name(self.backend_real) if self.backend_real is not None else "default")

    # ---------- Internals ----------
    def _run(self) -> None:
        import numpy as np
        next_t = time.time()
        while not self._stop.is_set():
            if self._cap is None:
                break
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            ts = time.time()
            if self.frame_cb:
                try:
                    self.frame_cb(ts, frame)  # frame BGR
                except Exception:
                    pass
            next_t += self.target_dt
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.time()

    def _disable_autos(self) -> None:
        if self._cap is None:
            return
        try:
            # Variantes de autoexposición
            for val in (0, 1, 0.25, 0.75):
                self._cap.set(getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", 0), val)
            # AutoWB / AutoFocus
            try: self._cap.set(getattr(cv2, "CAP_PROP_AUTO_WB", 0), 0)
            except Exception: pass
            try: self._cap.set(getattr(cv2, "CAP_PROP_AUTOFOCUS", 0), 0)
            except Exception: pass
            # Backlight compensation a 0
            try: self._cap.set(getattr(cv2, "CAP_PROP_BACKLIGHT", 0), 0)
            except Exception: pass
        except Exception:
            pass

    def _apply_controls(self, C: Dict) -> None:
        if self._cap is None:
            return
        MAP = {
            "brightness": getattr(cv2, "CAP_PROP_BRIGHTNESS", None),
            "contrast": getattr(cv2, "CAP_PROP_CONTRAST", None),
            "saturation": getattr(cv2, "CAP_PROP_SATURATION", None),
            "gain": getattr(cv2, "CAP_PROP_GAIN", None),
            "exposure": getattr(cv2, "CAP_PROP_EXPOSURE", None),
            "wb_temp": getattr(cv2, "CAP_PROP_WB_TEMPERATURE", None),
            "focus": getattr(cv2, "CAP_PROP_FOCUS", None),
            "sharpness": getattr(cv2, "CAP_PROP_SHARPNESS", None),
            "gamma": getattr(cv2, "CAP_PROP_GAMMA", None),
        }
        for k, v in (C or {}).items():
            prop = MAP.get(k)
            if prop is None or v is None:
                continue
            try:
                self._cap.set(prop, float(v))
            except Exception:
                pass

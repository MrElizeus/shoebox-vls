# -*- coding: utf-8 -*-
# spectro_rt/processing/spectrum_extractor.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None
    _cv_err = e


class SpectrumExtractor:
    """
    Pipeline tipo ImageJ:
      - ROI -> gris
      - Gamma (pow normalizado)
      - Resta fondo (constante)
      - Floor mínimo
      - Log10
      - Resize a width y promedio vertical -> vector (1×W)
    Devuelve (linea_cruda, linea_procesada)
    """

    def __init__(
        self,
        roi: Dict[str, int],
        resize_w: int = 500,
        gamma: float = 2.2222,
        bg_sub: float = 300.0,
        min_floor: float = 1e-3,
        log10: bool = True,
        baseline_mode: str = "ema",
        alpha: float = 0.01,
    ):
        if cv2 is None:
            raise RuntimeError(f"OpenCV no disponible: {_cv_err}")
        self.roi = {k: int(roi.get(k, 0)) for k in ("x", "y", "w", "h")}
        self.resize_w = int(resize_w)
        self.gamma = float(gamma)
        self.bg_sub = float(bg_sub)
        self.min_floor = float(min_floor)
        self.log10 = bool(log10)
        self.baseline_mode = str(baseline_mode)
        self.alpha = float(alpha)
        self._ema: np.ndarray | None = None

    @staticmethod
    def _to_gray(bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    @staticmethod
    def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
        if gamma is None or gamma == 1.0:
            return img
        mn, mx = img.min(), img.max()
        if mx <= mn:
            return img
        norm = (img - mn) / (mx - mn + 1e-12)
        out = np.power(norm, gamma) * 255.0
        return out

    def _post_log(self, vec: np.ndarray) -> np.ndarray:
        v = vec - self.bg_sub
        v[v < self.min_floor] = self.min_floor
        if self.log10:
            v = np.log(v) * 0.4343  # ln -> log10
        return v

    def _resize_to_w(self, img: np.ndarray, width: int) -> np.ndarray:
        if img.shape[1] == width:
            return img
        return cv2.resize(img, (width, img.shape[0]), interpolation=cv2.INTER_AREA)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = frame_bgr.shape[:2]
        x = max(0, min(self.roi["x"], W - 1))
        y = max(0, min(self.roi["y"], H - 1))
        w = max(1, min(self.roi["w"], W - x))
        h = max(1, min(self.roi["h"], H - y))
        crop = frame_bgr[y:y + h, x:x + w, :]

        gray = self._to_gray(crop)
        gamma_applied = self._apply_gamma(gray, self.gamma)

        # Promedio vertical (pre)
        raw_line = gamma_applied.mean(axis=0)  # 1×w

        # Resize + promedio vertical (post)
        resized = self._resize_to_w(gamma_applied, self.resize_w)
        proc_line = resized.mean(axis=0)  # 1×resize_w
        proc_line = self._post_log(proc_line)

        # EMA opcional
        if self.baseline_mode == "ema":
            if self._ema is None or self._ema.shape != proc_line.shape:
                self._ema = proc_line.copy()
            else:
                self._ema = self.alpha * proc_line + (1 - self.alpha) * self._ema
            out_line = self._ema
        else:
            out_line = proc_line

        return raw_line.astype(np.float32), out_line.astype(np.float32)

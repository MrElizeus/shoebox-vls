# spectro_rt/gui.py
# -*- coding: utf-8 -*-
"""
spectro-rt v0.3 (all-Python imaging)
- Preview en vivo centrado, visible incluso antes de iniciar (modo real)
- Cámara robusta (autodetección de backend/índice) sin auto-ajustes; controles manuales
- 3 subplots: T(t), A(λ_sel)(t), I/A(λ)
- ROI editable; marco rojo actualizado en preview
- Pop-ups si falla Arduino o la fuente de imagen (sin crashear)
- Modo espectro: I(λ) si no hay blanco; A(λ) si ya hay referencia
- Panel de medición: λ_sel, A, %T, C con ε·L configurable
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import yaml

# Tkinter / Matplotlib embed
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

# Preview rendering
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore


# ======================
# Acquisition Layer
# ======================

@dataclass
class RingBuffer:
    maxlen: int
    t: Deque[float] = field(default_factory=deque)
    y: Deque[float] = field(default_factory=deque)

    def push(self, ts: float, val: float) -> None:
        if len(self.t) >= self.maxlen:
            self.t.popleft()
            self.y.popleft()
        self.t.append(ts)
        self.y.append(val)

    def window(self, since_s: float) -> Tuple[np.ndarray, np.ndarray]:
        if not self.t:
            return np.array([]), np.array([])
        t0 = self.t[-1] - since_s
        idx = 0
        for i, ti in enumerate(self.t):
            if ti >= t0:
                idx = i
                break
        t_arr = np.fromiter(list(self.t)[idx:], dtype=np.float64)
        y_arr = np.fromiter(list(self.y)[idx:], dtype=np.float32)
        return t_arr, y_arr


@dataclass
class SpectraBuffer:
    max_frames: int
    lambdas: Optional[np.ndarray] = None
    frames: List[np.ndarray] = field(default_factory=list)

    def push(self, spectrum: np.ndarray, lambdas: np.ndarray) -> None:
        self.lambdas = lambdas.astype(np.float32)
        self.frames.append(spectrum.astype(np.float32))
        if len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames:]

    def latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.frames:
            return None, self.lambdas
        return self.frames[-1], self.lambdas

    def as_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.lambdas is None or not self.frames:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        M = np.vstack(self.frames)  # N×W
        return M, self.lambdas


class Acquisition:
    """Maneja adquisición sim/real y preview-only (cámara/pantalla)."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.mode = cfg.get("mode", "sim")
        self.running = False
        self.preview_only = False

        # Series buffers
        self.temp = RingBuffer(maxlen=20000)
        self.a_sel = RingBuffer(maxlen=20000)
        self.spec = SpectraBuffer(max_frames=20000)

        gcfg = cfg.get("graphics", {})
        self.lmin = float(gcfg.get("lambda_min", 400))
        self.lmax = float(gcfg.get("lambda_max", 700))
        self.w = int(cfg.get("imaging", {}).get("resize_w", 500))
        self.lambdas = np.linspace(self.lmin, self.lmax, self.w, dtype=np.float32)
        self.lambda_sel = float(gcfg.get("lambda_selected_nm", 520))

        # Preview frame (RGB) y líneas recientes
        self._preview_rgb: Optional[np.ndarray] = None
        self._last_I: Optional[np.ndarray] = None
        self._last_A: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (A, lam_dyn)

        # Referencias (dark y blank)
        self._dark: Optional[np.ndarray] = None
        self._blank: Optional[np.ndarray] = None
        self._blank_avg_n: int = 0
        self._blank_accum: Optional[np.ndarray] = None

        # Estado de modo de dibujo (I/A)
        self._plot_mode = "I"  # "I" intensidad o "A" absorbancia

        # Threads/sources
        self._threads: List[threading.Thread] = []
        self._stop_evt = threading.Event()
        self._arduino = None
        self._img_source = None
        self._extractor = None

        # Controles actuales de cámara (manuales)
        self._cam_controls: Dict = dict(cfg.get("imaging", {}).get("camera_controls", {}))
        paths_cfg = cfg.get("paths", {})
        raw_dir_cfg = paths_cfg.get("raw_roi_dir")
        self._raw_dir: Optional[Path] = None
        if raw_dir_cfg:
            try:
                raw_path = Path(raw_dir_cfg)
                raw_path.mkdir(parents=True, exist_ok=True)
                self._raw_dir = raw_path
            except Exception:
                self._raw_dir = None
        imaging_cfg = cfg.get("imaging", {})
        self._save_raw = (self.mode == "real") and bool(imaging_cfg.get("save_raw_roi", False))
        self._save_png = (self.mode == "real") and bool(imaging_cfg.get("save_raw_png", False)) and Image is not None
        self._raw_counter = 0
        self._raw_info_written = False

    # ---------- Public API (referencias) ----------

    def set_dark(self) -> None:
        if self._last_I is not None:
            self._dark = self._last_I.copy()

    def clear_dark(self) -> None:
        self._dark = None

    def start_blank(self, avg_n: int = 1) -> None:
        """Inicia captura de blanco; si avg_n>1 acumula y promedia."""
        self._blank_avg_n = max(1, int(avg_n))
        self._blank_accum = None

    def stop_blank(self) -> None:
        """Termina y fija el blanco con lo acumulado o el último I."""
        if self._blank_accum is not None and self._blank_avg_n >= 0:
            total = max(1, int(self.cfg.get("imaging", {}).get("blank_avg_total", 1)))
            self._blank = self._blank_accum / float(total)
        elif self._last_I is not None:
            self._blank = self._last_I.copy()
        self._blank_accum = None
        self._blank_avg_n = 0

    def clear_blank(self) -> None:
        self._blank = None
        self._blank_accum = None
        self._blank_avg_n = 0

    def get_plot_mode(self) -> str:
        return self._plot_mode

    def latest_intensity(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        if self._last_I is None:
            return None, self.lambdas
        return self._last_I, self.lambdas

    def latest_preview(self) -> Optional[np.ndarray]:
        return self._preview_rgb

    def latest_temp(self) -> Optional[float]:
        return self.temp.y[-1] if self.temp.y else None

    def latest_asel(self) -> Optional[float]:
        return self.a_sel.y[-1] if self.a_sel.y else None

    def latest_spectrum(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        spec, lam = self.spec.latest()
        return spec, (lam if lam is not None else self.lambdas)

    def export_csv(self, out_path: Path) -> None:
        M, lambdas = self.spec.as_matrix()
        tT = np.array(self.temp.t, dtype=np.float64)
        TT = np.array(self.temp.y, dtype=np.float32)
        tA = np.array(self.a_sel.t, dtype=np.float64)
        AA = np.array(self.a_sel.y, dtype=np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Time series
        np.savetxt(out_path.with_suffix(".temp.csv"), np.c_[tT, TT], delimiter=",",
                   header="t_s,TempC", comments="")
        np.savetxt(out_path.with_suffix(".asel.csv"), np.c_[tA, AA], delimiter=",",
                   header="t_s,A_sel", comments="")
        # Spectra matrix
        if M.size:
            head = ",".join(["lambda_nm"] + [f"frame_{i}" for i in range(M.shape[0])])
            mat = np.vstack([lambdas, M.T]).T  # W × (1+N)
            np.savetxt(out_path.with_suffix(".spectra.csv"), mat, delimiter=",",
                       header=head, comments="")

    def export_npy(self, out_path: Path) -> None:
        M, lambdas = self.spec.as_matrix()
        meta = dict(lambda_min=float(self.lmin), lambda_max=float(self.lmax),
                    width=int(self.w), lambda_selected=float(self.lambda_sel))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".spectra.npy"), M)
        np.save(out_path.with_suffix(".lambdas.npy"), lambdas)
        np.save(out_path.with_suffix(".meta.npy"), meta)

    # ---------- Internals (A = -log10((I-D)/(B-D))) ----------

    def _apply_dark(self, I: np.ndarray) -> np.ndarray:
        if self._dark is None or self._dark.shape != I.shape:
            return I
        return np.clip(I - self._dark, 1e-6, None)

    def _accumulate_blank_if_needed(self, I_corr: np.ndarray) -> None:
        if self._blank_avg_n > 0:
            if self._blank_accum is None:
                self._blank_accum = np.zeros_like(I_corr, dtype=np.float32)
            self._blank_accum += I_corr
            self._blank_avg_n -= 1
            if self._blank_avg_n == 0:
                # fijar automáticamente (stop_blank) con el promedio acumulado
                total = max(1, int(self.cfg.get("imaging", {}).get("blank_avg_total", 1)))
                self._blank = self._blank_accum.copy() / float(total)
                self._blank_accum = None

    def _compute_A_from_I(self, I_raw: np.ndarray) -> Optional[np.ndarray]:
        I_corr = self._apply_dark(I_raw)
        if self._blank is None or self._blank.shape != I_corr.shape:
            return None
        denom = np.clip(self._blank, 1e-6, None)
        T = np.clip(I_corr / denom, 1e-6, 1e6)
        A = -np.log10(T)
        return A.astype(np.float32)

    def _write_roi_raw(self, frame_bgr: np.ndarray, roi: Dict[str, int], ts: float) -> None:
        if (not self._save_raw and not self._save_png) or self._raw_dir is None:
            return
        try:
            H, W = frame_bgr.shape[:2]
            if W == 0 or H == 0:
                return
            x = int(roi.get("x", 0))
            y = int(roi.get("y", 0))
            w = int(roi.get("w", W - x))
            h = int(roi.get("h", H - y))
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            if w <= 0 or h <= 0:
                return
            crop = frame_bgr[y:y + h, x:x + w]
            if crop.size == 0:
                return
            ts_ms = int(ts * 1000)
            base_name = f"roi_{ts_ms:013d}_{self._raw_counter:06d}"
            formats: List[str] = []
            if self._save_raw:
                raw_path = self._raw_dir / f"{base_name}.raw"
                crop.tofile(raw_path)
                formats.append("RAW uint8 BGR" if crop.ndim == 3 and crop.shape[2] == 3 else "RAW uint8")
            if self._save_png:
                try:
                    if crop.ndim == 3 and crop.shape[2] == 3:
                        img = Image.fromarray(crop[..., ::-1], mode="RGB")
                    elif crop.ndim == 2:
                        img = Image.fromarray(crop, mode="L")
                    else:
                        img = Image.fromarray(crop.reshape(h, w), mode="L")
                    png_path = self._raw_dir / f"{base_name}.png"
                    img.save(png_path)
                    formats.append("PNG RGB" if img.mode == "RGB" else f"PNG {img.mode}")
                except Exception:
                    pass
            if not self._raw_info_written:
                info = (
                    f"ROI RAW frames\n"
                    f"Resolución ROI: ancho={w} px, alto={h} px, canales={crop.shape[2] if crop.ndim == 3 else 1}\n"
                    f"Formatos guardados: {', '.join(formats) if formats else 'N/A'}\n"
                    "Archivos RAW: uint8 sin cabecera, orden fila a fila (row-major). PNG: RGB estándar.\n"
                    "Puedes reconstruir los RAW con numpy.fromfile(..., dtype=np.uint8).reshape(h, w, canales).\n"
                )
                (self._raw_dir / "README.txt").write_text(info, encoding="utf-8")
                self._raw_info_written = True
            self._raw_counter += 1
        except Exception:
            pass

    # ---------- Camera controls (manuales) ----------

    def update_camera_controls(self, controls: Dict, disable_auto: bool = True) -> None:
        self._cam_controls = dict(controls or {})
        try:
            if self._img_source and hasattr(self._img_source, "update_controls"):
                self._img_source.update_controls(self._cam_controls, disable_auto=disable_auto)
        except Exception:
            pass
    def adjust_focus(self, delta: float) -> float:
        """Ajusta el control de enfoque manual (si la cÁmara lo soporta) y devuelve el valor fijado."""
        if not isinstance(delta, (int, float)):
            delta = 0.0
        current = float(self._cam_controls.get("focus") or 0.0)
        new_val = current + float(delta)
        self._cam_controls["focus"] = new_val
        self.update_camera_controls(self._cam_controls, disable_auto=False)
        return new_val

    # ---------- Start/Stop ----------

    def start_preview(self) -> None:
        """Inicia fuente de imagen SOLO para previsualización (y espectro de prueba)."""
        if self.preview_only or self.running or self.mode != "real":
            return

        # Lazy imports extractor y fuentes
        try:
            from .processing.spectrum_extractor import SpectrumExtractor  # type: ignore
        except Exception:
            from spectro_rt.processing.spectrum_extractor import SpectrumExtractor  # type: ignore

        source = self.cfg.get("imaging", {}).get("source", "camera")
        if source == "screen":
            try:
                from .controllers.image_sources.screen_reader import ScreenReader  # type: ignore
            except Exception:
                from spectro_rt.controllers.image_sources.screen_reader import ScreenReader  # type: ignore
        else:
            try:
                from .controllers.image_sources.camera_reader import CameraReader  # type: ignore
            except Exception:
                from spectro_rt.controllers.image_sources.camera_reader import CameraReader  # type: ignore

        # Imaging config
        icfg = self.cfg.get("imaging", {})
        fps = float(icfg.get("fps", 12))
        roi_cfg = icfg.get("roi", {"x": 0, "y": 0, "w": 500, "h": 86})
        resize_w = int(icfg.get("resize_w", 500))
        gamma = float(icfg.get("gamma", 2.2222))
        bg = float(icfg.get("background_subtract", 300.0))
        min_floor = float(icfg.get("min_floor", 1e-3))
        log10 = bool(icfg.get("log10", True))
        baseline = icfg.get("baseline", {"mode": "ema", "alpha": 0.01})

        # Extractor
        try:
            self._extractor = SpectrumExtractor(
                roi=roi_cfg, resize_w=resize_w, gamma=gamma, bg_sub=bg, min_floor=min_floor,
                log10=log10, baseline_mode=baseline.get("mode", "ema"), alpha=baseline.get("alpha", 0.01)
            )
        except TypeError:
            self._extractor = SpectrumExtractor(
                roi=roi_cfg, resize_w=resize_w, gamma=gamma, bg_sub=bg, min_floor=min_floor, log10=log10
            )

        def on_frame(ts: float, frame_bgr: np.ndarray) -> None:
            # ROI actual
            roi = getattr(self._extractor, "roi", roi_cfg)
            # Preview: BGR→RGB con marco ROI
            rgb = frame_bgr[..., ::-1].copy()
            try:
                y, x, h, w = roi.get("y", 0), roi.get("x", 0), roi.get("h", 86), roi.get("w", 500)
                H, W = rgb.shape[:2]
                y0, y1 = max(0, y), min(H - 1, y + h)
                x0, x1 = max(0, x), min(W - 1, x + w)
                rgb[y0:y0+2, x0:x1, :] = (255, 0, 0)
                rgb[y1-2:y1, x0:x1, :] = (255, 0, 0)
                rgb[y0:y1, x0:x0+2, :] = (255, 0, 0)
                rgb[y0:y1, x1-2:x1, :] = (255, 0, 0)
            except Exception:
                pass
            self._preview_rgb = rgb
            self._write_roi_raw(frame_bgr, roi, ts)

            # Intensidad para vista previa
            try:
                I_line, _ = self._extractor.process_frame(frame_bgr)  # I crudo corregido (según extractor)
            except Exception:
                I_line = None

            if I_line is not None:
                self._last_I = I_line.astype(np.float32)
                # Acumular blanco si se está promediando
                self._accumulate_blank_if_needed(self._apply_dark(self._last_I))
                # Si hay blanco válido, computa A pero no guarda a buffers (es preview)
                A_line = self._compute_A_from_I(self._last_I)
                if A_line is not None:
                    lam_dyn = self.lambdas if A_line.size == self.lambdas.size else \
                              np.linspace(self.lmin, self.lmax, A_line.size, dtype=np.float32)
                    self._last_A = (A_line, lam_dyn)
                    self._plot_mode = "A"
                else:
                    self._last_A = None
                    self._plot_mode = "I"

        # Fuente
        if source == "screen":
            self._img_source = ScreenReader(roi=roi_cfg, fps=fps, frame_cb=on_frame)
        else:
            cam_index = int(icfg.get("camera_index", -1))  # -1 = auto
            self._img_source = CameraReader(
                cam_index=cam_index,
                backend=icfg.get("backend", "auto"),
                fps=fps,
                frame_cb=on_frame,
                initial_controls=self._cam_controls,
                disable_auto_on_start=True
            )

        # Arranca
        self._img_source.start()
        # Aplica controles manuales vigentes
        self.update_camera_controls(self._cam_controls, disable_auto=True)
        self.preview_only = True

    def start(self) -> None:
        if self.running:
            return
        if self.preview_only:
            # detiene preview-only (para reiniciar limpio)
            self.stop()
        self._stop_evt.clear()
        self.running = True
        if self.mode == "real":
            self._start_real()
        else:
            self._start_sim()

    def stop(self) -> None:
        self._stop_evt.set()
        for th in self._threads:
            th.join(timeout=1.0)
        self._threads.clear()
        try:
            if self._img_source and hasattr(self._img_source, "stop"):
                self._img_source.stop()
        except Exception:
            pass
        try:
            if self._arduino and hasattr(self._arduino, "stop"):
                self._arduino.stop()
        except Exception:
            pass
        self.running = False
        self.preview_only = False

    # ---------- SIM ----------

    def _start_sim(self) -> None:
        def temp_thread():
            t0 = time.time()
            while not self._stop_evt.is_set():
                t = time.time() - t0
                T = 25 + 0.02 * t + 2.0 * np.sin(2*np.pi*t/30.0) + np.random.randn() * 0.05
                self.temp.push(time.time(), float(T))
                time.sleep(0.2)

        def spec_thread():
            t0 = time.time()
            h = int(self.cfg.get("imaging", {}).get("roi", {}).get("h", 86))
            h = max(40, min(h, 240))
            while not self._stop_evt.is_set():
                t = time.time() - t0
                center = 520 + 10 * np.sin(2*np.pi*t/60.0)
                sigma = 12.0
                line = np.exp(-0.5 * ((self.lambdas - center)/sigma)**2)
                A_sel = float(np.interp(self.lambda_sel, self.lambdas, line))
                now = time.time()
                self.spec.push(line.astype(np.float32), self.lambdas)
                self.a_sel.push(now, A_sel)
                # Preview sintético
                img = (np.tile(line, (h, 1)) * 255.0).astype(np.uint8)
                rgb = np.dstack([img, img, img])
                self._preview_rgb = rgb
                time.sleep(1.0)

        for fn in (temp_thread, spec_thread):
            th = threading.Thread(target=fn, daemon=True)
            th.start()
            self._threads.append(th)

    # ---------- REAL ----------

    def _start_real(self) -> None:
        # Lazy imports
        try:
            from .controllers.arduino_protocol import ArduinoProtocol  # type: ignore
        except Exception:
            from spectro_rt.controllers.arduino_protocol import ArduinoProtocol  # type: ignore

        source = self.cfg.get("imaging", {}).get("source", "camera")
        if source == "screen":
            try:
                from .controllers.image_sources.screen_reader import ScreenReader  # type: ignore
            except Exception:
                from spectro_rt.controllers.image_sources.screen_reader import ScreenReader  # type: ignore
        else:
            try:
                from .controllers.image_sources.camera_reader import CameraReader  # type: ignore
            except Exception:
                from spectro_rt.controllers.image_sources.camera_reader import CameraReader  # type: ignore

        try:
            from .processing.spectrum_extractor import SpectrumExtractor  # type: ignore
        except Exception:
            from spectro_rt.processing.spectrum_extractor import SpectrumExtractor  # type: ignore

        # Arduino (tolerante)
        scfg = self.cfg.get("serial", {})
        port = scfg.get("port", "COM3")
        baud = int(scfg.get("baud", 115200))

        def on_line(ts: float, tempC: float, heater: bool) -> None:
            self.temp.push(ts, tempC)

        self._arduino = None
        try:
            self._arduino = ArduinoProtocol(port=port, baud=baud, line_cb=on_line)
            self._arduino.start()
        except Exception as e:
            try:
                messagebox.showwarning(
                    "Arduino no disponible",
                    f"No se pudo abrir {port} @ {baud}.\n\nLa adquisición continuará sin T°.\n\nDetalle: {e}"
                )
            except Exception:
                pass
            print(f"[WARN] Arduino no disponible ({e}). Continuando sin T°.")
            self._arduino = None

        # Imaging config
        icfg = self.cfg.get("imaging", {})
        fps = float(icfg.get("fps", 12))
        roi_cfg = icfg.get("roi", {"x": 0, "y": 0, "w": 500, "h": 86})
        resize_w = int(icfg.get("resize_w", 500))
        gamma = float(icfg.get("gamma", 2.2222))
        bg = float(icfg.get("background_subtract", 300.0))
        min_floor = float(icfg.get("min_floor", 1e-3))
        log10 = bool(icfg.get("log10", True))
        baseline = icfg.get("baseline", {"mode": "ema", "alpha": 0.01})

        try:
            self._extractor = SpectrumExtractor(
                roi=roi_cfg, resize_w=resize_w, gamma=gamma, bg_sub=bg, min_floor=min_floor,
                log10=log10, baseline_mode=baseline.get("mode", "ema"), alpha=baseline.get("alpha", 0.01)
            )
        except TypeError:
            self._extractor = SpectrumExtractor(
                roi=roi_cfg, resize_w=resize_w, gamma=gamma, bg_sub=bg, min_floor=min_floor, log10=log10
            )

        def on_frame(ts: float, frame_bgr: np.ndarray) -> None:
            # ROI ACTUAL + Preview
            roi = getattr(self._extractor, "roi", roi_cfg)
            rgb = frame_bgr[..., ::-1].copy()
            try:
                y, x, h, w = roi.get("y", 0), roi.get("x", 0), roi.get("h", 86), roi.get("w", 500)
                H, W = rgb.shape[:2]
                y0, y1 = max(0, y), min(H - 1, y + h)
                x0, x1 = max(0, x), min(W - 1, x + w)
                rgb[y0:y0+2, x0:x1, :] = (255, 0, 0)
                rgb[y1-2:y1, x0:x1, :] = (255, 0, 0)
                rgb[y0:y1, x0:x0+2, :] = (255, 0, 0)
                rgb[y0:y1, x1-2:x1, :] = (255, 0, 0)
            except Exception:
                pass
            self._preview_rgb = rgb
            self._write_roi_raw(frame_bgr, roi, ts)

            # Línea de intensidad (I)
            try:
                I_line, _ = self._extractor.process_frame(frame_bgr)
            except Exception:
                I_line = None

            if I_line is None:
                return

            self._last_I = I_line.astype(np.float32)
            # Si estamos promediando blanco, acumula
            self._accumulate_blank_if_needed(self._apply_dark(self._last_I))

            # Calcula A si hay blanco; si no, modo I
            A_line = self._compute_A_from_I(self._last_I)
            if A_line is None:
                self._plot_mode = "I"
                return

            self._plot_mode = "A"
            lam_dyn = self.lambdas if A_line.size == self.lambdas.size else \
                      np.linspace(self.lmin, self.lmax, A_line.size, dtype=np.float32)
            self.spec.push(A_line, lam_dyn)
            A_sel = float(np.interp(self.lambda_sel, lam_dyn, A_line))
            self.a_sel.push(ts, A_sel)

        # Fuente
        if source == "screen":
            self._img_source = ScreenReader(roi=roi_cfg, fps=fps, frame_cb=on_frame)
        else:
            cam_index = int(icfg.get("camera_index", -1))  # -1 = auto
            self._img_source = CameraReader(
                cam_index=cam_index,
                backend=icfg.get("backend", "auto"),
                fps=fps,
                frame_cb=on_frame,
                initial_controls=self._cam_controls,
                disable_auto_on_start=True
            )

        # Start imaging (con manejo de error)
        try:
            self._img_source.start()
            # Aplica controles manuales vigentes
            self.update_camera_controls(self._cam_controls, disable_auto=True)
        except Exception as e:
            try:
                messagebox.showerror(
                    "Fuente de imagen",
                    f"No se pudo iniciar la fuente '{source}'.\n\nDetalle: {e}"
                )
            except Exception:
                pass
            self._stop_evt.set()
            self.running = False
            return


# ======================
# GUI
# ======================

class SpectroRTApp:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.mode = cfg.get("mode", "sim")
        meta = self.cfg.get("_meta", {})
        self._config_path = Path(meta.get("user_config_path", "config/config.local.yaml"))
        self.root = tk.Tk()
        self.root.title("spectro-rt v0.3 (all-Python)")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Acquisition
        self.acq = Acquisition(cfg)

        # Graphics config
        gcfg = cfg.get("graphics", {})
        self.refresh_ms = int(gcfg.get("refresh_ms", 200))
        self.window_s = float(gcfg.get("window_s", 120.0))
        self.lambda_sel = float(gcfg.get("lambda_selected_nm", 520))

        # UI state vars
        self._grid_var = tk.IntVar(value=1)
        self._autoscale_var = tk.IntVar(value=1)
        self._source_var = tk.StringVar(value=self.cfg.get("imaging", {}).get("source", "camera"))
        self._preview_var = tk.IntVar(value=1)
        self._tk_preview = None  # ImageTk ref
        self.preview_canvas: Optional[tk.Canvas] = None
        self._preview_image_id: Optional[int] = None
        self._preview_hint_id: Optional[int] = None
        self._preview_native_size: Tuple[int, int] = (0, 0)
        self._preview_disp_size: Tuple[int, int] = (0, 0)
        self._preview_scale: Tuple[float, float] = (1.0, 1.0)
        self._focused_ax: Optional[object] = None
        self._axes: List[object] = []
        self._axes_positions: Dict[object, Bbox] = {}
        self._roi_drag_start: Optional[Tuple[int, int]] = None
        self._roi_drag_rect: Optional[int] = None
        self._canvas_roi_id: Optional[int] = None

        # Build UI
        self._build_menubar()
        self._build_toolbar()
        self._build_preview()   # preview arriba (centrado)
        self._build_plots()
        self._build_measure_panel()
        self._status_vars()

        # Shortcuts
        self._bind_shortcuts()

        # refresco continuo
        self._refresh_job = None
        self._schedule_refresh()

        # arranca preview-only si modo real
        if self.mode == "real":
            try:
                self.acq.start_preview()
                self.var_status.set("Preview activo (fuente de imagen) — aún sin adquisición")
            except Exception as e:
                try:
                    messagebox.showwarning(
                        "Cámara/Pantalla no disponible",
                        f"No se pudo iniciar el preview.\n\nDetalle: {e}"
                    )
                except Exception:
                    pass

    # ---------- Menubar ----------

    def _build_menubar(self) -> None:
        m = tk.Menu(self.root)

        # Archivo
        m_file = tk.Menu(m, tearoff=False)
        m_file.add_command(label="Nuevo run", command=self._on_new_session)
        m_file.add_separator()
        m_file.add_command(label="Exportar CSV…", accelerator="Ctrl+S", command=self._on_export_csv)
        m_file.add_command(label="Exportar NPY…", accelerator="Ctrl+E", command=self._on_export_npy)
        m_file.add_command(label="Exportar Figura…", accelerator="Ctrl+G", command=self._on_export_figure)
        m_file.add_separator()
        m_file.add_command(label="Salir", accelerator="Ctrl+Q", command=self._on_quit)
        m.add_cascade(label="Archivo", menu=m_file)

        # Adquisición
        m_acq = tk.Menu(m, tearoff=False)
        m_acq.add_command(label="Iniciar", accelerator="Ctrl+R", command=self._on_start)
        m_acq.add_command(label="Detener", accelerator="Ctrl+Shift+R", command=self._on_stop)

        # Fuente de imagen
        m_src = tk.Menu(m_acq, tearoff=False)
        m_src.add_radiobutton(label="Cámara", value="camera", variable=self._source_var, command=self._on_change_source)
        m_src.add_radiobutton(label="Pantalla", value="screen", variable=self._source_var, command=self._on_change_source)
        m_acq.add_cascade(label="Fuente de imagen", menu=m_src)

        m_acq.add_command(label="Configurar puerto serie…", command=self._show_serial_dialog)

        # Cámara (controles manuales)
        m_cam = tk.Menu(m_acq, tearoff=False)
        m_cam.add_command(label="Controles de cámara…", command=self._show_camera_controls)
        m_acq.add_cascade(label="Cámara", menu=m_cam)

        m.add_cascade(label="Adquisición", menu=m_acq)

        # Vista
        m_view = tk.Menu(m, tearoff=False)
        m_view.add_command(label="Cambiar λ seleccionada…", command=self._prompt_lambda_change)
        m_view.add_command(label="Cambiar ventana (s)…", command=self._prompt_window_change)
        m_view.add_checkbutton(label="Grilla", variable=self._grid_var, command=self._toggle_grid)
        m_view.add_checkbutton(label="Autoescala", variable=self._autoscale_var, command=self._on_toggle_autoscale)
        m_view.add_checkbutton(label="Mostrar preview", variable=self._preview_var, command=self._toggle_preview)
        m_view.add_command(label="Reset zoom", command=self._reset_view)
        m.add_cascade(label="Vista", menu=m_view)

        # Herramientas
        m_tools = tk.Menu(m, tearoff=False)
        m_tools.add_command(label="Editar ROI…", command=self._prompt_roi_dialog)
        m_tools.add_separator()
        m_tools.add_command(label="Capturar oscuro (0% ADJ)", command=self._capture_dark)
        m_tools.add_command(label="Limpiar oscuro", command=self._clear_dark)
        m_tools.add_separator()
        m_tools.add_command(label="Capturar blanco (100% ADJ)", command=self._capture_blank_single)
        m_tools.add_command(label="Promediar blanco…", command=self._avg_blank_dialog)
        m_tools.add_command(label="Borrar blanco", command=self._clear_blank)
        m_tools.add_separator()
        m_tools.add_command(label="Preferencias de imagen…", command=self._show_preferences_dialog)
        m.add_cascade(label="Herramientas", menu=m_tools)

        # Ayuda
        m_help = tk.Menu(m, tearoff=False)
        m_help.add_command(label="Acerca de…", command=self._show_about)
        m.add_cascade(label="Ayuda", menu=m_help)

        self.root.config(menu=m)

    # ---------- Toolbar ----------

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)

        self.btn_start = ttk.Button(bar, text="Iniciar", command=self._on_start)
        self.btn_stop = ttk.Button(bar, text="Detener", command=self._on_stop, state=tk.DISABLED)
        self.btn_csv = ttk.Button(bar, text="Exportar CSV", command=self._on_export_csv)
        self.btn_npy = ttk.Button(bar, text="Exportar NPY", command=self._on_export_npy)
        self.btn_fig = ttk.Button(bar, text="Exportar Grafica", command=self._on_export_figure)
        self.chk_autoscale = ttk.Checkbutton(bar, text="Autoescala", variable=self._autoscale_var,
                                             command=self._on_toggle_autoscale)
        self.btn_focus_dec = ttk.Button(bar, text="<<", width=4, command=lambda: self._adjust_focus(-5.0))
        self.btn_focus_inc = ttk.Button(bar, text=">>", width=4, command=lambda: self._adjust_focus(5.0))

        self.win_var = tk.StringVar(value=str(int(self.window_s)))
        ttk.Label(bar, text="Ventana (s)").pack(side=tk.RIGHT)
        ttk.Entry(bar, textvariable=self.win_var, width=6).pack(side=tk.RIGHT, padx=(4, 12))

        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop.pack(side=tk.LEFT, padx=(6, 12))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self.btn_csv.pack(side=tk.LEFT)
        self.btn_npy.pack(side=tk.LEFT, padx=(6, 0))
        self.btn_fig.pack(side=tk.LEFT, padx=(6, 0))
        self.chk_autoscale.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(bar, text="Focus").pack(side=tk.LEFT)
        self.btn_focus_dec.pack(side=tk.LEFT, padx=(4, 0))
        self.btn_focus_inc.pack(side=tk.LEFT, padx=(2, 0))
        if self.mode != "real":
            self.btn_focus_dec.state(["disabled"])
            self.btn_focus_inc.state(["disabled"])

    # ---------- Preview ----------

    def _build_preview(self) -> None:
        self.preview_frame = ttk.Frame(self.root, padding=(8, 0))
        self.preview_frame.pack(side=tk.TOP, fill=tk.X)
        self.preview_canvas = tk.Canvas(self.preview_frame, highlightthickness=0, background="black", height=220)
        self.preview_canvas.pack(side=tk.TOP, anchor="center", pady=(2, 4))
        if not ImageTk:
            self._preview_hint_id = self.preview_canvas.create_text(
                10, 10, anchor="nw", fill="#cccccc",
                text="Instala Pillow (pip install Pillow) para ver el preview")
        else:
            self._preview_hint_id = self.preview_canvas.create_text(
                10, 10, anchor="nw", fill="#cccccc",
                text="Arrastra con el mouse para redefinir el ROI")
        self.preview_canvas.bind("<ButtonPress-1>", self._on_preview_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_preview_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_preview_release)

    def _toggle_preview(self) -> None:
        if self._preview_var.get():
            self.preview_frame.pack(side=tk.TOP, fill=tk.X)
        else:
            self.preview_frame.forget()

    def _update_preview(self) -> None:
        if not self._preview_var.get() or not ImageTk or self.preview_canvas is None:
            return
        frame = self.acq.latest_preview()
        if frame is None:
            return
        try:
            img = Image.fromarray(frame)  # RGB
            orig_w, orig_h = img.size
            if orig_w <= 0 or orig_h <= 0:
                return
            target_w = max(320, self.root.winfo_width() - 40)
            ratio = target_w / max(1, orig_w)
            target_h = int(orig_h * ratio)
            max_h = 220
            if target_h > max_h:
                ratio = max_h / max(1, orig_h)
                target_h = max_h
                target_w = int(orig_w * ratio)
            target_w = max(1, target_w)
            target_h = max(1, target_h)
            img_resized = img.resize((target_w, target_h), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img_resized)
            self._tk_preview = photo
            self.preview_canvas.config(width=target_w, height=target_h)
            if self._preview_image_id is None:
                self._preview_image_id = self.preview_canvas.create_image(0, 0, image=photo, anchor="nw")
            else:
                self.preview_canvas.itemconfig(self._preview_image_id, image=photo)
            if self._preview_hint_id is not None:
                self.preview_canvas.delete(self._preview_hint_id)
                self._preview_hint_id = None
            self._preview_native_size = (orig_w, orig_h)
            self._preview_disp_size = (target_w, target_h)
            self._preview_scale = (
                target_w / max(1, orig_w),
                target_h / max(1, orig_h),
            )
            self._update_canvas_roi_overlay()
        except Exception:
            pass

    def _clamp_preview_point(self, x: int, y: int) -> Tuple[int, int]:
        w, h = self._preview_disp_size
        if w <= 0 or h <= 0:
            return 0, 0
        return max(0, min(int(x), w - 1)), max(0, min(int(y), h - 1))

    def _update_canvas_roi_overlay(self) -> None:
        if self.preview_canvas is None or self._preview_image_id is None:
            return
        roi = self.cfg.get("imaging", {}).get("roi")
        if not roi:
            if self._canvas_roi_id is not None:
                self.preview_canvas.delete(self._canvas_roi_id)
                self._canvas_roi_id = None
            return
        if self._preview_disp_size == (0, 0) or self._preview_native_size == (0, 0):
            return
        sx, sy = self._preview_scale
        x0 = roi.get("x", 0) * sx
        y0 = roi.get("y", 0) * sy
        x1 = (roi.get("x", 0) + roi.get("w", 0)) * sx
        y1 = (roi.get("y", 0) + roi.get("h", 0)) * sy
        x0, y0 = self._clamp_preview_point(x0, y0)
        x1, y1 = self._clamp_preview_point(x1, y1)
        if self._canvas_roi_id is None:
            self._canvas_roi_id = self.preview_canvas.create_rectangle(x0, y0, x1, y1, outline="#ff4444", width=2)
        else:
            self.preview_canvas.coords(self._canvas_roi_id, x0, y0, x1, y1)
        if self._roi_drag_rect is not None:
            self.preview_canvas.tag_raise(self._roi_drag_rect)

    def _on_preview_press(self, event: tk.Event) -> None:
        if self.preview_canvas is None:
            return
        self._roi_drag_start = self._clamp_preview_point(event.x, event.y)
        if self._roi_drag_rect is not None:
            self.preview_canvas.delete(self._roi_drag_rect)
            self._roi_drag_rect = None

    def _on_preview_drag(self, event: tk.Event) -> None:
        if self.preview_canvas is None or self._roi_drag_start is None:
            return
        x0, y0 = self._roi_drag_start
        x1, y1 = self._clamp_preview_point(event.x, event.y)
        if self._roi_drag_rect is None:
            self._roi_drag_rect = self.preview_canvas.create_rectangle(
                x0, y0, x1, y1, outline="#00ff99", width=2, dash=(6, 3))
        else:
            self.preview_canvas.coords(self._roi_drag_rect, x0, y0, x1, y1)

    def _on_preview_release(self, event: tk.Event) -> None:
        if self.preview_canvas is None or self._roi_drag_start is None:
            return
        x0, y0 = self._roi_drag_start
        x1, y1 = self._clamp_preview_point(event.x, event.y)
        if self._roi_drag_rect is not None:
            self.preview_canvas.delete(self._roi_drag_rect)
            self._roi_drag_rect = None
        self._roi_drag_start = None
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return
        self._apply_roi_from_canvas(x0, y0, x1, y1)

    def _apply_roi_from_canvas(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self._preview_disp_size == (0, 0) or self._preview_native_size == (0, 0):
            return
        inv_sx = self._preview_native_size[0] / max(1, self._preview_disp_size[0])
        inv_sy = self._preview_native_size[1] / max(1, self._preview_disp_size[1])
        left = int(round(min(x0, x1) * inv_sx))
        top = int(round(min(y0, y1) * inv_sy))
        right = int(round(max(x0, x1) * inv_sx))
        bottom = int(round(max(y0, y1) * inv_sy))
        native_w, native_h = self._preview_native_size
        left = max(0, min(left, native_w - 1))
        top = max(0, min(top, native_h - 1))
        right = max(left + 1, min(right, native_w))
        bottom = max(top + 1, min(bottom, native_h))
        new_roi = {"x": left, "y": top, "w": right - left, "h": bottom - top}
        self._apply_roi(new_roi)

    def _apply_roi(self, new_roi: Dict[str, int], persist: bool = True) -> None:
        roi = {k: max(0, int(new_roi.get(k, 0))) for k in ("x", "y", "w", "h")}
        roi["w"] = max(1, roi["w"])
        roi["h"] = max(1, roi["h"])
        self.cfg.setdefault("imaging", {})["roi"] = roi
        try:
            if getattr(self.acq, "_extractor", None) is not None:
                self.acq._extractor.roi = roi  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if hasattr(getattr(self.acq, "_img_source", None), "update_roi"):
                self.acq._img_source.update_roi(roi)  # type: ignore[attr-defined]
        except Exception:
            pass
        self.var_status.set(f"ROI actualizado: {roi}")
        self._update_canvas_roi_overlay()
        if persist:
            self._save_config()

    def _save_config(self) -> None:
        data = {k: v for k, v in self.cfg.items() if k != "_meta"}
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with self._config_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
            self.cfg.setdefault("_meta", {})["user_config_path"] = str(self._config_path)
        except Exception as exc:
            print(f"[WARN] No se pudo guardar config: {exc}")

    # ---------- Plots ----------

    def _build_plots(self) -> None:
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.ax_T = self.fig.add_subplot(4, 1, 1)
        self.ax_A = self.fig.add_subplot(4, 1, 2)
        self.ax_S = self.fig.add_subplot(4, 1, 3)
        self.ax_SA = self.fig.add_subplot(4, 1, 4)

        self.ax_T.set_ylabel('T (°C)')
        self.ax_A.set_ylabel('A(λ_sel)')
        self.ax_S.set_ylabel('I(λ)')
        self.ax_SA.set_ylabel('A(λ)')
        self.ax_SA.set_xlabel('λ (nm)')

        for ax in (self.ax_T, self.ax_A, self.ax_S, self.ax_SA):
            ax.grid(True, linestyle=':', linewidth=0.6)

        self.l_T, = self.ax_T.plot([], [], lw=1.5)
        self.l_A, = self.ax_A.plot([], [], lw=1.5)
        self.l_S, = self.ax_S.plot([], [], lw=1.5)
        self.l_SA, = self.ax_SA.plot([], [], lw=1.5, color='#ff7f0e')
        self.cursor_S = self.ax_S.axvline(self.lambda_sel, ls='--')
        self.cursor_SA = self.ax_SA.axvline(self.lambda_sel, ls='--', color='#ff7f0e')

        # Fija los límites iniciales de λ según la configuración
        self.ax_S.set_xlim(self.acq.lmin, self.acq.lmax)
        self.ax_SA.set_xlim(self.acq.lmin, self.acq.lmax)
        self._axes = [self.ax_T, self.ax_A, self.ax_S, self.ax_SA]
        self._axes_positions = {ax: ax.get_position().frozen() for ax in self._axes}

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    # ---------- Panel de medición ----------

    def _build_measure_panel(self) -> None:
        pane = ttk.Frame(self.root, padding=(8, 4))
        pane.pack(side=tk.TOP, fill=tk.X)
        # λ
        ttk.Label(pane, text="λ_sel (nm):").pack(side=tk.LEFT)
        self.lam_var = tk.StringVar(value=f"{self.lambda_sel:.1f}")
        lam_entry = ttk.Entry(pane, textvariable=self.lam_var, width=7)
        lam_entry.pack(side=tk.LEFT, padx=(4, 10))

        # A / %T / C
        ttk.Label(pane, text="A=").pack(side=tk.LEFT)
        self.A_var = tk.StringVar(value='--')
        ttk.Label(pane, textvariable=self.A_var, width=8).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(pane, text="%T=").pack(side=tk.LEFT)
        self.T_var = tk.StringVar(value='--')
        ttk.Label(pane, textvariable=self.T_var, width=10).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(pane, text="ε·L=").pack(side=tk.LEFT)
        self.k_var = tk.StringVar(value="1.0")
        ttk.Entry(pane, textvariable=self.k_var, width=8).pack(side=tk.LEFT, padx=(4, 6))

        ttk.Label(pane, text="C=").pack(side=tk.LEFT)
        self.C_var = tk.StringVar(value='--')
        ttk.Label(pane, textvariable=self.C_var, width=12).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Button(pane, text="Aplicar λ", command=self._apply_lambda_from_panel).pack(side=tk.LEFT, padx=(12, 0))

    # ---------- Status ----------

    def _status_vars(self) -> None:
        stat = ttk.Frame(self.root, padding=(8, 4))
        stat.pack(side=tk.BOTTOM, fill=tk.X)
        self.var_status = tk.StringVar(value="Listo")
        self.var_last = tk.StringVar(value='T=-- °C   A=--')
        ttk.Label(stat, textvariable=self.var_status).pack(side=tk.LEFT)
        ttk.Label(stat, textvariable=self.var_last).pack(side=tk.RIGHT)

    # ---------- Shortcuts ----------

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-r>", lambda e: self._on_start())
        self.root.bind("<Control-R>", lambda e: self._on_start())
        self.root.bind("<Control-Shift-r>", lambda e: self._on_stop())
        self.root.bind("<Control-Shift-R>", lambda e: self._on_stop())
        self.root.bind("<Control-s>", lambda e: self._on_export_csv())
        self.root.bind("<Control-e>", lambda e: self._on_export_npy())
        self.root.bind("<Control-g>", lambda e: self._on_export_figure())
        self.root.bind("<Control-q>", lambda e: self._on_quit())

    # ---------- Events ----------

    def _on_start(self) -> None:
        try:
            self.window_s = float(self.win_var.get())
        except Exception:
            self.window_s = 120.0
            self.win_var.set("120")
        self.acq.start()
        if self.acq.mode == "real" and self.acq._arduino is None:
            self.var_status.set("Adquisición en marcha (sin Arduino: T° no disponible)")
        else:
            self.var_status.set(f"Adquisición en marcha (modo: {self.acq.mode})")
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)

    def _on_stop(self) -> None:
        self.acq.stop()
        self.var_status.set("Detenido. Puedes reiniciar la adquisición cuando desees.")
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def _on_new_session(self) -> None:
        self.acq = Acquisition(self.cfg)
        self.var_status.set("Sesión reiniciada")
        if self.mode == "real":
            # relanza preview
            try:
                self.acq.start_preview()
            except Exception:
                pass

    def _on_export_csv(self) -> None:
        ts = int(time.time())
        init_dir = Path(self.cfg.get("paths", {}).get("export_dir", "data/exports"))
        init_dir.mkdir(parents=True, exist_ok=True)
        f = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialdir=str(init_dir),
            initialfile=f"run_{ts}",
            title="Guardar CSV base",
        )
        if not f:
            return
        try:
            self.acq.export_csv(Path(f))
            self.var_status.set("CSV exportado")
        except Exception as e:
            messagebox.showerror("Exportar CSV", str(e))

    def _on_export_npy(self) -> None:
        ts = int(time.time())
        init_dir = Path(self.cfg.get("paths", {}).get("export_dir", "data/exports"))
        init_dir.mkdir(parents=True, exist_ok=True)
        f = filedialog.asksaveasfilename(
            defaultextension=".npy",
            initialdir=str(init_dir),
            initialfile=f"run_{ts}",
            title="Guardar base NPY (se crearán 3 archivos)",
        )
        if not f:
            return
        try:
            self.acq.export_npy(Path(f))
            self.var_status.set("NPY exportado")
        except Exception as e:
            messagebox.showerror("Exportar NPY", str(e))

    def _on_export_figure(self) -> None:
        f = filedialog.asksaveasfilename(defaultextension=".png", title="Guardar figura")
        if not f:
            return
        try:
            self.fig.savefig(f, bbox_inches="tight", dpi=200)
            self.var_status.set("Figura exportada")
        except Exception as e:
            messagebox.showerror("Exportar figura", str(e))

    def _on_change_source(self) -> None:
        src = self._source_var.get()
        self.cfg.setdefault("imaging", {})["source"] = src
        self._save_config()
        messagebox.showinfo("Fuente de imagen", "La fuente cambiará al reiniciar la adquisición.")

    def _prompt_lambda_change(self) -> None:
        val = simpledialog.askfloat('λ seleccionada', 'Nueva λ (nm):',
                                    initialvalue=self.lambda_sel, minvalue=200, maxvalue=1100)
        if val is None:
            return
        self.lambda_sel = float(val)
        self.cfg.setdefault('graphics', {})['lambda_selected_nm'] = self.lambda_sel
        self.cursor_S.set_xdata([self.lambda_sel])
        if hasattr(self, 'cursor_SA'):
            self.cursor_SA.set_xdata([self.lambda_sel])
        self.lam_var.set(f'{self.lambda_sel:.1f}')
        self.var_status.set(f'λ_sel = {self.lambda_sel:.1f} nm')
        self._save_config()
    def _prompt_window_change(self) -> None:
        val = simpledialog.askfloat('Ventana (s)', 'Segundos visibles:',
                                    initialvalue=self.window_s, minvalue=5, maxvalue=3600)
        if val is None:
            return
        self.window_s = float(val)
        self.win_var.set(str(int(self.window_s)))
        self.cfg.setdefault('graphics', {})['window_s'] = self.window_s
        self._save_config()
    # ===== ROI EDITABLE =====
    def _prompt_roi_dialog(self) -> None:
        """Diálogo para editar el ROI y aplicarlo en vivo (preview, extractor y screen-capture)."""
        roi = self.cfg.get("imaging", {}).get("roi", {"x": 0, "y": 0, "w": 500, "h": 86})
        top = tk.Toplevel(self.root)
        top.title("Editar ROI")
        top.resizable(False, False)
        frm = ttk.Frame(top, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        vars_ = {k: tk.IntVar(value=int(roi.get(k, 0))) for k in ("x", "y", "w", "h")}
        for i, k in enumerate(["x", "y", "w", "h"]):
            ttk.Label(frm, text=k.upper()).grid(row=i, column=0, sticky="e", padx=6, pady=4)
            ttk.Entry(frm, textvariable=vars_[k], width=10).grid(row=i, column=1, sticky="w", padx=6, pady=4)

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, pady=(10, 0))

        def apply_and_close():
            new_roi = {k: int(vars_[k].get()) for k in ("x", "y", "w", "h")}
            self._apply_roi(new_roi)
            top.destroy()

        ttk.Button(btns, text="Aceptar", command=apply_and_close).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Cancelar", command=top.destroy).pack(side=tk.LEFT)

    # ===== Blanco/Oscuro =====

    def _capture_dark(self) -> None:
        self.acq.set_dark()
        self.var_status.set("Oscuro (0%) capturado")

    def _clear_dark(self) -> None:
        self.acq.clear_dark()
        self.var_status.set("Oscuro eliminado")

    def _capture_blank_single(self) -> None:
        self.acq.start_blank(avg_n=1)
        self.acq.stop_blank()
        self.var_status.set("Blanco (100%) capturado")

    def _avg_blank_dialog(self) -> None:
        n = simpledialog.askinteger("Promediar blanco", "Nº de frames a promediar:", initialvalue=30, minvalue=1, maxvalue=10000)
        if n is None:
            return
        # guardamos el total para etiqueta correcta al finalizar
        self.cfg.setdefault("imaging", {})["blank_avg_total"] = n
        self.acq.start_blank(avg_n=n)
        messagebox.showinfo("Promedio de blanco", f"Promediando {n} frames… se fijará automáticamente al completar.")

    def _clear_blank(self) -> None:
        self.acq.clear_blank()
        self.var_status.set("Blanco eliminado")

    # ===== Cámara: controles manuales =====

    def _show_camera_controls(self) -> None:
        icfg = self.cfg.setdefault("imaging", {})
        C = dict(icfg.get("camera_controls", {}))

        top = tk.Toplevel(self.root); top.title("Controles de cámara (manual)"); top.resizable(False, False)
        frm = ttk.Frame(top, padding=12); frm.pack(fill=tk.BOTH, expand=True)

        fields = [
            ("exposure", "Exposición"),
            ("gain", "Ganancia"),
            ("brightness", "Brillo"),
            ("contrast", "Contraste"),
            ("saturation", "Saturación"),
            ("wb_temp", "WB Temp (K)"),
            ("focus", "Enfoque"),
            ("sharpness", "Nitidez"),
            ("gamma", "Gamma (HW)"),
        ]
        vars_: Dict[str, tk.StringVar] = {}
        for i, (k, label) in enumerate(fields):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="e", padx=6, pady=3)
            v = tk.StringVar(value=str(C.get(k, "")))
            vars_[k] = v
            ttk.Entry(frm, textvariable=v, width=12).grid(row=i, column=1, sticky="w", padx=6, pady=3)

        def apply_controls():
            out = {}
            for k, v in vars_.items():
                s = v.get().strip()
                out[k] = None if s == "" else float(s)
            icfg["camera_controls"] = out
            self.acq.update_camera_controls(out, disable_auto=True)
            self._save_config()
            self.var_status.set("Controles de cámara aplicados (auto OFF)")
            top.destroy()

        ttk.Button(frm, text="Aplicar", command=apply_controls).grid(row=len(fields)+1, column=0, columnspan=2, pady=(8, 0))

    # ===== Preferencias imagen (pre-proc extractor) =====

    def _show_preferences_dialog(self) -> None:
        icfg = self.cfg.get("imaging", {})
        top = tk.Toplevel(self.root)
        top.title("Preferencias de imagen")
        top.resizable(False, False)
        frm = ttk.Frame(top, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)
        gamma_var = tk.DoubleVar(value=float(icfg.get("gamma", 2.2222)))
        bg_var = tk.DoubleVar(value=float(icfg.get("background_subtract", 300.0)))
        min_var = tk.DoubleVar(value=float(icfg.get("min_floor", 1e-3)))
        log_var = tk.IntVar(value=1 if icfg.get("log10", True) else 0)

        ttk.Label(frm, text="Gamma").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm, textvariable=gamma_var, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm, text="Resta fondo").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm, textvariable=bg_var, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm, text="Min floor").grid(row=2, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm, textvariable=min_var, width=10).grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(frm, text="Log10", variable=log_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        def apply_and_close():
            icfg["gamma"] = float(gamma_var.get())
            icfg["background_subtract"] = float(bg_var.get())
            icfg["min_floor"] = float(min_var.get())
            icfg["log10"] = bool(log_var.get())
            try:
                if self.acq._extractor is not None:
                    self.acq._extractor.gamma = icfg["gamma"]
                    self.acq._extractor.bg_sub = icfg["background_subtract"]
                    self.acq._extractor.min_floor = icfg["min_floor"]
                    self.acq._extractor.log10 = icfg["log10"]
            except Exception:
                pass
            self.var_status.set("Preferencias actualizadas")
            top.destroy()

        ttk.Button(frm, text="Guardar", command=apply_and_close).grid(row=10, column=0, columnspan=2, pady=(10, 0))

    # ===== Puerto serie (NUEVO: faltaba y causaba el crash) =====

    def _show_serial_dialog(self) -> None:
        scfg = self.cfg.setdefault("serial", {})
        top = tk.Toplevel(self.root)
        top.title("Puerto serie (Arduino)")
        top.resizable(False, False)
        frm = ttk.Frame(top, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        # Intentar listar puertos si pyserial.tools está disponible
        ports = []
        try:
            from serial.tools import list_ports  # type: ignore
            ports = [p.device for p in list_ports.comports()]
        except Exception:
            ports = []

        ttk.Label(frm, text="Port").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        port_var = tk.StringVar(value=str(scfg.get("port", "COM3")))
        if ports:
            cb = ttk.Combobox(frm, values=ports, textvariable=port_var, width=16, state="readonly")
            cb.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        else:
            ttk.Entry(frm, textvariable=port_var, width=16).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(frm, text="Baud").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        baud_var = tk.IntVar(value=int(scfg.get("baud", 115200)))
        ttk.Entry(frm, textvariable=baud_var, width=16).grid(row=1, column=1, sticky="w", padx=6, pady=4)

        tip = ttk.Label(frm, text="Los cambios aplican al reiniciar la adquisición.", foreground="#555")
        tip.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 10))

        def apply_and_close():
            self.cfg.setdefault("serial", {})["port"] = port_var.get()
            self.cfg["serial"]["baud"] = int(baud_var.get())
            self._save_config()
            messagebox.showinfo("Puerto serie", f"Guardado: {port_var.get()} @ {baud_var.get()}.\n"
                                                "Se aplicará al reiniciar la adquisición.")
            top.destroy()

        btns = ttk.Frame(frm); btns.grid(row=3, column=0, columnspan=2, pady=(4, 0))
        ttk.Button(btns, text="Aceptar", command=apply_and_close).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Cancelar", command=top.destroy).pack(side=tk.LEFT, padx=6)

    # ---------- Helpers ----------

    def _apply_lambda_from_panel(self) -> None:
        try:
            val = float(self.lam_var.get())
        except Exception:
            messagebox.showerror("λ", "Valor no válido.")
            return
        self.lambda_sel = val
        self.cfg.setdefault("graphics", {})["lambda_selected_nm"] = self.lambda_sel
        self.cursor_S.set_xdata([self.lambda_sel])
        if hasattr(self, 'cursor_SA'):
            self.cursor_SA.set_xdata([self.lambda_sel])
        self.var_status.set(f"λ_sel = {self.lambda_sel:.1f} nm")
        self._save_config()

    def _toggle_grid(self) -> None:
        on = bool(self._grid_var.get())
        for ax in (self.ax_T, self.ax_A, self.ax_S, getattr(self, "ax_SA", None)):
            if ax is None:
                continue
            ax.grid(on, linestyle=":", linewidth=0.6)
        self.canvas.draw_idle()

    def _on_toggle_autoscale(self) -> None:
        if self._autoscale_var.get():
            self._reset_view()
            self.var_status.set("Autoescala activada")
        else:
            self.var_status.set("Autoescala desactivada (mantén zoom actual)")

    def _adjust_focus(self, delta: float) -> None:
        if not hasattr(self.acq, "adjust_focus"):
            self.var_status.set("Control de enfoque no disponible")
            return
        try:
            new_val = self.acq.adjust_focus(delta)
        except Exception as exc:
            self.var_status.set(f"No se pudo ajustar el enfoque: {exc}")
            return
        self.cfg.setdefault("imaging", {}).setdefault("camera_controls", {})["focus"] = new_val
        self._save_config()
        self.var_status.set(f"Focus manual: {new_val:.2f}")

    def _reset_view(self) -> None:
        for ax in (self.ax_T, self.ax_A, self.ax_S, getattr(self, "ax_SA", None)):
            if ax is None:
                continue
            ax.relim()
            ax.autoscale_view()
        # Asegura el rango completo de λ en los espectros
        self.ax_S.set_xlim(self.acq.lmin, self.acq.lmax)
        self.ax_SA.set_xlim(self.acq.lmin, self.acq.lmax)
        self.canvas.draw_idle()

    def _schedule_refresh(self) -> None:
        self._refresh()
        self._refresh_job = self.root.after(self.refresh_ms, self._schedule_refresh)

    def _update_blank_status(self) -> None:
        has_blank = self.acq._blank is not None
        if self.acq.get_plot_mode() == 'I' and not has_blank:
            self.var_status.set('Modo I(λ): captura oscuro/blanco para habilitar A(λ)')
    def _refresh(self) -> None:
        # preview primero
        self._update_preview()

        # series T y A_sel
        tT, TT = self.acq.temp.window(self.window_s)
        tA, AA = self.acq.a_sel.window(self.window_s)

        if tT.size:
            tT = tT - tT[0]
        if tA.size:
            tA = tA - tA[0]

        autoscale = bool(self._autoscale_var.get())

        self.l_T.set_data(tT, TT)
        if autoscale and self.ax_T.get_visible():
            self.ax_T.relim()
            self.ax_T.autoscale_view()

        self.l_A.set_data(tA, AA)
        if autoscale and self.ax_A.get_visible():
            self.ax_A.relim()
            self.ax_A.autoscale_view()

        # Espectros en vivo
        I_line, lam_I = self.acq.latest_intensity()
        if I_line is not None:
            lam_plot_I = lam_I if lam_I.size == I_line.size else np.linspace(self.acq.lmin, self.acq.lmax, I_line.size)
            self.l_S.set_data(lam_plot_I, I_line)
            self.cursor_S.set_xdata([self.lambda_sel])
            if autoscale and self.ax_S.get_visible():
                self.ax_S.relim()
                self.ax_S.autoscale_view()
            self.ax_S.set_xlim(self.acq.lmin, self.acq.lmax)
        else:
            self.l_S.set_data([], [])
            self.ax_S.set_xlim(self.acq.lmin, self.acq.lmax)

        spec, lam_A = self.acq.latest_spectrum()
        Tlast = self.acq.latest_temp()
        if spec is not None:
            lam_plot_A = lam_A if lam_A.size == spec.size else np.linspace(self.acq.lmin, self.acq.lmax, spec.size)
            self.l_SA.set_data(lam_plot_A, spec)
            if hasattr(self, 'cursor_SA'):
                self.cursor_SA.set_xdata([self.lambda_sel])
            if autoscale and self.ax_SA.get_visible():
                self.ax_SA.relim()
                self.ax_SA.autoscale_view()
            self.ax_SA.set_xlim(self.acq.lmin, self.acq.lmax)
            A_sel = float(np.interp(self.lambda_sel, lam_plot_A, spec))
            self.A_var.set(f'{A_sel:.4f}')
            self.T_var.set(f'{100.0*(10**(-A_sel)):.2f} %')
            try:
                k = float(self.k_var.get())
            except Exception:
                k = 0.0
            self.C_var.set(f'{(A_sel/k):.6g}' if k > 0 else '--')
            if Tlast is not None:
                self.var_last.set(f'T={Tlast:.2f} °C   A={A_sel:.4f}')
            else:
                self.var_last.set(f'T=-- °C   A={A_sel:.4f}')
        else:
            self.l_SA.set_data([], [])
            self.A_var.set('--'); self.T_var.set('--'); self.C_var.set('--')
            if Tlast is not None:
                self.var_last.set(f'T={Tlast:.2f} °C   A=--')
            else:
                self.var_last.set('T=-- °C   A=--')
            self.ax_SA.set_xlim(self.acq.lmin, self.acq.lmax)
        self._update_blank_status()
        self.canvas.draw_idle()

    def _set_focus(self, ax: Optional[object]) -> None:
        target = ax
        if target is not None and target is self._focused_ax:
            target = None

        if target is None:
            if self._focused_ax is not None:
                for ax_i, pos in self._axes_positions.items():
                    ax_i.set_visible(True)
                    ax_i.set_position(pos)
                if self._autoscale_var.get():
                    self._reset_view()
                self.canvas.draw_idle()
                self.var_status.set("Vista completa")
            self._focused_ax = None
            return

        if target not in self._axes_positions:
            return

        self._focused_ax = target
        focus_box = Bbox.from_bounds(0.12, 0.08, 0.82, 0.84)
        for ax_i, pos in self._axes_positions.items():
            if ax_i is target:
                ax_i.set_visible(True)
                ax_i.set_position(focus_box)
            else:
                ax_i.set_visible(False)
        if self._autoscale_var.get():
            self._reset_view()
        label = getattr(target, "get_ylabel", lambda: "")()
        if not label:
            label = getattr(target, "get_title", lambda: "")()
        if not label:
            label = "gráfico"
        self.var_status.set(f"Vista enfocada: {label}")
        self.canvas.draw_idle()

    def _on_plot_click(self, event) -> None:
        axes_tuple = tuple(self._axes) if self._axes else (self.ax_T, self.ax_A, self.ax_S, self.ax_SA)
        if event.inaxes not in axes_tuple:
            return
        if getattr(event, "button", None) == 3:
            self._set_focus(None)
            return
        self._set_focus(event.inaxes)

    def _show_about(self) -> None:
        msg = (
            "spectro-rt v0.3 (all-Python)\n"
            "Preview activo antes de iniciar + tolerante a hardware.\n\n"
            "Atajos:\n"
            "  Ctrl+R: Iniciar    |  Ctrl+Shift+R: Detener\n"
            "  Ctrl+S: Exportar CSV   Ctrl+E: NPY\n"
            "  Ctrl+G: Exportar Figura   Ctrl+Q: Salir\n\n"
            "Menús:\n"
            "  Archivo, Adquisición, Vista, Herramientas y Ayuda.\n\n"
            "Datos en memoria (sin guardar 50k imágenes)."
        )
        messagebox.showinfo("Acerca de spectro-rt", msg)

    def _on_quit(self) -> None:
        self._on_close()

    def _on_close(self) -> None:
        try:
            self.acq.stop()
        finally:
            self.root.destroy()

    # ---------- Public ----------

    def run(self) -> None:
        self.root.mainloop()



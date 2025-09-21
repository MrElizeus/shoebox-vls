# spectro_rt/gui.py
# -*- coding: utf-8 -*-
"""
spectro-rt v0.4.2
- Preview en vivo con bucle propio (15 fps por defecto) + ref. fuerte de PhotoImage
- Layout: preview arriba, plots abajo
- ROI editable; pop-ups si falla hardware
- Menú "Blanco" (single / mean N / autoN, dominio log/linear, import/export/clear)
- Panel de medición: λ (clic o entrada), A, %T, factor k, C=A/k
- Botones 0% ADJ (dark) y 100% ADJ (blank)
- FIX: tolerante a desajustes de tamaño entre λ y A(λ) (recalcula eje λ si hace falta)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

# Tkinter / Matplotlib embed
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Preview rendering
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

EPS = 1e-6


# ======================
# Buffers
# ======================
@dataclass
class RingBuffer:
    maxlen: int
    t: Deque[float] = field(default_factory=deque)
    y: Deque[float] = field(default_factory=deque)

    def push(self, ts: float, val: float) -> None:
        if len(self.t) >= self.maxlen:
            self.t.popleft(); self.y.popleft()
        self.t.append(ts); self.y.append(val)

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


# ======================
# Acquisition
# ======================
class Acquisition:
    """Adquisición sim/real + preview + cálculo de A con dark/blanco/baseline EMA."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.mode = cfg.get("mode", "sim")
        self.running = False
        self.preview_only = False

        # Series buffers
        self.temp = RingBuffer(maxlen=20000)
        self.a_sel = RingBuffer(maxlen=20000)
        self.spec = SpectraBuffer(max_frames=20000)

        # Eje espectral
        gcfg = cfg.get("graphics", {})
        self.lmin = float(gcfg.get("lambda_min", 400))
        self.lmax = float(gcfg.get("lambda_max", 700))
        self.w = int(cfg.get("imaging", {}).get("resize_w", 500))
        self.lambdas = np.linspace(self.lmin, self.lmax, self.w, dtype=np.float32)
        self.lambda_sel = float(gcfg.get("lambda_selected_nm", 520))

        # Preview
        self._preview_rgb: Optional[np.ndarray] = None
        self._last_I: Optional[np.ndarray] = None

        # BLANCO / DARK / Dominio
        bcfg = cfg.get("blank", {})
        self.blank_domain: str = bcfg.get("domain", "log")  # "log" | "linear"
        self.blank_line_linear: Optional[np.ndarray] = None
        self.blank_log_avg: Optional[np.ndarray] = None
        self.dark_line: Optional[np.ndarray] = None

        # Acumulador mean N
        self._blank_accum_active = False
        self._blank_accum_N = int(bcfg.get("N", 200))
        self._blank_accum_count = 0
        self._blank_accum_sum: Optional[np.ndarray] = None

        # Auto-blanco al iniciar
        self._auto_blank_N = int(bcfg.get("autoN", 0))  # 0 = off
        self._auto_blank_running = False

        # Baseline EMA (fallback)
        baseline = cfg.get("imaging", {}).get("baseline", {"mode": "ema", "alpha": 0.01})
        self._ema_alpha = float(baseline.get("alpha", 0.01))
        self._ema_log_ref: Optional[np.ndarray] = None

        # Threads/sources
        self._threads: List[threading.Thread] = []
        self._stop_evt = threading.Event()
        self._arduino = None
        self._img_source = None
        self._extractor = None

        # Mensajes a UI
        self.last_event: Optional[str] = None

    # ---------- Control ----------
    def start_preview(self) -> None:
        """Inicia fuente de imagen SOLO para previsualización."""
        if self.preview_only or self.running or self.mode != "real":
            return

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

        icfg = self.cfg.get("imaging", {})
        fps = float(icfg.get("fps", 12))
        roi_cfg = icfg.get("roi", {"x": 0, "y": 0, "w": 500, "h": 86})
        resize_w = int(icfg.get("resize_w", 500))
        gamma = float(icfg.get("gamma", 2.2222))
        bg = float(icfg.get("background_subtract", 300.0))
        min_floor = float(icfg.get("min_floor", 1e-3))
        log10_flag = bool(icfg.get("log10", True))

        try:
            self._extractor = SpectrumExtractor(
                roi=roi_cfg, resize_w=resize_w, gamma=gamma, bg_sub=bg,
                min_floor=min_floor, log10=log10_flag
            )
        except TypeError:
            self._extractor = SpectrumExtractor(
                roi=roi_cfg, resize_w=resize_w, gamma=gamma, bg_sub=bg, min_floor=min_floor, log10=log10_flag
            )

        def on_frame(ts: float, frame_bgr: np.ndarray) -> None:
            # ROI
            roi = getattr(self._extractor, "roi", roi_cfg)
            # BGR→RGB + marco ROI
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

            # Línea cruda para vista previa
            try:
                I_line, _ = self._extractor.process_frame(frame_bgr)
                if I_line is not None:
                    self._last_I = I_line.astype(np.float32)
            except Exception:
                pass

        if source == "screen":
            self._img_source = ScreenReader(roi=roi_cfg, fps=fps, frame_cb=on_frame)
        else:
            cam_index = int(icfg.get("camera_index", 0))
            self._img_source = CameraReader(cam_index=cam_index, fps=fps, frame_cb=on_frame)

        self._img_source.start()
        self.preview_only = True

    def start(self) -> None:
        if self.running:
            return
        if self.preview_only:
            self.stop()
        self._stop_evt.clear()
        self.running = True
        self._prepare_auto_blank()
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
        self._auto_blank_running = False

    # ---------- Export ----------
    def export_csv(self, out_path: Path) -> None:
        M, lambdas = self.spec.as_matrix()
        tT = np.array(self.temp.t, dtype=np.float64)
        TT = np.array(self.temp.y, dtype=np.float32)
        tA = np.array(self.a_sel.t, dtype=np.float64)
        AA = np.array(self.a_sel.y, dtype=np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path.with_suffix(".temp.csv"), np.c_[tT, TT], delimiter=",",
                   header="t_s,TempC", comments="")
        np.savetxt(out_path.with_suffix(".asel.csv"), np.c_[tA, AA], delimiter=",",
                   header="t_s,A_sel", comments="")
        if M.size:
            head = ",".join(["lambda_nm"] + [f"frame_{i}" for i in range(M.shape[0])])
            mat = np.vstack([lambdas, M.T]).T
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

    def latest_temp(self) -> Optional[float]:
        return self.temp.y[-1] if self.temp.y else None

    def latest_asel(self) -> Optional[float]:
        return self.a_sel.y[-1] if self.a_sel.y else None

    def latest_spectrum(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        spec, lam = self.spec.latest()
        return spec, (lam if lam is not None else self.lambdas)

    def latest_preview(self) -> Optional[np.ndarray]:
        return self._preview_rgb

    # ---------- BLANK/DARK ----------
    def clear_blank(self) -> None:
        self.blank_line_linear = None
        self.blank_log_avg = None
        self._blank_accum_active = False
        self._blank_accum_count = 0
        self._blank_accum_sum = None
        self.last_event = "Blanco borrado"

    def set_blank_single(self) -> None:
        if self._last_I is None:
            self.last_event = "No hay frame para fijar blanco"
            return
        Ieff = self._apply_dark(self._last_I)
        if self.blank_domain == "linear":
            self.blank_line_linear = Ieff.copy()
            self.blank_log_avg = None
        else:
            self.blank_log_avg = np.log10(np.clip(Ieff, EPS, None))
            self.blank_line_linear = None
        self.last_event = "Blanco fijado (frame actual)"

    def begin_blank_meanN(self, N: int) -> None:
        self._blank_accum_active = True
        self._blank_accum_N = int(N)
        self._blank_accum_count = 0
        self._blank_accum_sum: Optional[np.ndarray] = None
        self.last_event = f"Acumulando blanco (N={N}, dominio={self.blank_domain})"

    def set_dark_from_current(self) -> None:
        if self._last_I is None:
            self.last_event = "No hay frame para 0% ADJ"
            return
        self.dark_line = self._last_I.copy()
        self.last_event = "Dark (0% ADJ) fijado"

    def set_blank_100_from_current(self) -> None:
        self.set_blank_single()
        self.last_event = "100% ADJ fijado"

    def import_blank(self, path: Path) -> None:
        arr = np.load(path)
        if arr.ndim == 1:
            if self.blank_domain == "linear":
                self.blank_line_linear = arr.astype(np.float32)
                self.blank_log_avg = None
                self.last_event = "Blanco (lineal) importado"
            else:
                self.blank_log_avg = arr.astype(np.float32)
                self.blank_line_linear = None
                self.last_event = "Blanco (log) importado"
        else:
            self.last_event = "Archivo incompatible"

    def export_blank(self, path: Path) -> None:
        if self.blank_domain == "linear" and self.blank_line_linear is not None:
            np.save(path, self.blank_line_linear); self.last_event = "Blanco (lineal) exportado"
        elif self.blank_domain == "log" and self.blank_log_avg is not None:
            np.save(path, self.blank_log_avg); self.last_event = "Blanco (log) exportado"
        else:
            self.last_event = "No hay blanco para exportar"

    # ---------- Internals: cálculo A ----------
    def _apply_dark(self, I: np.ndarray) -> np.ndarray:
        if self.dark_line is None:
            return np.maximum(I, EPS)
        return np.maximum(I - self.dark_line, EPS)

    def _accumulate_blank_if_needed(self, Ieff: np.ndarray) -> None:
        if self._auto_blank_running and not self._blank_accum_active and self._auto_blank_N > 0:
            self.begin_blank_meanN(self._auto_blank_N)
            self._auto_blank_running = False

        if not self._blank_accum_active:
            return
        v = np.log10(Ieff) if self.blank_domain == "log" else Ieff
        if self._blank_accum_sum is None:
            self._blank_accum_sum = v.copy()
        else:
            self._blank_accum_sum += v
        self._blank_accum_count += 1
        if self._blank_accum_count >= self._blank_accum_N:
            avg = self._blank_accum_sum / max(1, self._blank_accum_count)
            if self.blank_domain == "log":
                self.blank_log_avg = avg.astype(np.float32); self.blank_line_linear = None
            else:
                self.blank_line_linear = avg.astype(np.float32); self.blank_log_avg = None
            self._blank_accum_active = False
            self.last_event = f"Blanco promedio fijado (N={self._blank_accum_count}, dominio={self.blank_domain})"

    def _compute_A_from_I(self, I: np.ndarray) -> Optional[np.ndarray]:
        Ieff = self._apply_dark(I)
        self._accumulate_blank_if_needed(Ieff)

        if self.blank_domain == "linear" and self.blank_line_linear is not None:
            denom = np.maximum(self.blank_line_linear, EPS)
            ratio = np.clip(Ieff / denom, EPS, 1e9)
            return -np.log10(ratio).astype(np.float32)

        if self.blank_domain == "log" and self.blank_log_avg is not None:
            return (-np.log10(Ieff) + self.blank_log_avg).astype(np.float32)

        # Fallback: baseline EMA en log
        logI = np.log10(Ieff)
        if self._ema_log_ref is None:
            self._ema_log_ref = logI.copy()
        else:
            a = float(self._ema_alpha)
            self._ema_log_ref = (1.0 - a) * self._ema_log_ref + a * logI
        return (-np.log10(Ieff) + self._ema_log_ref).astype(np.float32)

    def _prepare_auto_blank(self) -> None:
        if self._auto_blank_N > 0:
            self._auto_blank_running = True

    # ---------- Sources ----------
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
                I_line = np.exp(-0.5 * ((self.lambdas - center)/sigma)**2).astype(np.float32)
                self._last_I = I_line.copy()
                A_line = self._compute_A_from_I(I_line)
                now = time.time()
                if A_line is not None:
                    lam_dyn = (
                        self.lambdas if A_line.size == self.lambdas.size
                        else np.linspace(self.lmin, self.lmax, A_line.size, dtype=np.float32)
                    )
                    self.spec.push(A_line, lam_dyn)
                    A_sel = float(np.interp(self.lambda_sel, lam_dyn, A_line))
                    self.a_sel.push(now, A_sel)
                img = (np.tile(I_line, (h, 1)) * 255.0).astype(np.uint8)
                rgb = np.dstack([img, img, img])
                self._preview_rgb = rgb
                time.sleep(1.0)

        for fn in (temp_thread, spec_thread):
            th = threading.Thread(target=fn, daemon=True); th.start(); self._threads.append(th)

    def _start_real(self) -> None:
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

        # Arduino tolerante
        scfg = self.cfg.get("serial", {})
        port = scfg.get("port", "COM3"); baud = int(scfg.get("baud", 115200))

        def on_line(ts: float, tempC: float, heater: bool) -> None:
            self.temp.push(ts, tempC)

        self._arduino = None
        try:
            self._arduino = ArduinoProtocol(port=port, baud=baud, line_cb=on_line); self._arduino.start()
        except Exception as e:
            try:
                messagebox.showwarning("Arduino no disponible",
                                       f"No se pudo abrir {port} @ {baud}.\n\nLa adquisición continuará sin T°.\n\nDetalle: {e}")
            except Exception:
                pass
            print(f"[WARN] Arduino no disponible ({e}). Continuando sin T°.")
            self._arduino = None

        # Imaging
        icfg = self.cfg.get("imaging", {})
        fps = float(icfg.get("fps", 12))
        roi_cfg = icfg.get("roi", {"x": 0, "y": 0, "w": 500, "h": 86})
        resize_w = int(icfg.get("resize_w", 500))
        gamma = float(icfg.get("gamma", 2.2222))
        bg = float(icfg.get("background_subtract", 300.0))
        min_floor = float(icfg.get("min_floor", 1e-3))
        log10_flag = bool(icfg.get("log10", True))

        try:
            self._extractor = SpectrumExtractor(
                roi=roi_cfg, resize_w=resize_w, gamma=gamma, bg_sub=bg,
                min_floor=min_floor, log10=log10_flag
            )
        except TypeError:
            self._extractor = SpectrumExtractor(
                roi=roi_cfg, resize_w=resize_w, gamma=gamma, bg_sub=bg, min_floor=min_floor, log10=log10_flag
            )

        def on_frame(ts: float, frame_bgr: np.ndarray) -> None:
            # Preview con ROI
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

            # Línea cruda + push con λ dinámico si hace falta
            try:
                I_line, _ = self._extractor.process_frame(frame_bgr)
                if I_line is None:
                    return
                self._last_I = I_line.astype(np.float32)
                A_line = self._compute_A_from_I(self._last_I)
                if A_line is None:
                    return

                lam_dyn = (
                    self.lambdas if A_line.size == self.lambdas.size
                    else np.linspace(self.lmin, self.lmax, A_line.size, dtype=np.float32)
                )
                self.spec.push(A_line, lam_dyn)
                A_sel = float(np.interp(self.lambda_sel, lam_dyn, A_line))
                self.a_sel.push(ts, A_sel)
            except Exception:
                pass

        if source == "screen":
            self._img_source = ScreenReader(roi=roi_cfg, fps=fps, frame_cb=on_frame)
        else:
            cam_index = int(icfg.get("camera_index", 0))
            self._img_source = CameraReader(cam_index=cam_index, fps=fps, frame_cb=on_frame)

        try:
            self._img_source.start()
        except Exception as e:
            try:
                messagebox.showerror("Fuente de imagen",
                                     f"No se pudo iniciar la fuente '{source}'.\n\nDetalle: {e}")
            except Exception:
                pass
            self._stop_evt.set(); self.running = False
            return


# ======================
# GUI
# ======================
class SpectroRTApp:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.mode = cfg.get("mode", "sim")
        self.root = tk.Tk()
        self.root.title("spectro-rt v0.4.2 (all-Python)")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Acquisition
        self.acq = Acquisition(cfg)

        # Graphics config
        gcfg = cfg.get("graphics", {})
        self.refresh_ms = int(gcfg.get("refresh_ms", 200))   # plots
        self.preview_ms = int(cfg.get("imaging", {}).get("preview_ms", 66))  # ~15 fps
        self.window_s = float(gcfg.get("window_s", 120.0))
        self.lambda_sel = float(gcfg.get("lambda_selected_nm", 520))

        # UI state vars
        self._grid_var = tk.IntVar(value=1)
        self._source_var = tk.StringVar(value=self.cfg.get("imaging", {}).get("source", "camera"))
        self._preview_var = tk.IntVar(value=1)
        self._blank_domain_var = tk.StringVar(value=self.acq.blank_domain)  # "log" | "linear"

        # Medición
        self._tk_preview = None
        self.k_var = tk.DoubleVar(value=float(self.cfg.get("chemistry", {}).get("k_factor", 1.0)))
        self.lambda_var = tk.DoubleVar(value=self.lambda_sel)
        self.A_var = tk.StringVar(value="—")
        self.T_var = tk.StringVar(value="—")
        self.C_var = tk.StringVar(value="—")
        self.blank_status_var = tk.StringVar(value="BLANCO: Off")

        # Build UI
        self._build_menubar()
        self._build_toolbar()
        self._build_measure_panel()
        self._build_preview()
        self._build_plots()
        self._status_vars()

        # Shortcuts
        self._bind_shortcuts()

        # Matplotlib click para mover λ
        self._mpl_cid = None

        # Schedulers
        self._refresh_job = None
        self._preview_job = None
        self._schedule_refresh()       # plots/T/A
        self._schedule_preview_loop()  # preview en vivo independiente

        # Preview activo si modo real
        if self.mode == "real":
            try:
                self.acq.start_preview()
                self.var_status.set("Preview activo (fuente de imagen) — aún sin adquisición")
            except Exception as e:
                try:
                    messagebox.showwarning("Cámara/Pantalla no disponible",
                                           f"No se pudo iniciar el preview.\n\nDetalle: {e}")
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
        m_src = tk.Menu(m_acq, tearoff=False)
        m_src.add_radiobutton(label="Cámara", value="camera", variable=self._source_var, command=self._on_change_source)
        m_src.add_radiobutton(label="Pantalla", value="screen", variable=self._source_var, command=self._on_change_source)
        m_acq.add_cascade(label="Fuente de imagen", menu=m_src)
        m_acq.add_command(label="Configurar puerto serie…", command=self._show_serial_dialog)
        m.add_cascade(label="Adquisición", menu=m_acq)

        # Vista
        m_view = tk.Menu(m, tearoff=False)
        m_view.add_command(label="Cambiar λ seleccionada…", command=self._prompt_lambda_change)
        m_view.add_command(label="Cambiar ventana (s)…", command=self._prompt_window_change)
        m_view.add_checkbutton(label="Grilla", variable=self._grid_var, command=self._toggle_grid)
        m_view.add_checkbutton(label="Mostrar preview", variable=self._preview_var, command=self._toggle_preview)
        m_view.add_command(label="Reset zoom", command=self._reset_view)
        m.add_cascade(label="Vista", menu=m_view)

        # Herramientas
        m_tools = tk.Menu(m, tearoff=False)
        m_tools.add_command(label="Editar ROI…", command=self._prompt_roi_dialog)
        m_tools.add_command(label="Preferencias…", command=self._show_preferences_dialog)

        # Blanco
        m_blank = tk.Menu(m_tools, tearoff=False)
        m_blank.add_command(label="Fijar blanco (frame actual)", accelerator="Ctrl+B", command=self._on_blank_single)
        m_blank.add_command(label="Fijar blanco (promedio de N…)", accelerator="Ctrl+Shift+B", command=self._on_blank_meanN)
        m_blank.add_command(label="Auto-blanco (primeros N…)", command=self._on_blank_autoN)
        m_blank.add_separator()
        m_blank.add_radiobutton(label="Dominio: Log (compat. ImageJ)", value="log",
                                variable=self._blank_domain_var, command=self._on_blank_domain_change)
        m_blank.add_radiobutton(label="Dominio: Lineal (I/I0)", value="linear",
                                variable=self._blank_domain_var, command=self._on_blank_domain_change)
        m_blank.add_separator()
        m_blank.add_command(label="Importar blanco…", command=self._on_blank_import)
        m_blank.add_command(label="Exportar blanco…", command=self._on_blank_export)
        m_blank.add_separator()
        m_blank.add_command(label="Borrar blanco", command=self._on_blank_clear)
        m_tools.add_cascade(label="Blanco", menu=m_blank)

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
        self.btn_fig = ttk.Button(bar, text="Exportar Gráfica", command=self._on_export_figure)

        self.win_var = tk.StringVar(value=str(int(self.window_s)))
        ttk.Label(bar, text="Ventana (s)").pack(side=tk.RIGHT)
        ttk.Entry(bar, textvariable=self.win_var, width=6).pack(side=tk.RIGHT, padx=(4, 12))

        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop.pack(side=tk.LEFT, padx=(6, 12))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self.btn_csv.pack(side=tk.LEFT)
        self.btn_npy.pack(side=tk.LEFT, padx=(6, 0))
        self.btn_fig.pack(side=tk.LEFT, padx=(6, 0))

    # ---------- Panel de Medición ----------
    def _build_measure_panel(self) -> None:
        pane = ttk.Frame(self.root, padding=(8, 2))
        pane.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(pane, text="λ (nm):").pack(side=tk.LEFT)
        entry = ttk.Entry(pane, textvariable=self.lambda_var, width=7)
        entry.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Button(pane, text="Aplicar λ", command=self._apply_lambda_from_entry).pack(side=tk.LEFT)

        ttk.Separator(pane, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(pane, text="0% ADJ", command=self._on_dark_adj).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(pane, text="100% ADJ", command=self._on_blank_100).pack(side=tk.LEFT)

        ttk.Separator(pane, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(pane, text="k:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(pane, textvariable=self.k_var, width=8).pack(side=tk.LEFT)
        ttk.Label(pane, text="C = A/k:").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Label(pane, textvariable=self.C_var).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Separator(pane, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(pane, text="A:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(pane, textvariable=self.A_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(pane, text="%T:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(pane, textvariable=self.T_var).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Separator(pane, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(pane, textvariable=self.blank_status_var).pack(side=tk.LEFT)

    # ---------- Preview ----------
    def _build_preview(self) -> None:
        self.preview_frame = ttk.Frame(self.root, padding=(8, 0))
        self.preview_frame.pack(side=tk.TOP, fill=tk.X)
        self.preview_label = ttk.Label(self.preview_frame, text="(preview)")
        self.preview_label.pack(side=tk.TOP, anchor="center", pady=(2, 4))
        if not ImageTk:
            self.preview_label.config(text="(Instala Pillow para ver el preview: pip install Pillow)")

    def _toggle_preview(self) -> None:
        if self._preview_var.get():
            self.preview_frame.pack(side=tk.TOP, fill=tk.X)
        else:
            self.preview_frame.forget()

    def _update_preview(self) -> None:
        if not self._preview_var.get() or not ImageTk:
            return
        frame = self.acq.latest_preview()
        if frame is None:
            return
        try:
            img = Image.fromarray(frame)  # RGB
            target_w = max(320, self.root.winfo_width() - 40)
            ratio = target_w / max(1, img.width)
            target_h = int(img.height * ratio)
            max_h = 220
            if target_h > max_h:
                ratio = max_h / max(1, img.height)
                target_h = max_h
                target_w = int(img.width * ratio)
            img = img.resize((target_w, target_h), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            # Referencias fuertes para evitar GC
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo
            self._tk_preview = photo
            self.preview_label.update_idletasks()
        except Exception:
            pass

    def _schedule_preview_loop(self) -> None:
        self._update_preview()
        self._preview_job = self.root.after(self.preview_ms, self._schedule_preview_loop)

    # ---------- Plots ----------
    def _build_plots(self) -> None:
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.ax_T = self.fig.add_subplot(3, 1, 1)
        self.ax_A = self.fig.add_subplot(3, 1, 2)
        self.ax_S = self.fig.add_subplot(3, 1, 3)

        self.ax_T.set_ylabel("T (°C)")
        self.ax_A.set_ylabel("A(λ_sel)")
        self.ax_S.set_ylabel("A(λ)")
        self.ax_S.set_xlabel("λ (nm)")

        for ax in (self.ax_T, self.ax_A, self.ax_S):
            ax.grid(True, linestyle=":", linewidth=0.6)

        self.l_T, = self.ax_T.plot([], [], lw=1.5)
        self.l_A, = self.ax_A.plot([], [], lw=1.5)
        self.l_S, = self.ax_S.plot([], [], lw=1.5)
        self.cursor_S = self.ax_S.axvline(self.lambda_sel, ls="--")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Click para mover λ
        self._mpl_cid = self.fig.canvas.mpl_connect("button_press_event", self._on_mpl_click)

    def _on_mpl_click(self, event):
        if event.inaxes != self.ax_S:
            return
        x = float(event.xdata)
        x = max(self.acq.lmin, min(self.acq.lmax, x))
        self.lambda_sel = x
        self.lambda_var.set(round(x, 2))
        self.cursor_S.set_xdata([self.lambda_sel])
        self.cfg.setdefault("graphics", {})["lambda_selected_nm"] = self.lambda_sel
        self.var_status.set(f"λ_sel = {self.lambda_sel:.2f} nm")

    def _status_vars(self) -> None:
        stat = ttk.Frame(self.root, padding=(8, 4))
        stat.pack(side=tk.BOTTOM, fill=tk.X)
        self.var_status = tk.StringVar(value="Listo")
        self.var_last = tk.StringVar(value="T=—  A=—")
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
        self.root.bind("<Control-b>", lambda e: self._on_blank_single())
        self.root.bind("<Control-Shift-b>", lambda e: self._on_blank_meanN())

    # ---------- Events ----------
    def _on_start(self) -> None:
        try:
            self.window_s = float(self.win_var.get())
        except Exception:
            self.window_s = 120.0; self.win_var.set("120")
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

    def _on_export_csv(self) -> None:
        ts = int(time.time())
        init_dir = Path(self.cfg.get("paths", {}).get("export_dir", "data/exports"))
        init_dir.mkdir(parents=True, exist_ok=True)
        f = filedialog.asksaveasfilename(defaultextension=".csv", initialdir=str(init_dir),
                                         initialfile=f"run_{ts}", title="Guardar CSV base")
        if not f: return
        try:
            self.acq.export_csv(Path(f)); self.var_status.set("CSV exportado")
        except Exception as e:
            messagebox.showerror("Exportar CSV", str(e))

    def _on_export_npy(self) -> None:
        ts = int(time.time())
        init_dir = Path(self.cfg.get("paths", {}).get("export_dir", "data/exports"))
        init_dir.mkdir(parents=True, exist_ok=True)
        f = filedialog.asksaveasfilename(defaultextension=".npy", initialdir=str(init_dir),
                                         initialfile=f"run_{ts}",
                                         title="Guardar base NPY (se crearán 3 archivos)")
        if not f: return
        try:
            self.acq.export_npy(Path(f)); self.var_status.set("NPY exportado")
        except Exception as e:
            messagebox.showerror("Exportar NPY", str(e))

    def _on_export_figure(self) -> None:
        f = filedialog.asksaveasfilename(defaultextension=".png", title="Guardar figura")
        if not f: return
        try:
            self.fig.savefig(f, bbox_inches="tight", dpi=200); self.var_status.set("Figura exportada")
        except Exception as e:
            messagebox.showerror("Exportar figura", str(e))

    def _on_change_source(self) -> None:
        src = self._source_var.get()
        self.cfg.setdefault("imaging", {})["source"] = src
        messagebox.showinfo("Fuente de imagen", "La fuente cambiará al reiniciar la adquisición.")

    def _prompt_lambda_change(self) -> None:
        val = simpledialog.askfloat("λ seleccionada", "Nueva λ (nm):",
                                    initialvalue=self.lambda_sel, minvalue=200, maxvalue=1100)
        if val is None: return
        self.lambda_sel = float(val)
        self.lambda_var.set(self.lambda_sel)
        self.cfg.setdefault("graphics", {})["lambda_selected_nm"] = self.lambda_sel
        self.cursor_S.set_xdata([self.lambda_sel])
        self.var_status.set(f"λ_sel = {self.lambda_sel:.1f} nm")

    def _apply_lambda_from_entry(self) -> None:
        try:
            val = float(self.lambda_var.get())
        except Exception:
            return
        val = max(self.acq.lmin, min(self.acq.lmax, val))
        self.lambda_sel = val
        self.cfg.setdefault("graphics", {})["lambda_selected_nm"] = self.lambda_sel
        self.cursor_S.set_xdata([self.lambda_sel])
        self.var_status.set(f"λ_sel = {self.lambda_sel:.2f} nm")

    def _prompt_window_change(self) -> None:
        val = simpledialog.askfloat("Ventana (s)", "Segundos visibles:",
                                    initialvalue=self.window_s, minvalue=5, maxvalue=3600)
        if val is None: return
        self.window_s = float(val); self.win_var.set(str(int(self.window_s)))

    # ROI editable
    def _prompt_roi_dialog(self) -> None:
        roi = self.cfg.get("imaging", {}).get("roi", {"x": 0, "y": 0, "w": 500, "h": 86})
        top = tk.Toplevel(self.root); top.title("Editar ROI"); top.resizable(False, False)
        frm = ttk.Frame(top, padding=12); frm.pack(fill=tk.BOTH, expand=True)
        vars_ = {k: tk.IntVar(value=int(roi.get(k, 0))) for k in ("x", "y", "w", "h")}
        for i, k in enumerate(["x", "y", "w", "h"]):
            ttk.Label(frm, text=k.upper()).grid(row=i, column=0, sticky="e", padx=6, pady=4)
            ttk.Entry(frm, textvariable=vars_[k], width=10).grid(row=i, column=1, sticky="w", padx=6, pady=4)
        btns = ttk.Frame(frm); btns.grid(row=5, column=0, columnspan=2, pady=(10, 0))

        def apply_and_close():
            new_roi = {k: int(vars_[k].get()) for k in ("x", "y", "w", "h")}
            self.cfg.setdefault("imaging", {})["roi"] = new_roi
            try:
                if self.acq._extractor is not None: self.acq._extractor.roi = new_roi
            except Exception: pass
            try:
                if hasattr(self.acq._img_source, "update_roi"): self.acq._img_source.update_roi(new_roi)
            except Exception: pass
            self.acq.clear_blank()
            messagebox.showinfo("ROI", "ROI actualizado. El blanco se ha borrado para evitar inconsistencias.")
            top.destroy()

        ttk.Button(btns, text="Aceptar", command=apply_and_close).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Cancelar", command=top.destroy).pack(side=tk.LEFT)

    def _show_serial_dialog(self) -> None:
        scfg = self.cfg.get("serial", {})
        top = tk.Toplevel(self.root); top.title("Puerto serie"); top.resizable(False, False)
        frm = ttk.Frame(top, padding=12); frm.pack(fill=tk.BOTH, expand=True)
        port_var = tk.StringVar(value=str(scfg.get("port", "COM3")))
        baud_var = tk.IntVar(value=int(scfg.get("baud", 115200)))
        ttk.Label(frm, text="Port").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm, textvariable=port_var, width=12).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm, text="Baud").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm, textvariable=baud_var, width=12).grid(row=1, column=1, sticky="w", padx=6, pady=4)

        def apply_and_close():
            self.cfg.setdefault("serial", {})["port"] = port_var.get()
            self.cfg["serial"]["baud"] = int(baud_var.get())
            messagebox.showinfo("Puerto serie", "Los cambios se aplican al reiniciar la adquisición.")
            top.destroy()

        ttk.Button(frm, text="Aceptar", command=apply_and_close).grid(row=3, column=0, columnspan=2, pady=(10, 0))

    def _show_preferences_dialog(self) -> None:
        icfg = self.cfg.get("imaging", {})
        top = tk.Toplevel(self.root); top.title("Preferencias de imagen"); top.resizable(False, False)
        frm = ttk.Frame(top, padding=12); frm.pack(fill=tk.BOTH, expand=True)
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
        ttk.Checkbutton(frm, text="Log10 (solo extractor interno)", variable=log_var)\
            .grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        def apply_and_close():
            icfg["gamma"] = float(gamma_var.get())
            icfg["background_subtract"] = float(bg_var.get())
            icfg["min_floor"] = float(min_var.get())
            icfg["log10"] = bool(log_var.get())
            self.acq.clear_blank()
            self.var_status.set("Preferencias actualizadas. El blanco se ha borrado para evitar inconsistencias.")
            top.destroy()

        ttk.Button(frm, text="Guardar", command=apply_and_close).grid(row=10, column=0, columnspan=2, pady=(10, 0))

    # ----- Blanco -----
    def _on_blank_domain_change(self):
        self.acq.blank_domain = self._blank_domain_var.get()
        self.acq.clear_blank()
        self.var_status.set(f"Dominio de blanco: {self.acq.blank_domain}. Se borró el blanco actual.")

    def _on_blank_single(self):
        self.acq.set_blank_single(); self._poke_status_from_acq()

    def _on_blank_meanN(self, N: Optional[int] = None):
        if N is None:
            val = simpledialog.askinteger("Promedio de blanco", "N frames:", initialvalue=200, minvalue=2, maxvalue=10000)
            if val is None: return
            N = int(val)
        self.acq.begin_blank_meanN(N); self._poke_status_from_acq()

    def _on_blank_autoN(self):
        val = simpledialog.askinteger("Auto-blanco", "Promediar los primeros N frames al iniciar:",
                                      initialvalue=200, minvalue=2, maxvalue=10000)
        if val is None: return
        self.acq._auto_blank_N = int(val)
        self.var_status.set(f"Auto-blanco configurado (N={val}). Se aplicará al iniciar.")

    def _on_blank_clear(self):
        self.acq.clear_blank(); self._poke_status_from_acq()

    def _on_blank_import(self):
        f = filedialog.askopenfilename(title="Importar blanco (.npy)", filetypes=[("NumPy .npy", "*.npy")])
        if not f: return
        self.acq.import_blank(Path(f)); self._poke_status_from_acq()

    def _on_blank_export(self):
        f = filedialog.asksaveasfilename(title="Exportar blanco (.npy)", defaultextension=".npy",
                                         filetypes=[("NumPy .npy", "*.npy")])
        if not f: return
        self.acq.export_blank(Path(f)); self._poke_status_from_acq()

    # 0% / 100% ADJ
    def _on_dark_adj(self):
        self.acq.set_dark_from_current(); self._poke_status_from_acq()

    def _on_blank_100(self):
        self.acq.set_blank_100_from_current(); self._poke_status_from_acq()

    # ---------- Refresh loops ----------
    def _schedule_refresh(self) -> None:
        self._refresh()
        self._refresh_job = self.root.after(self.refresh_ms, self._schedule_refresh)

    def _refresh(self) -> None:
        # series y plots (el preview tiene bucle propio)
        tT, TT = self.acq.temp.window(self.window_s)
        tA, AA = self.acq.a_sel.window(self.window_s)
        spec, lam = self.acq.latest_spectrum()

        if tT.size: tT = tT - tT[0]
        if tA.size: tA = tA - tA[0]

        self.l_T.set_data(tT, TT); self.ax_T.relim(); self.ax_T.autoscale_view()
        self.l_A.set_data(tA, AA); self.ax_A.relim(); self.ax_A.autoscale_view()

        if spec is not None:
            # ---- FIX: asegurar misma longitud de X e Y ----
            lam_plot = lam
            if lam_plot.size != spec.size:
                lam_plot = np.linspace(self.acq.lmin, self.acq.lmax, spec.size, dtype=np.float32)

            self.l_S.set_data(lam_plot, spec)
            self.cursor_S.set_xdata([self.lambda_sel])
            self.ax_S.relim(); self.ax_S.autoscale_view()

            A_sel = float(np.interp(self.lambda_sel, lam_plot, spec))
            self.A_var.set(f"{A_sel:.4f}")
            self.T_var.set(f"{100.0*(10**(-A_sel)):.2f} %")
            k = float(self.k_var.get()) if self.k_var.get() not in ("", None) else 0.0
            self.C_var.set(f"{(A_sel/k):.6g}" if k > 0 else "—")

            Tlast = self.acq.latest_temp()
            if Tlast is not None:
                self.var_last.set(f"T={Tlast:.2f} °C   A={A_sel:.4f}")

        self._update_blank_status()
        self._poke_status_from_acq()
        self.canvas.draw_idle()

    def _update_blank_status(self):
        mode = "Off"
        if self.acq.blank_domain == "linear" and self.acq.blank_line_linear is not None:
            mode = "Single (linear)"
        elif self.acq.blank_domain == "log" and self.acq.blank_log_avg is not None:
            mode = "Single (log)"
        if self.acq._blank_accum_active:
            mode = f"Mean N={self.acq._blank_accum_N} ({self.acq.blank_domain})"
        self.blank_status_var.set(f"BLANCO: {mode}")

    def _poke_status_from_acq(self):
        if self.acq.last_event:
            self.var_status.set(self.acq.last_event)
            self.acq.last_event = None

    # ---------- Otros ----------
    def _toggle_grid(self) -> None:
        on = True if self._grid_var.get() else False
        for ax in (self.ax_T, self.ax_A, self.ax_S):
            ax.grid(on, linestyle=":", linewidth=0.6)
        self.canvas.draw_idle()

    def _reset_view(self) -> None:
        for ax in (self.ax_T, self.ax_A, self.ax_S):
            ax.relim(); ax.autoscale_view()
        self.canvas.draw_idle()

    def _show_about(self) -> None:
        msg = (
            "spectro-rt v0.4.2 (all-Python)\n"
            "Preview activo antes de iniciar + tolerante a hardware.\n\n"
            "Atajos:\n"
            "  Ctrl+R: Iniciar    |  Ctrl+Shift+R: Detener\n"
            "  Ctrl+S: Exportar CSV   Ctrl+E: NPY\n"
            "  Ctrl+G: Exportar Figura   Ctrl+Q: Salir\n"
            "  Ctrl+B: Fijar blanco (actual)   Ctrl+Shift+B: Blanco mean N\n\n"
            "Menús: Archivo, Adquisición, Vista, Herramientas (ROI/Preferencias/Blanco), Ayuda.\n"
            "Blanco: modo Log (compat. ImageJ) o Lineal (I/I0). 0%/100% ADJ incluidos.\n"
            "Datos en memoria (sin guardar 50k imágenes)."
        )
        messagebox.showinfo("Acerca de spectro-rt", msg)

    def _on_quit(self) -> None:
        self._on_close()

    def _on_close(self) -> None:
        try:
            if self._preview_job: self.root.after_cancel(self._preview_job)
            if self._refresh_job: self.root.after_cancel(self._refresh_job)
            self.acq.stop()
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

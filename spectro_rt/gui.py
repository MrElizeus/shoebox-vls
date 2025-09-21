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

    # ---------- Camera controls (manuales) ----------

    def update_camera_controls(self, controls: Dict, disable_auto: bool = True) -> None:
        self._cam_controls = dict(controls or {})
        try:
            if self._img_source and hasattr(self._img_source, "update_controls"):
                self._img_source.update_controls(self._cam_controls, disable_auto=disable_auto)
        except Exception:
            pass

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
        self._source_var = tk.StringVar(value=self.cfg.get("imaging", {}).get("source", "camera"))
        self._preview_var = tk.IntVar(value=1)
        self._tk_preview = None  # ImageTk ref

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
            self.preview_label.configure(image=photo, text="")
            self._tk_preview = photo
        except Exception:
            pass

    # ---------- Plots ----------

    def _build_plots(self) -> None:
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.ax_T = self.fig.add_subplot(3, 1, 1)
        self.ax_A = self.fig.add_subplot(3, 1, 2)
        self.ax_S = self.fig.add_subplot(3, 1, 3)

        self.ax_T.set_ylabel("T (°C)")
        self.ax_A.set_ylabel("A(λ_sel)")
        self.ax_S.set_ylabel("I/A(λ)")
        self.ax_S.set_xlabel("λ (nm)")

        for ax in (self.ax_T, self.ax_A, self.ax_S):
            ax.grid(True, linestyle=":", linewidth=0.6)

        self.l_T, = self.ax_T.plot([], [], lw=1.5)
        self.l_A, = self.ax_A.plot([], [], lw=1.5)
        self.l_S, = self.ax_S.plot([], [], lw=1.5)
        self.cursor_S = self.ax_S.axvline(self.lambda_sel, ls="--")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
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
        self.A_var = tk.StringVar(value="—")
        ttk.Label(pane, textvariable=self.A_var, width=8).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(pane, text="%T=").pack(side=tk.LEFT)
        self.T_var = tk.StringVar(value="—")
        ttk.Label(pane, textvariable=self.T_var, width=10).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(pane, text="ε·L=").pack(side=tk.LEFT)
        self.k_var = tk.StringVar(value="1.0")
        ttk.Entry(pane, textvariable=self.k_var, width=8).pack(side=tk.LEFT, padx=(4, 6))

        ttk.Label(pane, text="C=").pack(side=tk.LEFT)
        self.C_var = tk.StringVar(value="—")
        ttk.Label(pane, textvariable=self.C_var, width=12).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Button(pane, text="Aplicar λ", command=self._apply_lambda_from_panel).pack(side=tk.LEFT, padx=(12, 0))

    # ---------- Status ----------

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
        messagebox.showinfo("Fuente de imagen", "La fuente cambiará al reiniciar la adquisición.")

    def _prompt_lambda_change(self) -> None:
        val = simpledialog.askfloat("λ seleccionada", "Nueva λ (nm):",
                                    initialvalue=self.lambda_sel, minvalue=200, maxvalue=1100)
        if val is None:
            return
        self.lambda_sel = float(val)
        self.cfg.setdefault("graphics", {})["lambda_selected_nm"] = self.lambda_sel
        self.cursor_S.set_xdata([self.lambda_sel])
        self.lam_var.set(f"{self.lambda_sel:.1f}")
        self.var_status.set(f"λ_sel = {self.lambda_sel:.1f} nm")

    def _prompt_window_change(self) -> None:
        val = simpledialog.askfloat("Ventana (s)", "Segundos visibles:",
                                    initialvalue=self.window_s, minvalue=5, maxvalue=3600)
        if val is None:
            return
        self.window_s = float(val)
        self.win_var.set(str(int(self.window_s)))

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
            # 1) Guardar en cfg
            self.cfg.setdefault("imaging", {})["roi"] = new_roi
            # 2) Actualizar extractor en caliente (impacta preview + espectro)
            try:
                if self.acq._extractor is not None:
                    self.acq._extractor.roi = new_roi
            except Exception:
                pass
            # 3) Si la fuente es pantalla, actualizar capturador
            try:
                if hasattr(self.acq._img_source, "update_roi"):
                    self.acq._img_source.update_roi(new_roi)  # ScreenReader
            except Exception:
                pass

            self.var_status.set(f"ROI actualizado: {new_roi}")
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
        self.var_status.set(f"λ_sel = {self.lambda_sel:.1f} nm")

    def _toggle_grid(self) -> None:
        on = bool(self._grid_var.get())
        for ax in (self.ax_T, self.ax_A, self.ax_S):
            ax.grid(on, linestyle=":", linewidth=0.6)
        self.canvas.draw_idle()

    def _reset_view(self) -> None:
        for ax in (self.ax_T, self.ax_A, self.ax_S):
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()

    def _schedule_refresh(self) -> None:
        self._refresh()
        self._refresh_job = self.root.after(self.refresh_ms, self._schedule_refresh)

    def _update_blank_status(self) -> None:
        has_blank = self.acq._blank is not None
        mode = self.acq.get_plot_mode()
        if mode == "I" and not has_blank:
            self.var_status.set("Modo I(λ): fija 0%/100% o 'Capturar blanco' para ver A(λ)")

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

        self.l_T.set_data(tT, TT)
        self.ax_T.relim(); self.ax_T.autoscale_view()

        self.l_A.set_data(tA, AA)
        self.ax_A.relim(); self.ax_A.autoscale_view()

        # Espectro (A si hay, sino I)
        spec, lam = self.acq.latest_spectrum()
        mode = self.acq.get_plot_mode()

        if spec is None and mode == "I":
            I, lamI = self.acq.latest_intensity()
            if I is not None:
                spec, lam = I, lamI

        if spec is not None:
            lam_plot = lam if lam.size == spec.size else np.linspace(self.acq.lmin, self.acq.lmax, spec.size)
            self.l_S.set_data(lam_plot, spec)
            self.cursor_S.set_xdata([self.lambda_sel])
            self.ax_S.set_ylabel("A(λ)" if mode == "A" else "I(λ)")
            self.ax_S.relim(); self.ax_S.autoscale_view()

            # Panel de medición
            if mode == "A":
                A_sel = float(np.interp(self.lambda_sel, lam_plot, spec))
                self.A_var.set(f"{A_sel:.4f}")
                self.T_var.set(f"{100.0*(10**(-A_sel)):.2f} %")
                try:
                    k = float(self.k_var.get())
                except Exception:
                    k = 0.0
                self.C_var.set(f"{(A_sel/k):.6g}" if k > 0 else "—")
                Tlast = self.acq.latest_temp()
                if Tlast is not None:
                    self.var_last.set(f"T={Tlast:.2f} °C   A={A_sel:.4f}")
            else:
                self.A_var.set("—"); self.T_var.set("—"); self.C_var.set("—")

        self._update_blank_status()
        self.canvas.draw_idle()

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

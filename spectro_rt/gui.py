# spectro_rt/gui.py
# -*- coding: utf-8 -*-
"""
spectro-rt v0.3 (all-Python imaging)
- 3 subplots: T(t), A(λ_sel)(t), A(λ)
- Menú + Toolbar estilo app
- Preview en vivo centrado (cámara/pantalla) visible incluso antes de iniciar
- Pop-ups si falla Arduino o la fuente de imagen (sin crashear)
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

        # Preview frame (RGB) y espectro de preview (no se guarda)
        self._preview_rgb: Optional[np.ndarray] = None
        self._probe_line: Optional[np.ndarray] = None

        # Threads/sources
        self._threads: List[threading.Thread] = []
        self._stop_evt = threading.Event()
        self._arduino = None
        self._img_source = None
        self._extractor = None

    # ---------- Public API ----------

    def start_preview(self) -> None:
        """Inicia fuente de imagen SOLO para previsualización (y espectro de prueba)."""
        if self.preview_only or self.running or self.mode != "real":
            return

        # Lazy imports
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

        # Imaging config
        icfg = self.cfg.get("imaging", {})
        fps = float(icfg.get("fps", 8))
        roi = icfg.get("roi", {"x": 0, "y": 0, "w": 500, "h": 86})
        resize_w = int(icfg.get("resize_w", 500))
        gamma = float(icfg.get("gamma", 2.2222))
        bg = float(icfg.get("background_subtract", 300.0))
        min_floor = float(icfg.get("min_floor", 1e-3))
        log10 = bool(icfg.get("log10", True))
        baseline = icfg.get("baseline", {"mode": "ema", "alpha": 0.01})

        # Extractor
        try:
            self._extractor = SpectrumExtractor(
                roi=roi, resize_w=resize_w, gamma=gamma, bg_sub=bg, min_floor=min_floor,
                log10=log10, baseline_mode=baseline.get("mode", "ema"), alpha=baseline.get("alpha", 0.01)
            )
        except TypeError:
            self._extractor = SpectrumExtractor(
                roi=roi, resize_w=resize_w, gamma=gamma, bg_sub=bg, min_floor=min_floor, log10=log10
            )

        def on_frame(ts: float, frame_bgr: np.ndarray) -> None:
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

            # Espectro de prueba (NO se guarda en buffers)
            try:
                _, corrected = self._extractor.process_frame(frame_bgr)
                self._probe_line = corrected
            except Exception:
                pass

        # Fuente
        if source == "screen":
            self._img_source = ScreenReader(roi=roi, fps=fps, frame_cb=on_frame)
        else:
            cam_index = int(icfg.get("camera_index", 0))
            self._img_source = CameraReader(cam_index=cam_index, fps=fps, frame_cb=on_frame)

        # Arranca
        self._img_source.start()
        self.preview_only = True

    def start(self) -> None:
        if self.running:
            return
        # apaga preview-only si estaba activo
        if self.preview_only:
            self.stop()
        self._stop_evt.clear()
        self.running = True
        if self.mode == "real":
            self._start_real()
        else:
            self._start_sim()

    def stop(self) -> None:
        # detiene tanto preview como adquisición
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

    def latest_temp(self) -> Optional[float]:
        return self.temp.y[-1] if self.temp.y else None

    def latest_asel(self) -> Optional[float]:
        return self.a_sel.y[-1] if self.a_sel.y else None

    def latest_spectrum(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        spec, lam = self.spec.latest()
        return spec, (lam if lam is not None else self.lambdas)

    def latest_probe_spectrum(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        return self._probe_line, self.lambdas

    def latest_preview(self) -> Optional[np.ndarray]:
        return self._preview_rgb

    # ---------- Internals ----------

    def _start_sim(self) -> None:
        # Temperature thread: ramp + oscillation + noise
        def temp_thread():
            t0 = time.time()
            while not self._stop_evt.is_set():
                t = time.time() - t0
                T = 25 + 0.02 * t + 2.0 * np.sin(2*np.pi*t/30.0) + np.random.randn() * 0.05
                self.temp.push(time.time(), float(T))
                time.sleep(0.2)

        # Spectrum + synthetic preview thread
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
                # Synthetic preview (RGB)
                img = (np.tile(line, (h, 1)) * 255.0).astype(np.uint8)
                rgb = np.dstack([img, img, img])
                self._preview_rgb = rgb
                time.sleep(1.0)

        for fn in (temp_thread, spec_thread):
            th = threading.Thread(target=fn, daemon=True)
            th.start()
            self._threads.append(th)

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
                    f"""No se pudo abrir {port} @ {baud}.

La adquisición continuará sin T°.

Detalle: {e}"""
                )
            except Exception:
                pass
            print(f"[WARN] Arduino no disponible ({e}). Continuando sin T°.")
            self._arduino = None

        # Imaging config
        icfg = self.cfg.get("imaging", {})
        fps = float(icfg.get("fps", 8))
        roi = icfg.get("roi", {"x": 0, "y": 0, "w": 500, "h": 86})
        resize_w = int(icfg.get("resize_w", 500))
        gamma = float(icfg.get("gamma", 2.2222))
        bg = float(icfg.get("background_subtract", 300.0))
        min_floor = float(icfg.get("min_floor", 1e-3))
        log10 = bool(icfg.get("log10", True))
        baseline = icfg.get("baseline", {"mode": "ema", "alpha": 0.01})

        try:
            self._extractor = SpectrumExtractor(
                roi=roi, resize_w=resize_w, gamma=gamma, bg_sub=bg, min_floor=min_floor,
                log10=log10, baseline_mode=baseline.get("mode", "ema"), alpha=baseline.get("alpha", 0.01)
            )
        except TypeError:
            self._extractor = SpectrumExtractor(
                roi=roi, resize_w=resize_w, gamma=gamma, bg_sub=bg, min_floor=min_floor, log10=log10
            )

        def on_frame(ts: float, frame_bgr: np.ndarray) -> None:
            # Preview: BGR→RGB + ROI
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

            # Espectro (sí se guarda)
            _, corrected = self._extractor.process_frame(frame_bgr)
            A_sel = float(np.interp(self.lambda_sel, self.lambdas, corrected))
            self.spec.push(corrected, self.lambdas)
            self.a_sel.push(ts, A_sel)

        # Fuente
        if source == "screen":
            self._img_source = ScreenReader(roi=roi, fps=fps, frame_cb=on_frame)
        else:
            cam_index = int(icfg.get("camera_index", 0))
            self._img_source = CameraReader(cam_index=cam_index, fps=fps, frame_cb=on_frame)

        # Start imaging (con manejo de error)
        try:
            self._img_source.start()
        except Exception as e:
            try:
                messagebox.showerror(
                    "Fuente de imagen",
                    f"""No se pudo iniciar la fuente '{source}'.

Detalle: {e}"""
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
                        f"""No se pudo iniciar el preview.

Detalle: {e}"""
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
        # Label centrado
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
            # ancho de ventana menos márgenes, altura máx 220 px
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
            self.preview_label.configure(image=photo, text="")  # centrado por pack(anchor='center')
            self._tk_preview = photo  # evita GC
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
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
        # arrancar adquisición (si había preview, se reinicia)
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
        self.var_status.set(f"λ_sel = {self.lambda_sel:.1f} nm")

    def _prompt_window_change(self) -> None:
        val = simpledialog.askfloat("Ventana (s)", "Segundos visibles:",
                                    initialvalue=self.window_s, minvalue=5, maxvalue=3600)
        if val is None:
            return
        self.window_s = float(val)
        self.win_var.set(str(int(self.window_s)))

    def _prompt_roi_dialog(self) -> None:
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
            new_roi = {k: int(v.get()) for k, v in vars_.items()}
            self.cfg.setdefault("imaging", {})["roi"] = new_roi
            try:
                if self.acq._extractor is not None:
                    self.acq._extractor.roi = new_roi
            except Exception:
                pass
            self.var_status.set(f"ROI = {new_roi}")
            top.destroy()

        ttk.Button(btns, text="Aceptar", command=apply_and_close).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Cancelar", command=top.destroy).pack(side=tk.LEFT)

    def _show_serial_dialog(self) -> None:
        scfg = self.cfg.get("serial", {})
        top = tk.Toplevel(self.root)
        top.title("Puerto serie")
        top.resizable(False, False)
        frm = ttk.Frame(top, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)
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

    def _refresh(self) -> None:
        # preview primero
        self._update_preview()

        # series
        tT, TT = self.acq.temp.window(self.window_s)
        tA, AA = self.acq.a_sel.window(self.window_s)
        spec, lam = self.acq.latest_spectrum()
        if spec is None:
            # usa espectro del preview antes de iniciar
            p_spec, p_lam = self.acq.latest_probe_spectrum()
            if p_spec is not None:
                spec, lam = p_spec, p_lam

        if tT.size:
            tT = tT - tT[0]
        if tA.size:
            tA = tA - tA[0]

        self.l_T.set_data(tT, TT)
        self.ax_T.relim(); self.ax_T.autoscale_view()

        self.l_A.set_data(tA, AA)
        self.ax_A.relim(); self.ax_A.autoscale_view()

        if spec is not None:
            self.l_S.set_data(lam, spec)
            self.cursor_S.set_xdata([self.lambda_sel])
            self.ax_S.relim(); self.ax_S.autoscale_view()

        Tlast = self.acq.latest_temp()
        Alast = self.acq.latest_asel()
        if Tlast is not None and Alast is not None:
            self.var_last.set(f"T={Tlast:.2f} °C   A={Alast:.4f}")

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

# spectro_rt/gui.py
# GUI inicial: Tkinter + Matplotlib con dos gráficos (Temperatura y Absorbancia)
# Espera un "manager" con:
#   - start()/stop()
#   - get_plot_series(window_secs) -> {"T": (tT, TT), "A": (tA, AA)}
#   - export_csv(path)
#
# Uso típico:
#   from spectro_rt.controllers.acquisition_manager import AcquisitionManager
#   from spectro_rt.io.config_loader import load_config
#   cfg = load_config("config/config.default.yaml")
#   app = SpectroApp(cfg)
#   manager = AcquisitionManager(cfg)
#   app.attach_manager(manager)
#   manager.start()
#   app.run()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SpectroApp:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = tk.Tk()
        self.root.title("spectro-rt — Monitor en tiempo real (v0.1)")
        self.root.geometry("1000x650")
        self.manager = None

        # Intervalo de refresco (ms) configurable en config.yaml
        self.refresh_ms = int(self.cfg.get("graphics", {}).get("refresh_ms", 200))

        # Ventana de tiempo visible en los plots (segundos)
        self.buffer_secs = 120

        # Estado
        self.running = False

        # Construir UI
        self._build_toolbar()
        self._build_plots()
        self._build_statusbar()

        # Planificar primer refresco
        self._schedule_update()

        # Cierre limpio
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # === Inyección del manager (orquestador de adquisición) ===
    def attach_manager(self, manager):
        """Conecta el AcquisitionManager. No lo inicia/para (lo decide main)."""
        self.manager = manager

    # === Construcción de UI ===
    def _build_toolbar(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.btn_start = ttk.Button(top, text="Iniciar", command=self._on_start)
        self.btn_stop  = ttk.Button(top, text="Detener", command=self._on_stop, state=tk.DISABLED)
        self.btn_export_csv = ttk.Button(top, text="Exportar CSV", command=self._export_csv)
        self.btn_export_fig = ttk.Button(top, text="Exportar Gráfica", command=self._export_fig)
        self.btn_exit  = ttk.Button(top, text="Salir", command=self._on_close)

        for w in (self.btn_start, self.btn_stop, self.btn_export_csv, self.btn_export_fig, self.btn_exit):
            w.pack(side=tk.LEFT, padx=5)

        # Control simple para la ventana visible (seg)
        ttk.Label(top, text=" Ventana (s):").pack(side=tk.LEFT, padx=(12,4))
        self.spin_window = ttk.Spinbox(top, from_=10, to=600, width=6, command=self._on_change_window)
        self.spin_window.set(str(self.buffer_secs))
        self.spin_window.pack(side=tk.LEFT)

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

    def _build_plots(self):
        center = ttk.Frame(self.root)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.fig = Figure(figsize=(8,5), dpi=100)
        self.ax_T = self.fig.add_subplot(211)  # Temperatura vs tiempo
        self.ax_A = self.fig.add_subplot(212)  # Absorbancia vs tiempo

        self.ax_T.set_title("Temperatura vs tiempo")
        self.ax_T.set_xlabel("Tiempo (s)")
        self.ax_T.set_ylabel("T (°C)")
        self.ax_A.set_title("Absorbancia (λ seleccionada) vs tiempo")
        self.ax_A.set_xlabel("Tiempo (s)")
        self.ax_A.set_ylabel("A (u.a.)")

        self.line_T, = self.ax_T.plot([], [], lw=1.5)
        self.line_A, = self.ax_A.plot([], [], lw=1.5)

        self.ax_T.grid(True, alpha=0.3)
        self.ax_A.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_statusbar(self):
        bottom = ttk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)
        self.lbl_status = ttk.Label(bottom, text=f"Listo. Modo: {self.cfg.get('mode','sim')}")
        self.lbl_status.pack(side=tk.LEFT)
        self.lbl_values = ttk.Label(bottom, text="T=— °C   A=— u.a.")
        self.lbl_values.pack(side=tk.RIGHT)

    # === Handlers de botones ===
    def _on_start(self):
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._set_status("Adquisición iniciada.")

    def _on_stop(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self._set_status("Adquisición detenida.")

    def _export_csv(self):
        if not self.manager:
            messagebox.showwarning("Exportar CSV", "No hay datos/manager conectado.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")],
            title="Guardar datos sincronizados"
        )
        if not path:
            return
        try:
            self.manager.export_csv(path)
            messagebox.showinfo("Exportar CSV", f"Datos guardados en:\n{path}")
        except Exception as e:
            messagebox.showerror("Exportar CSV", f"Error: {e}")

    def _export_fig(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("SVG","*.svg")],
            title="Guardar figura"
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Exportar Gráfica", f"Figura guardada en:\n{path}")
        except Exception as e:
            messagebox.showerror("Exportar Gráfica", f"Error: {e}")

    def _on_change_window(self):
        try:
            val = int(self.spin_window.get())
            self.buffer_secs = max(10, min(600, val))
        except ValueError:
            pass

    # === Bucle de refresco ===
    def _schedule_update(self):
        self.root.after(self.refresh_ms, self._update_loop)

    def _update_loop(self):
        # reprogramar siguiente tick
        self._schedule_update()

        if not (self.manager and self.running):
            return

        try:
            series = self.manager.get_plot_series(window_secs=self.buffer_secs)
        except Exception as e:
            self._set_status(f"Error actualizando: {e}")
            return

        tT, TT = series.get("T", ([], []))
        tA, AA = series.get("A", ([], []))

        # Actualizamos líneas
        self.line_T.set_data(tT, TT)
        self.line_A.set_data(tA, AA)

        # Autoscale por ventana
        self._autoscale(self.ax_T, tT, TT, self.buffer_secs)
        self._autoscale(self.ax_A, tA, AA, self.buffer_secs)

        # Estado numérico (últimos valores)
        t_val = TT[-1] if TT else None
        a_val = AA[-1] if AA else None
        tstr = "—" if t_val is None else f"{t_val:.2f}"
        astr = "—" if a_val is None else f"{a_val:.3f}"
        self.lbl_values.config(text=f"T={tstr} °C   A={astr} u.a.")

        self.canvas.draw_idle()

    @staticmethod
    def _autoscale(ax, x, y, window_secs):
        if len(x) >= 2:
            ax.set_xlim(max(0, x[-1]-window_secs), x[-1] + 1e-6)
            if y:
                ymin, ymax = min(y), max(y)
                if ymin == ymax:
                    ymin -= 1; ymax += 1
                margin = 0.1*(ymax - ymin)
                ax.set_ylim(ymin - margin, ymax + margin)

    def _set_status(self, txt: str):
        self.lbl_status.config(text=txt)

    def _on_close(self):
        try:
            if self.manager:
                self.manager.stop()
        finally:
            self.root.destroy()

    def run(self):
        """Entra al loop principal de Tkinter."""
        self.root.mainloop()

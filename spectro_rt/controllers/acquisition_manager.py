# spectro_rt/controllers/acquisition_manager.py
import threading
import time
from collections import deque
from typing import Dict, Tuple, List
import csv

from spectro_rt.controllers.simulators.arduino_sim import ArduinoSim
from spectro_rt.controllers.simulators.imagej_sim import ImageJSim


class AcquisitionManager:
    """Orquesta la adquisición (por ahora solo simulación).
       Mantiene buffers y expone series listas para la GUI.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = cfg.get("mode", "sim")
        self.lock = threading.Lock()
        self.t0 = time.time()

        # Buffers con ventana limitada
        # guardamos (t_rel_s, valor)
        self.buff_T: deque = deque(maxlen=10_000)  # Temperatura
        self.buff_A: deque = deque(maxlen=10_000)  # Absorbancia (λ seleccionada)

        # Parámetros gráficos
        g = cfg.get("graphics", {})
        self.lambda_min = float(g.get("lambda_min", 400))
        self.lambda_max = float(g.get("lambda_max", 700))
        self.lambda_selected_nm = float(g.get("lambda_selected_nm", 520))

        # Simuladores / lectores reales
        self.arduino = None
        self.imagej = None
        self.running = False

        # (opcional) último espectro completo, por si luego pintamos A(λ)
        self.last_spectrum = None  # (wavelengths, A_vector)

    def start(self):
        """Arranca la adquisición (sim o real)."""
        if self.mode == "sim":
            self.running = True
            # Lanzar simuladores
            self.arduino = ArduinoSim(
                callback=self._on_arduino_row,
                period_s=0.2
            )
            self.imagej = ImageJSim(
                callback=self._on_image_row,
                period_s=1.0,
                n_pixels=500,
                lam_min=self.lambda_min,
                lam_max=self.lambda_max,
                lam_sel=self.lambda_selected_nm,
            )
            self.arduino.start()
            self.imagej.start()
        else:
            # Aquí luego iniciarás los hilos reales (serial/watchdog)
            self.running = True

    def stop(self):
        """Detiene todo limpiamente."""
        self.running = False
        if self.arduino:
            self.arduino.stop()
        if self.imagej:
            self.imagej.stop()

    # === Callbacks de entrada de datos ===
    def _on_arduino_row(self, ts_abs: float, temp_c: float):
        """Llamado por ArduinoSim/lector real con timestamp absoluto y temperatura."""
        t_rel = ts_abs - self.t0
        with self.lock:
            self.buff_T.append((t_rel, float(temp_c)))

    def _on_image_row(self, ts_abs: float, a_vector, wavelengths, a_at_sel: float):
        """Llamado por ImageJSim/lector real con espectro completo y A(λ_sel)."""
        t_rel = ts_abs - self.t0
        with self.lock:
            self.buff_A.append((t_rel, float(a_at_sel)))
            self.last_spectrum = (wavelengths, a_vector)

    # === Series listas para la GUI ===
    def get_plot_series(self, window_secs: float) -> Dict[str, Tuple[List[float], List[float]]]:
        t_now = time.time() - self.t0
        t_min = max(0.0, t_now - window_secs)
        with self.lock:
            tT = [t for (t, _) in self.buff_T if t >= t_min]
            TT = [v for (t, v) in self.buff_T if t >= t_min]
            tA = [t for (t, _) in self.buff_A if t >= t_min]
            AA = [v for (t, v) in self.buff_A if t >= t_min]
        return {"T": (tT, TT), "A": (tA, AA)}

    # === Exportación simple de CSV ===
    def export_csv(self, path: str):
        """Exporta t, T y A emparejando por tiempo (tolerancia 100 ms)."""
        with self.lock:
            T_list = list(self.buff_T)
            A_list = list(self.buff_A)

        rows = []
        i, j = 0, 0
        while i < len(T_list) and j < len(A_list):
            tT, TT = T_list[i]
            tA, AA = A_list[j]
            if abs(tT - tA) <= 0.1:  # 100 ms
                rows.append((tT, TT, AA))
                i += 1
                j += 1
            elif tT < tA:
                i += 1
            else:
                j += 1

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "TempC", "A_sel"])
            w.writerows(rows)

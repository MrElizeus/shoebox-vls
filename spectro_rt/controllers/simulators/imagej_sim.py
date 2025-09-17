# spectro_rt/controllers/simulators/imagej_sim.py
import threading
import time
import math
import numpy as np


class ImageJSim:
    """Simula espectros de 1×N con un pico gaussiano desplazándose lentamente."""

    def __init__(self, callback, period_s=1.0, n_pixels=500,
                 lam_min=400.0, lam_max=700.0, lam_sel=520.0):
        """
        callback(ts_abs, A_vector (np.ndarray), wavelengths (np.ndarray), A_at_sel (float))
        """
        self.callback = callback
        self.period_s = float(period_s)
        self.n_pixels = int(n_pixels)
        self.lam = np.linspace(float(lam_min), float(lam_max), self.n_pixels)
        self.lam_sel = float(lam_sel)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.t0 = time.time()
        self._stop.clear()
        if not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            ts = time.time()
            t = ts - self.t0

            # Pico gaussiano que se desplaza ±10 nm en ~2 min
            lam0 = 520 + 10 * math.sin(2 * math.pi * t / 120.0)
            sigma = 12.0
            Amax = 0.8 + 0.1 * math.sin(2 * math.pi * t / 45.0)
            A = Amax * np.exp(-0.5 * ((self.lam - lam0) / sigma) ** 2)

            # Baseline + ruido blanco suave
            A += 0.02 + 0.01 * np.random.randn(self.n_pixels)

            # A en λ seleccionada
            a_sel = float(np.interp(self.lam_sel, self.lam, A))

            self.callback(ts, A, self.lam, a_sel)
            time.sleep(self.period_s)

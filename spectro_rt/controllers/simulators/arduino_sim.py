# spectro_rt/controllers/simulators/arduino_sim.py
import threading
import time
import math
import random


class ArduinoSim:
    """Simula temperatura: rampa a ~45°C con oscilación y ruido."""

    def __init__(self, callback, period_s=0.2):
        """
        callback(ts_abs, temp_c)
        period_s: intervalo entre muestras
        """
        self.callback = callback
        self.period_s = float(period_s)
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

            # Modelo simple: aproximación sigmoide a 45°C + oscilación lenta + ruido
            base = 25 + 20 * (1 - math.exp(-t / 60.0))  # asintótico hacia 45°C
            osc = 0.6 * math.sin(2 * math.pi * t / 30.0)
            noise = random.uniform(-0.15, 0.15)
            temp_c = base + osc + noise

            self.callback(ts, temp_c)
            time.sleep(self.period_s)

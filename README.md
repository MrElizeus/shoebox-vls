# spectro-rt — Monitor en tiempo real para espectrofotómetro (v0.1)

Interfaz de escritorio en **Python (Tkinter + Matplotlib)** con **simuladores internos** para desarrollar y probar el pipeline de adquisición/visualización **sin hardware**. La app está pensada para integrarse con:

* **Arduino UNO** (temperatura y control de calentador) por puerto serie.
* **ImageJ** (macro que guarda espectros `.tif` y un log de tiempos).

Esta versión **v0.1** ya abre la GUI, corre simuladores y muestra **gráficos en tiempo real** de Temperatura y Absorbancia (a una λ seleccionada), y permite **exportar CSV** y **guardar la figura**.

---

## Tabla de contenidos

* [Estado actual](#estado-actual)
* [Arquitectura y estructura del repo](#arquitectura-y-estructura-del-repo)
* [Requisitos y compatibilidad](#requisitos-y-compatibilidad)
* [Instalación](#instalación)
* [Ejecución (modo simulación)](#ejecución-modo-simulación)
* [Interfaz de usuario](#interfaz-de-usuario)
* [Configuración (YAML)](#configuración-yaml)
* [Datos y formatos](#datos-y-formatos)
* [Roadmap / Próximos pasos](#roadmap--próximos-pasos)
* [Cómo integrar hardware real](#cómo-integrar-hardware-real)
* [Flujo de desarrollo](#flujo-de-desarrollo)
* [Solución de problemas](#solución-de-problemas)
* [Licencia](#licencia)
* [Changelog](#changelog)

---

## Estado actual

* ✅ **GUI** (Tkinter + Matplotlib) con 2 subplots:

  * Temperatura vs tiempo
  * Absorbancia (λ seleccionada) vs tiempo
* ✅ **Simuladores** internos:

  * ArduinoSim (temperatura con rampa + oscilación + ruido)
  * ImageJSim (espectros 1×500 con pico gaussiano y A(λ\_sel))
* ✅ **Exportación**:

  * CSV: `t_s, TempC, A_sel`
  * Figura: PNG/SVG
* ✅ **Gestor de adquisición**: orquesta hilos, buffers y expone series para la GUI
* ⛳ **Listo para extender** a: tercer subplot con espectro A(λ), lectores reales, sincronización avanzada, ajuste cinético, etc.

---

## Arquitectura y estructura del repo

```
spectro-rt/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ config/
│  ├─ config.default.yaml
│  └─ profiles/
│     ├─ sim.yaml
│     └─ lab.yaml
├─ data/
│  ├─ spectra/      # (ImageJ real guardará .tif aquí)
│  └─ logs/         # (img_log.tsv)
└─ spectro_rt/
   ├─ main.py       # Punto de entrada (CLI)
   ├─ gui.py        # Interfaz Tkinter + Matplotlib
   ├─ controllers/
   │  ├─ acquisition_manager.py
   │  └─ simulators/
   │     ├─ arduino_sim.py
   │     └─ imagej_sim.py
   └─ io/
      └─ config_loader.py
```

### Flujo (v0.1)

```
Simuladores (hilos) ──> AcquisitionManager (buffers) ──> GUI (after/refresh)
           |                                                 |
           └── export_csv() <────────────────────────────────┘
```

---

## Requisitos y compatibilidad

* **Python 3.8+** (probado con 3.13 en Windows; compatible con Ubuntu)
* **Tkinter** (en Linux: `sudo apt install python3-tk` si hace falta)
* Librerías (ver `requirements.txt`):

  * `numpy`, `matplotlib`, `Pillow`, `tifffile`, `watchdog`, `pyserial`, `scipy`, `pyyaml`

> **Multiplataforma:** Desarrolla en **Windows** y corre en **Linux**. Usa rutas con `pathlib`/`os.path.join` y configura el **puerto serie** por YAML (Windows `COM3`, Linux `/dev/ttyACM0`).

---

## Instalación

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Si PowerShell bloquea el script:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Linux (Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
sudo apt-get install -y python3-tk
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Ejecución (modo simulación)

```bash
# Activar venv y ejecutar:
python -m spectro_rt.main --sim
```

Pasos en la GUI:

1. **Iniciar** → empiezan a llegar datos simulados
2. **Exportar CSV** / **Exportar Gráfica**
3. **Detener** / **Salir**

---

## Interfaz de usuario

* **Botones superiores**

  * **Iniciar / Detener**: controla la adquisición (simulada por ahora)
  * **Exportar CSV**: guarda datos sincronizados (T y A) en el intervalo activo
  * **Exportar Gráfica**: guarda la figura tal como se ve (PNG/SVG)
  * **Salir**
  * **Ventana (s)**: ancho temporal visible en los plots (p. ej. 120 s)

* **Gráficos**

  * **T vs t**: curva de temperatura simulada (actualización \~200 ms)
  * **A vs t** (λ seleccionada): absorbancia a una longitud de onda (actualización \~1 s)

* **Barra inferior**

  * Estado (mensajes)
  * Últimos valores: `T=… °C`, `A=… u.a.`

---

## Configuración (YAML)

`config/config.default.yaml` (y perfiles en `config/profiles/`):

```yaml
mode: sim            # sim | real
graphics:
  refresh_ms: 200
  lambda_min: 400
  lambda_max: 700
  lambda_selected_nm: 520
paths:
  spectra_dir: "data/spectra"
  img_log: "data/logs/img_log.tsv"
processing:
  median_window: 5
  use_led_sync: true
kinetics:
  enabled: false
```

Para **Windows + Arduino real**, en `profiles/lab.yaml`:

```yaml
mode: real
serial:
  port: "COM3"
  baud: 115200
```

Para **Linux + Arduino real**:

```yaml
mode: real
serial:
  port: "/dev/ttyACM0"
  baud: 115200
```

> En v0.1 `mode: real` aún no lanza lectores reales (solo estructura). Próximo paso.

---

## Datos y formatos

### Exportación CSV (v0.1)

* Columnas: `t_s, TempC, A_sel`
* Fusión simple por **tiempo** (tolerancia 100 ms) de las dos series

### Simuladores

* **ArduinoSim**: emite `TempC` cada \~200 ms (rampa → \~45 °C con oscilación + ruido).
* **ImageJSim**: espectro 1×500 cada \~1 s (pico gaussiano que se desplaza ±10 nm). Devuelve también **A(λ\_sel)** por interpolación lineal.

> Cuando conectemos hardware real:
>
> * Arduino: CSV tipo `millis,tempC,heater,beacon`.
> * ImageJ: `.tif` en `data/spectra/` y `img_log.tsv` con `i`, `t_img_ms`, `led_mean`.

---

## Roadmap / Próximos pasos

1. **Tercer subplot**: espectro A(λ) actual y cursor en λ\_sel
2. **Selector de λ** en la GUI
3. **Lectores reales**:

   * `arduino_reader.py` (pyserial)
   * `imagej_reader.py` (watchdog + tifffile + parser de `img_log.tsv`)
4. **Sincronización**:

   * Ajuste `t_ard ≈ a·t_img + b` con balizas LED (`beacon` ↔ `led_mean`)
5. **Cálculos**:

   * Beer–Lambert (si llegan intensidades)
   * `C(t) = A/(ε·L)`
   * Ajuste cinético (Euler/Heun con Arrhenius) como opción en vivo
6. **Reportes**:

   * Exportación de espectro vs tiempo (matriz) y resúmenes
7. **Robustez**:

   * Manejo de errores de puerto serie / permisos
   * Persistencia de sesiones / logs

---

## Cómo integrar hardware real

### Arduino (lectura serie)

Crear `spectro_rt/controllers/arduino_reader.py` (sugerido):

```python
import serial, time, threading

class ArduinoReader:
    def __init__(self, port, baud, callback, period_s=0.2):
        self.port = port; self.baud = baud
        self.callback = callback
        self.period_s = period_s
        self._stop = threading.Event()

    def start(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        self._stop.clear()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()
        try: self.ser.close()
        except: pass

    def _run(self):
        while not self._stop.is_set():
            line = self.ser.readline().decode(errors="ignore").strip()
            # parsea: millis,tempC,heater,beacon
            try:
                ms,temp,heater,beacon = line.split(",")
                ts = time.time()  # o usa ms relativo
                self.callback(ts, float(temp))
            except:
                pass
```

Conectar en `AcquisitionManager.start()` si `mode=="real"`.

### ImageJ (watchdog + `.tif` + `img_log.tsv`)

Crear `spectro_rt/controllers/imagej_reader.py` que:

* Monitorea `paths.spectra_dir` (carpeta) con **watchdog**
* Al ver un `spectrum<i>.tif`: lo abre (tifffile), extrae vector 1×N (float)
* Busca `t_img_ms` y `led_mean` en `paths.img_log`
* Emite callback: `(ts_abs, A_vector, wavelengths, A_at_sel)`
  (usar `lambda_min/max` y `N` para mapear pixel→λ lineal)

---

## Flujo de desarrollo

**Ramas recomendadas**:

* `main`: estable
* `feature/gui-spectrum-plot`
* `feature/arduino-reader`
* `feature/imagej-reader`
* `feature/sync-beacons`
* `feature/kinetics`

**Commits**: mensajes cortos y claros (en español o inglés, pero consistentes).
**Tags**: versiones `v0.1`, `v0.2`, …

---

## Solución de problemas

* **PowerShell no activa venv**
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

* **`ImportError: cannot import name 'load_config'`**
  Verifica el contenido de `spectro_rt/io/config_loader.py` y que existan:

  * `spectro_rt/__init__.py`
  * `spectro_rt/io/__init__.py`
    Limpia caché:

  ```powershell
  Get-ChildItem -Recurse -Force -Include __pycache__ | Remove-Item -Recurse -Force
  ```

* **Tkinter en Linux**
  Instala: `sudo apt install python3-tk`

* **Permisos del puerto serie (Linux)**
  `sudo usermod -a -G dialout $USER` (cerrar sesión/entrar)

* **Matplotlib no dibuja fluido**
  Reduce `refresh_ms` o limita la ventana visible (p. ej. 60–120 s)

---

## Licencia

> (Añadir archivo `LICENSE`. Sugerencia: **MIT** para un proyecto abierto y flexible.)

---

## Changelog

* **v0.1**

  * GUI inicial (Tkinter + Matplotlib)
  * Simuladores de Arduino e ImageJ
  * Exportación CSV y figura
  * Estructura modular lista para integrar hardware real

---
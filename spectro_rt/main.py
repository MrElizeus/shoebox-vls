# spectro_rt/main.py
"""
Punto de entrada de la app (GUI + adquisición).

Uso típico:
  # modo simulación (recomendado para primera prueba)
  python -m spectro_rt.main --sim

  # con perfil específico
  python -m spectro_rt.main --config config/config.default.yaml --sim
"""

import argparse
from spectro_rt.io.config_loader import load_config
from spectro_rt.controllers.acquisition_manager import AcquisitionManager
from spectro_rt.gui import SpectroApp


def _parse_args():
    ap = argparse.ArgumentParser(description="spectro-rt — monitor en tiempo real")
    ap.add_argument(
        "--config",
        default="config/config.default.yaml",
        help="Ruta al archivo de configuración YAML",
    )
    ap.add_argument(
        "--sim",
        action="store_true",
        help="Usar simuladores internos (sin Arduino/ImageJ)",
    )
    return ap.parse_args()


def main():
    args = _parse_args()

    # 1) Cargar configuración (con defaults seguros)
    cfg = load_config(args.config)
    if args.sim:
        cfg["mode"] = "sim"

    # 2) Crear GUI
    app = SpectroApp(cfg)

    # 3) Crear manager de adquisición y conectarlo a la GUI
    manager = AcquisitionManager(cfg)
    app.attach_manager(manager)

    # 4) Arrancar adquisición (sim o real según cfg)
    manager.start()

    # 5) Entrar al loop de Tkinter y detener al salir
    try:
        app.run()
    finally:
        manager.stop()


if __name__ == "__main__":
    main()

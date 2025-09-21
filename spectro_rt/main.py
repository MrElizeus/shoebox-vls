# spectro_rt/main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

try:
    from .gui import SpectroRTApp  # type: ignore
except Exception:
    from spectro_rt.gui import SpectroRTApp  # type: ignore


def load_config() -> dict:
    # Carga config por defecto si existe; si no, crea una mínima.
    base = {
        "mode": "sim",
        "graphics": {"refresh_ms": 200, "window_s": 120, "lambda_min": 400, "lambda_max": 700, "lambda_selected_nm": 520},
        "paths": {"export_dir": "data/exports"},
        "serial": {"port": "COM3", "baud": 115200},
        "imaging": {
            "source": "camera",        # "camera" | "screen"
            "camera_index": 0,
            "fps": 12,
            "preview_ms": 66,          # ~15 fps
            "resize_w": 500,
            "roi": {"x": 0, "y": 0, "w": 500, "h": 86},
            "gamma": 2.2222,
            "background_subtract": 300.0,
            "min_floor": 1e-3,
            "log10": True,
            "baseline": {"mode": "ema", "alpha": 0.01},
        },
        "blank": {"domain": "log", "N": 200, "autoN": 0},
        "chemistry": {"k_factor": 1.0},
    }
    path = Path("config/config.default.yaml")
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                disk = yaml.safe_load(f) or {}
            # merge simple (shallow)
            for k, v in disk.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    base[k].update(v)
                else:
                    base[k] = v
        except Exception:
            pass
    return base


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="spectro-rt launcher")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true", help="modo simulación")
    g.add_argument("--real", action="store_true", help="modo real")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    if args.sim:
        cfg["mode"] = "sim"
    if args.real:
        cfg["mode"] = "real"

    app = SpectroRTApp(cfg)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

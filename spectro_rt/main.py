# spectro_rt/main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    from .gui import SpectroRTApp  # cuando se ejecuta como paquete: python -m spectro_rt.main
except Exception:
    from gui import SpectroRTApp  # fallback si se ejecuta dentro del directorio del paquete


DEFAULT_CFG = {
    "mode": "sim",
    "graphics": {
        "refresh_ms": 200,
        "lambda_min": 400,
        "lambda_max": 700,
        "lambda_selected_nm": 520,
        "window_s": 120.0,
    },
    "imaging": {
        "source": "camera",         # camera | screen
        "camera_index": 0,
        "fps": 10,
        "roi": {"x": 100, "y": 200, "w": 800, "h": 80},
        "resize_w": 500,
        "gamma": 2.2222,
        "background_subtract": 300.0,
        "min_floor": 1e-3,
        "log10": True,
    },
    "serial": {"port": "COM3", "baud": 115200},
    "paths": {"export_dir": "data/exports"},
}


def _merge_dicts(dst: dict, src: dict) -> dict:
    out = dict(dst)
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict:
    cfg = dict(DEFAULT_CFG)
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg = _merge_dicts(cfg, file_cfg)
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="spectro-rt launcher")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true", help="Fuerza modo simulación")
    g.add_argument("--real", action="store_true", help="Fuerza modo real")
    p.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("config/config.default.yaml"),
        help="Ruta a YAML de configuración",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.sim:
        cfg["mode"] = "sim"
    elif args.real:
        cfg["mode"] = "real"
    app = SpectroRTApp(cfg)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

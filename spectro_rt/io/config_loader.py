# spectro_rt/io/config_loader.py
import os
import yaml

_DEFAULT = {
    "mode": "sim",
    "graphics": {
        "refresh_ms": 200,
        "lambda_min": 400.0,
        "lambda_max": 700.0,
        "lambda_selected_nm": 520.0,
    },
    "paths": {
        "spectra_dir": "data/spectra",
        "img_log": "data/logs/img_log.tsv",
    },
    "processing": {
        "median_window": 5,
        "use_led_sync": True,
    },
    "kinetics": {
        "enabled": False
    },
}

def load_config(path: str) -> dict:
    """Carga YAML de config y hace merge con defaults seguros."""
    cfg = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    merged = dict(_DEFAULT)
    for k, v in cfg.items():
        if isinstance(v, dict) and k in _DEFAULT:
            nv = dict(_DEFAULT[k])
            nv.update(v)
            merged[k] = nv
        else:
            merged[k] = v
    return merged

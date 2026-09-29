"""
โหลดไฟล์ตั้งค่า YAML ให้เป็น dict ที่ใช้ได้ทุกโมดูล

ใช้งาน:
    from convrain.config import load_config
    cfg = load_config("config/default.yaml")
    cfg["detect"]["thresholds_k"]
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# ตำแหน่งไฟล์ตั้งค่า default ใน repository
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    """รวม dict แบบซ้อนชั้น: ค่าใน override ทับค่าใน base เฉพาะ key ที่ระบุ"""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """
    อ่าน config/default.yaml ก่อน แล้วทับด้วยไฟล์ที่ระบุ (ถ้ามี) และ overrides (ถ้ามี)
    ทำให้ไฟล์ของแต่ละการทดลองระบุเฉพาะค่าที่ต่างจาก default ได้
    """
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        with open(path, encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    return cfg

"""ฟังก์ชันใช้ร่วมกันของสคริปต์ (หาไฟล์ตามเวลา, โหลด rain LUT + ค่าคงที่ของส่วนทำนายเวลา)"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STAMP_RE = re.compile(r"(\d{8}T\d{4}Z)")


def parse_stamp(name: str) -> datetime | None:
    """ดึงเวลาจากชื่อไฟล์รูปแบบ ..._YYYYMMDDTHHMMZ..."""
    m = STAMP_RE.search(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc)


def files_by_time(directory: str | Path, pattern: str) -> list[tuple[datetime, Path]]:
    out = []
    for p in Path(directory).glob(pattern):
        t = parse_stamp(p.name)
        if t is not None:
            out.append((t, p))
    return sorted(out)


TIMING_PARAMS = "timing_params.json"
OUTLOOK_PARAMS = "outlook_params.json"
RAINMAP_PARAMS = "rainmap_params.json"


def load_models(models_dir: str | Path | None, cfg: dict):
    """
    โฟลเดอร์โมเดล (ทางเลือก):
        rain_lut.npz        จาก build_lut.py (rain rate ของขั้น 6)
        timing_params.json  จาก calibrate_timing.py (ค่าคงที่ของส่วนทำนายเวลา; ไม่มี = ค่าที่ติดมากับโค้ด)
        outlook_params.json จาก calibrate_outlook.py (คำเตือน 3 ระดับ + เวลาพร้อมความมั่นใจ)
        rainmap_params.json แผนที่ฝนตอนนี้ที่ปรับเทียบเอง (ทางเลือก)
        meta.json           ค่าเสริม เช่น warn_threshold
    ตั้ง cfg["timing"] ตามไฟล์ที่มี แล้วคืน (lut, meta)
    """
    from convrain.intensity import RainRateLUT

    if models_dir is None:
        return None, {}
    d = Path(models_dir)
    if (d / TIMING_PARAMS).exists():
        cfg["timing"]["params_file"] = str(d / TIMING_PARAMS)
    if (d / OUTLOOK_PARAMS).exists():
        cfg["timing"]["outlook_file"] = str(d / OUTLOOK_PARAMS)
    if (d / RAINMAP_PARAMS).exists():
        cfg["timing"]["rainmap_file"] = str(d / RAINMAP_PARAMS)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    if meta.get("warn_threshold") is not None:
        cfg["timing"]["warn_threshold"] = meta["warn_threshold"]
    lut = RainRateLUT.load(d / "rain_lut.npz") if (d / "rain_lut.npz").exists() else None
    return lut, meta

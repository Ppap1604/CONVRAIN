"""ฟังก์ชันใช้ร่วมกันของสคริปต์ (หาไฟล์ตามเวลา, โหลดโมเดล)"""
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


def load_models(models_dir: str | Path | None):
    """
    โหลดโมเดลทั้งหมดจากโฟลเดอร์ที่ train.py สร้าง
    คืน (onset, peak, stop, lut, meta) ตัวที่ไม่มีไฟล์จะเป็น None
    """
    from convrain.intensity import RainRateLUT
    from convrain.models.hazard import HazardModel

    if models_dir is None:
        return None, None, None, None, {}
    d = Path(models_dir)

    def opt(name):
        p = d / f"model_{name}.joblib"
        return HazardModel.load(p) if p.exists() else None

    lut = RainRateLUT.load(d / "rain_lut.npz") if (d / "rain_lut.npz").exists() else None
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return opt("onset"), opt("peak"), opt("stop"), lut, meta

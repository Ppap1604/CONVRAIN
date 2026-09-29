"""ตัวช่วยสำหรับ unit test: สร้าง Scan จำลองจากภาพ BT อย่างเดียว"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convrain.config import load_config  # noqa: E402
from convrain.ingest import Scan, prepare_geometry  # noqa: E402


def blob(shape, r, c, radius, core_bt, bg=296.0):
    """เมฆรูป Gaussian: BT ต่ำสุด core_bt ที่ (r, c)"""
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    g = np.exp(-(((rr - r) ** 2 + (cc - c) ** 2) / radius**2))
    return (bg - (bg - core_bt) * g).astype(np.float32)


def make_scan(bt13: np.ndarray, t: datetime, cfg: dict) -> Scan:
    ny, nx = bt13.shape
    rows, cols = np.mgrid[0:ny, 0:nx].astype(np.float32)
    lat = 15.0 - rows * 0.02
    lon = 100.0 + cols * 0.02
    # band อื่นเท่ากับ B13 → BTD = 0 (ไม่มี cirrus ถูกตัด) ยกเว้น B08 ให้ BTD_WV สมจริงเล็กน้อย
    bt = {b: bt13.copy() for b in ["B10", "B11", "B13", "B14", "B15", "B16"]}
    bt["B08"] = bt13 - 20.0
    scan = Scan(time=t, bt=bt, lat=lat.astype(np.float32), lon=lon.astype(np.float32))
    return prepare_geometry(scan, cfg)


@pytest.fixture
def cfg():
    return load_config(overrides={"domain": {"lat_min": 12.0, "lat_max": 15.0, "lon_min": 100.0, "lon_max": 103.0}})


@pytest.fixture
def t0():
    return datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc)

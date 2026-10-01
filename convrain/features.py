"""
ขั้น 4: Feature extraction (interest fields)  (เอกสาร หัวข้อ 7)

สรุปสถานะและแนวโน้มของเมฆแต่ละก้อนเป็น feature vector ขนาดเล็ก (~30 ค่า)

หลักการ
    - คำนวณเฉพาะ pixel ใน mask ของ object (ไม่ใช่ทั้งภาพ)
    - แนวโน้มคิดแบบ Lagrangian: เทียบกับ object เดียวกันใน scan ก่อนหน้า (จาก track.history)
    - ใช้ทั้งค่าดิบที่ 10/20/30 นาที และค่าที่ผ่าน Kalman filter

อ้างอิงกลุ่ม feature: Mecikalski & Bedka (2006), Sieglaff et al. (2011),
Zhuge & Zou (2018), Li et al. (2024), Senf & Deneke (2017)
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from .ingest import Scan
from .motion import MotionField
from .track import Track

# feature ที่ส่งเข้าโมเดล (ขั้น 5) ค่า default — ปรับได้ตอน train
MODEL_FEATURES = [
    # ความสูงยอดเมฆ
    "min_bt", "cold_bt", "mean_bt", "below_0c", "min_since_below_0c", "cold_bt_kf",
    # การโตของ updraft
    "cooling_rate_kf", "cooling_rate_std", "d_cold_bt_10", "d_cold_bt_20", "d_cold_bt_30",
    "log_area", "area_growth_kf", "d_log_area_30",
    # ความสูงเทียบ troposphere
    "btd_wv_cold", "btd_wv_max", "btd_73_cold", "btd_co2_cold",
    "d_btd_wv_10", "d_btd_wv_30", "d_btd_co2_30", "d_btd_73_10", "d_btd_73_30",
    # phase ยอดเมฆ
    "ttd_cold", "ttd_max", "d_ttd_10", "d_ttd_30",
    # ความหนาเมฆ
    "btd_sw_cold", "d_btd_sw_10", "d_btd_sw_30",
    # บริบท
    "age_min", "lst_sin", "lst_cos", "land_frac", "elev_mean", "sat_zenith",
    # track ขั้นก่อนหน้า
    "min_cold_bt_track",
]


# =====================================================================
# [ขั้น 4.1] สถิติของ pixel ใน object
# =====================================================================
def _cold_mean(values: np.ndarray, cold_idx: np.ndarray) -> float:
    v = values[cold_idx]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    return float(v.max()) if v.size else float("nan")


# =====================================================================
# [ขั้น 4.2] แนวโน้ม Lagrangian จาก history ของ track
# =====================================================================
def _value_at(history: list[dict], target: datetime, key: str, tol_min: float = 3.0) -> float:
    """หาค่าของ feature ณ เวลา target (±tol) ใน history ของ track เดียวกัน"""
    for h in reversed(history):
        if abs((h["time"] - target).total_seconds()) <= tol_min * 60:
            return h.get(key, float("nan"))
        if h["time"] < target - timedelta(minutes=tol_min):
            break
    return float("nan")


def _trend(history: list[dict], now: datetime, current: float, key: str, window_min: int) -> float:
    """current − ค่าเมื่อ window_min นาทีก่อน (NaN ถ้า track อายุไม่ถึง)"""
    past = _value_at(history, now - timedelta(minutes=window_min), key)
    return current - past if np.isfinite(past) else float("nan")


# =====================================================================
# [ขั้น 4] ฟังก์ชันหลัก: feature ของ track 1 ก้อน ณ scan ปัจจุบัน
# =====================================================================
def compute_features(track: Track, scan: Scan, motion: MotionField | None, cfg: dict) -> dict:
    """
    คืน dict ของ feature ณ เวลา scan และต่อท้ายไว้ใน track.history
    (history เป็นแหล่งข้อมูลของแนวโน้มใน scan ถัดไป และเป็นแถวของตาราง object-scan)
    """
    obj = track.obj
    rows, cols = obj.rows, obj.cols
    now = scan.time
    ir = scan.ir[rows, cols]

    # pixel ที่เย็นที่สุด 10% ของ object = "แกน" ของเมฆ
    n_cold = max(1, int(math.ceil(cfg["features"]["cold_fraction"] * ir.size)))
    cold_idx = np.argpartition(ir, n_cold - 1)[:n_cold]

    btd_wv = scan.btd_wv[rows, cols]
    btd_73 = scan.btd_73[rows, cols]
    btd_sw = scan.btd_sw[rows, cols]
    ttd = scan.ttd[rows, cols]
    btd_co2 = scan.btd_co2[rows, cols]

    f: dict = {
        "time": now,
        "object_id": track.track_id,
        "parent_id": track.parent_id,
        "lat": obj.lat,
        "lon": obj.lon,
        "area_km2": obj.area_km2,
        "area_px": obj.area_px,
        # ---- ความสูงยอดเมฆ ----
        "min_bt": obj.min_bt,
        "cold_bt": obj.cold_mean_bt,
        "mean_bt": obj.mean_bt,
        "below_0c": float(obj.min_bt < 273.15),
        "min_since_below_0c": ((now - track.first_below_0c).total_seconds() / 60.0
                               if track.first_below_0c else float("nan")),
        # ---- ความสูงเทียบ troposphere / phase / ความหนา ----
        "btd_wv_cold": _cold_mean(btd_wv, cold_idx),
        "btd_wv_max": _finite_max(btd_wv),
        "btd_73_cold": _cold_mean(btd_73, cold_idx),
        "btd_co2_cold": _cold_mean(btd_co2, cold_idx),
        "ttd_cold": _cold_mean(ttd, cold_idx),
        "ttd_max": _finite_max(ttd),
        "btd_sw_cold": _cold_mean(btd_sw, cold_idx),
        # ---- ค่าที่ผ่าน Kalman (หน่วย: ต่อ 10 นาที) ----
        "cold_bt_kf": track.kf_bt.value,
        "cooling_rate_kf": track.kf_bt.rate,
        "cooling_rate_std": math.sqrt(max(track.kf_bt.rate_var, 0.0)),
        "log_area": math.log(max(obj.area_km2, 1e-3)),
        "area_growth_kf": track.kf_area.rate,
        # ---- track ----
        "age_min": track.age_min,
        "n_scans": track.n_scans,
        "has_parent": float(track.parent_id is not None),
        "min_cold_bt_track": track.min_cold_bt,
        # ---- [CST / Adler-Negri] แกนพาความร้อน & Parallax Ground Coordinates ----
        "core_lat": getattr(obj, "core_lat", obj.lat),
        "core_lon": getattr(obj, "core_lon", obj.lon),
        "lat_pc": getattr(obj, "lat_pc", obj.lat),
        "lon_pc": getattr(obj, "lon_pc", obj.lon),
        "core_lat_pc": getattr(obj, "core_lat_pc", getattr(obj, "core_lat", obj.lat)),
        "core_lon_pc": getattr(obj, "core_lon_pc", getattr(obj, "core_lon", obj.lon)),
        "cth_km": getattr(obj, "cth_km", 0.0),
        "parallax_shift_km": getattr(obj, "parallax_shift_km", 0.0),
        # ---- จุลฟิสิกส์เมฆและสัญญาณพายุรุนแรง ----
        "has_overshooting": float(getattr(obj, "has_overshooting", False)),
        "ot_pixel_count": float(getattr(obj, "ot_pixel_count", 0)),
        "core_thermal_depth": getattr(obj, "core_thermal_depth", 0.0),
        "core_gradient_k_per_km": getattr(obj, "core_gradient_k_per_km", 0.0),
    }

    # ---- [ขั้น 4.2] แนวโน้มตามเวลา (Lagrangian) ----
    for w in cfg["features"]["trend_windows_min"]:
        f[f"d_cold_bt_{w}"] = _trend(track.history, now, f["cold_bt"], "cold_bt", w)
        f[f"d_btd_wv_{w}"] = _trend(track.history, now, f["btd_wv_cold"], "btd_wv_cold", w)
        f[f"d_ttd_{w}"] = _trend(track.history, now, f["ttd_cold"], "ttd_cold", w)
        f[f"d_btd_co2_{w}"] = _trend(track.history, now, f["btd_co2_cold"], "btd_co2_cold", w)
        f[f"d_log_area_{w}"] = _trend(track.history, now, f["log_area"], "log_area", w)
        f[f"d_btd_sw_{w}"] = _trend(track.history, now, f["btd_sw_cold"], "btd_sw_cold", w)
        f[f"d_btd_73_{w}"] = _trend(track.history, now, f["btd_73_cold"], "btd_73_cold", w)

    # ---- [ขั้น 4.3] บริบท: เวลาท้องถิ่น, พื้นผิว, มุมมอง ----
    lst_hours = (now.hour + now.minute / 60.0 + obj.lon / 15.0) % 24.0  # local solar time
    f["lst_hours"] = lst_hours
    f["lst_sin"] = math.sin(2 * math.pi * lst_hours / 24.0)
    f["lst_cos"] = math.cos(2 * math.pi * lst_hours / 24.0)
    f["land_frac"] = float(scan.land_mask[rows, cols].mean()) if scan.land_mask is not None else float("nan")
    f["elev_mean"] = float(scan.elevation_m[rows, cols].mean()) if scan.elevation_m is not None else float("nan")
    f["sat_zenith"] = float(scan.sat_zenith[rows, cols].mean()) if scan.sat_zenith is not None else float("nan")

    # ---- ทางเลือกกลางวัน: reflectance 0.64 µm (NaN ตอนกลางคืนหรือไม่มีข้อมูล) ----
    is_day = 7.0 <= lst_hours <= 17.0
    f["is_day"] = float(is_day)
    f["refl_b03_cold"] = (_cold_mean(scan.refl["B03"][rows, cols], cold_idx)
                          if is_day and "B03" in scan.refl else float("nan"))

    # ---- motion ของ object (ใช้ใน output และ extrapolation) ----
    if motion is not None:
        f["speed_ms"], f["direction_deg"] = motion.speed_direction(rows, cols)
    else:
        f["speed_ms"], f["direction_deg"] = float("nan"), float("nan")

    track.history.append(f)
    return f

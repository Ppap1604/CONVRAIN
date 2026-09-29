"""
Parallax correction  (เอกสาร หัวข้อ 9.4)

ดาวเทียมค้างฟ้ามองยอดเมฆสูงแบบเฉียง ทำให้เมฆปรากฏไกลจากจุดใต้ดาวเทียมกว่าตำแหน่งจริง
ประเทศไทยมองจาก Himawari (140.7°E) ด้วยมุม ~45–55° ยอดเมฆสูง 12 km จึงเลื่อนได้ ~13 km

ขั้นตอน
    1. ประเมินความสูงยอดเมฆ h จาก BT(10.4) เทียบ profile อุณหภูมิเขตร้อน (ไม่ต้องใช้ NWP)
    2. หาจุดบนเส้นสายตาดาวเทียม→ตำแหน่งปรากฏ ที่อยู่สูง h จากผิวโลก = ตำแหน่งจริงของยอดเมฆ
       (โลกทรงกลม, ค่าคลาดเคลื่อนเล็กมากเทียบกับขนาด pixel; ดู Bieliński 2020 สำหรับแบบ ellipsoid)
"""
from __future__ import annotations

import numpy as np

from .ingest import EARTH_RADIUS_KM, Scan

# =====================================================================
# [9.4 ข้อ 1] profile อุณหภูมิบรรยากาศเขตร้อนมาตรฐาน (AFGL tropical, ประมาณ)
# ความสูง (km) → อุณหภูมิ (K) ใช้ช่วง 0–17 km ที่อุณหภูมิลดลงตามความสูง
# ควรแทนด้วย profile ภูมิอากาศรายเดือนของประเทศไทยเมื่อมีข้อมูล
# =====================================================================
TROPICAL_PROFILE_KM = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], float)
TROPICAL_PROFILE_K = np.array([300.0, 294.0, 288.0, 284.0, 277.0, 270.0, 264.0, 257.0, 250.0,
                               244.0, 237.0, 230.0, 224.0, 217.0, 210.0, 204.0, 197.0, 195.0])


def cloud_top_height_km(bt: np.ndarray | float,
                        profile_km: np.ndarray = TROPICAL_PROFILE_KM,
                        profile_k: np.ndarray = TROPICAL_PROFILE_K) -> np.ndarray | float:
    """
    แปลง BT (K) เป็นความสูงยอดเมฆ (km) ด้วยการ interpolate profile
    BT อุ่นกว่าผิวพื้น → 0 km, เย็นกว่า tropopause → ความสูง tropopause
    หมายเหตุ: เมฆบางที่ไม่เป็น blackbody จะได้ความสูงต่ำกว่าจริง (เอกสาร หัวข้อ 12 ข้อ 6)
    """
    # np.interp ต้องการแกน x เพิ่มขึ้น จึงกลับลำดับ (อุณหภูมิลดลงตามความสูง)
    return np.interp(bt, profile_k[::-1], profile_km[::-1])


# =====================================================================
# [9.4 ข้อ 2] เรขาคณิต parallax
# =====================================================================
def parallax_correct(lat: np.ndarray | float, lon: np.ndarray | float, height_km: np.ndarray | float,
                     sub_lon: float = 140.7, orbit_radius_km: float = 42164.0):
    """
    คืนตำแหน่งจริง (lat, lon) ของยอดเมฆที่ปรากฏในภาพดาวเทียมที่ (lat, lon) และสูง height_km

    วิธี: S = ตำแหน่งดาวเทียม, G = ตำแหน่งปรากฏบนผิวโลก
          หา t ที่ |S + t(G − S)| = R + h (จุดตัดแรกจากฝั่งดาวเทียม)
    """
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    sl = np.deg2rad(sub_lon)
    R = EARTH_RADIUS_KM
    S = np.array([orbit_radius_km * np.cos(sl), orbit_radius_km * np.sin(sl), 0.0])
    G = np.stack([R * np.cos(lat_r) * np.cos(lon_r),
                  R * np.cos(lat_r) * np.sin(lon_r),
                  R * np.sin(lat_r)], axis=-1)
    D = G - S
    a = np.sum(D * D, axis=-1)
    b = 2.0 * np.sum(D * S, axis=-1)
    c = np.dot(S, S) - (R + np.asarray(height_km, float)) ** 2
    t = (-b - np.sqrt(np.maximum(b * b - 4 * a * c, 0.0))) / (2 * a)
    X = S + t[..., None] * D if np.ndim(t) else S + t * D
    lat_true = np.rad2deg(np.arcsin(X[..., 2] / np.linalg.norm(X, axis=-1)))
    lon_true = np.rad2deg(np.arctan2(X[..., 1], X[..., 0]))
    return lat_true, lon_true


def shift_km(lat, lon, lat_true, lon_true) -> np.ndarray | float:
    """ระยะที่เลื่อน (km) ระหว่างตำแหน่งปรากฏกับตำแหน่งจริง (haversine)"""
    p1, p2 = np.deg2rad(lat), np.deg2rad(lat_true)
    dl = np.deg2rad(np.asarray(lon_true) - np.asarray(lon))
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


# =====================================================================
# แปลงการเลื่อนเป็นหน่วย pixel (ใช้ย้าย mask ของ object ลงพื้นตอนจับคู่กับเรดาร์)
# =====================================================================
def pixel_offset(scan: Scan, r0: float, c0: float, lat_true: float, lon_true: float) -> tuple[float, float]:
    """
    แก้สมการเชิงเส้นด้วย Jacobian ของ (lat, lon) ต่อ (row, col) ที่ centroid
    คืน (drow, dcol) ที่ต้องเลื่อน pixel ของ object ให้ไปอยู่ตำแหน่งจริง
    """
    ny, nx = scan.lat.shape
    r = int(np.clip(round(r0), 1, ny - 2))
    c = int(np.clip(round(c0), 1, nx - 2))
    J = np.array([
        [(scan.lat[r + 1, c] - scan.lat[r - 1, c]) / 2, (scan.lat[r, c + 1] - scan.lat[r, c - 1]) / 2],
        [(scan.lon[r + 1, c] - scan.lon[r - 1, c]) / 2, (scan.lon[r, c + 1] - scan.lon[r, c - 1]) / 2],
    ])
    d = np.array([lat_true - scan.lat[r, c], lon_true - scan.lon[r, c]])
    try:
        drow, dcol = np.linalg.solve(J, d)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    return float(drow), float(dcol)


def correct_object(scan: Scan, lat: float, lon: float, cold_bt: float, r0: float, c0: float,
                   cfg: dict) -> dict:
    """
    แก้ parallax ของ object เดียว: ใช้ BT ของแกนเย็นประเมินความสูง
    คืน dict: lat/lon ที่แก้แล้ว, ความสูง, ระยะเลื่อน, offset หน่วย pixel
    """
    sat = cfg["satellite"]
    h = float(cloud_top_height_km(cold_bt))
    if not cfg["parallax"]["enabled"]:
        return {"lat": lat, "lon": lon, "cth_km": h, "shift_km": 0.0, "drow": 0.0, "dcol": 0.0}
    lat_t, lon_t = parallax_correct(lat, lon, h, sat["sub_lon"], sat["orbit_radius_km"])
    lat_t, lon_t = float(lat_t), float(lon_t)
    drow, dcol = pixel_offset(scan, r0, c0, lat_t, lon_t)
    return {"lat": lat_t, "lon": lon_t, "cth_km": h, "shift_km": float(shift_km(lat, lon, lat_t, lon_t)),
            "drow": drow, "dcol": dcol}

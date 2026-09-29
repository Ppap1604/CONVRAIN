"""
การสร้าง label จาก ground truth  (เอกสาร หัวข้อ 10.1–10.2)

ใช้เฉพาะตอน train / validate เท่านั้น ระบบตอน inference ใช้ดาวเทียมอย่างเดียว

ขั้นตอน
    1. regrid เรดาร์ลง grid ของดาวเทียม (regrid_to_satellite)
    2. ระหว่าง replay ทีละ scan: ย้าย mask ของ object ไปตำแหน่งจริง (parallax)
       ขยายขอบ buffer แล้วหา max dBZ ในแต่ละ frame เรดาร์ (object_radar_stats)
       pipeline บันทึกค่าเหล่านี้ลงตาราง object-scan
    3. สรุปต่อ track: t_onset, t_peak, t_stop, censor_time (build_track_labels)
    4. เติม label กลับเข้าไปในตาราง object-scan (attach_labels) → ใช้ train hazard model

นิยาม onset: dBZ ≥ 35 ครั้งแรก (Mecikalski & Bedka 2006)
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy import ndimage


# =====================================================================
# [10.1] เรดาร์ → grid ของดาวเทียม
# =====================================================================
def regrid_to_satellite(radar_lat: np.ndarray, radar_lon: np.ndarray, radar_values: np.ndarray,
                        sat_lat: np.ndarray, sat_lon: np.ndarray, max_dist_km: float = 3.0) -> np.ndarray:
    """
    nearest-neighbour จาก grid เรดาร์ (หรือจุด) ไปยัง pixel ของดาวเทียม
    pixel ที่ไกลจากจุดเรดาร์เกิน max_dist_km (นอกรัศมีเรดาร์) = NaN
    ควรแก้ clutter / beam blockage ก่อนเรียกฟังก์ชันนี้ (หัวข้อ 14, Phase 0.4)
    """
    from scipy.spatial import cKDTree

    def xyz(lat, lon):
        la, lo = np.deg2rad(lat.ravel()), np.deg2rad(lon.ravel())
        return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]) * 6371.0

    ok = np.isfinite(radar_values.ravel())
    tree = cKDTree(xyz(radar_lat, radar_lon)[ok])
    dist, idx = tree.query(xyz(sat_lat, sat_lon), distance_upper_bound=max_dist_km)
    vals = np.full(dist.shape, np.nan, dtype=np.float32)
    hit = np.isfinite(dist)
    vals[hit] = radar_values.ravel()[ok][idx[hit]]
    return vals.reshape(sat_lat.shape)


def dbz_to_rainrate(dbz: np.ndarray, a: float = 300.0, b: float = 1.4) -> np.ndarray:
    """
    Z-R: Z = a·R^b → R = (Z/a)^(1/b)  [mm/h]
    ค่า default (300, 1.4) คือความสัมพันธ์ convective ที่ใช้ทั่วไป ควรปรับด้วย gauge ของไทย
    """
    z = np.power(10.0, np.asarray(dbz, float) / 10.0)
    return np.power(z / a, 1.0 / b)


# =====================================================================
# [10.2 ข้อ 1–2] สถิติเรดาร์ในพื้นที่ของ object (หลังแก้ parallax)
# =====================================================================
def shifted_mask_pixels(rows: np.ndarray, cols: np.ndarray, drow: float, dcol: float,
                        shape: tuple[int, int], buffer_px: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """ย้าย pixel ของ object ไปตำแหน่งจริงบนพื้น แล้วขยายขอบ buffer_px pixel"""
    ny, nx = shape
    r = np.rint(rows + drow).astype(int)
    c = np.rint(cols + dcol).astype(int)
    ok = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
    r, c = r[ok], c[ok]
    if r.size == 0 or buffer_px <= 0:
        return r, c
    r0, c0 = max(r.min() - buffer_px, 0), max(c.min() - buffer_px, 0)
    r1, c1 = min(r.max() + buffer_px + 1, ny), min(c.max() + buffer_px + 1, nx)
    m = np.zeros((r1 - r0, c1 - c0), bool)
    m[r - r0, c - c0] = True
    m = ndimage.binary_dilation(m, iterations=buffer_px)
    rr, cc = np.nonzero(m)
    return rr + r0, cc + c0


def object_radar_stats(rows, cols, drow, dcol, frames: list[tuple[datetime, np.ndarray]],
                       onset_dbz: float, buffer_px: int = 1) -> dict:
    """
    frames: เรดาร์ทุก frame ในช่วง (scan ก่อนหน้า, scan นี้] ที่ regrid แล้ว
    คืน max dBZ ในช่วง, เวลา frame แรกที่ ≥ onset_dbz, และสัดส่วน pixel ที่มีข้อมูลเรดาร์
    """
    if not frames:
        return {"radar_max_dbz": np.nan, "radar_onset_time": pd.NaT, "radar_coverage": 0.0}
    rr, cc = shifted_mask_pixels(rows, cols, drow, dcol, frames[0][1].shape, buffer_px)
    best, first, cov = -np.inf, pd.NaT, 0.0
    for t, dbz in sorted(frames, key=lambda x: x[0]):
        v = dbz[rr, cc]
        fin = np.isfinite(v)
        cov = max(cov, float(fin.mean()) if v.size else 0.0)
        if not fin.any():
            continue
        m = float(v[fin].max())
        best = max(best, m)
        if pd.isna(first) and m >= onset_dbz:
            first = pd.Timestamp(t)
    return {"radar_max_dbz": best if np.isfinite(best) else np.nan,
            "radar_onset_time": first, "radar_coverage": cov}


def lut_pairs(bt: np.ndarray, btd: np.ndarray, rows, cols, drow, dcol, rain_rate: np.ndarray):
    """
    คู่ (BT, BTD, rain rate) ของ pixel ใน object สำหรับสร้าง RainRateLUT (หัวข้อ 16, Phase 5)
    ค่าดาวเทียมอ่านที่ตำแหน่งปรากฏ ค่าเรดาร์อ่านที่ตำแหน่งจริง (เลื่อน parallax)
    """
    ny, nx = rain_rate.shape
    r2 = np.clip(np.rint(rows + drow).astype(int), 0, ny - 1)
    c2 = np.clip(np.rint(cols + dcol).astype(int), 0, nx - 1)
    rr = rain_rate[r2, c2]
    ok = np.isfinite(rr)
    return bt[rows, cols][ok], btd[rows, cols][ok], rr[ok]


# =====================================================================
# [10.2 ข้อ 2–4] สรุป label ต่อ track
# =====================================================================
def build_track_labels(ts: pd.DataFrame, tracks: pd.DataFrame | None, cfg: dict,
                       min_coverage: float = 0.5) -> pd.DataFrame:
    """
    ts     : ตาราง object-scan ที่มี radar_max_dbz, radar_onset_time, radar_coverage
    tracks : ตาราง track ที่ปิดแล้ว (object_id, end_time, end_reason) จาก pipeline

    คืนตารางต่อ track: t_onset, t_peak, t_stop, censor_time, onset_at_birth, radar_ok
        t_onset : frame เรดาร์แรกที่ ≥ 35 dBZ
        t_peak  : scan ที่ radar_max_dbz สูงสุด (เฉพาะ track ที่มี onset)
        t_stop  : scan แรกหลัง peak ที่ dBZ < stop_dbz ต่อเนื่อง stop_consecutive_scans scan
        censor  : เวลาสุดท้ายที่ติดตามได้ ถ้า track จบด้วย merge / ออกนอกโดเมน / ข้อมูลขาด
                  / หมดข้อมูล (ไม่รู้ว่าหลังจากนั้นจะตกหรือไม่)
    track ที่ onset ตั้งแต่เกิด (onset_at_birth) ไม่มีช่วงก่อนฝนให้เรียน จึงถูกตัดจากชุด onset
    """
    lab = cfg["labels"]
    ts = ts.copy()
    ts["time"] = pd.to_datetime(ts["time"], utc=True)
    ts["radar_onset_time"] = pd.to_datetime(ts["radar_onset_time"], utc=True)
    out = []
    for oid, g in ts.sort_values("time").groupby("object_id", sort=False):
        birth = g["time"].iloc[0]
        dbz = g["radar_max_dbz"].values
        radar_ok = bool(np.nanmean(g["radar_coverage"].values) >= min_coverage) if "radar_coverage" in g else True
        onset = g["radar_onset_time"].dropna().min() if g["radar_onset_time"].notna().any() else pd.NaT
        t_peak = t_stop = pd.NaT
        if pd.notna(onset) and np.isfinite(dbz).any():
            ip = int(np.nanargmax(dbz))
            t_peak = g["time"].iloc[ip]
            below = np.nan_to_num(dbz[ip + 1:], nan=np.inf) < lab["stop_dbz"]
            n = int(lab["stop_consecutive_scans"])
            for j in range(len(below) - n + 1):
                if below[j:j + n].all():
                    t_stop = g["time"].iloc[ip + 1 + j]
                    break
        out.append({
            "object_id": oid, "birth_time": birth, "last_time": g["time"].iloc[-1],
            "t_onset": onset, "t_peak": t_peak, "t_stop": t_stop,
            "onset_at_birth": bool(pd.notna(onset) and onset <= birth),
            "radar_ok": radar_ok,
        })
    labels = pd.DataFrame(out)
    for col in ("birth_time", "last_time", "t_onset", "t_peak", "t_stop"):
        labels[col] = pd.to_datetime(labels[col], utc=True)
    m = pd.Series(False, index=labels.index)
    if tracks is not None and len(tracks):
        censored = tracks.loc[tracks["end_reason"].isin(["merged", "left_domain", "data_gap", "end_of_data"]),
                              "object_id"]
        m = labels["object_id"].isin(censored)
    labels["censor_time"] = labels["last_time"].where(m)  # NaT ถ้า track จบเองตามธรรมชาติ
    return labels


def attach_labels(ts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    เติม t_onset / t_peak / t_stop / censor_time ลงในตาราง object-scan
    ตัด track ที่ไม่มีเรดาร์ครอบคลุม (label ไม่น่าเชื่อถือ) ออก
    """
    keep = labels[labels["radar_ok"]]
    cols = ["object_id", "t_onset", "t_peak", "t_stop", "censor_time", "onset_at_birth"]
    df = ts.merge(keep[cols], on="object_id", how="inner")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df

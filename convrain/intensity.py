"""
ขั้น 6: ความรุนแรง และสถานะ lifecycle  (เอกสาร หัวข้อ 9.1–9.3)

ทำเฉพาะ object ที่ผ่านขั้น 5 หรือกำลังให้ฝนอยู่แล้ว (ลด compute)

9.1 Rain rate: lookup table 2 มิติ RR = LUT[BT(10.4), BTD(6.2−10.4)]
    - สร้างเองจากคู่ข้อมูล AHI (แก้ parallax แล้ว) กับเรดาร์ไทย/GPM ด้วย build_lut.py
    - ก่อน calibrate ใช้ prior = GOES auto-estimator (Vicente et al. 1998) ซึ่งใช้ BT อย่างเดียว
      R = 1.1183e11 · exp(−3.6382e−2 · T^1.2)  [mm/h, T หน่วย K]
      ค่านี้ calibrate สำหรับสหรัฐฯ ใช้เป็นจุดเริ่มต้นเท่านั้น
9.2–9.3 สถานะ: developing → raining → peak → decaying (→ dissipated เมื่อ track ปิด)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .ingest import Scan
from .track import Track


# =====================================================================
# [9.1] prior: GOES auto-estimator
# =====================================================================
def auto_estimator_mmh(bt: np.ndarray | float) -> np.ndarray:
    bt = np.asarray(bt, dtype=float)
    rr = 1.1183e11 * np.exp(-3.6382e-2 * np.power(np.clip(bt, 150.0, 330.0), 1.2))
    return np.clip(rr, 0.0, 150.0)


# =====================================================================
# [9.1] lookup table 2 มิติ
# =====================================================================
class RainRateLUT:
    def __init__(self, bt_edges=None, btd_edges=None, min_count: int = 20):
        self.bt_edges = np.arange(180.0, 302.5, 2.5) if bt_edges is None else np.asarray(bt_edges)
        self.btd_edges = np.arange(-46.0, 8.0, 2.0) if btd_edges is None else np.asarray(btd_edges)
        self.min_count = min_count
        self.table: np.ndarray | None = None      # (n_bt, n_btd)
        self.table_1d: np.ndarray | None = None   # สำรองเมื่อ BTD เป็น NaN

    @property
    def calibrated(self) -> bool:
        return self.table is not None

    def fit(self, bt: np.ndarray, btd: np.ndarray, rr: np.ndarray, quantile: float = 0.5) -> "RainRateLUT":
        """
        bt, btd, rr: อาร์เรย์ 1 มิติของ pixel ที่จับคู่กับเรดาร์/gauge แล้ว
        ค่าใน bin = quantile ของ rain rate (มัธยฐานเป็นค่า default)
        bin ที่ข้อมูลน้อยกว่า min_count เติมด้วยตาราง 1 มิติตาม BT แล้วด้วย prior
        """
        ok = np.isfinite(bt) & np.isfinite(rr)
        bt, btd, rr = bt[ok], btd[ok], np.clip(rr[ok], 0, None)
        nb, nd = len(self.bt_edges) - 1, len(self.btd_edges) - 1
        ib = np.clip(np.digitize(bt, self.bt_edges) - 1, 0, nb - 1)
        centers = 0.5 * (self.bt_edges[:-1] + self.bt_edges[1:])

        # ตาราง 1 มิติตาม BT
        t1 = np.full(nb, np.nan)
        for i in range(nb):
            sel = rr[ib == i]
            if sel.size >= self.min_count:
                t1[i] = np.quantile(sel, quantile)
        t1 = np.where(np.isfinite(t1), t1, auto_estimator_mmh(centers))
        self.table_1d = t1

        # ตาราง 2 มิติ
        good = np.isfinite(btd)
        idd = np.clip(np.digitize(btd[good], self.btd_edges) - 1, 0, nd - 1)
        t2 = np.full((nb, nd), np.nan)
        flat = ib[good] * nd + idd
        order = np.argsort(flat)
        flat_s, rr_s = flat[order], rr[good][order]
        uniq, start, counts = np.unique(flat_s, return_index=True, return_counts=True)
        for u, s, cnt in zip(uniq, start, counts):
            if cnt >= self.min_count:
                t2.flat[u] = np.quantile(rr_s[s:s + cnt], quantile)
        self.table = np.where(np.isfinite(t2), t2, t1[:, None])
        return self

    def predict(self, bt: np.ndarray, btd: np.ndarray | None = None) -> np.ndarray:
        bt = np.asarray(bt, dtype=float)
        if self.table is None:
            return auto_estimator_mmh(bt)
        nb, nd = self.table.shape
        ib = np.clip(np.digitize(bt, self.bt_edges) - 1, 0, nb - 1)
        if btd is None:
            return self.table_1d[ib]
        btd = np.asarray(btd, dtype=float)
        idd = np.clip(np.digitize(np.nan_to_num(btd, nan=-99.0), self.btd_edges) - 1, 0, nd - 1)
        return np.where(np.isfinite(btd), self.table[ib, idd], self.table_1d[ib])

    def bt_threshold(self, rate_mmh: float) -> float:
        """BT ที่อุ่นที่สุดที่ยังให้ rain rate ≥ rate_mmh (ใช้ประเมินเวลาหยุดตก)"""
        centers = 0.5 * (self.bt_edges[:-1] + self.bt_edges[1:])
        curve = self.table_1d if self.table_1d is not None else auto_estimator_mmh(centers)
        wet = centers[curve >= rate_mmh]
        return float(wet.max()) if wet.size else float(centers.min())

    def save(self, path: str | Path) -> None:
        np.savez(path, bt_edges=self.bt_edges, btd_edges=self.btd_edges,
                 table=self.table if self.table is not None else np.array([]),
                 table_1d=self.table_1d if self.table_1d is not None else np.array([]),
                 min_count=self.min_count)

    @staticmethod
    def load(path: str | Path) -> "RainRateLUT":
        z = np.load(path)
        lut = RainRateLUT(z["bt_edges"], z["btd_edges"], int(z["min_count"]))
        lut.table = z["table"] if z["table"].size else None
        lut.table_1d = z["table_1d"] if z["table_1d"].size else None
        return lut


# =====================================================================
# [9.1] rain rate ของ object
# =====================================================================
def object_rain(track: Track, scan: Scan, lut: RainRateLUT, cfg: dict) -> dict:
    rows, cols = track.obj.rows, track.obj.cols
    rr = lut.predict(scan.ir[rows, cols], scan.btd_wv[rows, cols])
    thr = cfg["intensity"]["rain_threshold_mmh"]
    wet = rr >= thr
    area = scan.pixel_area_km2[rows, cols] if scan.pixel_area_km2 is not None else np.full(rr.shape, 4.0)
    return {
        "rain_rate_max_mmh": float(rr.max()) if rr.size else 0.0,
        "rain_rate_mean_mmh": float(rr[wet].mean()) if wet.any() else 0.0,
        "rain_area_km2": float(area[wet].sum()),
    }


# =====================================================================
# [9.2–9.3] อัปเดตสถานะ lifecycle (ค่าเหนียว: เดินหน้าอย่างเดียว)
# =====================================================================
def update_status(track: Track, feats: dict, rain: dict, time: datetime, cfg: dict,
                  onset_eta: datetime | None = None) -> str:
    """
    developing → raining : rain rate จาก LUT ≥ เกณฑ์ หรือเลยเวลา onset ที่โมเดลคาดไว้
    raining → peak       : cooling rate ใกล้ 0 และ BT แกนเย็นอยู่ใกล้ค่าต่ำสุดของ track
    raining/peak → decaying : ยอดเมฆอุ่นขึ้นเร็วกว่าเกณฑ์ หรืออุ่นกว่าค่าต่ำสุดเกิน 3 K
    decaying → raining   : กลับมาเย็นลงเร็วและยังมีฝน (เซลล์ใหม่ก่อตัวซ้อน)
    """
    c = cfg["intensity"]
    rate = feats["cooling_rate_kf"]          # ติดลบ = เย็นลง (updraft ยังแรง)
    raining_now = rain["rain_rate_max_mmh"] >= c["rain_threshold_mmh"]
    s = track.status

    if s == "developing":
        if raining_now or (onset_eta is not None and time >= onset_eta):
            s, track.rain_started = "raining", time
    elif s == "raining":
        since = (time - track.rain_started).total_seconds() / 60.0 if track.rain_started else 0.0
        if rate > c["decay_rate_k"]:
            s = "decaying"
        elif since >= 10 and abs(rate) < c["peak_rate_tol_k"] and feats["cold_bt"] <= track.min_cold_bt + 1.0:
            s, track.peak_time = "peak", time
    elif s == "peak":
        if rate > c["decay_rate_k"] or feats["cold_bt"] - track.min_cold_bt > 3.0:
            s = "decaying"
    elif s == "decaying":
        # กลับมาแรงอีกครั้ง (เช่น รวมกับเซลล์ใหม่): ยอดเมฆเย็นลงเร็วและยังมีฝน
        if rate < -c["decay_rate_k"] and raining_now:
            s = "raining"
    track.status = s
    return s


def stop_eta_fallback(feats: dict, lut: RainRateLUT, time: datetime, cfg: dict) -> datetime | None:
    """
    ใช้เมื่อยังไม่มีโมเดล stop: extrapolate การอุ่นขึ้นของแกนเย็นไปหา BT ที่ rain rate ต่ำกว่าเกณฑ์
    (หัวข้อ 9.3 — ค่านี้หยาบ ควรแทนด้วย hazard model ของ stop เมื่อ train แล้ว)
    """
    rate = feats["cooling_rate_kf"]
    if not np.isfinite(rate) or rate <= 0.2:
        return None
    bt_dry = lut.bt_threshold(cfg["intensity"]["rain_threshold_mmh"])
    if feats["cold_bt_kf"] >= bt_dry:
        return None
    minutes = max(0.0, (bt_dry - feats["cold_bt_kf"]) / rate * 10.0)
    return time + timedelta(minutes=float(min(minutes, 180.0)))

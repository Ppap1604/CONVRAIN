"""
ข้อมูลจำลอง (synthetic) สำหรับทดสอบ pipeline ตั้งแต่ต้นจนจบ โดยยังไม่มีข้อมูลจริง

ไม่ใช่แบบจำลองฟิสิกส์ของเมฆ ใช้เพื่อตรวจว่าโค้ดทุกขั้นทำงานร่วมกันได้ถูกต้อง
(detect → track → features → label → train → evaluate → output)
ผลความแม่นยำบนข้อมูลจำลองจึงไม่ได้บอกความแม่นยำกับข้อมูลจริง

สิ่งที่จำลอง
    - เมฆหลายก้อน เกิดต่างเวลา เคลื่อนที่ตามลม เย็นลง → คงที่ → อุ่นขึ้น (lifecycle)
    - ก้อนที่ให้ฝน: onset เมื่อ BT แกนเย็นต่ำกว่าค่าสุ่ม (~243–255 K) + lag สุ่ม
      บางก้อนเป็น warm rain (onset ก่อนยอดเมฆเย็นจัด) ตามข้อจำกัดในหัวข้อ 12 ข้อ 1
    - ก้อนที่ไม่ให้ฝน: หยุดเย็นที่ ~250–268 K แล้วสลาย
    - BTD ของแต่ละ band สัมพันธ์กับ BT และ glaciation
    - ภาพดาวเทียมเลื่อนตาม parallax ส่วนเรดาร์อยู่ตำแหน่งจริงบนพื้น
    - เรดาร์ทุก 5 นาที
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .ingest import Scan
from .labels import dbz_to_rainrate
from .parallax import cloud_top_height_km, parallax_correct

BG_BT = 296.0


@dataclass
class SynthCloud:
    cid: int
    birth_min: float
    r0: float           # ตำแหน่งจริงบนพื้น (pixel) ตอนเกิด
    c0: float
    u: float            # pixel ต่อ 10 นาที
    v: float
    raining: bool
    cool_rate: float    # K ต่อ 10 นาที
    t_start: float
    t_min: float
    hold_min: float
    warm_rate: float
    radius0: float
    growth: float       # pixel ต่อ 10 นาที
    onset_bt: float
    onset_lag_min: float

    def core_bt(self, age: float) -> float | None:
        if age < 0:
            return None
        t_cool = (self.t_start - self.t_min) / self.cool_rate * 10.0
        if age <= t_cool:
            return self.t_start - self.cool_rate * age / 10.0
        if age <= t_cool + self.hold_min:
            return self.t_min
        bt = self.t_min + self.warm_rate * (age - t_cool - self.hold_min) / 10.0
        return None if bt > 286.0 else bt

    def onset_age(self) -> float | None:
        if not self.raining:
            return None
        bt_hit = max(self.onset_bt, self.t_min)
        return (self.t_start - bt_hit) / self.cool_rate * 10.0 + self.onset_lag_min

    def max_dbz(self, age: float) -> float:
        bt = self.core_bt(age)
        if bt is None:
            return 0.0
        if not self.raining:
            return float(np.clip(10 + (self.t_start - bt) * 0.5, 0, 28))
        on = self.onset_age()
        t_cool = (self.t_start - self.t_min) / self.cool_rate * 10.0
        if age < on:
            return float(np.clip(10 + (age / max(on, 1)) * 18, 0, 30))
        peak_end = t_cool + self.hold_min + 10
        if age <= peak_end:
            return float(min(52.0, 36 + (age - on) * 0.6))
        return float(max(0.0, 52.0 - (age - peak_end) * 0.7))

    def radius(self, age: float) -> float:
        return float(min(self.radius0 + self.growth * age / 10.0, 16.0))

    def pos(self, age: float) -> tuple[float, float]:
        return self.r0 + self.v * age / 10.0, self.c0 + self.u * age / 10.0


class SyntheticDay:
    """สร้างข้อมูล 1 วัน: scan ทุก 10 นาที พร้อมเรดาร์ทุก 5 นาที"""

    def __init__(self, seed: int, start: datetime, n_scans: int = 30, ny: int = 160, nx: int = 160,
                 lat0: float = 15.5, lon0: float = 99.5, dlat: float = 0.02, dlon: float = 0.02,
                 n_clouds: tuple[int, int] = (14, 20), sub_lon: float = 140.7):
        self.rng = np.random.default_rng(seed)
        self.start = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        self.n_scans, self.ny, self.nx = n_scans, ny, nx
        rows, cols = np.mgrid[0:ny, 0:nx].astype(np.float32)
        self.lat = (lat0 - rows * dlat).astype(np.float32)
        self.lon = (lon0 + cols * dlon).astype(np.float32)
        self.lat0, self.lon0, self.dlat, self.dlon = lat0, lon0, dlat, dlon
        self.sub_lon = sub_lon
        self.rows, self.cols = rows, cols
        self.clouds = self._make_clouds(n_clouds)

    def _make_clouds(self, n_range) -> list[SynthCloud]:
        r = self.rng
        n = int(r.integers(*n_range))
        dur = self.n_scans * 10
        wu, wv = r.uniform(-2.5, 2.5), r.uniform(-2.0, 2.0)   # ลมของวัน (pixel/10 นาที)
        clouds = []
        for i in range(n):
            raining = r.random() < 0.55
            warm_rain = raining and r.random() < 0.2
            clouds.append(SynthCloud(
                cid=i, birth_min=float(r.uniform(0, dur - 70)),
                r0=float(r.uniform(25, self.ny - 25)), c0=float(r.uniform(25, self.nx - 25)),
                u=float(wu + r.normal(0, 0.3)), v=float(wv + r.normal(0, 0.3)), raining=raining,
                cool_rate=float(r.uniform(4.0, 9.0) if raining else r.uniform(2.0, 5.0)),
                t_start=float(r.uniform(282, 287)),
                t_min=float(r.uniform(198, 225) if raining else r.uniform(250, 268)),
                hold_min=float(r.uniform(10, 50)), warm_rate=float(r.uniform(2.5, 6.0)),
                radius0=float(r.uniform(2.5, 4.0)), growth=float(r.uniform(0.6, 1.6)),
                onset_bt=float(r.uniform(258, 268) if warm_rain else r.uniform(240, 252)),
                onset_lag_min=float(r.uniform(0, 12)),
            ))
        return clouds

    # -----------------------------------------------------------------
    def _to_px(self, lat, lon):
        return (self.lat0 - lat) / self.dlat, (lon - self.lon0) / self.dlon

    def _to_ll(self, r, c):
        return self.lat0 - r * self.dlat, self.lon0 + c * self.dlon

    def _apparent(self, r, c, bt):
        """ตำแหน่งที่ดาวเทียมเห็น (เลื่อนออกจากจุดใต้ดาวเทียมตาม parallax) จากตำแหน่งจริง"""
        h = float(cloud_top_height_km(bt))
        lat_t, lon_t = self._to_ll(r, c)
        lat_a, lon_a = lat_t, lon_t
        for _ in range(4):  # fixed-point: หา apparent ที่แก้ parallax แล้วได้ตำแหน่งจริง
            la, lo = parallax_correct(lat_a, lon_a, h, self.sub_lon)
            lat_a, lon_a = lat_a + (lat_t - float(la)), lon_a + (lon_t - float(lo))
        return self._to_px(lat_a, lon_a)

    def _render(self, minutes: float):
        bt = np.full((self.ny, self.nx), BG_BT, np.float32)
        dbz = np.zeros((self.ny, self.nx), np.float32)
        for cl in self.clouds:
            age = minutes - cl.birth_min
            core = cl.core_bt(age)
            if core is None:
                continue
            rad = cl.radius(age)
            r, c = cl.pos(age)
            ra, ca = self._apparent(r, c, core)
            g = np.exp(-(((self.rows - ra) ** 2 + (self.cols - ca) ** 2) / rad**2))
            bt = np.minimum(bt, BG_BT - (BG_BT - core) * g)
            dmax = cl.max_dbz(age)
            if dmax > 0:
                gr = np.exp(-(((self.rows - r) ** 2 + (self.cols - c) ** 2) / (0.6 * rad) ** 2))
                dbz = np.maximum(dbz, dmax * gr)
        return bt, dbz

    def _bands(self, bt: np.ndarray) -> dict[str, np.ndarray]:
        r = self.rng
        noise = lambda: r.normal(0, 0.3, bt.shape).astype(np.float32)  # noqa: E731
        bt13 = bt + noise()
        s = np.clip((BG_BT - bt) / (BG_BT - 195.0), 0, 1) ** 1.5
        glac = 1.0 / (1.0 + np.exp(-(258.0 - bt) / 4.0))
        btd_sw = 0.5 + 2.0 * (1 - s)
        b14 = bt13 - 0.3
        b15 = bt13 - btd_sw + noise()
        ttd = -1.5 + 3.0 * glac
        b11 = b14 + ttd + (b14 - b15) + noise()
        return {
            "B13": bt13.astype(np.float32),
            "B08": (bt13 + (-45 + 46 * s) + noise()).astype(np.float32),
            "B10": (bt13 + (-25 + 25 * s) + noise()).astype(np.float32),
            "B16": (bt13 + (-25 + 23 * s) + noise()).astype(np.float32),
            "B14": b14.astype(np.float32), "B15": b15.astype(np.float32), "B11": b11.astype(np.float32),
        }

    # -----------------------------------------------------------------
    def scans(self):
        """
        ให้ผลทีละ scan: (Scan, radar_frames, radar_rainrate)
        radar_frames = เรดาร์ 2 frame ในช่วง (scan ก่อน, scan นี้] ห่างกัน 5 นาที
        """
        for k in range(self.n_scans):
            m = k * 10.0
            t = self.start + timedelta(minutes=m)
            bt, dbz_now = self._render(m)
            _, dbz_mid = self._render(m - 5.0)
            scan = Scan(time=t, bt=self._bands(bt), lat=self.lat, lon=self.lon,
                        land_mask=np.ones_like(bt, np.float32), elevation_m=np.zeros_like(bt, np.float32))
            frames = [(t - timedelta(minutes=5), dbz_mid), (t, dbz_now)]
            rr = np.where(dbz_now >= 15, dbz_to_rainrate(dbz_now), 0.0).astype(np.float32)
            yield scan, frames, rr

    def domain(self) -> dict:
        return {"lat_min": float(self.lat.min()), "lat_max": float(self.lat.max()),
                "lon_min": float(self.lon.min()), "lon_max": float(self.lon.max())}

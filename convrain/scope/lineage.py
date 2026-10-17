"""
สายตระกูลเมฆพาความร้อน + นาฬิกา t0 จาก track ของ Tracker (ขั้น 3) ทีละ scan

  รอยพาความร้อน = พิกเซลของวัตถุที่ B13 < footprint_k        แกนลึก = รอย ∩ B13 < core_k
  t0 ของ track    = scan แรกที่แกนมี ≥ core_min_px พิกเซล (จังหวะที่ยอดเมฆทะลุระดับ deep convection)
  การแตกตัว      : track ลูก (parent_id) รับ t0 และประวัติของพ่อแม่ ถ้าพ่อแม่มี t0 แล้วก่อนลูกเกิด
                   → การแตก / รวมก้อนของตัวติดตามไม่ทำให้นาฬิกาเริ่มนับใหม่
  t0_censored    : เห็นครั้งแรกก็มีแกนแล้ว และไม่มีพ่อแม่ (ไม่เห็นจังหวะทะลุระดับ) — ยังใช้ทายได้ แต่ t0 อาจช้ากว่าจริง
สถิติของรอย: T_p5 (เปอร์เซ็นไทล์ 5), จุดศูนย์กลางหลังแก้ parallax (เวกเตอร์เดียวจาก T_p5 + SAZ),
จุดศูนย์กลางและโมเมนต์ที่สองของแกน (km²) → วงรีแกนที่ใช้วัดว่าจุดอยู่ใกล้แกนแค่ไหน
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .geometry import moments, parallax_correct


@dataclass
class LineageState:
    """สถานะของ track 1 ตัว ณ scan ปัจจุบัน (เฉพาะ track ที่เห็นและรอยไม่ว่าง)"""
    track_id: str
    t0: datetime | None
    t0_censored: bool
    n_px: int
    core_n_px: int
    T_p5: float
    lat_s: float
    lon_s: float
    core_lat_s: float = np.nan
    core_lon_s: float = np.nan
    core_sxx: float = np.nan
    core_syy: float = np.nan
    core_sxy: float = np.nan

    @property
    def has_core(self) -> bool:
        return bool(np.isfinite(self.core_lat_s))


@dataclass
class _Book:
    first_time: datetime
    has_parent: bool
    t0: datetime | None = None
    t0_censored: bool = False
    last_seen: datetime | None = None
    history: list = field(default_factory=list)          # (เวลา, T_p5) ของสายตระกูล


class LineageBook:
    def __init__(self, tc: dict):
        self.tc = tc
        self.book: dict[str, _Book] = {}

    def reset(self) -> None:
        self.book = {}

    def _book_for(self, tr, t: datetime) -> _Book:
        b = self.book.get(tr.track_id)
        if b is None:
            parent = self.book.get(tr.parent_id) if tr.parent_id else None
            b = _Book(first_time=t, has_parent=parent is not None)
            if parent is not None and parent.t0 is not None and parent.t0 <= t:      # แตกตัว: รับนาฬิกา + ประวัติ
                b.t0, b.t0_censored = parent.t0, parent.t0_censored
                b.history = [h for h in parent.history if h[0] >= parent.t0]
            self.book[tr.track_id] = b
        return b

    def update(self, tracks, scan) -> dict[str, LineageState]:
        """tracks: Track ที่เห็นใน scan นี้ (upd.observed) → {track_id: LineageState}"""
        # track เดิมก่อน track ใหม่: พ่อแม่ที่เพิ่งได้ t0 ใน scan นี้ส่งต่อให้ลูกที่แตกออกใน scan เดียวกันได้
        out = {}
        for tr in sorted(tracks, key=lambda x: x.track_id not in self.book):
            st = self._state(tr, scan)
            if st is not None:
                out[tr.track_id] = st
        # ลืม track ที่ไม่เห็นเกิน 2 ชม. (ลูกที่แตกออกต้องการแค่พ่อแม่ที่ยังอยู่ ณ ตอนแตก)
        self.book = {k: v for k, v in self.book.items()
                     if v.last_seen is None or (scan.time - v.last_seen).total_seconds() <= 7200}
        return out

    def _state(self, tr, scan) -> LineageState | None:
        tc, b13 = self.tc, scan.bt["B13"]
        b = self._book_for(tr, scan.time)
        b.last_seen = scan.time
        r, c = tr.obj.rows, tr.obj.cols
        tb = b13[r, c]
        foot = tb < tc["footprint_k"]
        if not foot.any():
            return None
        r, c, tb = r[foot], c[foot], tb[foot]
        core = tb < tc["core_k"]
        if b.t0 is None and core.sum() >= tc["core_min_px"]:
            b.t0 = scan.time
            b.t0_censored = bool(scan.time == b.first_time and not b.has_parent)
        T_p5 = float(np.percentile(tb, 5))
        b.history.append((scan.time, T_p5))
        lat_px, lon_px = scan.lat[r, c].astype(float), scan.lon[r, c].astype(float)
        lat_c, lon_c = float(lat_px.mean()), float(lon_px.mean())
        saz = float(scan.sat_zenith[int(round(r.mean())), int(round(c.mean()))])
        lat_s, lon_s, _ = parallax_correct(lat_c, lon_c, T_p5, saz)
        st = LineageState(track_id=tr.track_id, t0=b.t0, t0_censored=b.t0_censored, n_px=int(foot.sum()),
                          core_n_px=int(core.sum()), T_p5=T_p5, lat_s=float(lat_s), lon_s=float(lon_s))
        if core.any():
            clat, clon = float(lat_px[core].mean()), float(lon_px[core].mean())
            st.core_lat_s, st.core_lon_s = clat + (lat_s - lat_c), clon + (lon_s - lon_c)
            st.core_sxx, st.core_syy, st.core_sxy = moments(lat_px[core], lon_px[core], clat, clon)
        return st

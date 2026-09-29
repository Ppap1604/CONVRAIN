"""
ขั้น 3: Tracking แบบ online + Kalman filter  (เอกสาร หัวข้อ 6)

เชื่อม object ข้าม scan ให้มี ID คงที่ เพื่อคำนวณแนวโน้มของ "เมฆก้อนเดิม" (Lagrangian)

ขั้นตอนต่อ scan
    1. Advect  : เลื่อน pixel ของ track เดิมตาม motion field (ขั้น 2)
    2. Match   : นับ overlap กับ object ใหม่ แล้วจับคู่ 1:1 ด้วย Hungarian assignment
    3. Split/merge : ก้อนที่แตก → ID ใหม่ + parent_id / ก้อนที่รวม → ปิด track ที่อายุน้อยกว่า
    4. Birth/death : object ไม่มีคู่ = เกิดใหม่ / track หายเกิน max_missed = ปิด
    5. Kalman  : อัปเดต state [BT แกนเย็น, อัตราเปลี่ยน] และ [log พื้นที่, อัตราโต]

อ้างอิง: Dixon & Wiener (1993, TITAN), Núñez Ocasio et al. (2020, TAMS: ต้อง advect ก่อน match)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import count

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import DetectedObject, Detections
from .motion import MotionField


# =====================================================================
# [ขั้น 3.5] Kalman filter แบบ constant-velocity (2 state)
# =====================================================================
class KalmanCV:
    """
    state x = [ค่า, อัตราเปลี่ยนต่อ 10 นาที]
    ใช้ smooth BT ของแกนเย็นและ log(พื้นที่) เพราะข้อมูลทุก 10 นาทีมี noise สูง
    ถ้าคำนวณ ΔBT/Δt จากแค่ 2 ภาพค่าจะแกว่งมาก (เอกสาร หัวข้อ 12 ข้อ 2)
    """

    def __init__(self, x0: float, q: float, r: float, p0_rate: float = 25.0):
        self.q, self.r = float(q), float(r)
        self.x = np.array([x0, 0.0], dtype=float)
        self.P = np.diag([r, p0_rate]).astype(float)
        self.n_updates = 1

    def _predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        return F @ self.x, F @ self.P @ F.T + Q

    def update(self, z: float, dt: float) -> None:
        """dt ในหน่วย "ช่วง 10 นาที" (1.0 = scan ปกติ, 2.0 = ข้ามไป 1 scan)"""
        x, P = self._predict(dt)
        S = P[0, 0] + self.r
        K = P[:, 0] / S
        x = x + K * (z - x[0])
        P = P - np.outer(K, P[0, :])
        self.x, self.P = x, P
        self.n_updates += 1

    def forecast(self, dt: float) -> tuple[float, float]:
        """ค่าที่คาดในอีก dt ช่วง และ variance ของค่านั้น"""
        x, P = self._predict(dt)
        return float(x[0]), float(P[0, 0])

    @property
    def value(self) -> float:
        return float(self.x[0])

    @property
    def rate(self) -> float:
        return float(self.x[1])

    @property
    def rate_var(self) -> float:
        return float(self.P[1, 1])


# =====================================================================
# โครงสร้างข้อมูลของ track
# =====================================================================
@dataclass
class Track:
    track_id: str
    birth_time: datetime
    obj: DetectedObject
    kf_bt: KalmanCV
    kf_area: KalmanCV
    parent_id: str | None = None
    last_time: datetime | None = None
    n_scans: int = 1
    missed: int = 0
    active: bool = True
    end_reason: str | None = None          # dissipated / merged / left_domain
    merged_into: str | None = None
    rows: np.ndarray | None = None          # pixel ล่าสุด (ถ้าหายไปจะเป็นตำแหน่งที่ advect แล้ว)
    cols: np.ndarray | None = None
    history: list[dict] = field(default_factory=list)  # feature ต่อ scan (เติมโดยขั้น 4)
    first_below_0c: datetime | None = None
    min_cold_bt: float = np.inf
    # สถานะ lifecycle ที่ขั้น 6 ดูแล (ค่าเหนียว: เข้าแล้วไม่ย้อนกลับ)
    status: str = "developing"
    rain_started: datetime | None = None
    peak_time: datetime | None = None
    expected_onset: datetime | None = None  # เวลา onset (มัธยฐาน) ล่าสุดที่โมเดลคาด

    @property
    def age_min(self) -> float:
        end = self.last_time or self.birth_time
        return (end - self.birth_time).total_seconds() / 60.0


@dataclass
class TrackUpdate:
    observed: list[Track]     # track ที่มี object ใน scan นี้ (ทั้งเดิมและเกิดใหม่)
    closed: list[Track]       # track ที่ปิดใน scan นี้
    births: list[Track]


class Tracker:
    """Tracker แบบ online: เรียก update() ทีละ scan แล้ว state ต่อยอดจาก scan ก่อน"""

    def __init__(self, cfg: dict):
        self.cfg = cfg["track"]
        self.tracks: dict[str, Track] = {}
        self._counter = count(1)
        self.shape: tuple[int, int] | None = None

    def reset(self) -> None:
        """ใช้เมื่อ scan ขาดนานเกิน max_gap_min: ปิดทุก track แล้วเริ่มใหม่"""
        for t in self.tracks.values():
            t.active, t.end_reason = False, t.end_reason or "data_gap"
        self.tracks = {}

    # -----------------------------------------------------------------
    # สร้าง track ใหม่
    # -----------------------------------------------------------------
    def _new_track(self, obj: DetectedObject, time: datetime, parent: str | None = None) -> Track:
        k = self.cfg["kalman"]
        tid = f"obj_{time:%Y%m%d_%H%M}_{next(self._counter):05d}"
        t = Track(
            track_id=tid, birth_time=time, last_time=time, obj=obj, parent_id=parent,
            kf_bt=KalmanCV(obj.cold_mean_bt, k["q_bt"], k["r_bt"]),
            kf_area=KalmanCV(np.log(max(obj.area_km2, 1e-3)), k["q_area"], k["r_area"], p0_rate=0.25),
            rows=obj.rows, cols=obj.cols,
        )
        self._after_observe(t, obj, time)
        return t

    @staticmethod
    def _after_observe(t: Track, obj: DetectedObject, time: datetime) -> None:
        if t.first_below_0c is None and obj.min_bt < 273.15:
            t.first_below_0c = time
        t.min_cold_bt = min(t.min_cold_bt, obj.cold_mean_bt)

    # =================================================================
    # [ขั้น 3] ฟังก์ชันหลัก
    # =================================================================
    def update(self, det, motion: MotionField | None, time: datetime, dt_min: float = 10.0) -> TrackUpdate:
        c = self.cfg
        ny, nx = det.labels.shape
        self.shape = (ny, nx)
        prev = [t for t in self.tracks.values() if t.active]
        objs = det.objects
        P, K = len(prev), len(objs)
        dt_steps = dt_min / 10.0

        # ---------- [ขั้น 3.1] Advect pixel ของ track เดิม ----------
        adv: list[tuple[np.ndarray, np.ndarray]] = []
        inside_frac = np.zeros(P)
        for i, t in enumerate(prev):
            if motion is not None:
                dv, du = motion.displacement(t.rows, t.cols, dt_min)
                rr = np.rint(t.rows + dv).astype(np.int64)
                cc = np.rint(t.cols + du).astype(np.int64)
            else:
                rr, cc = t.rows.astype(np.int64), t.cols.astype(np.int64)
            inside = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
            inside_frac[i] = inside.mean() if inside.size else 0.0
            adv.append((rr[inside], cc[inside]))

        # ---------- [ขั้น 3.2] overlap ratio ระหว่าง track เดิม × object ใหม่ ----------
        overlap = np.zeros((P, K))
        obj_area = np.array([o.area_px for o in objs], dtype=float)
        for i, (rr, cc) in enumerate(adv):
            if rr.size == 0 or K == 0:
                continue
            hits = np.bincount(det.labels[rr, cc], minlength=K + 1)[1:]
            nz = np.nonzero(hits)[0]
            # หารด้วยพื้นที่ที่เล็กกว่า เพื่อให้ก้อนที่กำลังโตเร็วยังจับคู่ได้
            overlap[i, nz] = hits[nz] / np.minimum(max(rr.size, 1), obj_area[nz])
        overlap = np.clip(overlap, 0.0, 1.0)

        # Hungarian assignment: cost ต่ำ = overlap สูง, track อายุมากได้เปรียบเล็กน้อย (กรณี merge)
        matched_prev: dict[int, int] = {}
        if P and K:
            ages = np.array([min(t.age_min / 60.0, 1.0) for t in prev])
            cost = 1.0 - overlap - c["age_bonus"] * ages[:, None]
            cost[overlap < c["min_overlap"]] = 1e6
            ri, ci = linear_sum_assignment(cost)
            for i, k in zip(ri, ci):
                if cost[i, k] < 1e5:
                    matched_prev[i] = k
        matched_obj = {k: i for i, k in matched_prev.items()}

        observed: list[Track] = []
        closed: list[Track] = []
        births: list[Track] = []

        # ---------- track เดิมที่จับคู่ได้: อัปเดต state + Kalman ----------
        for i, k in matched_prev.items():
            t, o = prev[i], objs[k]
            t.obj, t.rows, t.cols = o, o.rows, o.cols
            t.kf_bt.update(o.cold_mean_bt, dt_steps)
            t.kf_area.update(np.log(max(o.area_km2, 1e-3)), dt_steps)
            t.last_time, t.n_scans, t.missed = time, t.n_scans + 1, 0
            self._after_observe(t, o, time)
            observed.append(t)

        # ---------- [ขั้น 3.3–3.4] track เดิมที่ไม่มีคู่: merge / หาย / ออกนอกโดเมน ----------
        for i, t in enumerate(prev):
            if i in matched_prev:
                continue
            best_k = int(np.argmax(overlap[i])) if K else -1
            if K and overlap[i, best_k] >= c["min_overlap"] and best_k in matched_obj:
                # รวมเข้ากับก้อนอื่น: ก้อนที่ Hungarian เลือก (อายุมากกว่า/overlap สูงกว่า) สืบ ID
                t.active, t.end_reason = False, "merged"
                t.merged_into = prev[matched_obj[best_k]].track_id
                closed.append(t)
            elif inside_frac[i] < 0.5:
                t.active, t.end_reason = False, "left_domain"
                closed.append(t)
            else:
                t.missed += 1
                t.rows, t.cols = adv[i]
                if t.missed > c["max_missed_scans"] or t.rows.size == 0:
                    t.active, t.end_reason = False, "dissipated"
                    closed.append(t)

        # ---------- object ใหม่ที่ไม่มีคู่: split (มี parent) หรือเกิดใหม่ ----------
        for k, o in enumerate(objs):
            if k in matched_obj:
                continue
            parent = None
            if P:
                best_i = int(np.argmax(overlap[:, k]))
                if overlap[best_i, k] >= c["min_overlap"]:
                    parent = prev[best_i].track_id
            t = self._new_track(o, time, parent)
            self.tracks[t.track_id] = t
            observed.append(t)
            births.append(t)

        # เก็บเฉพาะ track ที่ยัง active ไว้ใน state (track ที่ปิดแล้วส่งออกผ่าน closed)
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.active}
        return TrackUpdate(observed=observed, closed=closed, births=births)

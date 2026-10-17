"""
ผลิตภัณฑ์รายจุดที่ไม่ต้องรอให้มีเมฆติดตามอยู่เหนือจุด — ออกใหม่ทุกภาพ ให้ทุกจุด

  1. คำเตือนฝนเริ่ม 3 ระดับ (จุดที่ยังไม่มีฝน)
        เฝ้าระวัง (watch)      ฝนอาจเริ่มภายใน 3 ชม.   — สภาพแวดล้อมรอบ ~50 km: เมฆเย็นขยายตัว, มีส่วนที่เย็นลงเร็ว
        เตือน (warning)        ฝนอาจเริ่มภายใน 60 นาที  — เมฆลึก / หนาเข้ามาใกล้ในรัศมี ~25 km
        ใกล้ตก (imminent)      ฝนอาจเริ่มภายใน 30 นาที  — คอลัมน์อากาศเหนือจุดและรอบ ~10 km หนาและเย็นลง
  2. เวลาฝนเริ่ม / ฝนหยุด พร้อมความมั่นใจ
        "ฝนจะเริ่มราว HH:MM  ความมั่นใจ Y%"   (จุดที่ยังไม่มีฝน)
        "ฝนจะหยุดราว HH:MM  ความมั่นใจ Y%"    (จุดที่แผนที่ฝนตอนนี้บอกว่ากำลังมีฝน)

หลักการ: สำหรับระยะ b = 10 … 180 นาที โมเดล logistic หนึ่งตัวต่อระยะให้ P(ฝนเริ่มภายใน b นาที); เรียงเป็นเส้นโค้งสะสม F(L)
(บังคับไม่ลดลง) → ระดับคำเตือนอ่านจาก F(30), F(60), F(180) เทียบเกณฑ์ ส่วนเวลาที่บอก = กึ่งกลางของหน้าต่าง ±15 นาทีที่มี
ความน่าจะเป็นมากที่สุด และความมั่นใจ = ความน่าจะเป็นในหน้าต่างนั้น ปรับให้ตรงกับความถี่จริง (ข้อมูลที่ไม่ได้ใช้ฟิต)
คำเตือนและเวลาจึงมาจากเส้นโค้งเดียวกัน ไม่ขัดกันเอง; F(180) ที่ต่ำ = "ไม่น่าจะตก" ซึ่งความมั่นใจของเวลาก็จะต่ำตาม

ตัวแปรทุกตัวมาจากภาพ Himawari (ภาพนี้และภาพ ≤ 2 ชม. ก่อน) ที่พิกเซลเหนือจุด (ย้อน parallax ของยอดเมฆเย็นคงที่ของภูมิภาค)
และรอบจุด ร่วมกับแผนที่ฝนตอนนี้ (rainmap.py); ระยะ > 60 นาทีใช้ค่าคงที่แยกกลางวัน / กลางคืน (วัฏจักรรายวันของการพาความร้อน)
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
from scipy.special import expit

from .geometry import GroundIndex, shifted_pixels

CONSTANTS = Path(__file__).resolve().parent / "constants"
H_ON = (10, 20, 30, 45, 60, 90, 120, 180)
H_ST = (10, 20, 30, 45, 60, 90, 120)
LEVELS = (("imminent", 30), ("warning", 60), ("watch", 180))      # เรียงจากเร่งด่วนสุด
R10, R25, R50 = 5, 12, 24                                         # ครึ่งความกว้างกรอบ (พิกเซล 2 km) ≈ 10 / 25 / 50 km
SWING_DISC, SWING_SMOOTH, SWING_SPAN = 5, 3, 13                   # จาน ~10 km, เฉลี่ย 3 ภาพ, ย้อน 2 ชม.
DAY_COS = 0.2
WINDOW = 15.0

MAP3 = ["rain_now", "rain_near", "rain_near25"]
EXTRA = ["st_ttd", "st_b16_13"]
SETS = {
    "watch": ["cold_frac_50", "cold_frac_change_50", "cooling_frac_50", "coldest_50", "swing", "cos_sun"] + MAP3,
    "warning": ["cold_frac_25", "coldest_25", "thickest_25", "cooling_frac_25", "cold_frac_change_25", "st_b13"]
               + MAP3 + EXTRA,
    "imminent": ["st_b13", "st_btd", "st_db13", "st_dbtd", "coldest_10", "thickest_10", "cold_frac_10"] + MAP3 + EXTRA,
}
STOP_SET = SETS["imminent"]
FEATURES = sorted({f for v in SETS.values() for f in v})


def onset_set(b: float) -> list:
    return SETS["imminent"] if b <= 30 else (SETS["warning"] if b <= 60 else SETS["watch"])


def cos_sun_simple(lat, lon, t: pd.Timestamp) -> np.ndarray:
    """cos มุมเซนิธดวงอาทิตย์แบบสูตรสั้น (ตัวแปร "ช่วงเวลาของวัน" ของระดับเฝ้าระวัง)"""
    dec = np.radians(23.44) * np.sin(2 * np.pi * (284 + t.dayofyear) / 365)
    ha = np.radians(15 * (t.hour + t.minute / 60 + np.asarray(lon, float) / 15 - 12))
    la = np.radians(np.asarray(lat, float))
    return np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)


# ====================================================================== ตัวแปรรายจุด
class PointFeatures:
    """
    ตัวแปรรอบจุดคงที่ชุดหนึ่ง (จุดผู้ใช้ / สถานี) — ส่งภาพตามลำดับเวลา
      st_*            พิกเซลเหนือจุด: B13, B13 − B15, การเปลี่ยนใน 10 นาที, ดัชนีเฟสสามช่อง, B16 − B13
      coldest_R       B13 ต่ำสุดในกรอบ R km          thickest_R   B13 − B15 ต่ำสุดในกรอบ (เมฆหนาสุด)
      cold_frac_R     สัดส่วนพิกเซล B13 < 235 K       cold_frac_change_R   การเปลี่ยนของสัดส่วนนั้น (25 km: 10 นาที, 50 km: 30 นาที)
      cooling_frac_R  สัดส่วนพิกเซลที่เย็นลง ≥ 4 K ใน 10 นาที ในบรรดาพิกเซลที่ B13 < 273 K (ยอดที่กำลังพุ่งขึ้น)
      swing           อุณหภูมิเฉลี่ยรอบจุดตอนนี้ต่ำกว่าค่าสูงสุดใน 2 ชม. ที่ผ่านมากี่ K (ช่วงอุ่น → เย็นลงของเมฆที่เข้ามา)
      cos_sun         ช่วงเวลาของวัน
    อุณหภูมิในภูมิภาคเขตร้อนเลื่อนด้วย offset ก่อน (เกณฑ์ 235 / 273 K เป็นค่าของเขตอบอุ่น)
    """

    def __init__(self, lat, lon, top_k: float, offset: float = 0.0):
        self.lat, self.lon = np.asarray(lat, float), np.asarray(lon, float)
        self.top_k, self.offset = float(top_k), float(offset)
        self.prev: dict = {}                                       # เวลา → ฟิลด์ (สำหรับการเปลี่ยน 10 / 30 นาที)
        self.raw = deque(maxlen=SWING_SMOOTH)
        self.smooth = deque(maxlen=SWING_SPAN)
        self._key = None
        self.r = self.c = None

    def reset(self) -> None:
        self.prev, self._key = {}, None
        self.raw.clear()
        self.smooth.clear()

    def _pixels(self, scan):
        key = (scan.lat.shape, float(scan.lat.flat[0]), float(scan.lon.flat[0]))
        if key != self._key:
            idx = GroundIndex(scan.lat, scan.lon)
            self.r, self.c = (np.asarray(a) for a in shifted_pixels(idx, scan.lat.shape, self.lat, self.lon,
                                                                       scan.sat_zenith, self.top_k))
            self._key = key
            self.raw.clear()
            self.smooth.clear()

    def _lag(self, t, minutes):
        for tp, v in self.prev.items():
            if abs((t - tp).total_seconds() / 60 - minutes) <= 1:
                return v
        return None

    @staticmethod
    def _fields(B13, B15):
        b = np.where(np.isfinite(B13), B13, 300.0)
        btd = np.where(np.isfinite(B15), b - B15, 10.0)
        F = {"b": b, "btd": btd}
        for R, n in ((R10, "10"), (R25, "25"), (R50, "50")):
            F[f"cold_frac_{n}"] = uniform_filter((b < 235).astype(np.float32), 2 * R + 1)
            F[f"coldest_{n}"] = minimum_filter(b, 2 * R + 1)
        for R, n in ((R10, "10"), (R25, "25")):
            F[f"thickest_{n}"] = minimum_filter(btd, 2 * R + 1)
        return F

    def update(self, scan, rain: pd.DataFrame | None = None, rain_p: np.ndarray | None = None) -> pd.DataFrame:
        """rain = ค่าที่จุดจาก RainMap.point_features; rain_p = แผนที่ความน่าจะเป็นทั้งภาพ (สำหรับ rain_near25)"""
        self._pixels(scan)
        r, c = self.r, self.c
        t = scan.time
        o = self.offset
        g = lambda k: (scan.bt[k].astype(float) + o) if k in scan.bt else np.full(scan.shape, np.nan)  # noqa: E731
        B13, B15 = g("B13"), g("B15")
        F = self._fields(B13, B15)
        P10, P30 = self._lag(t, 10), self._lag(t, 30)
        out = {n: F[n][r, c] for n in ("cold_frac_10", "cold_frac_25", "cold_frac_50", "coldest_10", "coldest_25",
                                       "coldest_50", "thickest_10", "thickest_25")}
        out["st_b13"], out["st_btd"] = F["b"][r, c], F["btd"][r, c]
        nan = np.full(len(r), np.nan)
        if P10 is not None:
            cool = ((F["b"] - P10["b"]) <= -4.0) & (F["b"] < 273)
            warm = (F["b"] < 273).astype(np.float32)
            for R, n in ((R25, "25"), (R50, "50")):
                num = uniform_filter(cool.astype(np.float32), 2 * R + 1)
                den = uniform_filter(warm, 2 * R + 1)
                out[f"cooling_frac_{n}"] = np.where(den > 0, num / np.maximum(den, 1e-6), 0.0)[r, c]
            out["cold_frac_change_25"] = (F["cold_frac_25"] - P10["cold_frac_25"])[r, c]
            out["st_db13"] = (F["b"] - P10["b"])[r, c]
            out["st_dbtd"] = (F["btd"] - P10["btd"])[r, c]
        else:
            for n in ("cooling_frac_25", "cooling_frac_50", "cold_frac_change_25", "st_db13", "st_dbtd"):
                out[n] = nan
        out["cold_frac_change_50"] = (F["cold_frac_50"] - P30["cold_frac_50"])[r, c] if P30 is not None else nan
        ts = pd.Timestamp(t).tz_convert(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t)
        out["cos_sun"] = cos_sun_simple(self.lat, self.lon, ts)
        out["st_ttd"] = ((g("B11") - g("B14")) - (g("B14") - B15))[r, c]
        out["st_b16_13"] = (g("B16") - B13)[r, c]
        # swing: เฉลี่ยจาน ~10 km → เฉลี่ยเคลื่อนที่ 3 ภาพ → ค่าสูงสุดใน 2 ชม. − ตอนนี้
        b290 = np.where(np.isfinite(B13), B13, 290.0)
        yy, xx = np.mgrid[-SWING_DISC:SWING_DISC + 1, -SWING_DISC:SWING_DISC + 1]
        dk = yy ** 2 + xx ** 2 <= SWING_DISC ** 2
        rr = np.clip(r[:, None] + yy[dk][None], 0, B13.shape[0] - 1)
        cc = np.clip(c[:, None] + xx[dk][None], 0, B13.shape[1] - 1)
        self.raw.append(b290[rr, cc].mean(1))
        self.smooth.append(np.mean(np.stack(self.raw), 0))
        out["swing"] = np.max(np.stack(self.smooth), 0) - self.smooth[-1]
        # แผนที่ฝนตอนนี้
        if rain is not None:
            out["rain_now"], out["rain_near"] = rain.rain_now.to_numpy(), rain.rain_near.to_numpy()
            out["daylit"] = rain.daylit.to_numpy()
        else:
            out["rain_now"] = out["rain_near"] = nan
            out["daylit"] = np.zeros(len(r), bool)
        out["rain_near25"] = maximum_filter(rain_p, 2 * R25 + 1)[r, c] if rain_p is not None else nan
        self.prev[t] = {k: F[k] for k in ("b", "btd", "cold_frac_25", "cold_frac_50")}
        self.prev = {k: v for k, v in self.prev.items() if (t - k).total_seconds() / 60 <= 31}
        return pd.DataFrame(out)


# ====================================================================== โมเดล
class LR:
    """logistic (ปรับมาตรฐาน, ค่าหาย → มัธยฐาน) — ค่าคงที่ cols, med, mu, sd, w (ตัวแรก = ค่าตัด)"""

    def __init__(self, P: dict):
        self.cols = list(P["cols"])
        self.med, self.mu, self.sd, self.w = (np.asarray(P[k], float) for k in ("med", "mu", "sd", "w"))

    def prob(self, D: pd.DataFrame) -> np.ndarray:
        X = D.reindex(columns=self.cols).to_numpy(float)
        X = np.where(np.isfinite(X), X, self.med)
        return expit(np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd]) @ self.w)


def curve_at(Fk: np.ndarray, knots: np.ndarray, x: float) -> np.ndarray:
    """ค่าของเส้นโค้งสะสม (แถว × ปม, เส้นตรงระหว่างปม) ที่ระยะ x นาที ของทุกแถว"""
    x = float(np.clip(x, knots[0], knots[-1]))
    j = int(np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2))
    f = (x - knots[j]) / (knots[j + 1] - knots[j])
    return Fk[:, j] + f * (Fk[:, j + 1] - Fk[:, j])


def window_statement(F: np.ndarray, H, lmax: float, half: float = WINDOW):
    """
    เส้นโค้งสะสม F (แถว × len(H), F(0) = 0) → (นาทีจากเวลาภาพถึงเวลาที่บอก, ความน่าจะเป็นในหน้าต่าง ±half)
    เวลาที่บอก = กึ่งกลางหน้าต่าง 30 นาทีที่มวลมากที่สุด (เท่ากันหลายหน้าต่าง → กึ่งกลางของช่วงที่เท่ากัน)
    """
    knots = np.r_[0.0, np.asarray(H, float)]
    Fk = np.hstack([np.zeros((len(F), 1)), F])
    Ls = np.arange(5.0, lmax - half + 1, 5.0)
    M = np.column_stack([curve_at(Fk, knots, L + half) - curve_at(Fk, knots, max(0.0, L - half)) for L in Ls])
    c = M.max(1)
    top = M >= (c[:, None] - 1e-9)
    first = np.argmax(top, 1)
    last = top.shape[1] - 1 - np.argmax(top[:, ::-1], 1)
    L = np.where(c > 0, (Ls[first] + Ls[last]) / 2, np.nan)
    return L, c


class Outlook:
    """ค่าคงที่ทั้งชุด (constants/outlook_<ภูมิภาค>.json) → คำเตือน + เวลาพร้อมความมั่นใจ"""

    def __init__(self, P: dict):
        self.P = P
        self.onset = {int(b): {m: (LR(v) if v else None) for m, v in d.items()} for b, d in P["onset"].items()}
        self.stop = {int(b): LR(v) for b, v in P["stop"].items()}
        self.levels = P["levels"]                                   # {"imminent": เกณฑ์ F(30), ...}
        self.calib = P["confidence"]                                # {"onset": {"x": [...], "y": [...]}, "stop": ...}
        self.stop_needs = P.get("stop_requires", "map")

    @classmethod
    def load(cls, name_or_path) -> "Outlook":
        p = Path(name_or_path)
        if not p.suffix:
            p = CONSTANTS / f"outlook_{name_or_path}.json"
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def onset_curve(self, D: pd.DataFrame) -> np.ndarray:
        day = (D.cos_sun.to_numpy(float) >= DAY_COS) if "cos_sun" in D else np.zeros(len(D), bool)
        cols = []
        for b in H_ON:
            m = self.onset[b]
            p = m["all"].prob(D)
            for mode, sel in (("day", day), ("night", ~day)):
                if m.get(mode) is not None and sel.any():
                    p[sel] = m[mode].prob(D[sel])
            cols.append(p)
        return np.maximum.accumulate(np.column_stack(cols), axis=1)

    def stop_curve(self, D: pd.DataFrame) -> np.ndarray:
        return np.maximum.accumulate(np.column_stack([self.stop[b].prob(D) for b in H_ST]), axis=1)

    def _conf(self, kind, c_raw):
        k = self.calib[kind]
        return np.interp(c_raw, k["x"], k["y"])

    def predict(self, D: pd.DataFrame, raining: np.ndarray, t0) -> pd.DataFrame:
        """D = ตัวแปรรายจุดของภาพนี้; raining = แผนที่ฝนตอนนี้บอกว่าจุดนี้มีฝนอยู่; t0 = เวลาภาพ (UTC)"""
        n = len(D)
        out = pd.DataFrame(index=D.index)
        F = self.onset_curve(D)
        knots = np.r_[0.0, H_ON]
        Fk = np.hstack([np.zeros((n, 1)), F])
        for b in (30, 60, 180):
            out[f"p_onset_{b}"] = Fk[:, list(knots).index(b)]
        lvl = np.full(n, "none", dtype=object)
        for name, b in reversed(LEVELS):                            # ระดับเร่งด่วนกว่าทับระดับที่เบากว่า
            lvl = np.where(out[f"p_onset_{b}"].to_numpy() >= self.levels[name], name, lvl)
        L, c_raw = window_statement(F, H_ON, H_ON[-1])
        conf = self._conf("onset", c_raw)
        dry = ~np.asarray(raining, bool)
        out["warning_level"] = np.where(dry, lvl, "raining")
        out["onset_in_min"] = np.where(dry, L, np.nan)
        out["onset_time"] = pd.to_datetime(t0) + pd.to_timedelta(np.where(dry, L, np.nan), unit="m")
        out["onset_confidence"] = np.where(dry, conf, np.nan)
        Fs = self.stop_curve(D)
        Ls, cs_raw = window_statement(Fs, H_ST, H_ST[-1])
        cs = self._conf("stop", cs_raw)
        wet = np.asarray(raining, bool)
        out["stop_in_min"] = np.where(wet, Ls, np.nan)
        out["stop_time"] = pd.to_datetime(t0) + pd.to_timedelta(np.where(wet, Ls, np.nan), unit="m")
        out["stop_confidence"] = np.where(wet, cs, np.nan)
        return out


def statement_text(row) -> str:
    """ข้อความสั้นสำหรับหน้าจอ / แจ้งเตือน (เวลาท้องถิ่นให้ผู้เรียกแปลงเอง)"""
    if pd.notna(row.get("onset_time")):
        return f"ฝนจะเริ่มราว {row['onset_time']:%H:%M} ความมั่นใจ {100 * row['onset_confidence']:.0f}%"
    if pd.notna(row.get("stop_time")):
        return f"ฝนจะหยุดราว {row['stop_time']:%H:%M} ความมั่นใจ {100 * row['stop_confidence']:.0f}%"
    return ""

"""
แผนที่ "ฝนตอนนี้" (rain-now map) แบบโหมดกลางวัน / กลางคืน และค่าที่จุดบนพื้น

ทุกพิกเซลได้ความน่าจะเป็นที่ฝนกำลังตกถึงพื้นอยู่ ณ เวลาภาพ จากภาพ Himawari อย่างเดียว

  โหมดกลางคืน (ใช้ได้ 24 ชม.)  ช่องอินฟราเรด: B13, ความต่าง split window (B13 − B15), ไอน้ำ (B08 − B13),
                    ค่าต่ำสุด / เฉลี่ย / ความแปรปรวนของ B13 รอบพิกเซล, สัดส่วนเมฆเย็นรอบพิกเซล, ความชันของ B13,
                    ระยะถึงยอดเมฆเย็นเฉพาะที่ที่ใกล้ที่สุดและอุณหภูมิของมัน, การเย็นลงใน 10 / 20 นาที,
                    ดัชนีเฟสยอดเมฆสามช่อง (B11 − B14) − (B14 − B15), B11 − B13, B16 − B13, B10 − B13
  โหมดกลางวัน (ดวงอาทิตย์สูงพอ cos θ0 ≥ 0.2 และมีช่อง B04 / B05 / B07)
                    = ทุกตัวของกลางคืน + ค่าสะท้อน 0.86 µm (ความหนาเชิงแสง), อัตราส่วน 1.6 / 0.86 µm (เฟส / ขนาด
                    อนุภาค), ค่าสะท้อน 3.9 µm หลังหักส่วนความร้อน (อนุภาคใหญ่ = ค่าต่ำ)
แต่ละโหมดเป็น logistic regression หนึ่งชุด (ปรับมาตรฐาน + ธงค่าหาย + B13 เป็นช่วง 5 K) — ค่าคงที่อยู่ใน constants/

ค่าที่จุด (ใช้ภาพปัจจุบันและภาพ 10 / 20 นาทีก่อนเท่านั้น):
  rain_now        ความน่าจะเป็นที่พิกเซลเหนือจุด (ย้อน parallax ของยอดเมฆเย็นคงที่ของภูมิภาค)
  rain_near       ค่าสูงสุดในกรอบ ~10 km รอบจุด
  arrival_min     เวลา L ∈ {0, 10, 20, 30, 45, 60} นาทีแรกที่จุดเหนือลม (ย้อนตามการเคลื่อนที่ของยอดเมฆ) เป็นฝน; 120 = ไม่มี
  departure_min   เวลาแรกที่จุดเหนือลม "ไม่เป็นฝน" (ใช้กับฝนหยุด)
  rain_arrival    ln(arrival_min + 10)  (ตัวแปรที่โมเดลเวลาใช้)
  daylit          จุดนี้ใช้โหมดกลางวันหรือไม่
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter, minimum_filter, uniform_filter
from scipy.special import expit

from .geometry import GroundIndex, parallax_correct, shifted_pixels

BINS = np.arange(190, 305, 5)
LEADS = (10, 20, 30, 45, 60)
COS_MIN = 0.2
F0_39, LAM39 = 10.0, 3.885                 # ฟลักซ์อาทิตย์ที่ 3.9 µm (W m⁻² µm⁻¹ ที่ 1 AU), ความยาวคลื่น B07 (µm)
C1, C2 = 1.191042e8, 1.4387752e4
R10 = 5                                    # ครึ่งความกว้างกรอบ ~10 km (พิกเซล 2 km)
NO_LEAD = 120.0
IR_BANDS = ("B07", "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16")
CONSTANTS = Path(__file__).resolve().parent / "constants"


def planck(lam_um, T):
    return C1 / (lam_um ** 5 * np.expm1(C2 / (lam_um * T)))


def sun_dist_au(t: datetime) -> float:
    return 1 - 0.01672 * np.cos(np.deg2rad(0.9856 * (t.timetuple().tm_yday - 4)))


def cos_zenith(t: datetime, lat, lon) -> np.ndarray:
    """cos ของมุมเซนิธดวงอาทิตย์ ณ กลางการสแกน (เวลาภาพ + 1.25 นาที)"""
    tt = (t + timedelta(minutes=1.25)).replace(tzinfo=None)
    try:
        from pyorbital.astronomy import cos_zen
        return np.asarray(cos_zen(tt, lon, lat), float)
    except ImportError:                                        # สูตรดาราศาสตร์อย่างง่าย (ต่างกันไม่เกิน ~0.01)
        doy = tt.timetuple().tm_yday
        g = 2 * np.pi / 365.0 * (doy - 1 + (tt.hour - 12) / 24.0)
        dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
               + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
        eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                        - 0.040849 * np.sin(2 * g))
        tst = tt.hour * 60 + tt.minute + tt.second / 60 + eqt + 4 * np.asarray(lon, float)
        ha = np.radians(tst / 4 - 180)
        la = np.radians(np.asarray(lat, float))
        return np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)


def grid_spacing_km(lat, lon) -> tuple[float, float]:
    """ระยะระหว่างพิกเซล (แถว, คอลัมน์) โดยประมาณ (km)"""
    dy = abs(float(np.nanmedian(np.diff(lat, axis=0)))) * 111.32
    dx = abs(float(np.nanmedian(np.diff(lon, axis=1)))) * 111.32 * np.cos(np.radians(float(np.nanmean(lat))))
    return dy, dx


class MapModel:
    """logistic ของแผนที่หนึ่งโหมด: ค่าคงที่ (cols, med, mu, sd, b, thr)"""

    def __init__(self, P: dict):
        self.cols = list(P["cols"])
        self.med, self.mu, self.sd, self.b = (np.asarray(P[k], float) for k in ("med", "mu", "sd", "b"))
        self.thr = float(P["thr"])

    def design(self, X):
        na = ~np.isfinite(X)
        Xf = np.where(na, self.med, X)
        Z = (Xf - self.mu) / self.sd
        k = np.clip(np.digitize(Xf[:, 0], BINS), 0, len(BINS))
        oh = np.zeros((len(X), len(BINS) + 1))
        oh[np.arange(len(X)), k] = 1
        return np.hstack([Z, na[:, -2:].astype(float), oh, np.ones((len(X), 1))])

    def prob(self, X):
        return expit(self.design(np.asarray(X, float)) @ self.b)


def ir_fields(bands: dict, lat, lon, prev10, prev20) -> dict:
    nan = lambda: np.full(bands["B13"].shape, np.nan)  # noqa: E731
    B13 = bands["B13"].astype(float)
    B15 = bands.get("B15", nan()).astype(float)
    B08 = bands.get("B08", nan()).astype(float)
    b = np.where(np.isfinite(B13), B13, 290.0)
    btd = B13 - B15
    m5 = uniform_filter(b, 5)
    F = {"b13": B13, "btd": btd, "wv": B08 - B13, "min5": minimum_filter(b, 5), "mean5": m5,
         "std5": np.sqrt(np.maximum(uniform_filter(b * b, 5) - m5 * m5, 0)),
         "btdmin5": minimum_filter(np.where(np.isfinite(btd), btd, 10.0), 5),
         "frac235": uniform_filter((b < 235).astype(np.float32), 11),
         "grad": np.hypot(*np.gradient(b)),
         "d10": B13 - prev10 if prev10 is not None else nan(),
         "d20": B13 - prev20 if prev20 is not None else nan()}
    bc = np.where(np.isfinite(B13), B13, 320.0)
    core = (bc == minimum_filter(bc, 11)) & (bc < 235.0)
    if core.any():
        dy, dx = grid_spacing_km(lat, lon)
        d, (ii, jj) = distance_transform_edt(~core, sampling=(dy, dx), return_indices=True)
        F["dcore"], F["tcore"] = np.minimum(d, 200.0), bc[ii, jj]
    else:
        F["dcore"], F["tcore"] = np.full(b.shape, 200.0), np.full(b.shape, 260.0)
    F["wvmax11"] = maximum_filter(np.where(np.isfinite(F["wv"]), F["wv"], -50.0), 11)
    X = {k: bands.get(k, nan()).astype(float) for k in ("B10", "B11", "B14", "B16")}
    ttd = (X["B11"] - X["B14"]) - (X["B14"] - B15)
    F.update({"ttd": ttd, "b11_13": X["B11"] - B13, "b16_13": X["B16"] - B13, "b10_13": X["B10"] - B13,
              "ttdmin5": minimum_filter(np.where(np.isfinite(ttd), ttd, 10.0), 5)})
    return F


def day_fields(bands: dict, refl: dict, cz: np.ndarray, t: datetime):
    """ช่องกลางวัน: refl["B04"], refl["B05"] = ค่าสะท้อน (สัดส่วน 0–1), bands["B07"] = อุณหภูมิสว่าง (K)"""
    shape = bands["B13"].shape
    if not ("B04" in refl and "B05" in refl and "B07" in bands):
        return {}, np.zeros(shape, bool)
    with np.errstate(all="ignore"):
        ok = (cz >= COS_MIN) & np.isfinite(refl["B04"])
        r086 = np.where(ok, refl["B04"] / cz, np.nan)
        r16 = np.where(ok, refl["B05"] / cz, np.nan)
        q16 = np.where(r086 > 0.05, r16 / r086, np.nan)
        be = planck(LAM39, bands["B13"].astype(float))
        sol = F0_39 * cz / (np.pi * sun_dist_au(t) ** 2)
        r39 = np.where(ok, (planck(LAM39, bands["B07"].astype(float)) - be) / (sol - be), np.nan)
        F = {"r086": r086, "q16": q16, "r39": r39,
             "r086max5": np.where(ok, maximum_filter(np.where(np.isfinite(r086), r086, 0.0), 5), np.nan),
             "r39min5": np.where(ok, minimum_filter(np.where(np.isfinite(r39), r39, 1.0), 5), np.nan)}
    return F, ok


def flow_pair(prev, now):
    """การเคลื่อนที่ของยอดเมฆ (DIS optical flow ของ B13; เมฆเย็น = สว่าง) จากภาพก่อน → ภาพนี้: พิกเซลต่อคู่ภาพ"""
    import cv2
    s = lambda x: (np.clip((300.0 - np.nan_to_num(x, nan=300.0)) / 120.0, 0, 1) * 255).astype(np.uint8)  # noqa: E731
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = dis.calc(s(prev), s(now), None)
    return f[..., 0].astype(float), f[..., 1].astype(float)


def load_constants(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = CONSTANTS / f"rainmap_{name_or_path}.json"
    return json.loads(p.read_text(encoding="utf-8"))


class RainMap:
    """ส่งภาพตามลำดับเวลาเข้า update() แล้วอ่านแผนที่ (last) หรือค่าที่จุด (point_features)"""

    def __init__(self, params: dict, top_k: float):
        self.N, self.D = MapModel(params["N"]), MapModel(params["D"])
        self.top_k = float(top_k)
        # ค่าคงที่ปรับเทียบที่อุณหภูมิของเขตอบอุ่น; เขตร้อนโทรโพพอสสูงกว่า ยอดเมฆเย็นกว่าราว 10 K → เลื่อนช่อง IR ก่อนเข้าแผนที่
        self.offset = float(params.get("band_offset_k", 0.0))
        self.past: dict[datetime, np.ndarray] = {}
        self.last: dict | None = None
        self._index = None
        self._key = None

    @classmethod
    def from_constants(cls, name_or_path, top_k: float) -> "RainMap":
        return cls(load_constants(name_or_path), top_k)

    def reset(self) -> None:
        self.past, self.last = {}, None

    def _lag(self, t, minutes):
        for tp, v in self.past.items():
            if abs((t - tp).total_seconds() / 60 - minutes) <= 1:
                return v
        return None

    def bands(self, scan) -> dict:
        return {k: (v.astype(float) + self.offset if k in IR_BANDS else v) for k, v in scan.bt.items()} \
            if self.offset else scan.bt

    def update(self, scan) -> dict:
        t = scan.time
        bands = self.bands(scan)
        B13 = bands["B13"].astype(float)
        p10, p20 = self._lag(t, 10), self._lag(t, 20)
        F = ir_fields(bands, scan.lat, scan.lon, p10, p20)
        refl = getattr(scan, "refl", {}) or {}
        if "B04" in refl:
            Fd, okd = day_fields(bands, refl, cos_zenith(t, scan.lat, scan.lon), t)
        else:
            Fd, okd = {}, np.zeros(B13.shape, bool)
        F.update(Fd)
        full = np.isfinite(B13)
        p = np.zeros(B13.shape)
        e = np.zeros(B13.shape, bool)
        if full.any():
            pn = self.N.prob(np.column_stack([F[c][full] for c in self.N.cols]))
            p[full], e[full] = pn, pn >= self.N.thr
            dm = full & okd
            if dm.any():
                pd_ = self.D.prob(np.column_stack([F[c][dm] for c in self.D.cols]))
                p[dm], e[dm] = pd_, pd_ >= self.D.thr
        U = V = None
        if p10 is not None:
            u, v = flow_pair(p10, B13)
            U, V = gaussian_filter(u / 10.0, 3), gaussian_filter(v / 10.0, 3)     # พิกเซลต่อนาที
        self.past[t] = B13
        self.past = {k: v for k, v in self.past.items() if t - k <= timedelta(minutes=30)}
        key = (scan.lat.shape, float(scan.lat.flat[0]), float(scan.lon.flat[0]))
        if key != self._key:
            self._index, self._key = GroundIndex(scan.lat, scan.lon), key
        self.last = {"time": t, "p": p, "raining": e, "daylit": okd, "max10": maximum_filter(p, 2 * R10 + 1),
                     "U": U, "V": V, "lat": scan.lat, "lon": scan.lon, "saz": scan.sat_zenith}
        return self.last

    def ground_grid(self):
        """ตำแหน่งบนพื้นของทุกพิกเซล (แก้ parallax ด้วยยอดเมฆเย็นคงที่ top_k) — เปลี่ยนเฉพาะเมื่อกริดเปลี่ยน"""
        M = self.last
        if getattr(self, "_gkey", None) != self._key:
            la, lo, _ = parallax_correct(M["lat"], M["lon"], np.full(M["lat"].shape, self.top_k), M["saz"])
            self._ground = (la.astype(np.float32), lo.astype(np.float32))
            self._gkey = self._key
        return self._ground

    def rain_areas(self, pixel_area_km2=None, min_px: int = 2) -> list[dict]:
        """บริเวณที่ฝนกำลังตก (กลุ่มพิกเซลที่ติดกันซึ่งแผนที่บอกว่าเป็นฝน) → GeoJSON Feature บนพื้น (แก้ parallax แล้ว)"""
        from scipy.ndimage import label
        from ..output import footprint_polygon
        M = self.last
        lab, n = label(M["raining"], np.ones((3, 3)))
        if n == 0:
            return []
        gla, glo = self.ground_grid()
        t = pd.Timestamp(M["time"])
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        feats = []
        for k in range(1, n + 1):
            rows, cols = np.nonzero(lab == k)
            if len(rows) < min_px:
                continue
            ring = footprint_polygon(rows, cols, gla, glo)
            if not ring:
                continue
            p = M["p"][rows, cols]
            area = float(np.sum(pixel_area_km2[rows, cols])) if pixel_area_km2 is not None else np.nan
            feats.append({"type": "Feature", "id": f"rain_{t:%Y%m%dT%H%MZ}_{len(feats) + 1:03d}",
                          "geometry": {"type": "Polygon", "coordinates": [ring]},
                          "properties": {"scan_time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                         "area_km2": round(area, 1) if np.isfinite(area) else None,
                                         "n_pixels": int(len(rows)), "rain_prob_max": round(float(p.max()), 3),
                                         "rain_prob_mean": round(float(p.mean()), 3),
                                         "centroid_lat": round(float(gla[rows, cols].mean()), 4),
                                         "centroid_lon": round(float(glo[rows, cols].mean()), 4),
                                         "mode": "day" if M["daylit"][rows, cols].mean() >= 0.5 else "night"}})
        return feats

    def product(self, pixel_area_km2=None) -> dict:
        """ผลิตภัณฑ์แผนที่ฝนตอนนี้ของภาพล่าสุด: กริดความน่าจะเป็น / เป็นฝน / โหมด + บริเวณฝนบนพื้น"""
        M = self.last
        gla, glo = self.ground_grid()
        return {"time": M["time"], "p": M["p"], "raining": M["raining"], "daylit": M["daylit"],
                "lat": M["lat"], "lon": M["lon"], "lat_ground": gla, "lon_ground": glo,
                "areas": self.rain_areas(pixel_area_km2)}

    def pixels(self, pt_lat, pt_lon):
        """(แถว, คอลัมน์) ของพิกเซลเหนือจุด (ย้อน parallax ของยอดเมฆเย็นคงที่ top_k)"""
        M = self.last
        return shifted_pixels(self._index, M["p"].shape, np.asarray(pt_lat, float), np.asarray(pt_lon, float),
                              M["saz"], self.top_k)

    def point_features(self, pt_lat, pt_lon) -> pd.DataFrame:
        M = self.last
        r, c = self.pixels(pt_lat, pt_lon)
        r, c = np.asarray(r), np.asarray(c)
        ny, nx = M["p"].shape
        out = {"rain_now": M["p"][r, c], "rain_near": M["max10"][r, c], "daylit": M["daylit"][r, c]}
        Ls = np.array((0,) + LEADS, float)
        E = np.zeros((len(r), len(Ls)), bool)
        E[:, 0] = M["raining"][r, c]
        have = np.zeros(len(r), bool)
        if M["U"] is not None:
            yy, xx = np.mgrid[-R10:R10 + 1, -R10:R10 + 1]
            dk = yy ** 2 + xx ** 2 <= R10 * R10
            rr = np.clip(r[:, None] + yy[dk][None], 0, ny - 1)
            cc = np.clip(c[:, None] + xx[dk][None], 0, nx - 1)
            u, v = M["U"][rr, cc].mean(1), M["V"][rr, cc].mean(1)
            have = np.isfinite(u) & np.isfinite(v)
            for j, L in enumerate(LEADS, 1):
                ru = np.clip(np.rint(r - v * L).astype(int), 0, ny - 1)
                cu = np.clip(np.rint(c - u * L).astype(int), 0, nx - 1)
                E[:, j] = np.where(have, M["raining"][ru, cu], False)
        n_ok = np.where(have, len(Ls), 1)                      # ไม่มีการเคลื่อนที่ (ภาพแรก) → ดูเฉพาะจุดเอง
        arr, dep = np.full(len(r), NO_LEAD), np.full(len(r), NO_LEAD)
        for i in range(len(r)):
            w = np.nonzero(E[i, :n_ok[i]])[0]
            d = np.nonzero(~E[i, :n_ok[i]])[0]
            arr[i] = Ls[w[0]] if len(w) else NO_LEAD
            dep[i] = Ls[d[0]] if len(d) else NO_LEAD
        out.update({"arrival_min": arr, "rain_arrival": np.log(arr + 10), "departure_min": dep,
                    "rain_departure": np.log(dep + 10)})
        return pd.DataFrame(out)

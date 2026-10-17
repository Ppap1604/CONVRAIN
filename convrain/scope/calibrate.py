"""
ปรับเทียบค่าคงที่ของโมเดลเวลาด้วยข้อมูลฝนที่จุด (เรดาร์ / มาตรวัดฝน) — ใช้ offline เท่านั้น ไม่ใช่ข้อมูลเข้าตอนรันจริง

ขั้นตอน
  1. replay ด้วย ScopeTiming(collect_rows=True) + timing.points_file = จุดที่มีข้อมูลฝน → แถวรายจุด (rows_log)
  2. ฝนที่จุดเป็นช่วง 10 นาที (มม. / 10 นาที, ป้ายเวลา = กลางช่วง) → เหตุการณ์ด้วย point_events
  3. training_rows: ติดเวลาเหตุการณ์ให้แต่ละแถว (หน้าต่าง [t0 − 60, ออกคำทำนายครั้งสุดท้าย + 90 นาที])
  4. fit_event_models: ฟิต นาฬิกา Weibull + hazard รายช่วง ของแต่ละเหตุการณ์ → dict แบบเดียวกับ params.EVENT_MODELS (บันทึกเป็น JSON
     แล้วตั้ง timing.params_file)

นิยามเหตุการณ์ที่จุด (เกณฑ์ฝน 0.5 มม. / 10 นาที ≈ 3 มม./ชม. ≈ 30 dBZ):
  onset = กลางช่วงแรกที่ ≥ เกณฑ์;  peak = กลางช่วงที่ฝนมากสุด;
  stop  = ปลายช่วงเปียกสุดท้าย ก่อนช่วงแห้ง 2 ช่วงติดกันแรกหลัง peak (แห้ง ≥ 20 นาที = กันการหยุดหลอก)
  onset ในช่วงแรกของหน้าต่าง หรือ stop ที่ยังไม่จบก่อนหน้าต่างหมด → ไม่รู้ (ไม่ใช้)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clock import QCOLS, WeibullClock, widen
from .hazard import LightHazard, Switch
from .timing import EVENTS, to_min

RAIN_MIN = 0.5
GRID_C = np.round(np.arange(0.50, 1.501, 0.01), 2)


def radar_mm10(dbz) -> np.ndarray:
    """dBZ → มม. / 10 นาที (Marshall–Palmer Z = 200 R^1.6); 30 dBZ ≈ 0.5 มม. / 10 นาที"""
    z = np.asarray(dbz, dtype=float)
    return np.where(z > 0, (10 ** (z / 10) / 200.0) ** (1 / 1.6), 0.0) / 6.0


def ten_minute_series(times, values, tol_min: float = 8.0) -> pd.Series:
    """
    ค่าฝนที่จุด (เวลาใดก็ได้ เช่น เรดาร์ทุก 15 นาที) → ช่วง 10 นาที (ป้าย = กลางช่วง :05, :15, …):
    แต่ละช่วงใช้ค่าที่เวลาใกล้กลางช่วงที่สุดภายใน ± tol_min นาที ไม่มี → NaN
    """
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    v = np.asarray(values, dtype=float)
    order = np.argsort(t.asi8)
    t, v = t[order], v[order]
    mids = pd.date_range(t.min().floor("10min") + pd.Timedelta(minutes=5), t.max(), freq="10min")
    k = np.clip(np.searchsorted(t.asi8, mids.asi8), 1, len(t) - 1)
    near = np.where(np.abs(t.asi8[k - 1] - mids.asi8) <= np.abs(t.asi8[k] - mids.asi8), k - 1, k)
    ok = np.abs(t.asi8[near] - mids.asi8) <= tol_min * 60e9
    return pd.Series(np.where(ok, v[near], np.nan), index=mids)


def point_events(series: pd.Series, thr: float = RAIN_MIN) -> dict:
    """series: มม. / 10 นาที, index = เวลากลางช่วง (เรียงเวลา) → onset, peak, stop (+ ธงไม่รู้)"""
    s = series.dropna()
    out = {"onset": pd.NaT, "onset_censored": True, "peak": pd.NaT, "peak_mm": np.nan,
           "stop": pd.NaT, "stop_censored": True}
    if s.empty:
        return out
    v, t = s.to_numpy(float), s.index
    p = int(np.argmax(v))
    out["peak"], out["peak_mm"] = t[p], float(v[p])
    wet = np.nonzero(v >= thr)[0]
    if len(wet):
        out["onset"], out["onset_censored"] = t[wet[0]], bool(wet[0] == 0)
    for k in range(p + 1, len(v) - 1):
        if v[k] < thr and v[k + 1] < thr:
            last = [i for i in range(p, k) if v[i] >= thr]
            if last:
                out["stop"], out["stop_censored"] = t[last[-1]] + pd.Timedelta(minutes=5), False
            break
    return out


def training_rows(rows_log: pd.DataFrame, rain: dict, before_min: float = 60, after_min: float = 90) -> pd.DataFrame:
    """
    rows_log : แถวรายจุดจาก ScopeTiming.rows_log (คอลัมน์ cell_id, station, t, b, d2, btd, ตัวแปรคอลัมน์)
    rain     : {station: Series มม. / 10 นาที (index = เวลากลางช่วง UTC)}
    → แถวเดิม + onset / peak / stop (นาที; ไม่รู้ = NaN) เฉพาะคู่ที่มีฝนถึงเกณฑ์
    """
    out = []
    for (cid, st), g in rows_log.groupby(["cell_id", "station"]):
        s = rain.get(st)
        if s is None:
            continue
        lo = pd.Timestamp(g.b.min() * 60 - before_min * 60, unit="s", tz="UTC")
        hi = pd.Timestamp(g.t.max() * 60 + after_min * 60, unit="s", tz="UTC")
        idx = s.index.tz_localize("UTC") if s.index.tz is None else s.index
        ev = point_events(pd.Series(s.to_numpy(), index=idx)[(idx >= lo) & (idx <= hi)])
        if not (ev["peak_mm"] >= RAIN_MIN):
            continue
        g = g.copy()
        for x in EVENTS:
            cens = ev.get(f"{x}_censored", False)
            g[x] = np.nan if (pd.isna(ev[x]) or cens) else to_min(pd.Timestamp(ev[x]))
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def choose_c(Q: pd.DataFrame, obs: np.ndarray) -> float:
    """ตัวขยายช่วงที่ทำให้ช่วง 50 % / 80 % ครอบคลุมใกล้ 50 % / 80 % ที่สุด"""
    score = [abs(np.mean((obs >= W.q25) & (obs <= W.q75)) - .5) + abs(np.mean((obs >= W.q10) & (obs <= W.q90)) - .8)
             for W in (widen(Q, c) for c in GRID_C)]
    return float(GRID_C[int(np.argmin(score))])


def event_rows(rows: pd.DataFrame, x: str) -> pd.DataFrame:
    return rows[rows[x].notna() & (rows.t < rows[x])].reset_index(drop=True)


def fit_event_models(rows: pd.DataFrame, switch_minutes: float = 30.0) -> dict:
    """ฟิตโมเดลของทั้ง 3 เหตุการณ์ (รูปเดียวกับ params.EVENT_MODELS) → dict สำหรับ timing.params_file"""
    out = {}
    for x in EVENTS:
        I = event_rows(rows, x)
        if len(I) < 50:
            raise ValueError(f"{x}: แถวที่ใช้ได้มีแค่ {len(I)} (ต้อง ≥ 50) — เพิ่มวัน / จุด")
        clock = WeibullClock.fit(I, x)
        clock.c = choose_c(clock.quantiles(I), I[x].to_numpy(float))
        haz = LightHazard.for_event(clock, x).fit(I, x)
        out[x] = Switch(haz, clock, switch_minutes).to_dict() if x == "stop" else haz.to_dict()
    return out


def scores(obs: np.ndarray, Q: pd.DataFrame, lead: np.ndarray) -> dict:
    """MAE และ % ภายใน ±5 / 10 / 15 / 30 นาที + ความครอบคลุม แยกตามระยะเวลาก่อนเหตุการณ์จริง"""
    e = Q.q50.to_numpy() - obs
    res = {}
    for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 120), (0, 10 ** 6)):
        m = (lead >= lo) & (lead < hi)
        if not m.any():
            continue
        a = np.abs(e[m])
        res["all" if hi == 10 ** 6 else f"{lo}-{hi}"] = {
            "n": int(m.sum()), "MAE": float(a.mean()), **{f"w{w}": float((a <= w).mean()) for w in (5, 10, 15, 30)},
            "median_err": float(np.median(e[m])),
            "cov50": float(((obs[m] >= Q.q25.to_numpy()[m]) & (obs[m] <= Q.q75.to_numpy()[m])).mean()),
            "cov80": float(((obs[m] >= Q.q10.to_numpy()[m]) & (obs[m] <= Q.q90.to_numpy()[m])).mean())}
    return res


__all__ = ["radar_mm10", "point_events", "training_rows", "fit_event_models", "event_rows", "scores", "QCOLS"]

"""
ฟังก์ชันสำหรับรันการทดลอง (Phase 2–6 ในเอกสาร หัวข้อ 15–17)
ใช้ร่วมกันระหว่างสคริปต์ใน scripts/ และ demo ข้อมูลจำลอง

    replay_scans      รัน pipeline ทีละ scan แบบเดียวกับ real-time แล้วเก็บผล
    split_by_day      แบ่ง train / validation / test ตามวัน (ไม่สุ่มตามแถว, หัวข้อ 10.3)
    prefilter_rows    แถวที่ระบบจะส่งเข้าโมเดล (หัวข้อ 8.1)
    train_models      train onset / peak / stop + เลือก threshold เตือน
    predict_table     ใช้โมเดลหรือ baseline กับตาราง object-scan แบบ offline
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .evaluate import at_risk_rows, best_threshold
from .features import MODEL_FEATURES
from .intensity import RainRateLUT
from .models.hazard import HazardModel
from .pipeline import ConvRainPipeline


# =====================================================================
# [Phase 2–3] replay
# =====================================================================
def replay_scans(pipeline: ConvRainPipeline, items: Iterable, on_result=None) -> dict:
    """
    items: iterable ของ scan หรือ (scan, radar_frames, radar_rainrate)
    on_result: callback(result) เช่น เขียนไฟล์ output ทีละ scan
    คืน dict: timeseries (DataFrame), tracks (track ที่ปิดแล้ว), timings, errors
    """
    ts, closed, timings, errors = [], [], [], []
    for item in items:
        scan, frames, rr = (item if isinstance(item, tuple) else (item, None, None))
        res = pipeline.step(scan, radar_frames=frames, radar_rainrate=rr)
        if on_result is not None:
            on_result(res)
        if res.skipped:
            continue
        if res.timeseries is not None and len(res.timeseries):
            ts.append(res.timeseries)
        if res.closed_tracks is not None:
            closed.append(res.closed_tracks)
        timings.append(res.timings_ms)
        errors.extend(f"{res.time}: {e}" for e in res.errors)
    closed.append(pipeline.finalize())
    return {
        "timeseries": pd.concat(ts, ignore_index=True) if ts else pd.DataFrame(),
        "tracks": pd.concat([c for c in closed if len(c)], ignore_index=True) if any(len(c) for c in closed)
        else pd.DataFrame(columns=["object_id", "end_reason"]),
        "timings": pd.DataFrame(timings),
        "errors": errors,
    }


def fit_lut(pipeline_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], min_count: int = 20) -> RainRateLUT:
    """[Phase 5] สร้าง RainRateLUT จากคู่ pixel ที่ pipeline เก็บไว้ระหว่าง replay"""
    if not pipeline_pairs:
        return RainRateLUT()
    bt = np.concatenate([p[0] for p in pipeline_pairs])
    btd = np.concatenate([p[1] for p in pipeline_pairs])
    rr = np.concatenate([p[2] for p in pipeline_pairs])
    return RainRateLUT(min_count=min_count).fit(bt, btd, rr)


# =====================================================================
# [หัวข้อ 10.3] แบ่งข้อมูลตามวัน
# =====================================================================
def split_by_day(df: pd.DataFrame, fractions=(0.6, 0.2, 0.2), seed: int = 0):
    """
    แบ่งวันแบบเรียงตามเวลา (ค่า default) เพื่อไม่ให้ข้อมูลวันเดียวกันรั่วข้ามชุด
    ถ้าต้องการทดสอบข้ามปี ให้แบ่งเองตามช่วงวันที่แทนฟังก์ชันนี้
    """
    day = pd.to_datetime(df["time"], utc=True).dt.strftime("%Y-%m-%d")
    days = np.array(sorted(day.unique()))
    n = len(days)
    n_tr = max(1, int(round(fractions[0] * n)))
    n_va = max(1, int(round(fractions[1] * n)))
    sets = days[:n_tr], days[n_tr:n_tr + n_va], days[n_tr + n_va:]
    return tuple(df[day.isin(s)].copy() for s in sets)


# =====================================================================
# [หัวข้อ 8.1] prefilter
# =====================================================================
def prefilter_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    oc = cfg["onset"]
    return (df["min_bt"] < oc["prefilter_max_min_bt"]) & (df["n_scans"] >= oc["prefilter_min_age_scans"])


# =====================================================================
# [Phase 4–5] train
# =====================================================================
def train_models(train: pd.DataFrame, val: pd.DataFrame, cfg: dict, estimator: str = "lightgbm",
                 features: list[str] | None = None, balance: bool = False) -> dict:
    """
    train: ตาราง object-scan ที่ attach_labels แล้ว
    คืน dict: onset, peak, stop (HazardModel หรือ None), warn_threshold, val_csi
    """
    oc = cfg["onset"]
    kw = dict(features=features or MODEL_FEATURES, horizon_steps=oc["horizon_steps"], step_min=oc["step_min"],
              estimator=estimator, balance=balance, min_halfwidth_min=oc.get("eta_min_halfwidth_min", 5.0))
    out: dict = {}

    # ---- onset: แถวก่อนฝน ที่ผ่าน prefilter, ตัด track ที่ฝนตกตั้งแต่เกิด ----
    def onset_rows(d):
        d = at_risk_rows(d)
        return d[prefilter_mask(d, cfg)]

    tr_on, va_on = onset_rows(train), onset_rows(val)
    out["onset"] = HazardModel(name="onset", **kw).fit(tr_on, "t_onset", "censor_time", va_on)

    # ---- peak: แถวหลัง onset ก่อน peak ----
    def after(d, col):
        t = pd.to_datetime(d["time"], utc=True)
        return d[pd.to_datetime(d[col], utc=True).notna() & (t >= pd.to_datetime(d[col], utc=True))]

    out["peak"] = out["stop"] = None
    for name, start_col, event_col in (("peak", "t_onset", "t_peak"), ("stop", "t_peak", "t_stop")):
        tr_x, va_x = after(train, start_col), after(val, start_col)
        n_ev = pd.to_datetime(tr_x[event_col], utc=True).gt(pd.to_datetime(tr_x["time"], utc=True)).sum()
        if n_ev >= 10:  # ต้องมี event พอจึงจะ train
            out[name] = HazardModel(name=name, **kw).fit(tr_x, event_col, "censor_time", va_x)

    # ---- เลือก threshold คำเตือนจาก CSI บน validation ----
    pv = predict_table(out["onset"], val, cfg)
    out["warn_threshold"], out["val_csi"] = best_threshold(pv, "p30", 30.0)
    return out


# =====================================================================
# ใช้โมเดล / baseline แบบ offline กับตาราง object-scan
# =====================================================================
def predict_table(model, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    คืนตารางเดิม + p30, p60, eta_*_min
    แถวที่ไม่ผ่าน prefilter ได้ p = 0 (ระบบจริงจะไม่เตือน) เพื่อให้นับ miss ได้ถูกต้อง
    """
    d = at_risk_rows(df).reset_index(drop=True)
    for c in ("p30", "p60", "eta_lo_min", "eta_median_min", "eta_hi_min"):
        d[c] = 0.0 if c in ("p30", "p60") else np.nan
    m = prefilter_mask(d, cfg).values
    if m.any():
        pred = model.predict(d[m]).reset_index(drop=True)
        for c in ("p30", "p60", "eta_lo_min", "eta_median_min", "eta_hi_min"):
            d.loc[m, c] = pred[c].values
    return d

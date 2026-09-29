"""
การประเมินผล  (เอกสาร หัวข้อ 10.4 และ 17 Phase 6)

ใช้ฟังก์ชันชุดเดียวกันประเมินทุกโมเดลและ baseline บนแถวชุดเดียวกัน

Input หลัก: ตาราง object-scan ที่มีคอลัมน์
    object_id, time, t_onset (label), p30, p60, eta_lo_min, eta_median_min, eta_hi_min

Metric
    - Object-based: POD, FAR, CSI ที่ horizon 30 / 60 นาที
    - Timing: MAE และ bias ของเวลา onset (นาที), coverage ของช่วงความไม่แน่นอน
    - Lead time: การกระจายของ lead time ของเหตุการณ์ที่ตรวจจับได้
    - Probability: Brier score, reliability table
    - Bootstrap CI แบบสุ่มตามวัน (แถวในวันเดียวกันสัมพันธ์กัน)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def at_risk_rows(df: pd.DataFrame) -> pd.DataFrame:
    """แถวที่ยังไม่เกิด onset (object ที่ onset ตั้งแต่เกิดถูกตัดทิ้ง)"""
    t = pd.to_datetime(df["time"], utc=True)
    ev = pd.to_datetime(df["t_onset"], utc=True)
    m = (ev.isna() | (ev > t))
    if "onset_at_birth" in df:
        m &= ~df["onset_at_birth"].astype(bool)
    return df[m].copy()


# =====================================================================
# [10.4] object-based POD / FAR / CSI + timing + lead time
# =====================================================================
def object_scores(df: pd.DataFrame, prob_col: str = "p30", horizon_min: float = 30.0,
                  threshold: float = 0.5) -> dict:
    """
    ต่อ object:
        hit  = มีคำเตือน (prob ≥ threshold) อย่างน้อยหนึ่งครั้งภายใน horizon ก่อน onset
        miss = มี onset แต่ไม่มีคำเตือนในช่วงนั้น
        FA   = มีคำเตือนที่ไม่ตามด้วย onset ภายใน horizon (นับ object ละครั้ง)
    timing ประเมิน ณ คำเตือนแรกของ hit: error = eta_median − เวลาจริงถึง onset
    """
    d = at_risk_rows(df)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d["t_onset"] = pd.to_datetime(d["t_onset"], utc=True)
    hits = misses = fas = cns = 0
    leads, errs, covered = [], [], []
    for _, g in d.sort_values("time").groupby("object_id", sort=False):
        ev = g["t_onset"].iloc[0]
        warn = g[g[prob_col] >= threshold]
        to_onset = (ev - warn["time"]).dt.total_seconds() / 60.0 if pd.notna(ev) else pd.Series(dtype=float)
        good = warn[(to_onset > 0) & (to_onset <= horizon_min)] if pd.notna(ev) else warn.iloc[0:0]
        bad = len(warn) > len(good)
        if pd.notna(ev):
            if len(good):
                hits += 1
                first = good.iloc[0]
                lead = (ev - first["time"]).total_seconds() / 60.0
                leads.append(lead)
                if "eta_median_min" in first and np.isfinite(first.get("eta_median_min", np.nan)):
                    errs.append(first["eta_median_min"] - lead)
                    lo, hi = first.get("eta_lo_min", np.nan), first.get("eta_hi_min", np.nan)
                    if np.isfinite(lo) and np.isfinite(hi):
                        covered.append(lo <= lead <= hi)
            else:
                misses += 1
                # object ที่มีเหตุการณ์ แต่ไม่มีแถวในช่วง horizon ก่อน onset จะนับเป็น miss เช่นกัน
        if bad and not (pd.notna(ev) and len(good)):
            fas += 1
        if pd.isna(ev) and not len(warn):
            cns += 1
    pod = hits / (hits + misses) if hits + misses else np.nan
    far = fas / (hits + fas) if hits + fas else np.nan
    csi = hits / (hits + misses + fas) if hits + misses + fas else np.nan
    errs_a = np.array(errs, float)
    return {
        "hits": hits, "misses": misses, "false_alarms": fas, "correct_negatives": cns,
        "POD": pod, "FAR": far, "CSI": csi,
        "onset_MAE_min": float(np.mean(np.abs(errs_a))) if errs_a.size else np.nan,
        "onset_bias_min": float(np.mean(errs_a)) if errs_a.size else np.nan,
        "window_coverage": float(np.mean(covered)) if covered else np.nan,
        "lead_median_min": float(np.median(leads)) if leads else np.nan,
        "lead_p25_min": float(np.percentile(leads, 25)) if leads else np.nan,
        "lead_p75_min": float(np.percentile(leads, 75)) if leads else np.nan,
    }


# =====================================================================
# [10.4] Brier score และ reliability
# =====================================================================
def event_within(df: pd.DataFrame, horizon_min: float) -> np.ndarray:
    t = pd.to_datetime(df["time"], utc=True)
    ev = pd.to_datetime(df["t_onset"], utc=True)
    return ((ev > t) & (ev <= t + pd.Timedelta(minutes=horizon_min))).fillna(False).astype(int).values


def brier(df: pd.DataFrame, prob_col: str = "p30", horizon_min: float = 30.0) -> float:
    d = at_risk_rows(df).dropna(subset=[prob_col])
    if not len(d):
        return np.nan
    y = event_within(d, horizon_min)
    return float(np.mean((d[prob_col].values - y) ** 2))


def reliability_table(df: pd.DataFrame, prob_col: str = "p30", horizon_min: float = 30.0,
                      n_bins: int = 10) -> pd.DataFrame:
    """ความถี่ที่เกิดจริงในแต่ละช่วงความน่าจะเป็น (ควรใกล้เส้นทแยง)"""
    d = at_risk_rows(df).dropna(subset=[prob_col])
    y = event_within(d, horizon_min)
    bins = np.clip((d[prob_col].values * n_bins).astype(int), 0, n_bins - 1)
    t = pd.DataFrame({"bin": bins, "p": d[prob_col].values, "y": y})
    return t.groupby("bin").agg(p_mean=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


# =====================================================================
# [17 Phase 6] bootstrap CI แบบสุ่มตามวัน
# =====================================================================
def bootstrap_ci(df: pd.DataFrame, metric: Callable[[pd.DataFrame], float], n: int = 300,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    d = df.copy()
    d["_day"] = pd.to_datetime(d["time"], utc=True).dt.strftime("%Y-%m-%d")
    days = d["_day"].unique()
    groups = {k: g for k, g in d.groupby("_day")}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        # ต่อท้าย object_id ด้วยลำดับการสุ่ม เพื่อไม่ให้วันที่ถูกสุ่มซ้ำรวมเป็น object เดียว
        sample = pd.concat([groups[k].assign(object_id=groups[k]["object_id"] + f"#{i}")
                            for i, k in enumerate(pick)], ignore_index=True)
        vals.append(metric(sample))
    vals = np.array(vals, float)
    return float(metric(df)), float(np.nanpercentile(vals, 100 * alpha / 2)), \
        float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


# =====================================================================
# รวมทุก metric + แยกกลุ่ม
# =====================================================================
def evaluate_all(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    out = {}
    for h, col in ((30, "p30"), (60, "p60")):
        s = object_scores(df, col, h, threshold)
        out.update({f"{k}@{h}": v for k, v in s.items()})
        out[f"Brier@{h}"] = brier(df, col, h)
    return out


def stratified(df: pd.DataFrame, by: str, threshold: float = 0.5) -> pd.DataFrame:
    """แยกผลตามกลุ่ม เช่น is_day, land/sea (หัวข้อ 10.4 และ 17)"""
    rows = []
    for key, g in df.groupby(by):
        r = evaluate_all(g, threshold)
        r[by] = key
        rows.append(r)
    return pd.DataFrame(rows)


def best_threshold(df: pd.DataFrame, prob_col: str = "p30", horizon_min: float = 30.0,
                   grid=np.arange(0.1, 0.91, 0.05)) -> tuple[float, float]:
    """เลือก threshold คำเตือนที่ให้ CSI สูงสุดบนชุด validation (หัวข้อ 8.4)"""
    best = (0.5, -1.0)
    for thr in grid:
        csi = object_scores(df, prob_col, horizon_min, float(thr))["CSI"]
        if np.isfinite(csi) and csi > best[1]:
            best = (float(thr), float(csi))
    return best

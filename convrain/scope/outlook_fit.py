"""
ปรับเทียบและประเมินผลิตภัณฑ์รายจุด (คำเตือน 3 ระดับ + เวลาฝนเริ่ม / หยุดพร้อมความมั่นใจ) — ใช้ offline เท่านั้น
ข้อมูลฝนที่จุด (มาตรวัดฝน / เรดาร์) ใช้สร้างคำตอบสำหรับฝึกและตรวจเท่านั้น ไม่เป็นข้อมูลเข้าตอนรันจริง

ขั้นตอน (scripts/calibrate_outlook.py):
  1. collect_rows   เล่นภาพตามลำดับเวลา → ตัวแปรรายจุดทุกภาพ (ใช้โค้ดเดียวกับตอนรันจริง)
  2. label_rows     ฝนที่จุดเป็นช่วง 10 นาที (ป้าย = กลางช่วง) → ตัวอย่างฝนเริ่ม / ฝนหยุด
                      ฝนเริ่ม: จุดแห้งครบ 60 นาทีถึงเวลาภาพ; คำตอบ = กลางช่วงแรกที่เปียก (หลังแห้ง 60 นาที) − เวลาภาพ ภายใน 180 นาที
                      ฝนหยุด: ช่วงที่จบ ณ เวลาภาพเปียก; คำตอบ = กลางช่วงแรกของช่วงแห้ง 2 ช่วงติดกัน − เวลาภาพ ภายใน 120 นาที
                      ข้อมูลขาดจนตัดสินไม่ได้ → ไม่ใช้
  3. fit_outlook    แบ่งวันเป็น k กลุ่ม: ฟิตบนกลุ่มอื่น ทายกลุ่มที่เว้นไว้ (ไม่เคยเห็นวันนั้น) → เลือกเกณฑ์คำเตือน, ปรับความมั่นใจ,
                    และวัดผล; สุดท้ายฟิตบนทุกวัน → ค่าคงที่ (JSON) + รายงานผล
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .calibrate import ten_minute_series
from .hazard import _Logit
from .outlook import (DAY_COS, H_ON, H_ST, LEVELS, STOP_SET, PointFeatures, curve_at, onset_set, window_statement)
from .rainmap import RainMap

MIN_MODE_ROWS, MIN_MODE_EVENTS = 300, 50


# ====================================================================== 1. ตัวแปรรายจุด
def collect_rows(scans, points: pd.DataFrame, rain_constants: dict, top_k: float, offset: float | None = None,
                 on_scan=None) -> pd.DataFrame:
    """scans = Scan เรียงเวลา (วันเดียวกัน); points = key, lat, lon → แถว (key, time, ตัวแปร, raining)"""
    rm = RainMap(rain_constants, top_k)
    pf = PointFeatures(points.lat, points.lon, top_k, rm.offset if offset is None else offset)
    out, last = [], None
    for s in scans:
        if last is not None and (s.time - last).total_seconds() / 60 > 30:
            rm.reset()
            pf.reset()
        last = s.time
        rm.update(s)
        rain = rm.point_features(points.lat.to_numpy(float), points.lon.to_numpy(float))
        D = pf.update(s, rain, rm.last["p"])
        r, c = pf.r, pf.c
        D["raining"] = rm.last["raining"][r, c]
        D.insert(0, "time", pd.Timestamp(s.time))
        D.insert(0, "key", points.key.to_numpy())
        out.append(D)
        if on_scan:
            on_scan(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ====================================================================== 2. คำตอบจากฝนที่จุด
def rain_series(rain: pd.DataFrame, value_col: str) -> dict:
    """rain: key, time, ค่า → {key: Series 10 นาที (ป้าย = กลางช่วง, UTC)}"""
    out = {}
    for k, g in rain.groupby("key"):
        out[k] = ten_minute_series(g.time, g[value_col])
    return out


def label_rows(rows: pd.DataFrame, series: dict, wet: float) -> pd.DataFrame:
    """เพิ่มคอลัมน์ onset_ok / onset_lead (นาที, NaN = ไม่ตกใน 180) และ stop_ok / stop_lead (NaN = ไม่หยุดใน 120)"""
    rows = rows.copy()
    n = len(rows)
    on_ok, on_lead = np.zeros(n, bool), np.full(n, np.nan)
    st_ok, st_lead = np.zeros(n, bool), np.full(n, np.nan)
    t = pd.DatetimeIndex(pd.to_datetime(rows.time, utc=True))
    for k, idx in rows.groupby("key").indices.items():
        s = series.get(k)
        if s is None or s.empty:
            continue
        si = s.index.tz_localize("UTC") if s.index.tz is None else s.index
        A = s.to_numpy(float)
        fin = np.isfinite(A)
        dry, wt = fin & (A < wet), fin & (A >= wet)
        prev6 = np.ones(len(A), bool)
        for j in range(1, 7):
            sh = np.zeros(len(A), bool)
            sh[j:] = dry[:-j]
            prev6 &= sh
        onset = wt & prev6
        base = si[0]
        i0 = np.rint(((t[idx] - pd.Timedelta(minutes=5)) - base).total_seconds().to_numpy() / 600.0).astype(int)
        for row, i in zip(idx, i0):
            if i < 5 or i >= len(A):
                continue
            if dry[i - 5:i + 1].all():                             # ฝนเริ่ม: แห้ง 60 นาทีถึงเวลาภาพ
                seg = onset[i + 1:i + 19]
                hit = np.nonzero(seg)[0]
                if len(hit):
                    on_ok[row], on_lead[row] = True, 10.0 * (hit[0] + 1) - 5
                elif len(seg) == 18 and fin[i + 1:i + 19].all():
                    on_ok[row] = True
            elif wt[i]:                                            # ฝนหยุด: ช่วงที่จบ ณ เวลาภาพเปียก
                d2 = dry[i + 1:i + 15]
                two = np.nonzero(d2[:-1] & d2[1:])[0]
                if len(two) and 10.0 * (two[0] + 1) - 5 <= 120:
                    st_ok[row], st_lead[row] = True, 10.0 * (two[0] + 1) - 5
                elif len(d2) == 14 and fin[i + 1:i + 14].all():
                    st_ok[row] = True
    rows["onset_ok"], rows["onset_lead"] = on_ok, on_lead
    rows["stop_ok"], rows["stop_lead"] = st_ok, st_lead
    return rows


# ====================================================================== 3. ฟิต
def fit_lr(D: pd.DataFrame, cols: list, y: np.ndarray) -> dict:
    X = D[cols].to_numpy(float)
    med = np.nanmedian(X, 0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xf = np.where(np.isfinite(X), X, med)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    w = _Logit(1.0).fit(np.hstack([np.ones((len(X), 1)), (Xf - mu) / sd]), y.astype(float)).w
    return {"cols": list(cols), "med": med.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def _prob(P: dict, D: pd.DataFrame) -> np.ndarray:
    from .outlook import LR
    return LR(P).prob(D)


def fit_onset(D: pd.DataFrame, lead: np.ndarray) -> dict:
    day = D.cos_sun.to_numpy(float) >= DAY_COS
    out = {}
    for b in H_ON:
        y = np.nan_to_num(lead, nan=1e9) <= b
        cols = onset_set(b)
        m = {"all": fit_lr(D, cols, y), "day": None, "night": None}
        if b > 60:                                                 # ระยะไกล: ค่าคงที่แยกกลางวัน / กลางคืน
            for mode, sel in (("day", day), ("night", ~day)):
                if sel.sum() >= MIN_MODE_ROWS and y[sel].sum() >= MIN_MODE_EVENTS:
                    m[mode] = fit_lr(D[sel], cols, y[sel])
        out[str(b)] = m
    return out


def onset_curve(M: dict, D: pd.DataFrame) -> np.ndarray:
    day = D.cos_sun.to_numpy(float) >= DAY_COS
    cols = []
    for b in H_ON:
        m = M[str(b)]
        p = _prob(m["all"], D)
        for mode, sel in (("day", day), ("night", ~day)):
            if m.get(mode) and sel.any():
                p[sel] = _prob(m[mode], D[sel])
        cols.append(p)
    return np.maximum.accumulate(np.column_stack(cols), axis=1)


def fit_stop(D: pd.DataFrame, lead: np.ndarray) -> dict:
    return {str(b): fit_lr(D, STOP_SET, np.nan_to_num(lead, nan=1e9) <= b) for b in H_ST}


def stop_curve(M: dict, D: pd.DataFrame) -> np.ndarray:
    return np.maximum.accumulate(np.column_stack([_prob(M[str(b)], D) for b in H_ST]), axis=1)


def best_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """เกณฑ์ที่ให้ CSI สูงสุด"""
    o = np.argsort(-p)
    ys = y[o].astype(float)
    hits = np.cumsum(ys)
    csi = hits / (np.arange(1, len(p) + 1) + (ys.sum() - hits))
    return float(p[o][int(np.argmax(csi))])


def _folds(days: np.ndarray, k: int) -> dict:
    ud = np.array(sorted(set(days)))
    return {d: i % k for i, d in enumerate(ud)}


def _scores(alarm, y):
    h, m, fa = np.sum(alarm & y), np.sum(~alarm & y), np.sum(alarm & ~y)
    return {"POD": float(h / max(h + m, 1)), "FAR": float(fa / max(h + fa, 1)), "CSI": float(h / max(h + m + fa, 1)),
            "base_rate": float(y.mean())}


def _ibs(F: np.ndarray, H, lead: np.ndarray) -> float:
    knots = np.r_[0.0, np.asarray(H, float)]
    tot = 0.0
    for j in range(1, len(knots)):
        y = (np.nan_to_num(lead, nan=1e9) <= knots[j]).astype(float)
        tot += (knots[j] - knots[j - 1]) * np.mean((F[:, j - 1] - y) ** 2)
    return float(tot / knots[-1])


def _statement_report(lead, L, c, days, H):
    ver = np.isfinite(lead) & np.isfinite(L) & (np.abs(lead - L) <= 15)
    rel = []
    for lo in np.arange(0, 1, 0.1):
        m = (c >= lo) & (c < lo + 0.1 + (1e-9 if lo >= 0.9 else 0))
        if m.sum():
            rel.append({"bin": f"{lo:.1f}-{lo + 0.1:.1f}", "n": int(m.sum()), "confidence": float(c[m].mean()),
                        "verified": float(ver[m].mean())})
    ece = float(sum(r["n"] * abs(r["confidence"] - r["verified"]) for r in rel) / max(len(c), 1))
    n_days = len(set(days))
    gate = {}
    for cs in (0.5, 0.7):
        m = c >= cs
        gate[str(cs)] = {"statements": int(m.sum()), "verified": float(ver[m].mean()) if m.any() else None}
    acc = {}
    for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 120)):
        m = np.isfinite(lead) & (lead > lo) & (lead <= hi) & np.isfinite(L)
        if m.any():
            e = np.abs(L[m] - lead[m])
            acc[f"{lo}-{hi}"] = {"n": int(m.sum()), "within_15": float(np.mean(e <= 15)), "MAE": float(e.mean())}
    m = np.isfinite(lead) & np.isfinite(L)
    pooled = {"n": int(m.sum()), "within_15": float(np.mean(np.abs(L[m] - lead[m]) <= 15)) if m.any() else None,
              "MAE": float(np.mean(np.abs(L[m] - lead[m]))) if m.any() else None}
    return {"verified_share_all": float(ver.mean()), "reliability": rel, "ECE": ece, "confidence_gate": gate,
            "days": n_days, "accuracy_by_true_lead": acc, "accuracy_events_pooled": pooled}


def fit_outlook(rows: pd.DataFrame, k: int = 5, log=print) -> tuple[dict, dict]:
    """rows จาก label_rows (+ คอลัมน์ day) → (ค่าคงที่, รายงานผลจากวันที่โมเดลไม่เคยเห็น)"""
    rep = {"folds": k}
    const = {"levels": {}, "confidence": {}}
    # ---------------- ฝนเริ่ม
    O = rows[rows.onset_ok & ~rows.raining.astype(bool)].reset_index(drop=True)
    lead = O.onset_lead.to_numpy(float)
    days = O.day.to_numpy()
    fm = _folds(days, k)
    fold = np.array([fm[d] for d in days])
    P = np.zeros((len(O), len(H_ON)))
    alarms = {name: np.zeros(len(O), bool) for name, _ in LEVELS}
    for f in range(k):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        M = fit_onset(O[tr], lead[tr])
        P[te] = onset_curve(M, O[te])
        Ptr = onset_curve(M, O[tr])
        for name, b in LEVELS:
            j = H_ON.index(b)
            y = np.nan_to_num(lead[tr], nan=1e9) <= b
            alarms[name][te] = P[te, j] >= best_threshold(Ptr[:, j], y)
        log(f"  onset fold {f + 1}/{k}")
    L, c_raw = window_statement(P, H_ON, H_ON[-1])
    hit = (np.isfinite(lead) & (np.abs(lead - L) <= 15)).astype(float)
    c = np.zeros(len(O))
    for f in range(k):
        tr, te = fold != f, fold == f
        if te.any():
            c[te] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(c_raw[tr], hit[tr]).predict(c_raw[te])
    clim = np.array([np.mean(np.nan_to_num(lead, nan=1e9) <= b) for b in H_ON])
    rep["onset"] = {"samples": int(len(O)), "event_within_180": float(np.mean(np.isfinite(lead))),
                    "IBS": _ibs(P, H_ON, lead), "IBS_climatology": _ibs(np.tile(clim, (len(O), 1)), H_ON, lead),
                    "levels": {name: _scores(alarms[name], np.nan_to_num(lead, nan=1e9) <= b) for name, b in LEVELS},
                    "statement": _statement_report(lead, L, c, days, H_ON)}
    rep["onset"]["IBS_skill"] = 1 - rep["onset"]["IBS"] / rep["onset"]["IBS_climatology"]
    for name, b in LEVELS:
        const["levels"][name] = best_threshold(P[:, H_ON.index(b)], np.nan_to_num(lead, nan=1e9) <= b)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(c_raw, hit)
    const["confidence"]["onset"] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    const["onset"] = fit_onset(O, lead)
    log("  onset done")
    # ---------------- ฝนหยุด
    S = rows[rows.stop_ok].reset_index(drop=True)
    sl = S.stop_lead.to_numpy(float)
    sd = S.day.to_numpy()
    fms = _folds(sd, k)
    sf = np.array([fms[d] for d in sd])
    PS = np.zeros((len(S), len(H_ST)))
    for f in range(k):
        tr, te = sf != f, sf == f
        if te.any():
            PS[te] = stop_curve(fit_stop(S[tr], sl[tr]), S[te])
    Ls, cs_raw = window_statement(PS, H_ST, H_ST[-1])
    hs = (np.isfinite(sl) & (np.abs(sl - Ls) <= 15)).astype(float)
    cs = np.zeros(len(S))
    for f in range(k):
        tr, te = sf != f, sf == f
        if te.any():
            cs[te] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cs_raw[tr], hs[tr]).predict(cs_raw[te])
    clims = np.array([np.mean(np.nan_to_num(sl, nan=1e9) <= b) for b in H_ST])
    rep["stop"] = {"samples": int(len(S)), "stop_within_120": float(np.mean(np.isfinite(sl))),
                   "IBS": _ibs(PS, H_ST, sl), "IBS_climatology": _ibs(np.tile(clims, (len(S), 1)), H_ST, sl),
                   "statement": _statement_report(sl, Ls, cs, sd, H_ST),
                   "map_says_raining_share": float(S.raining.astype(bool).mean())}
    rep["stop"]["IBS_skill"] = 1 - rep["stop"]["IBS"] / rep["stop"]["IBS_climatology"]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cs_raw, hs)
    const["confidence"]["stop"] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    const["stop"] = fit_stop(S, sl)
    log("  stop done")
    return const, rep


__all__ = ["collect_rows", "rain_series", "label_rows", "fit_outlook", "curve_at"]

"""
ฟังก์ชันสำหรับรันย้อนหลัง (replay) ปรับเทียบ และแบ่งข้อมูลตามวัน (Phase 2–6 ในเอกสาร หัวข้อ 15–17)
ใช้ร่วมกันระหว่างสคริปต์ใน scripts/ และ demo ข้อมูลจำลอง

    replay_scans      รัน pipeline ทีละ scan แบบเดียวกับ real-time แล้วเก็บผล
    fit_lut           สร้าง rain rate LUT จากคู่ pixel ที่เก็บระหว่าง replay
    split_by_day      แบ่ง train / validation / test ตามวัน (ไม่สุ่มตามแถว, หัวข้อ 10.3)
การปรับเทียบเวลาฝนเริ่ม / หนักสุด / หยุด อยู่ที่ convrain/scope/calibrate.py (scripts/calibrate_timing.py)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .intensity import RainRateLUT
from .pipeline import ConvRainPipeline


# =====================================================================
# [Phase 2–3] replay
# =====================================================================
def replay_scans(pipeline: ConvRainPipeline, items: Iterable, on_result=None) -> dict:
    """
    items: iterable ของ scan หรือ (scan, radar_frames, radar_rainrate)
    on_result: callback(result) เช่น เขียนไฟล์ output ทีละ scan
    คืน dict: timeseries (DataFrame), points (เวลาฝนรายจุด), tracks (track ที่ปิดแล้ว), timings, errors
    """
    ts, pts, closed, timings, errors = [], [], [], [], []
    for item in items:
        scan, frames, rr = (item if isinstance(item, tuple) else (item, None, None))
        res = pipeline.step(scan, radar_frames=frames, radar_rainrate=rr)
        if on_result is not None:
            on_result(res)
        if res.skipped:
            continue
        if res.timeseries is not None and len(res.timeseries):
            ts.append(res.timeseries)
        if res.points is not None and len(res.points):
            pts.append(res.points)
        if res.closed_tracks is not None:
            closed.append(res.closed_tracks)
        timings.append(res.timings_ms)
        errors.extend(f"{res.time}: {e}" for e in res.errors)
    closed.append(pipeline.finalize())
    return {
        "timeseries": pd.concat(ts, ignore_index=True) if ts else pd.DataFrame(),
        "points": pd.concat(pts, ignore_index=True) if pts else pd.DataFrame(),
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
def split_by_day(df: pd.DataFrame, fractions=(0.6, 0.2, 0.2), seed: int = 0, time_col: str = "time"):
    """
    แบ่งวันแบบเรียงตามเวลา (ค่า default) เพื่อไม่ให้ข้อมูลวันเดียวกันรั่วข้ามชุด
    ถ้าต้องการทดสอบข้ามปี ให้แบ่งเองตามช่วงวันที่แทนฟังก์ชันนี้
    """
    day = pd.to_datetime(df[time_col], utc=True).dt.strftime("%Y-%m-%d")
    days = np.array(sorted(day.unique()))
    n = len(days)
    n_tr = max(1, int(round(fractions[0] * n)))
    n_va = max(1, int(round(fractions[1] * n)))
    sets = days[:n_tr], days[n_tr:n_tr + n_va], days[n_tr + n_va:]
    return tuple(df[day.isin(s)].copy() for s in sets)

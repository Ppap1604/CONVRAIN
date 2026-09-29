"""การประเมินผล (evaluate.py) unit tests"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from convrain.evaluate import (
    at_risk_rows,
    best_threshold,
    bootstrap_ci,
    brier,
    evaluate_all,
    object_scores,
    reliability_table,
    stratified,
)

T0 = datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc)


def test_object_scores_exact_contingency():
    # ออกแบบ 4 object ให้ผลลัพธ์เป็น Hit, Miss, False Alarm, Correct Negative ชัดเจน
    rows = [
        # 1. Hit: มี onset ที่ 6:25 และมีการเตือนที่ 6:10 (lead time = 15 min, ใน horizon 30)
        {"object_id": "hit_obj", "time": T0 + timedelta(minutes=10), "t_onset": T0 + timedelta(minutes=25),
         "p30": 0.8, "eta_median_min": 15.0, "eta_lo_min": 10.0, "eta_hi_min": 20.0},

        # 2. Miss: มี onset ที่ 6:25 แต่ความน่าจะเป็นต่ำ ไม่เตือน (p30=0.2)
        {"object_id": "miss_obj", "time": T0 + timedelta(minutes=10), "t_onset": T0 + timedelta(minutes=25),
         "p30": 0.2, "eta_median_min": np.nan, "eta_lo_min": np.nan, "eta_hi_min": np.nan},

        # 3. False Alarm: ไม่มี onset (NaT) แต่เตือน (p30=0.9)
        {"object_id": "fa_obj", "time": T0 + timedelta(minutes=10), "t_onset": pd.NaT,
         "p30": 0.9, "eta_median_min": 20.0, "eta_lo_min": 10.0, "eta_hi_min": 30.0},

        # 4. Correct Negative: ไม่มี onset (NaT) และไม่เตือน (p30=0.1)
        {"object_id": "cn_obj", "time": T0 + timedelta(minutes=10), "t_onset": pd.NaT,
         "p30": 0.1, "eta_median_min": np.nan, "eta_lo_min": np.nan, "eta_hi_min": np.nan},
    ]
    df = pd.DataFrame(rows)

    scores = object_scores(df, prob_col="p30", horizon_min=30.0, threshold=0.5)

    assert scores["hits"] == 1
    assert scores["misses"] == 1
    assert scores["false_alarms"] == 1
    assert scores["correct_negatives"] == 1

    # POD = 1 / (1 + 1) = 0.5
    assert abs(scores["POD"] - 0.5) < 1e-5
    # FAR = 1 / (1 + 1) = 0.5
    assert abs(scores["FAR"] - 0.5) < 1e-5
    # CSI = 1 / (1 + 1 + 1) = 1/3
    assert abs(scores["CSI"] - 1.0 / 3.0) < 1e-5

    # Lead time = 15 นาที, Error = 15 - 15 = 0
    assert abs(scores["lead_median_min"] - 15.0) < 1e-5
    assert abs(scores["onset_MAE_min"] - 0.0) < 1e-5
    assert abs(scores["window_coverage"] - 1.0) < 1e-5


def test_brier_and_reliability():
    # เหตุการณ์เกิดจริงภายใน 30 นาที
    df = pd.DataFrame([
        {"object_id": "o1", "time": T0, "t_onset": T0 + timedelta(minutes=20), "p30": 1.0},
        {"object_id": "o2", "time": T0, "t_onset": pd.NaT, "p30": 0.0},
    ])
    # การพยากรณ์สมบูรณ์แบบ -> Brier score = 0.0
    bs = brier(df, prob_col="p30", horizon_min=30.0)
    assert abs(bs - 0.0) < 1e-5

    # ตาราง reliability
    rel = reliability_table(df, prob_col="p30", horizon_min=30.0, n_bins=2)
    assert len(rel) == 2


def test_bootstrap_ci_and_best_threshold():
    # สร้างข้อมูลสุ่ม 5 วัน
    rows = []
    for d in range(5):
        day_t = T0 + timedelta(days=d)
        for i in range(10):
            onset = day_t + timedelta(minutes=20) if i < 5 else pd.NaT
            rows.append({
                "object_id": f"d{d}_o{i}",
                "time": day_t,
                "t_onset": onset,
                "p30": 0.8 if i < 4 else 0.2,
                "p60": 0.9 if i < 4 else 0.3,
            })
    df = pd.DataFrame(rows)

    # ทดสอบ bootstrap_ci
    point, lo, hi = bootstrap_ci(df, lambda d: object_scores(d, "p30", 30.0, 0.5)["CSI"], n=50)
    assert lo <= point + 1e-5
    assert point <= hi + 1e-5

    # ทดสอบ best_threshold
    thr, max_csi = best_threshold(df, prob_col="p30", horizon_min=30.0)
    assert 0.1 <= thr <= 0.9
    assert max_csi > 0.0


def test_stratified_evaluation():
    df = pd.DataFrame([
        {"object_id": "o1", "time": T0, "t_onset": T0 + timedelta(minutes=20), "p30": 0.8, "p60": 0.9, "zone": "land"},
        {"object_id": "o2", "time": T0, "t_onset": pd.NaT, "p30": 0.2, "p60": 0.3, "zone": "land"},
        {"object_id": "o3", "time": T0, "t_onset": T0 + timedelta(minutes=20), "p30": 0.7, "p60": 0.8, "zone": "sea"},
        {"object_id": "o4", "time": T0, "t_onset": pd.NaT, "p30": 0.9, "p60": 0.9, "zone": "sea"},
    ])
    strat = stratified(df, by="zone", threshold=0.5)
    assert len(strat) == 2
    assert "zone" in strat.columns
    assert set(strat["zone"]) == {"land", "sea"}

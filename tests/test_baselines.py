"""ขั้น 5: Baseline models (MB06, Extrapolation, Climatology) unit tests"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from convrain.models.baseline import (
    ClimatologyBaseline,
    ExtrapolationBaseline,
    MB06InterestFields,
)

T0 = datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc)


def test_mb06_score_and_predict():
    mb = MB06InterestFields()
    # สร้างแถวที่ผ่านเกณฑ์เกือบหมด (เมฆกำลังพัฒนาอย่างรวดเร็ว)
    rows = pd.DataFrame([{
        "object_id": "c1",
        "min_bt": 250.0,
        "min_since_below_0c": 15.0,
        "d_cold_bt_10": -4.0,  # cool15 = -6 < -4
        "d_cold_bt_30": -10.0,
        "btd_wv_cold": -20.0,  # between -35 and -10
        "btd_co2_cold": -15.0, # between -25 and -5
        "d_btd_wv_10": 3.0,    # trend = 4.5 > 3.0
        "d_btd_co2_10": 3.0,   # trend = 4.5 > 3.0
    }])
    score_df = mb.score(rows)
    assert score_df.shape == (1, 8)
    assert score_df.iloc[0].sum() == 8

    pred = mb.predict(rows)
    assert pred.shape[0] == 1
    assert pred["p30"].iloc[0] == 1.0
    assert pred["p60"].iloc[0] == 1.0
    assert pred["eta_median_min"].iloc[0] == 30.0

    # แถวที่ไม่ผ่านเกณฑ์เลย (เมฆอุ่น นิ่ง)
    rows_dead = pd.DataFrame([{
        "object_id": "c2",
        "min_bt": 285.0,
        "min_since_below_0c": 99.0,
        "d_cold_bt_10": 1.0,
        "d_cold_bt_30": 2.0,
        "btd_wv_cold": 0.0,
        "btd_co2_cold": 0.0,
        "d_btd_wv_10": 0.0,
        "d_btd_co2_10": 0.0,
    }])
    score_dead = mb.score(rows_dead)
    assert score_dead.iloc[0].sum() == 0
    pred_dead = mb.predict(rows_dead)
    assert pred_dead["p30"].iloc[0] == 0.0
    assert np.isnan(pred_dead["eta_median_min"].iloc[0])


def test_extrapolation_baseline():
    extrap = ExtrapolationBaseline(onset_bt=240.0)

    rows = pd.DataFrame([
        # c1: เย็นลงเร็ว cold_bt_kf=260, rate=-5 K/10 min -> ETA = (240-260)/-5 * 10 = 40 นาที
        {"object_id": "c1", "cold_bt_kf": 260.0, "cooling_rate_kf": -5.0},
        # c2: ถึงจุด onset แล้ว cold_bt_kf=230 -> ETA = 0
        {"object_id": "c2", "cold_bt_kf": 230.0, "cooling_rate_kf": -2.0},
        # c3: กำลังอุ่นขึ้น cooling_rate_kf=2.0 -> ไม่เตือน (ETA = NaN)
        {"object_id": "c3", "cold_bt_kf": 270.0, "cooling_rate_kf": 2.0},
    ])

    pred = extrap.predict(rows)
    assert pred["p30"].iloc[0] == 0.0
    assert pred["p60"].iloc[0] == 1.0
    assert abs(pred["eta_median_min"].iloc[0] - 40.0) < 1e-3

    assert pred["p30"].iloc[1] == 1.0
    assert pred["eta_median_min"].iloc[1] == 0.0

    assert pred["p60"].iloc[2] == 0.0
    assert np.isnan(pred["eta_median_min"].iloc[2])


def test_climatology_baseline():
    clim = ClimatologyBaseline()

    # ข้อมูลฝึกสอน: ฝนเกิดช่วงบ่าย 14:00 - 16:00
    times = [T0 + timedelta(hours=i) for i in range(24)]
    train_df = pd.DataFrame({
        "object_id": [f"o{i}" for i in range(24)],
        "time": times,
        "lst_hours": [float(i) for i in range(24)],
        "t_onset": [times[i] + timedelta(minutes=15) if 13 <= i <= 16 else pd.NaT for i in range(24)],
    })

    clim.fit(train_df)

    # ทดสอบ predict
    test_rows = pd.DataFrame({
        "object_id": ["t_afternoon", "t_morning"],
        "lst_hours": [14.2, 5.0],
    })
    pred = clim.predict(test_rows)
    assert pred["p30"].iloc[0] > pred["p30"].iloc[1]
    assert (pred["p30"] <= pred["p60"]).all()
    assert (pred["p30"] >= 0.0).all() and (pred["p60"] <= 1.0).all()

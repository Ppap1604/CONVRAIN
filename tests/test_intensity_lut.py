"""ขั้น 6: ความรุนแรง (LUT) และสถานะ lifecycle unit tests"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from convrain.intensity import (
    RainRateLUT,
    auto_estimator_mmh,
    stop_eta_fallback,
    update_status,
)
from convrain.track import KalmanCV, Track


def test_auto_estimator_bounds():
    # Cold temperatures should produce high rain rate
    rr_cold = auto_estimator_mmh(200.0)
    assert rr_cold > 20.0
    assert rr_cold <= 150.0

    # Warm temperatures should produce zero or negligible rain rate
    rr_warm = auto_estimator_mmh(290.0)
    assert rr_warm < 0.1

    # Clamping behavior
    assert auto_estimator_mmh(100.0) == auto_estimator_mmh(150.0)
    assert auto_estimator_mmh(400.0) == auto_estimator_mmh(330.0)


def test_rain_rate_lut_fit_predict_save_load(tmp_path):
    lut = RainRateLUT(min_count=5)
    rng = np.random.default_rng(42)

    # สร้างคู่ข้อมูลจำลอง
    n = 200
    bt = rng.uniform(200.0, 280.0, n)
    btd = rng.uniform(-20.0, 2.0, n)
    # rain rate แปรผันตามความเย็น
    rr = np.clip((280.0 - bt) * 0.5 + rng.normal(0, 1, n), 0, 50)

    lut.fit(bt, btd, rr)
    assert lut.calibrated

    # ตรวจสอบ predict แบบ 2D
    pred_2d = lut.predict(np.array([210.0, 270.0]), np.array([-10.0, -5.0]))
    assert pred_2d[0] > pred_2d[1]

    # ตรวจสอบ fallback 1D เมื่อ BTD เป็น NaN
    pred_nan_btd = lut.predict(np.array([210.0]), np.array([np.nan]))
    assert np.isfinite(pred_nan_btd[0])
    assert pred_nan_btd[0] > 0.0

    # ตรวจสอบ bt_threshold
    thr_bt = lut.bt_threshold(5.0)
    assert 180.0 <= thr_bt <= 300.0

    # บันทึกและโหลดกลับมา (Roundtrip)
    save_file = tmp_path / "lut.npz"
    lut.save(save_file)
    lut_loaded = RainRateLUT.load(save_file)
    assert lut_loaded.calibrated
    np.testing.assert_allclose(lut.table, lut_loaded.table)
    np.testing.assert_allclose(lut.table_1d, lut_loaded.table_1d)


def test_lifecycle_state_machine(cfg):
    t0 = datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc)
    class MockObj:
        pass
    track = Track(
        track_id="t1",
        birth_time=t0,
        last_time=t0,
        obj=MockObj(),
        kf_bt=KalmanCV(240.0, 1.0, 1.0),
        kf_area=KalmanCV(10.0, 1.0, 1.0)
    )
    assert track.status == "developing"

    # 1. developing -> raining (เมื่อ rain_rate_max >= เกณฑ์ 1.0)
    feats = {"cooling_rate_kf": -5.0, "cold_bt": 230.0, "cold_bt_kf": 230.0}
    rain = {"rain_rate_max_mmh": 5.0, "rain_rate_mean_mmh": 4.0, "rain_area_km2": 20.0}
    s = update_status(track, feats, rain, t0, cfg)
    assert s == "raining"
    assert track.rain_started == t0

    # 2. raining -> peak (เมื่อ cooling rate เข้าใกล้ 0 และ BT เย็นสุด)
    t1 = t0 + timedelta(minutes=20)
    feats_peak = {"cooling_rate_kf": 0.1, "cold_bt": 218.0, "cold_bt_kf": 218.0}
    track.min_cold_bt = 218.0
    s = update_status(track, feats_peak, rain, t1, cfg)
    assert s == "peak"
    assert track.peak_time == t1

    # 3. peak -> decaying (เมื่อ cooling_rate > decay_rate_k หรืออุ่นขึ้น)
    t2 = t1 + timedelta(minutes=20)
    feats_decay = {"cooling_rate_kf": 3.0, "cold_bt": 225.0, "cold_bt_kf": 225.0}
    s = update_status(track, feats_decay, rain, t2, cfg)
    assert s == "decaying"

    # 4. decaying -> raining (re-intensification: กลับมาเย็นลงเร็วและยังมีฝน)
    t3 = t2 + timedelta(minutes=20)
    feats_reint = {"cooling_rate_kf": -4.0, "cold_bt": 220.0, "cold_bt_kf": 220.0}
    s = update_status(track, feats_reint, rain, t3, cfg)
    assert s == "raining"


def test_stop_eta_fallback(cfg):
    t0 = datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc)
    lut = RainRateLUT()
    # เมฆกำลังอุ่นขึ้น cold_bt_kf = 220.0 (ยังเปียก) warming rate = 2.0 K/10 min
    feats = {"cooling_rate_kf": 2.0, "cold_bt_kf": 220.0}
    stop_time = stop_eta_fallback(feats, lut, t0, cfg)
    assert stop_time is not None
    assert stop_time > t0

    # ถ้าเมฆยังเย็นลง cooling_rate_kf <= 0.2 -> คืน None
    feats_cooling = {"cooling_rate_kf": -1.0, "cold_bt_kf": 230.0}
    assert stop_eta_fallback(feats_cooling, lut, t0, cfg) is None

"""ขั้น 4: Feature extraction (interest fields) unit tests"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from conftest import blob, make_scan
from convrain.detect import detect_objects
from convrain.features import MODEL_FEATURES, _cold_mean, _finite_max, _trend, compute_features
from convrain.track import Tracker


def test_cold_mean_and_finite_max():
    # Normal case
    vals = np.array([200.0, 210.0, 220.0, 230.0])
    cold_idx = np.array([0, 1])
    assert _cold_mean(vals, cold_idx) == 205.0
    assert _finite_max(vals) == 230.0

    # With NaNs
    vals_nan = np.array([np.nan, 210.0, np.nan, 230.0])
    assert _cold_mean(vals_nan, np.array([0, 1])) == 210.0
    assert _finite_max(vals_nan) == 230.0

    # Empty / All NaNs
    all_nan = np.array([np.nan, np.nan])
    assert np.isnan(_cold_mean(all_nan, np.array([0])))
    assert np.isnan(_finite_max(all_nan))


def test_trend_lagrangian():
    now = datetime(2026, 6, 1, 6, 30, tzinfo=timezone.utc)
    history = [
        {"time": now - timedelta(minutes=30), "cold_bt": 270.0},
        {"time": now - timedelta(minutes=20), "cold_bt": 260.0},
        {"time": now - timedelta(minutes=10), "cold_bt": 245.0},
    ]
    # Current is 230.0 -> trend 10 min: 230 - 245 = -15
    assert _trend(history, now, 230.0, "cold_bt", 10) == -15.0
    # Trend 30 min: 230 - 270 = -40
    assert _trend(history, now, 230.0, "cold_bt", 30) == -40.0
    # Window beyond history -> returns NaN
    assert np.isnan(_trend(history, now, 230.0, "cold_bt", 40))


def test_compute_features_all_keys_present(cfg, t0):
    bt = blob((80, 80), 40, 40, 6, 225)
    scan = make_scan(bt, t0, cfg)
    det = detect_objects(scan, cfg)
    assert len(det.objects) == 1

    tracker = Tracker(cfg)
    tracker.update(det, None, scan.time, dt_min=10.0)
    track = list(tracker.tracks.values())[0]

    feats = compute_features(track, scan, None, cfg)

    # ตรวจสอบว่า MODEL_FEATURES ครบทุกตัว
    for k in MODEL_FEATURES:
        assert k in feats, f"Missing feature: {k}"

    # ตรวจสอบ Local Solar Time
    assert 0.0 <= feats["lst_hours"] <= 24.0
    assert -1.0 <= feats["lst_sin"] <= 1.0
    assert -1.0 <= feats["lst_cos"] <= 1.0

    # ตรวจสอบว่า track.history เก็บประวัติ
    assert len(track.history) == 1
    assert track.history[0]["min_bt"] == feats["min_bt"]

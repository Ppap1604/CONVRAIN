"""ขั้น 2–3: motion field และ tracking"""
from datetime import timedelta

import numpy as np
from conftest import blob, make_scan

from convrain.detect import detect_objects
from convrain.motion import MotionEstimator
from convrain.track import KalmanCV, Tracker


def textured_field(shape, seed=0):
    """สนามเมฆที่มี texture หลายก้อน ให้ optical flow ติดตามได้"""
    rng = np.random.default_rng(seed)
    bt = np.full(shape, 296.0, np.float32)
    for _ in range(25):
        r, c = rng.uniform(15, shape[0] - 15), rng.uniform(15, shape[1] - 15)
        bt = np.minimum(bt, blob(shape, r, c, rng.uniform(3, 6), rng.uniform(210, 265)))
    return bt


def test_dis_recovers_shift(cfg):
    a = textured_field((120, 120))
    b = np.roll(np.roll(a, 3, axis=0), 2, axis=1)   # เลื่อนลง 3 แถว ขวา 2 คอลัมน์
    me = MotionEstimator(cfg)
    mf = me.estimate(a, b, dt_min=10.0, pixel_km=2.0)
    cloudy = (a < 273)[20:100, 20:100]
    v = mf.v_px[20:100, 20:100][cloudy].mean()
    u = mf.u_px[20:100, 20:100][cloudy].mean()
    assert abs(v - 3) < 0.8 and abs(u - 2) < 0.8


def test_track_keeps_id_and_cooling_rate(cfg, t0):
    me, tr = MotionEstimator(cfg), Tracker(cfg)
    prev, ids = None, set()
    for k in range(6):
        bt = blob((100, 100), 30 + 3 * k, 30 + 2 * k, 5 + 0.3 * k, 270 - 5 * k)  # เคลื่อนที่และเย็นลง 5 K/scan
        scan = make_scan(bt, t0 + timedelta(minutes=10 * k), cfg)
        det = detect_objects(scan, cfg)
        mf = me.estimate(prev, scan.ir, 10.0, 2.0) if prev is not None else None
        upd = tr.update(det, mf, scan.time, 10.0)
        ids |= {t.track_id for t in upd.observed}
        prev = scan.ir
    assert len(ids) == 1
    track = next(iter(tr.tracks.values()))
    assert track.n_scans == 6
    assert -6.5 < track.kf_bt.rate < -3.5   # K ต่อ 10 นาที


def test_split_gets_parent(cfg, t0):
    tr = Tracker(cfg)
    one = np.minimum(blob((100, 100), 50, 44, 6, 220), blob((100, 100), 50, 56, 6, 220))
    scan1 = make_scan(one, t0, cfg)
    det1 = detect_objects(scan1, cfg)
    # ใช้ threshold เดียวเพื่อบังคับให้ scan แรกเป็นก้อนเดียว
    cfg1 = {**cfg, "detect": {**cfg["detect"], "thresholds_k": [273.0]}}
    det1 = detect_objects(scan1, cfg1)
    assert len(det1.objects) == 1
    tr.update(det1, None, t0, 10.0)
    two = np.minimum(blob((100, 100), 50, 40, 5, 220), blob((100, 100), 50, 60, 5, 220))
    det2 = detect_objects(make_scan(two, t0 + timedelta(minutes=10), cfg), cfg)
    assert len(det2.objects) == 2
    upd = tr.update(det2, None, t0 + timedelta(minutes=10), 10.0)
    parents = [t.parent_id for t in upd.observed]
    assert parents.count(None) == 1 and len([p for p in parents if p]) == 1


def test_kalman_smooths_noise():
    rng = np.random.default_rng(1)
    kf = KalmanCV(280.0, q=0.5, r=4.0)
    for k in range(1, 12):
        kf.update(280.0 - 4.0 * k + rng.normal(0, 2.0), 1.0)
    assert -5.5 < kf.rate < -2.5

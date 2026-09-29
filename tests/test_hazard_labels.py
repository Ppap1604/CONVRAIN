"""ขั้น 5 และหัวข้อ 10: hazard model, การสร้างแถว hazard, label ต่อ track"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from convrain.labels import build_track_labels
from convrain.models.hazard import HazardModel

T0 = datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc)


def toy_table(n_obj=200, seed=0):
    """ข้อมูลของเล่น: object ที่เย็นลงเร็ว (cooling ติดลบมาก) ฝนตกเร็ว"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_obj):
        rate = rng.uniform(-8, 0)
        rains = rate < -4
        onset = T0 + timedelta(minutes=10 * (3 + int((rate + 8) * 2))) if rains else pd.NaT
        for k in range(6):
            t = T0 + timedelta(minutes=10 * k)
            if pd.notna(onset) and t >= onset:
                break
            rows.append({"object_id": f"o{i}", "time": t, "cooling_rate_kf": rate + rng.normal(0, 0.3),
                         "cold_bt": 280 + rate * k, "t_onset": onset, "censor_time": pd.NaT})
    return pd.DataFrame(rows)


def test_expand_rules():
    m = HazardModel(features=["cooling_rate_kf"], horizon_steps=3)
    df = pd.DataFrame({
        "object_id": ["a", "a", "b"],
        "time": [T0, T0 + timedelta(minutes=10), T0],
        "cooling_rate_kf": [-5, -6, 0],
        "t_onset": [T0 + timedelta(minutes=15), T0 + timedelta(minutes=15), pd.NaT],
        "censor_time": [pd.NaT, pd.NaT, T0 + timedelta(minutes=20)],
    })
    X, y, meta = m.expand(df, "t_onset", "censor_time")
    a0 = meta[(meta.object_id == "a") & (meta.time == pd.Timestamp(T0).tz_convert(None))]
    # object a ที่ T0: k=1 ไม่เกิด, k=2 เกิด (15 นาที), k=3 ไม่อยู่ในความเสี่ยงแล้ว
    assert list(a0.k) == [1, 2]
    assert list(y[a0.index]) == [0, 1]
    # object b censor ที่ 20 นาที: เหลือ k=1, 2 (k=3 จบหลัง censor)
    assert list(meta[meta.object_id == "b"].k) == [1, 2]


def test_fit_predict_monotone():
    df = toy_table()
    tr, va = df[df.object_id.str[1:].astype(int) < 150], df[df.object_id.str[1:].astype(int) >= 150]
    m = HazardModel(features=["cooling_rate_kf", "cold_bt"], estimator="logistic").fit(tr, "t_onset", "censor_time", va)
    p = m.predict(va)
    assert (p.p30 <= p.p60 + 1e-9).all()
    fast = p[va.groupby("object_id").cooling_rate_kf.first().reindex(p.object_id).values < -6]
    slow = p[va.groupby("object_id").cooling_rate_kf.first().reindex(p.object_id).values > -2]
    assert fast.p60.mean() > slow.p60.mean()
    ok = p.eta_median_min.notna()
    assert (p.eta_lo_min[ok] <= p.eta_median_min[ok]).all() and (p.eta_median_min[ok] <= p.eta_hi_min[ok]).all()


def test_quantile_from_hazard():
    m = HazardModel(horizon_steps=6)
    F = m.cdf(np.array([[0.5, 0, 0, 0, 0, 0]]))
    assert abs(m.quantile_minutes(F, 0.5)[0] - 10.0) < 1e-9
    assert abs(m.quantile_minutes(F, 0.25)[0] - 5.0) < 1e-9
    assert np.isnan(m.quantile_minutes(F, 0.75)[0])   # ไม่ถึง 0.75 ภายใน 60 นาที


def test_track_labels(cfg):
    times = [T0 + timedelta(minutes=10 * k) for k in range(8)]
    dbz = [15, 22, 30, 38, 48, 44, 18, 12]
    ts = pd.DataFrame({
        "object_id": "x", "time": times, "radar_max_dbz": dbz, "radar_coverage": 1.0,
        "radar_onset_time": [t if d >= 35 else pd.NaT for t, d in zip(times, dbz)],
    })
    lab = build_track_labels(ts, None, cfg).iloc[0]
    assert lab.t_onset == pd.Timestamp(times[3])
    assert lab.t_peak == pd.Timestamp(times[4])
    assert lab.t_stop == pd.Timestamp(times[6])      # < 20 dBZ สอง scan ติดกัน
    assert not lab.onset_at_birth

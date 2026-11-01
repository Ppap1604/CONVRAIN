"""ขั้น 5: เวลาฝนเริ่ม / หนักสุด / หยุด (convrain/scope)"""
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from convrain.scope import load_event_models
from convrain.scope.calibrate import fit_event_models, point_events, radar_mm10, training_rows
from convrain.scope.clock import QCOLS, WeibullClock, from_dict
from convrain.scope.columns import ColumnSampler
from convrain.scope.lineage import LineageBook

from conftest import blob, make_scan


def rows_like(n=900, seed=1):
    """แถวสังเคราะห์: เหตุการณ์ = t0 + Weibull, ออกคำทำนายทุก 10 นาทีก่อนเหตุการณ์"""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n // 6):
        b = rng.uniform(0, 600)
        ev = b - 30 + rng.weibull(1.8) * 110
        for t in np.arange(b, ev, 10)[:6]:
            out.append({"cell_id": f"c{i}", "station": f"s{i}", "t": t, "b": b, "onset": ev, "peak": ev + 10,
                        "stop": ev + 30, "d2": min(rng.exponential(4), 16), "st_btd": rng.normal(2, 1),
                        "d_b13_10": rng.normal(0, 3), "d_btd_10": rng.normal(0, .5), "depth_6": rng.normal(0, 10),
                        "warm_20": abs(rng.normal(0, 2))})
    I = pd.DataFrame(out)
    I["btd"] = I.st_btd
    return I


# ------------------------------------------------------------------ โมเดล
def test_builtin_models_quantiles_and_probabilities():
    M = load_event_models()
    I = rows_like()
    for ev, m in M.items():
        Q = m.quantiles(I)
        assert (Q[QCOLS].diff(axis=1).iloc[:, 1:] >= -1e-9).all().all(), ev       # q10 ≤ … ≤ q90
        assert (Q.q10 >= I.t - 1e-6).all(), ev                                   # ไม่ทายย้อนอดีต
    p30, p60 = M["onset"].prob_within(I, 30), M["onset"].prob_within(I, 60)
    assert ((0 <= p30) & (p30 <= p60 + 1e-12) & (p60 <= 1)).all()


def test_weibull_probability_matches_quantile():
    I = rows_like()
    w1 = WeibullClock.fit(I, "onset")                    # c = 1: q50 ของนาฬิกา = จุดที่ P(เกิดภายใน) = 0.5
    Q = w1.quantiles(I)
    p = np.array([w1.prob_within(I.iloc[[k]], float(Q.q50.iloc[k] - I.t.iloc[k]))[0] for k in range(30)])
    assert np.allclose(p, 0.5, atol=1e-6)


def test_dict_roundtrip():
    I = rows_like()
    w1 = WeibullClock.fit(I, "stop")
    w1.c = 0.9
    w2 = from_dict(json.loads(json.dumps(w1.to_dict())))
    np.testing.assert_allclose(w2.quantiles(I).to_numpy(), w1.quantiles(I).to_numpy())


# ------------------------------------------------------------------ สายตระกูล / นาฬิกา t0
@dataclass
class _Obj:
    rows: np.ndarray
    cols: np.ndarray


@dataclass
class _Tr:
    track_id: str
    obj: _Obj
    parent_id: str | None = None


def _disc(r, c, rad):
    rr, cc = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    m = rr ** 2 + cc ** 2 <= rad ** 2
    return _Obj(r + rr[m], c + cc[m])


def test_lineage_clock_and_split_inheritance(cfg, t0):
    tc = cfg["timing"]
    book = LineageBook(tc)
    warm = make_scan(blob((60, 60), 30, 30, 6, 236.5), t0, cfg)                   # มีรอย < 238 K แต่ยังไม่มีแกน < 235 K
    cold = make_scan(blob((60, 60), 30, 30, 6, 215.0), t0 + timedelta(minutes=10), cfg)
    s1 = book.update([_Tr("a", _disc(30, 30, 5))], warm)
    assert s1["a"].t0 is None
    s2 = book.update([_Tr("a", _disc(30, 30, 5)), _Tr("b", _disc(30, 30, 3), parent_id="a")], cold)
    assert s2["a"].t0 == cold.time and not s2["a"].t0_censored
    assert s2["b"].t0 == cold.time                                                # ลูกที่แตกออกรับนาฬิกาของพ่อแม่
    s3 = book.update([_Tr("c", _disc(30, 30, 3))], make_scan(blob((60, 60), 30, 30, 6, 215.0),
                                                             t0 + timedelta(minutes=20), cfg))
    assert s3["c"].t0_censored                                                    # เกิดมาพร้อมแกน ไม่มีพ่อแม่
    assert s2["a"].has_core and s2["a"].core_sxx > 0


def test_columns_refuse_old_scan(cfg, t0):
    cs = ColumnSampler(cfg["timing"])
    cs.push(make_scan(blob((40, 40), 20, 20, 5, 230.0), t0, cfg))
    with pytest.raises(ValueError):
        cs.push(make_scan(blob((40, 40), 20, 20, 5, 230.0), t0 - timedelta(minutes=10), cfg))


# ------------------------------------------------------------------ pipeline ต้นจนจบ (ข้อมูลจำลอง)
def test_pipeline_end_to_end_synthetic(tmp_path):
    from convrain.config import load_config
    from convrain.experiment import replay_scans
    from convrain.pipeline import ConvRainPipeline
    from convrain.synthetic import SyntheticDay

    day = SyntheticDay(3, datetime(2026, 9, 25, 5, 0, tzinfo=timezone.utc), n_scans=24)
    la, lo = day.lat[::8, ::8].ravel(), day.lon[::8, ::8].ravel()
    pts = tmp_path / "points.csv"
    pd.DataFrame({"key": [f"p{i}" for i in range(la.size)], "lat": la, "lon": lo}).to_csv(pts, index=False)
    cfg = load_config(overrides={"domain": day.domain(), "timing": {"points_file": str(pts)}})
    pipe = ConvRainPipeline(cfg, collect_timing_rows=True)
    docs = []
    r = replay_scans(pipe, day.scans(), on_result=lambda res: docs.append(res))
    assert not r["errors"], r["errors"][:3]
    P = r["points"]
    assert len(P) and {"onset_q50", "peak_q50", "stop_q50", "onset_p30"} <= set(P.columns)
    assert (P.onset_q10 <= P.onset_q90).all() and (P.onset_p30 <= P.onset_p60 + 1e-12).all()
    objs = [o for d in docs if d.document for o in d.document["objects"]]
    assert any(o["onset"]["p_within_30min"] is not None for o in objs)
    assert len(pipe.timing.rows_log)
    # บันทึก / กู้ state แล้วทำต่อได้
    pipe.save_state(tmp_path / "state.pkl")
    pipe2 = ConvRainPipeline(cfg)
    pipe2.load_state(tmp_path / "state.pkl")
    assert pipe2.timing.lineages.book.keys() == pipe.timing.lineages.book.keys()


# ------------------------------------------------------------------ ปรับเทียบ
def test_point_events_rules():
    t = pd.date_range("2026-09-01 06:05", periods=10, freq="10min", tz="UTC")
    s = pd.Series([0, 0.6, 2.0, 0.7, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0], index=t)
    ev = point_events(s)
    assert ev["onset"] == t[1] and ev["peak"] == t[2]
    assert ev["stop"] == t[5] + pd.Timedelta(minutes=5)                          # แห้ง 1 ช่วงคั่น ยังไม่หยุด
    assert abs(radar_mm10(30.0) - 0.5) < 0.05


def test_calibration_loop():
    rng = np.random.default_rng(0)
    rows = rows_like(1200).drop(columns=["onset", "peak", "stop"])
    rain = {}
    for (cid, st), g in rows.groupby(["cell_id", "station"]):
        b = g.b.iloc[0]
        on = b + rng.uniform(10, 90)
        idx = pd.to_datetime((np.arange(b - 60, g.t.max() + 90, 10) + 5) * 60, unit="s", utc=True)
        m = (np.arange(len(idx)) * 10 + b - 55)
        rain[st] = pd.Series(np.where((m >= on) & (m < on + 30), 2.0, 0.0), index=idx)
    T = training_rows(rows, rain)
    assert len(T) and T.onset.notna().any()
    P = fit_event_models(T)
    M = {ev: from_dict(json.loads(json.dumps(P[ev]))) for ev in ("onset", "peak", "stop")}
    q = M["onset"].quantiles(T[T.onset.notna() & (T.t < T.onset)])
    assert q.notna().all().all()

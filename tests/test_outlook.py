"""แผนที่ฝนตอนนี้ + คำเตือน 3 ระดับ + เวลาพร้อมความมั่นใจ (convrain/scope/rainmap.py, outlook.py, outlook_fit.py)"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from convrain.ingest import Scan
from convrain.scope.outlook import H_ON, Outlook, PointFeatures, curve_at, window_statement
from convrain.scope.outlook_fit import fit_outlook, label_rows
from convrain.scope.rainmap import RainMap, load_constants


def _scan(t, b13, ny=60, nx=70, refl=None):
    lat, lon = np.meshgrid(np.linspace(17.0, 15.8, ny), np.linspace(99.5, 100.9, nx), indexing="ij")
    bt = {b: np.full((ny, nx), v, np.float32) for b, v in (("B08", 235.0), ("B10", 255.0), ("B11", 285.0),
                                                            ("B14", 288.0), ("B15", 287.0), ("B16", 265.0))}
    bt["B13"] = b13.astype(np.float32)
    return Scan(time=t, bt=bt, refl=refl or {}, lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                sat_zenith=np.full((ny, nx), 45.0, np.float32))


def _storm(k, ny=60, nx=70):
    yy, xx = np.mgrid[:ny, :nx]
    r2 = (yy - 30) ** 2 + (xx - 10 - 3 * k) ** 2
    return np.where(r2 < 64, 205.0, 295.0)


def test_curve_and_window():
    F = np.array([[0.05, 0.3, 0.6, 0.7, 0.75, 0.8, 0.8, 0.8]])                  # ฝนน่าจะมาใน 10–30 นาที
    knots = np.r_[0.0, H_ON]
    Fk = np.hstack([np.zeros((1, 1)), F])
    assert abs(curve_at(Fk, knots, 25.0)[0] - 0.45) < 1e-9
    L, c = window_statement(F, H_ON, 180)
    assert 10 <= L[0] <= 25 and 0.5 < c[0] <= 0.8
    flat = np.full((1, len(H_ON)), 0.0)
    L0, c0 = window_statement(flat, H_ON, 180)
    assert np.isnan(L0[0]) and c0[0] == 0


def test_rainmap_and_point_features_run():
    for region, top in (("thailand", 220.0), ("japan", 230.0)):
        rm = RainMap(load_constants(region), top)
        pf = PointFeatures(np.array([16.4, 16.0]), np.array([99.9, 100.6]), top, rm.offset)
        t0 = datetime(2026, 10, 5, 6, 0, tzinfo=timezone.utc)
        for k in range(14):
            s = _scan(t0 + timedelta(minutes=10 * k), _storm(k))
            rm.update(s)
            rain = rm.point_features(pf.lat, pf.lon)
            D = pf.update(s, rain, rm.last["p"])
        assert {"rain_now", "rain_near", "arrival_min", "daylit"} <= set(rain.columns)
        assert (rain.rain_now.between(0, 1)).all()
        assert np.isfinite(D.st_db13).all() and np.isfinite(D.cold_frac_change_50).all()
        assert (D.swing >= 0).all()
        assert rm.last["U"] is not None                                    # มีการเคลื่อนที่หลังภาพที่สอง


def test_rain_now_product_written(tmp_path):
    import json
    from convrain.output import write_scan_outputs
    rm = RainMap(load_constants("thailand"), 220.0)
    t0 = datetime(2026, 10, 5, 6, 0, tzinfo=timezone.utc)
    for k in range(3):
        s = _scan(t0 + timedelta(minutes=10 * k), _storm(k))
        rm.update(s)
    rm.last["raining"][:] = False
    rm.last["raining"][25:35, 12:22] = True                              # บังคับให้มีบริเวณฝนหนึ่งก้อน
    prod = rm.product(np.full(rm.last["p"].shape, 4.0))
    assert prod["lat_ground"].shape == prod["p"].shape and len(prod["areas"]) == 1
    a = prod["areas"][0]["properties"]
    assert a["area_km2"] == 400.0 and a["mode"] == "night" and 0 <= a["rain_prob_mean"] <= 1
    doc = {"scan_time": "2026-10-05T06:20:00Z", "model_version": "x", "objects": []}
    paths = write_scan_outputs(tmp_path, doc, None, None, rainmap=prod)
    z = np.load(paths["rainmap"])
    assert z["rain_prob_pct"].dtype == np.uint8 and z["raining"].sum() == 100
    fc = json.loads(paths["rainnow"].read_text())
    assert fc["features"][0]["geometry"]["type"] == "Polygon"
    g = np.load(tmp_path / "rainmap" / "grid.npz")
    assert {"lat", "lon", "lat_ground", "lon_ground"} <= set(g.files)


def test_label_rows_rules():
    t = pd.date_range("2026-10-05 06:05", periods=40, freq="10min", tz="UTC")       # กลางช่วง
    rain = np.zeros(40)
    rain[20:24] = 1.0                                                      # ฝน 06:05 + 200 นาที … (4 ช่วง)
    S = {"p": pd.Series(rain, index=t)}
    rows = pd.DataFrame({"key": "p", "time": pd.to_datetime(["2026-10-05 07:00", "2026-10-05 09:30"], utc=True),
                         "raining": False})
    L = label_rows(rows, S, 0.5)
    assert L.onset_ok.iat[0] and L.onset_lead.iat[0] == 145                 # กลางช่วงที่ 20 = 09:25 → 145 นาที
    assert L.stop_ok.iat[1] and L.stop_lead.iat[1] == 35                    # เปียก 09:25–09:55; แห้ง 2 ช่วงเริ่มกลางช่วง 10:05


def test_fit_and_predict_roundtrip():
    rng = np.random.default_rng(0)
    n = 6000
    D = pd.DataFrame({c: rng.normal(size=n) for c in ("cold_frac_10", "cold_frac_25", "cold_frac_50", "coldest_10",
                                                       "coldest_25", "coldest_50", "thickest_10", "thickest_25",
                                                       "st_b13", "st_btd", "st_db13", "st_dbtd", "cooling_frac_25",
                                                       "cooling_frac_50", "cold_frac_change_25", "cold_frac_change_50",
                                                       "swing", "rain_now", "rain_near", "rain_near25", "st_ttd",
                                                       "st_b16_13")})
    D["cos_sun"] = rng.uniform(-1, 1, n)
    comes = rng.uniform(size=n) < 1 / (1 + np.exp(-(2.0 * D.rain_now - 1.0)))      # ฝนมาหรือไม่ ขึ้นกับแผนที่ฝน
    lead = np.where(comes, np.clip(60 - 25 * D.rain_now + rng.normal(0, 10, n), 5, 175), np.nan)
    stop = np.where(rng.uniform(size=n) < 0.9, np.clip(30 + 10 * D.st_btd + rng.normal(0, 8, n), 5, 115), np.nan)
    rows = D.assign(day=np.repeat([f"d{i}" for i in range(10)], n // 10), raining=False,
                    onset_ok=True, onset_lead=lead, stop_ok=True, stop_lead=stop)
    const, rep = fit_outlook(rows, k=3, log=lambda *_: None)
    assert rep["onset"]["IBS_skill"] > 0 and set(const["levels"]) == {"imminent", "warning", "watch"}
    O = Outlook({**const, "stop_requires": "map"})
    out = O.predict(D.iloc[:50], np.r_[np.zeros(25, bool), np.ones(25, bool)], pd.Timestamp("2026-10-05 06:00", tz="UTC"))
    assert out.warning_level.iloc[25:].eq("raining").all() and out.onset_time.iloc[25:].isna().all()
    assert out.stop_time.iloc[:25].isna().all() and out.stop_confidence.iloc[25:].between(0, 1).all()
    assert (out.p_onset_30 <= out.p_onset_60 + 1e-12).all() and (out.p_onset_60 <= out.p_onset_180 + 1e-12).all()

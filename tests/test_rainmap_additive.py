"""แผนที่ฝนแบบโมเดลบวก (ตาราง) + ตัวแปรช่องเพิ่ม + ก้อนเมฆ / ช่วงก่อตัวรายจุด"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from convrain.scope.outlook import FORMATION_FEATURES, OBJECT_FEATURES, PointFeatures
from convrain.scope.rainmap import AdditiveModel, MapModel, make_model, n9_fields


def _table():
    # ตัวแปร a: ช่อง ≤ 0 / > 0 / NaN → −1 / +1 / 0; คู่ (a, b): ตาราง 3 × 3
    return {"kind": "additive", "cols": ["a", "b"], "thr": 0.5, "baseline": 0.2,
            "groups": [{"vars": [0], "cuts": [[0.0]], "table": [-1.0, 1.0, 0.0]},
                       {"vars": [0, 1], "cuts": [[0.0], [5.0]], "table": [[0.0, 0.5, 0.0], [0.1, 0.2, 0.0], [0.0, 0.0, 0.0]]}]}


def test_additive_cells_and_missing():
    m = make_model(_table())
    assert isinstance(m, AdditiveModel)
    X = np.array([[-1.0, 1.0], [2.0, 9.0], [np.nan, 1.0], [0.0, 5.0]])
    np.testing.assert_allclose(m.raw(X), [0.2 - 1.0 + 0.0, 0.2 + 1.0 + 0.2, 0.2 + 0.0 + 0.0, 0.2 - 1.0 + 0.0])
    assert ((m.prob(X) > 0) & (m.prob(X) < 1)).all()
    assert isinstance(make_model({"cols": ["a"], "med": [0], "mu": [0], "sd": [1], "b": [0] * 28, "thr": 0.5}), MapModel)


def test_n9_day_and_night():
    shape = (6, 6)
    bt = {b: np.full(shape, t) for b, t in (("B07", 230.0), ("B11", 225.0), ("B12", 240.0), ("B13", 220.0), ("B15", 219.0),
                                             ("B16", 228.0))}
    refl = {b: np.full(shape, r) for b, r in (("B01", 0.8), ("B02", 0.75), ("B03", 0.7), ("B06", 0.25))}
    F = n9_fields(bt, refl, np.full(shape, 0.5))
    np.testing.assert_allclose(F["n9_B03/B06"], 0.7 / 0.25)
    np.testing.assert_allclose(F["n9_B01-B06"], (0.8 - 0.25) / 0.5)
    np.testing.assert_allclose(F["n9_B13/B16"], 220.0 / 228.0)
    N = n9_fields(bt, refl, np.full(shape, 0.05))                    # กลางคืน: ช่องแสงไม่ใช้ แต่ IR ยังใช้ได้
    assert np.isnan(N["n9_B03/B06"]).all() and np.isfinite(N["n9_B07/B12"]).all()


class _Scan:
    def __init__(self, t, b13):
        self.time = t
        self.bt = {"B13": b13, "B15": b13 - 1.0, "B16": b13 + 5.0}
        self.refl = {}
        self.lat, self.lon = np.meshgrid(np.linspace(14, 15, 40), np.linspace(100, 101, 40), indexing="ij")
        self.sat_zenith = np.full(self.lat.shape, 40.0)
        self.shape = self.lat.shape


def test_cloud_object_and_formation_features():
    t0 = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)
    pf = PointFeatures([14.5, 14.05], [100.5, 100.05], 220.0)
    cloud = np.full((40, 40), 290.0)
    cloud[15:25, 15:25] = 230.0                                       # ก้อนเมฆเย็นกลางภาพ
    p = np.zeros((40, 40))
    p[18:20, 18:20] = 0.8
    D = None
    for k in range(4):                                                # เมฆเย็นลงทีละภาพ
        s = _Scan(pd.Timestamp(t0) + pd.Timedelta(minutes=10 * k), cloud - 2.0 * k)
        D = pf.update(s, None, p, p >= 0.5, (np.zeros((40, 40)), np.zeros((40, 40))))
    assert set(OBJECT_FEATURES + FORMATION_FEATURES) <= set(D.columns)
    inside = int(np.argmax(D.obj_p.to_numpy()))
    assert D.obj_p.iloc[inside] == 0.8 and D.obj_wet.iloc[inside] > 0
    assert D.obj_p.min() == 0.0                                       # จุดที่ไม่อยู่ใต้ก้อน
    assert np.isfinite(D["form_dmin_0"]).all() and (D["form_dmin_0"].iloc[inside] < 0)   # เย็นลงใน 30 นาที

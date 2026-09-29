"""ระดับ 4–5: Scientific Sanity & Meteorological Physical Constraints tests"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from convrain.models.baseline import (
    ClimatologyBaseline,
    ExtrapolationBaseline,
    MB06InterestFields,
)
from convrain.models.hazard import HazardModel
from convrain.parallax import cloud_top_height_km, parallax_correct


def test_monotonic_probability_across_all_models():
    """ตรวจสอบกฎความน่าจะเป็น $P(onset <= 30 min) <= P(onset <= 60 min)$ ในทุกโมเดล"""
    # 1. MB06
    mb = MB06InterestFields()
    rows_mb = pd.DataFrame([{
        "object_id": "c1", "min_bt": 240.0, "min_since_below_0c": 10.0, "d_cold_bt_10": -5.0,
        "d_cold_bt_30": -12.0, "btd_wv_cold": -18.0, "btd_co2_cold": -12.0,
        "d_btd_wv_10": 4.0, "d_btd_co2_10": 4.0,
    }])
    p_mb = mb.predict(rows_mb)
    assert (p_mb["p30"] <= p_mb["p60"] + 1e-6).all()

    # 2. Extrapolation
    extrap = ExtrapolationBaseline()
    rows_ex = pd.DataFrame([
        {"object_id": "e1", "cold_bt_kf": 250.0, "cooling_rate_kf": -4.0},
        {"object_id": "e2", "cold_bt_kf": 230.0, "cooling_rate_kf": -2.0},
    ])
    p_ex = extrap.predict(rows_ex)
    assert (p_ex["p30"] <= p_ex["p60"] + 1e-6).all()

    # 3. Climatology
    clim = ClimatologyBaseline()
    clim.table = pd.Series([0.2, 0.4, 0.6], index=[12, 13, 14])
    rows_cl = pd.DataFrame([{"object_id": "cl1", "lst_hours": 13.5}])
    p_cl = clim.predict(rows_cl)
    assert (p_cl["p30"] <= p_cl["p60"] + 1e-6).all()


def test_parallax_physical_shift_direction_and_scaling():
    """ตรวจสอบความถูกต้องทางฟิสิกส์ของการแก้ Parallax สำหรับประเทศไทย (Himawari 140.7°E)"""
    lat_app, lon_app = 13.7, 100.5  # กรุงเทพฯ

    # ที่ความสูง 0 km ตำแหน่งต้องไม่เปลี่ยน
    lat_0, lon_0 = parallax_correct(lat_app, lon_app, height_km=0.0)
    assert abs(lat_0 - lat_app) < 1e-6
    assert abs(lon_0 - lon_app) < 1e-6

    # ที่ความสูงยอดเมฆต่างๆ: การเลื่อนต้องเพิ่มขึ้นตามความสูง
    from convrain.parallax import shift_km
    heights = [4.0, 8.0, 12.0, 16.0]
    shifts_km = []
    for h in heights:
        lat_c, lon_c = parallax_correct(lat_app, lon_app, height_km=h)
        # ดาวเทียมอยู่ที่ 140.7°E บนเส้นศูนย์สูตร (0°N, 140.7°E) ทิศตะวันออกเฉียงใต้ของไทย
        # ภาพปรากฏถูกผลักไปทางตะวันตกเฉียงเหนือ ดังนั้นตำแหน่งจริงที่แก้แล้วจึงต้องเลื่อนเข้าหาจุดใต้ดาวเทียม (SE: lo > lon, la < lat)
        assert lon_c > lon_app, f"จริงต้องอยู่ทางตะวันออกกว่าที่ปรากฏ (h={h})"
        assert lat_c < lat_app, f"จริงต้องอยู่ทางใต้กว่าที่ปรากฏ (h={h})"

        dist_km = shift_km(lat_app, lon_app, lat_c, lon_c)
        shifts_km.append(dist_km)

    # ระยะการเลื่อนต้องเพิ่มขึ้นแบบ Monotonic ตามความสูงยอดเมฆ
    assert np.all(np.diff(shifts_km) > 0)
    # ที่ความสูง 12 km ระยะการเลื่อนต้องอยู่ในช่วง ~10–14 km
    assert 10.0 <= shifts_km[2] <= 15.0


def test_cloud_top_height_temperature_inversion_bounds():
    """ตรวจสอบการแปลงอุณหภูมิ BT เป็นความสูงยอดเมฆตามมาตรฐานเขตร้อน"""
    # พื้นผิวร้อน 300 K -> ความสูง 0 km
    assert cloud_top_height_km(300.0) == 0.0
    assert cloud_top_height_km(320.0) == 0.0

    # ยอดเมฆเย็นจัด 195 K -> ความสูงสูงสุดของ Tropopause (~17 km)
    assert abs(cloud_top_height_km(195.0) - 17.0) < 1e-2
    assert abs(cloud_top_height_km(180.0) - 17.0) < 1e-2

    # ระดับแช่เยือกแข็ง 0 °C (273.15 K) -> ความสูงประมาณ 4.5–5.5 km ในเขตร้อน
    h_0c = cloud_top_height_km(273.15)
    assert 4.0 <= h_0c <= 6.0

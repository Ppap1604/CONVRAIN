"""ขั้น 6.4: parallax correction และมุม satellite zenith"""
import numpy as np

from convrain.ingest import satellite_zenith
from convrain.parallax import cloud_top_height_km, parallax_correct, shift_km


def test_bangkok_zenith():
    z = satellite_zenith(np.array([13.75]), np.array([100.5]), 140.7)[0]
    assert 48.0 < z < 49.2   # เอกสารหัวข้อ 9.4: ประมาณ 48–49°


def test_bangkok_shift_12km():
    lat, lon = 13.75, 100.5
    la, lo = parallax_correct(lat, lon, 12.0, 140.7)
    d = shift_km(lat, lon, la, lo)
    assert 12.5 < d < 14.5                     # ~13–14 km ตามเอกสาร
    assert lo > lon and la < lat               # เลื่อนเข้าหาจุดใต้ดาวเทียม (ตะวันออกเฉียงใต้)


def test_zero_height_no_shift():
    la, lo = parallax_correct(13.75, 100.5, 0.0, 140.7)
    assert abs(la - 13.75) < 1e-6 and abs(lo - 100.5) < 1e-6


def test_height_from_bt():
    assert abs(cloud_top_height_km(224.0) - 12.0) < 0.01
    assert cloud_top_height_km(310.0) == 0.0
    assert cloud_top_height_km(180.0) == 17.0

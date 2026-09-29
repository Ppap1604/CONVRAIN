"""ขั้น 0: BTD, NetCDF round-trip, ไฟล์ข้อมูลพื้นผิวคงที่, QC"""
import numpy as np
import xarray as xr
from conftest import blob, make_scan

from convrain.ingest import Scan, load_scan_netcdf, prepare_geometry, quality_check, save_scan_netcdf


def test_btd_values(cfg, t0):
    bt = blob((40, 40), 20, 20, 5, 230)
    scan = make_scan(bt, t0, cfg)
    assert np.allclose(scan.btd_wv, -20.0)   # conftest ตั้ง B08 = B13 − 20
    assert np.allclose(scan.btd_sw, 0.0)


def test_netcdf_roundtrip(cfg, t0, tmp_path):
    scan = make_scan(blob((40, 40), 20, 20, 5, 230), t0, cfg)
    save_scan_netcdf(scan, tmp_path / "s.nc")
    back = load_scan_netcdf(tmp_path / "s.nc", cfg)
    assert back.time == scan.time
    assert np.max(np.abs(back.ir - scan.ir)) < 0.01     # int16 × 0.01 K


def test_static_file_attached(cfg, t0, tmp_path):
    bt = blob((40, 40), 20, 20, 5, 230)
    lm = np.zeros((40, 40), np.float32)
    lm[:, 20:] = 1
    xr.Dataset({"land_mask": (("y", "x"), lm), "elevation_m": (("y", "x"), lm * 500)}).to_netcdf(tmp_path / "st.nc")
    cfg2 = {**cfg, "ingest": {**cfg["ingest"], "static_file": str(tmp_path / "st.nc")}}
    scan = make_scan(bt, t0, cfg2)
    assert scan.land_mask is not None and scan.land_mask[:, 25].all()
    assert scan.elevation_m[0, 30] == 500


def test_quality_check_flags_bad_scan(cfg, t0):
    bt = blob((40, 40), 20, 20, 5, 230)
    bt[:20] = np.nan    # ครึ่งภาพเสีย
    scan = quality_check(prepare_geometry(Scan(time=t0, bt={"B13": bt}, lat=np.zeros((40, 40)) + 14,
                                               lon=np.zeros((40, 40)) + 100 + np.arange(40) * 0.02), cfg))
    assert not scan.quality_ok

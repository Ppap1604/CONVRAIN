"""
ขั้น 0: รับข้อมูล Himawari และ preprocessing  (เอกสาร หัวข้อ 3)

หน้าที่ของโมดูลนี้
    1. อ่านไฟล์ AHI (HSD ผ่าน Satpy หรือไฟล์ NetCDF ที่ crop ไว้แล้ว)
    2. Crop เฉพาะโดเมน และเก็บ lat/lon, มุม satellite zenith, พื้นที่ pixel ไว้ครั้งเดียว
    3. คำนวณ BTD ต่อ pixel (ลบกันธรรมดา ต้นทุนต่ำ)
    4. ตรวจคุณภาพ scan (pixel เสีย / scan หาย)

โครงสร้างข้อมูลหลักคือ `Scan` ซึ่งทุกขั้นถัดไปรับเป็น input
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

EARTH_RADIUS_KM = 6371.0

# ชื่อ band ของ AHI → ความยาวคลื่น (µm) เพื่อความชัดเจนในโค้ดส่วนอื่น
AHI_BANDS = {
    "B03": 0.64, "B07": 3.9, "B08": 6.2, "B09": 6.9, "B10": 7.3, "B11": 8.6,
    "B13": 10.4, "B14": 11.2, "B15": 12.4, "B16": 13.3,
}


# =====================================================================
# [ขั้น 0.1] โครงสร้างข้อมูล Scan
# =====================================================================
@dataclass
class Scan:
    """
    ข้อมูลดาวเทียม 1 scan หลัง crop

    bt        : dict ชื่อ band → อาร์เรย์ 2 มิติของ brightness temperature (K)
    lat, lon  : พิกัดของแต่ละ pixel (องศา) ในพิกัดดั้งเดิมของดาวเทียม (ไม่ remap)
    sat_zenith: มุม satellite zenith ต่อ pixel (องศา) ใช้ใน parallax
    pixel_area_km2 : พื้นที่ต่อ pixel (km²)
    refl      : reflectance ช่อง VIS (ทางเลือก กลางวันเท่านั้น)
    land_mask, elevation_m : ข้อมูลพื้นผิว (ทางเลือก ใช้เป็น feature บริบท)
    """

    time: datetime
    bt: dict[str, np.ndarray]
    lat: np.ndarray
    lon: np.ndarray
    sat_zenith: np.ndarray | None = None
    pixel_area_km2: np.ndarray | None = None
    refl: dict[str, np.ndarray] = field(default_factory=dict)
    land_mask: np.ndarray | None = None
    elevation_m: np.ndarray | None = None
    quality_ok: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            self.time = self.time.replace(tzinfo=timezone.utc)

    @property
    def shape(self) -> tuple[int, int]:
        return self.bt["B13"].shape

    # -----------------------------------------------------------------
    # [ขั้น 0.3] BTD ต่อ pixel (คำนวณครั้งแรกที่เรียกแล้ว cache ไว้)
    # -----------------------------------------------------------------
    def _band(self, name: str) -> np.ndarray:
        """คืน band ที่ขอ ถ้าไม่มีให้คืนอาร์เรย์ NaN (feature ที่ใช้ band นี้จะเป็น NaN)"""
        if name in self.bt:
            return self.bt[name]
        return np.full(self.shape, np.nan, dtype=np.float32)

    def _cached(self, key: str, fn) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = fn().astype(np.float32)
        return self._cache[key]

    @property
    def ir(self) -> np.ndarray:
        """BT ช่อง IR window 10.4 µm (B13) ช่องหลักของทุกขั้น"""
        return self.bt["B13"]

    @property
    def btd_wv(self) -> np.ndarray:
        """BT(6.2) − BT(10.4): ความสูงยอดเมฆเทียบ troposphere บน (เข้าใกล้ 0 = เมฆลึก)"""
        return self._cached("btd_wv", lambda: self._band("B08") - self.ir)

    @property
    def btd_73(self) -> np.ndarray:
        """BT(7.3) − BT(10.4): ความสูงยอดเมฆเทียบ troposphere กลาง-ล่าง (Li et al. 2024)"""
        return self._cached("btd_73", lambda: self._band("B10") - self.ir)

    @property
    def btd_sw(self) -> np.ndarray:
        """BT(10.4) − BT(12.4): split window บอกความหนาเมฆ (สูง + เย็น = cirrus บาง)"""
        return self._cached("btd_sw", lambda: self.ir - self._band("B15"))

    @property
    def ttd(self) -> np.ndarray:
        """Tri-spectral: [BT(8.6) − BT(11.2)] − [BT(11.2) − BT(12.4)] บอก phase ยอดเมฆ"""
        return self._cached(
            "ttd",
            lambda: (self._band("B11") - self._band("B14")) - (self._band("B14") - self._band("B15")),
        )

    @property
    def btd_co2(self) -> np.ndarray:
        """BT(13.3) − BT(10.4): เทียบกับช่อง 13.3 µm ของ GOES ใน Mecikalski & Bedka (2006)"""
        return self._cached("btd_co2", lambda: self._band("B16") - self.ir)


# =====================================================================
# [ขั้น 0.2] เรขาคณิตคงที่ของโดเมน (คำนวณครั้งเดียวแล้วใช้ซ้ำ)
# =====================================================================
def satellite_zenith(lat: np.ndarray, lon: np.ndarray, sub_lon: float = 140.7,
                     orbit_radius_km: float = 42164.0) -> np.ndarray:
    """
    มุม satellite zenith (องศา) ของแต่ละ pixel สำหรับดาวเทียมค้างฟ้าที่ sub_lon
    ใช้โลกทรงกลม: z = atan( sin(c) / (cos(c) − R/Rs) ), c = มุมศูนย์กลางจากจุดใต้ดาวเทียม
    ตัวอย่าง: กรุงเทพฯ (13.75N, 100.5E) กับ Himawari (140.7E) ได้ประมาณ 48.6°
    """
    la = np.deg2rad(lat)
    dlo = np.deg2rad(lon - sub_lon)
    cos_c = np.clip(np.cos(la) * np.cos(dlo), -1.0, 1.0)
    sin_c = np.sqrt(1.0 - cos_c**2)
    z = np.arctan2(sin_c, cos_c - EARTH_RADIUS_KM / orbit_radius_km)
    return np.rad2deg(z).astype(np.float32)


def pixel_area_km2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    พื้นที่โดยประมาณของแต่ละ pixel (km²) จากระยะห่างของ pixel ข้างเคียง
    (pixel ของ AHI เหนือไทยใหญ่กว่า 2 km เพราะมุมมองเฉียง จึงต้องคำนวณจริง)
    """
    lat_r = np.deg2rad(lat)
    dy_lat = np.gradient(lat, axis=0)
    dy_lon = np.gradient(lon, axis=0)
    dx_lat = np.gradient(lat, axis=1)
    dx_lon = np.gradient(lon, axis=1)
    k = np.deg2rad(1.0) * EARTH_RADIUS_KM
    dy = k * np.hypot(dy_lat, dy_lon * np.cos(lat_r))
    dx = k * np.hypot(dx_lat, dx_lon * np.cos(lat_r))
    return np.abs(dx * dy).astype(np.float32)


_STATIC_CACHE: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}


def prepare_geometry(scan: Scan, cfg: dict) -> Scan:
    """เติม sat_zenith, pixel_area_km2 และข้อมูลพื้นผิวคงที่ (ถ้าตั้งค่าไว้) ถ้ายังไม่มี"""
    sat = cfg["satellite"]
    if scan.sat_zenith is None:
        scan.sat_zenith = satellite_zenith(scan.lat, scan.lon, sat["sub_lon"], sat["orbit_radius_km"])
    if scan.pixel_area_km2 is None:
        scan.pixel_area_km2 = pixel_area_km2(scan.lat, scan.lon)
    attach_static(scan, cfg)
    return scan


def attach_static(scan: Scan, cfg: dict) -> Scan:
    """
    โหลด land/sea mask และความสูงภูมิประเทศจากไฟล์ NetCDF บน grid เดียวกับ scan (อ่านครั้งเดียวแล้ว cache)
    ไฟล์ต้องมีตัวแปร land_mask (0–1) และ elevation_m (y, x) ตั้งค่า path ที่ ingest.static_file
    สร้างได้จาก DEM (เช่น SRTM) โดย regrid ลง lat/lon ของ scan แรก
    """
    path = cfg.get("ingest", {}).get("static_file")
    if not path or (scan.land_mask is not None and scan.elevation_m is not None):
        return scan
    if path not in _STATIC_CACHE:
        import xarray as xr

        engine = _nc_engine()
        with xr.open_dataset(path, engine=engine) as ds:
            lm = ds["land_mask"].values.astype(np.float32) if "land_mask" in ds else None
            el = ds["elevation_m"].values.astype(np.float32) if "elevation_m" in ds else None
        _STATIC_CACHE[path] = (lm, el)
    lm, el = _STATIC_CACHE[path]
    if lm is not None and lm.shape == scan.shape and scan.land_mask is None:
        scan.land_mask = lm
    if el is not None and el.shape == scan.shape and scan.elevation_m is None:
        scan.elevation_m = el
    return scan


# =====================================================================
# [ขั้น 0.5] ตรวจคุณภาพ scan
# =====================================================================
def quality_check(scan: Scan, max_bad_fraction: float = 0.05) -> Scan:
    """
    ทำเครื่องหมาย scan ที่มี pixel เสีย (NaN หรือค่านอกช่วงฟิสิกส์) เกินสัดส่วนที่กำหนด
    scan ที่ไม่ผ่าน tracker จะข้ามไป และขยาย Δt ในการคำนวณแนวโน้มแทน
    """
    ir = scan.ir
    bad = ~np.isfinite(ir) | (ir < 150.0) | (ir > 340.0)
    scan.quality_ok = bool(bad.mean() <= max_bad_fraction)
    if scan.quality_ok and bad.any():
        # เติม pixel เสียจำนวนน้อยด้วยค่ามัธยฐาน เพื่อไม่ให้ขั้นถัดไปเจอ NaN
        fill = np.nanmedian(ir[~bad]) if (~bad).any() else 295.0
        for name, arr in scan.bt.items():
            arr[bad] = fill if name == "B13" else np.nanmedian(arr[~bad])
    return scan


# =====================================================================
# [ขั้น 0.1] อ่านไฟล์ AHI HSD ด้วย Satpy (ใช้กับข้อมูลจริง)
# =====================================================================
def read_ahi_hsd(filenames: Iterable[str | Path], cfg: dict) -> Scan:
    """
    อ่านไฟล์ Himawari Standard Data (HSD) ของ scan เดียว แล้ว crop ตามโดเมน

    ต้องติดตั้ง satpy (`pip install satpy`) และส่งไฟล์ segment ของทุก band ที่ต้องใช้
    Satpy จะถอดเฉพาะ segment ที่ครอบคลุมโดเมนหลังเรียก crop() (ลดเวลาอ่านไฟล์)
    """
    try:
        from satpy import Scene  # import ตอนใช้ เพื่อให้ส่วนอื่นรันได้โดยไม่ต้องมี satpy
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ต้องติดตั้ง satpy เพื่ออ่านไฟล์ HSD: pip install satpy") from exc

    d = cfg["domain"]
    bands = cfg["satellite"]["bands"]
    scn = Scene(reader="ahi_hsd", filenames=[str(f) for f in filenames])
    scn.load(bands, calibration="brightness_temperature")
    scn = scn.crop(ll_bbox=(d["lon_min"], d["lat_min"], d["lon_max"], d["lat_max"]))
    # band IR ทั้งหมด (B07–B16) อยู่บน grid 2 km เดียวกันอยู่แล้ว ถ้าเพิ่ม B03 (0.5 km)
    # ให้ resample ลงมาที่ grid หยาบสุด: scn = scn.resample(scn.coarsest_area(), resampler="native")

    lon, lat = scn["B13"].attrs["area"].get_lonlats()
    bt = {b: np.asarray(scn[b].values, dtype=np.float32) for b in bands}
    time = scn["B13"].attrs["start_time"]
    scan = Scan(time=time, bt=bt, lat=lat.astype(np.float32), lon=lon.astype(np.float32))
    return quality_check(prepare_geometry(scan, cfg))


# =====================================================================
# [ขั้น 0.3] เก็บ/อ่าน scan ที่ crop แล้วเป็น NetCDF (Phase 0 infrastructure)
# =====================================================================
def _nc_engine() -> str | None:
    try:
        import h5netcdf  # noqa: F401
        return "h5netcdf"
    except ImportError:
        return None


def save_scan_netcdf(scan: Scan, path: str | Path) -> None:
    """
    บันทึก scan ที่ crop แล้วเป็น NetCDF แบบบีบอัด
    BT เก็บเป็น int16 × 0.01 K ลดขนาดไฟล์ประมาณครึ่งหนึ่ง (เอกสาร หัวข้อ 14, Phase 0.3)
    """
    import xarray as xr

    data_vars = {name: (("y", "x"), arr) for name, arr in scan.bt.items()}
    for name, arr in scan.refl.items():
        data_vars[f"refl_{name}"] = (("y", "x"), arr)
    data_vars["lat"] = (("y", "x"), scan.lat)
    data_vars["lon"] = (("y", "x"), scan.lon)
    ds = xr.Dataset(data_vars, attrs={"time": scan.time.isoformat()})
    enc = {name: {"dtype": "int16", "scale_factor": 0.01, "add_offset": 250.0,
                  "_FillValue": -32768, "zlib": True} for name in scan.bt}
    engine = _nc_engine()
    ds.to_netcdf(path, engine=engine, encoding=enc)


def load_scan_netcdf(path: str | Path, cfg: dict) -> Scan:
    """อ่านไฟล์ NetCDF ที่สร้างด้วย save_scan_netcdf แล้วเติมเรขาคณิต + QC"""
    import xarray as xr

    engine = _nc_engine()
    with xr.open_dataset(path, engine=engine) as ds:
        bt = {v: ds[v].values.astype(np.float32) for v in ds.data_vars if v.startswith("B")}
        refl = {v[5:]: ds[v].values.astype(np.float32) for v in ds.data_vars if v.startswith("refl_")}
        time = datetime.fromisoformat(ds.attrs["time"])
        scan = Scan(time=time, bt=bt, refl=refl,
                    lat=ds["lat"].values.astype(np.float32), lon=ds["lon"].values.astype(np.float32))
    return quality_check(prepare_geometry(scan, cfg))

"""
ดาวน์โหลดข้อมูลฝน GPM IMERG Late Run (V07) จาก NASA GES DISC
และแปลงเป็น NetCDF เรดาร์ (ตัวแปร 'dbz') บนกริดพิกัดดาวเทียม Himawari สำหรับ convrain

คุณสมบัติ:
    - อ่าน Username / Password จาก .env (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
    - รองรับการแปลง Precipitation rate (mm/h) เป็นค่าการสะท้อนเรดาร์สมมูล (Equivalent dBZ)
      โดยใช้ความสัมพันธ์ Z = 300 * R^1.4
    - Regrid ลงบนกริดพิกัดดาวเทียม 2 km อัตโนมัติ

วิธีใช้:
    python scripts/download_gpm_imerg.py --date 20260928 --start-hour 6 --end-hour 8 --convert --scans-dir data/scans
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import xarray as xr

from convrain.labels import regrid_to_satellite

# โดเมนประเทศไทย (+ Margin 2 องศา)
DOMAIN = {"lat_min": 2.0, "lat_max": 24.0, "lon_min": 94.0, "lon_max": 109.0}


def load_credentials() -> tuple[str, str]:
    """โหลดข้อมูลบัญชี NASA Earthdata จาก .env หรือ environment"""
    user = os.environ.get("EARTHDATA_USERNAME") or os.environ.get("EARTHDATA_USER")
    pwd = os.environ.get("EARTHDATA_PASSWORD") or os.environ.get("EARTHDATA_PASS")
    if user and pwd:
        return user, pwd

    for candidate in [Path(".env"), Path(__file__).resolve().parent.parent / ".env"]:
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k in ("EARTHDATA_USERNAME", "EARTHDATA_USER") and not user:
                    user = v
                elif k in ("EARTHDATA_PASSWORD", "EARTHDATA_PASS") and not pwd:
                    pwd = v

    if user and pwd:
        os.environ["EARTHDATA_USERNAME"] = user
        os.environ["EARTHDATA_PASSWORD"] = pwd

    return user or "", pwd or ""


def convert_imerg_hdf5_to_radar_nc(imerg_path: Path, scan_lat: np.ndarray,
                                   scan_lon: np.ndarray, out_nc: Path,
                                   scan_time: datetime) -> bool:
    """แปลงไฟล์ HDF5 ของ GPM IMERG เป็น radar_YYYYMMDDTHHMMZ.nc"""
    try:
        with h5py.File(imerg_path, "r") as f:
            lons = f["Grid/lon"][:]
            lats = f["Grid/lat"][:]
            precip = f["Grid/precipitation"][0]  # shape (lon, lat)

        lon_mask = (lons >= DOMAIN["lon_min"]) & (lons <= DOMAIN["lon_max"])
        lat_mask = (lats >= DOMAIN["lat_min"]) & (lats <= DOMAIN["lat_max"])

        sub_lon = lons[lon_mask]
        sub_lat = lats[lat_mask]
        sub_precip = precip[np.ix_(lon_mask, lat_mask)].T  # เปลี่ยนเป็น (lat, lon)

        # แปลง mm/h -> dBZ: Z = 300 * R^1.4, dBZ = 10 * log10(Z)
        sub_precip = np.where(sub_precip < 0, 0.0, sub_precip)
        with np.errstate(divide="ignore"):
            z = 300.0 * np.power(np.clip(sub_precip, 0.01, None), 1.4)
            dbz = 10.0 * np.log10(np.clip(z, 1.0, None))
            dbz = np.where(sub_precip < 0.1, np.nan, dbz)

        lon_grid, lat_grid = np.meshgrid(sub_lon, sub_lat)
        radar_dbz = regrid_to_satellite(lat_grid, lon_grid, dbz.astype(np.float32),
                                        scan_lat, scan_lon, max_dist_km=15.0)

        out_nc.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {"dbz": (("y", "x"), radar_dbz.astype(np.float32))},
            attrs={"time": scan_time.isoformat()}
        )
        ds.to_netcdf(out_nc, engine="h5netcdf")
        print(f"  -> แปลงสำเร็จ: {out_nc.name} (max: {np.nanmax(radar_dbz):.1f} dBZ)")
        return True

    except Exception as exc:
        print(f"  [ERROR] แปลงไฟล์ {imerg_path.name} ล้มเหลว: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="ดาวน์โหลด GPM IMERG Late Run จาก NASA สำหรับ convrain ground truth")
    parser.add_argument("--date", required=True, help="วันที่ YYYYMMDD (เช่น 20260928)")
    parser.add_argument("--start-hour", type=int, default=6, help="ชั่วโมงเริ่มต้น UTC (default: 6)")
    parser.add_argument("--end-hour", type=int, default=8, help="ชั่วโมงสิ้นสุด UTC (default: 8)")
    parser.add_argument("--out-dir", default="data/gpm", help="โฟลเดอร์เก็บไฟล์ HDF5 (default: data/gpm)")
    parser.add_argument("--radar-dir", default="data/radar", help="โฟลเดอร์เก็บ radar NetCDF (default: data/radar)")
    parser.add_argument("--convert", action="store_true", help="แปลงเป็น radar NetCDF ทันทีหลังโหลด")
    parser.add_argument("--scans-dir", default="data/scans", help="โฟลเดอร์เก็บไฟล์ scan ดาวเทียม (สำหรับอ่านพิกัด)")

    args = parser.parse_args()

    user, pwd = load_credentials()
    if not user or not pwd:
        print("[ERROR] ไม่พบข้อมูลบัญชี NASA Earthdata ใน .env กรุณาระบุ EARTHDATA_USERNAME และ EARTHDATA_PASSWORD", file=sys.stderr)
        sys.exit(1)

    import earthaccess

    print(f"กำลังเข้าสู่ระบบ NASA Earthdata ด้วยผู้ใช้ {user}...")
    auth = earthaccess.login(strategy="environment")

    t_start = datetime.strptime(args.date, "%Y%m%d").replace(hour=args.start_hour, minute=0, tzinfo=timezone.utc)
    t_end = datetime.strptime(args.date, "%Y%m%d").replace(hour=args.end_hour, minute=59, tzinfo=timezone.utc)

    print(f"ค้นหาข้อมูล GPM IMERG ช่วง {t_start:%Y-%m-%d %H:%M} ถึง {t_end:%Y-%m-%d %H:%M} UTC...")
    results = earthaccess.search_data(
        short_name="GPM_3IMERGHHL",
        version="07",
        temporal=(t_start.strftime("%Y-%m-%d %H:%M:%S"), t_end.strftime("%Y-%m-%d %H:%M:%S")),
        count=100
    )
    print(f"พบข้อมูลทั้งหมด {len(results)} รายการ")

    out_raw = Path(args.out_dir) / args.date
    out_raw.mkdir(parents=True, exist_ok=True)

    downloaded = earthaccess.download(results, str(out_raw))
    print(f"ดาวน์โหลดสำเร็จ {len(downloaded)} ไฟล์")

    if args.convert and downloaded:
        scans_dir = Path(args.scans_dir)
        scan_files = sorted(scans_dir.glob("scan_*.nc"))
        if not scan_files:
            print(f"[WARN] ไม่พบไฟล์ scan ใน {scans_dir} ข้ามขั้นตอนการแปลง", file=sys.stderr)
            return

        with xr.open_dataset(scan_files[0], engine="h5netcdf") as sds:
            scan_lat = sds["lat"].values.astype(np.float32)
            scan_lon = sds["lon"].values.astype(np.float32)

        radar_dir = Path(args.radar_dir)
        for f in downloaded:
            p = Path(f)
            # ตัวอย่างชื่อไฟล์: 3B-HHR-L.MS.MRG.3IMERG.20260928-S063000-E065959.0390.V07C.HDF5
            name_parts = p.name.split(".")
            for part in name_parts:
                if "-" in part and part.startswith("2026"):
                    # 20260928-S063000-E065959
                    date_part, s_time, _ = part.split("-")
                    scan_time = datetime.strptime(f"{date_part}_{s_time[1:5]}", "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
                    out_nc = radar_dir / f"radar_{scan_time:%Y%m%dT%H%MZ}.nc"
                    convert_imerg_hdf5_to_radar_nc(p, scan_lat, scan_lon, out_nc, scan_time)


if __name__ == "__main__":
    main()

"""
ดาวน์โหลดข้อมูลเรดาร์ตรวจอากาศจริง (Thailand Composite Radar) ผ่าน RainViewer API
และแปลงเป็นไฟล์ NetCDF พร้อมตัวแปร 'dbz' บนกริดพิกัดดาวเทียม Himawari สำหรับ convrain

คุณสมบัติ:
    - ดึงภาพเรดาร์ Composite ที่รวมสถานีเรดาร์ทั่วไทย (TMD + กทม.) อัปเดตทุก 10 นาที
    - แปลงโค้ดสี RGBA เป็นค่าความสะท้อนเรดาร์จริง (Reflectivity ในหน่วย dBZ)
      โดยใช้ตารางเทียบสีทางการ (RainViewer Universal Blue Color Table)
    - Georeference จาก Web Mercator (EPSG:3857) ลงบนกริดพิกัด 2 km ของดาวเทียม Himawari
    - รองรับการดาวน์โหลดเฟรมล่าสุด หรือดึงย้อนหลัง 2 ชั่วโมง (~13 เฟรม) พร้อมกัน

วิธีใช้:
    # 1. ดึงเฟรมเรดาร์ล่าสุด 1 เฟรม
    python scripts/download_rainviewer.py --latest

    # 2. ดึงเฟรมเรดาร์ย้อนหลังทั้งหมดที่มีในระบบ (~2 ชั่วโมงล่าสุด, 12-13 เฟรม)
    python scripts/download_rainviewer.py --all

    # 3. กำหนดโฟลเดอร์ปลายทาง
    python scripts/download_rainviewer.py --all --radar-dir data/radar --scans-dir data/scans
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from PIL import Image
from scipy.spatial import cKDTree

# กำหนด sys.path ให้ import โมดูล convrain ได้
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAINVIEWER_MAPS_URL = "https://api.rainviewer.com/public/weather-maps.json"
COLOR_TABLE_URL = "https://www.rainviewer.com/files/rainviewer_api_colors_table.csv"


def load_or_fetch_color_palette(cache_path: Path) -> tuple[cKDTree, np.ndarray]:
    """โหลดหรือดาวน์โหลดตารางเทียบสี Universal Blue เป็นค่า dBZ"""
    if not cache_path.exists():
        print(f"กำลังดาวน์โหลดตารางเทียบสีเรดาร์จาก RainViewer...")
        resp = requests.get(COLOR_TABLE_URL, timeout=15)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)

    palette_rgb = []
    palette_dbz = []
    with open(cache_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dbz = float(row["dBZ / RGBA"])
            ub = row["Universal Blue"].strip().lower()
            if ub and ub != "#00000000":
                r = int(ub[1:3], 16)
                g = int(ub[3:5], 16)
                b = int(ub[5:7], 16)
                palette_rgb.append([r, g, b])
                palette_dbz.append(dbz)

    tree = cKDTree(np.array(palette_rgb, dtype=np.float32))
    return tree, np.array(palette_dbz, dtype=np.float32)


def process_frame(frame: dict, host: str, zoom: int,
                  sat_lat: np.ndarray, sat_lon: np.ndarray,
                  tile_x: np.ndarray, tile_y: np.ndarray,
                  px: np.ndarray, py: np.ndarray,
                  tree: cKDTree, palette_dbz: np.ndarray,
                  out_dir: Path) -> Path | None:
    """ประมวลผล 1 เฟรมเรดาร์: ดาวน์โหลด tile, แมปพิกัด และบันทึกเป็น NetCDF"""
    epoch = frame["time"]
    path = frame["path"]
    frame_time = datetime.fromtimestamp(epoch, tz=timezone.utc)
    out_nc = out_dir / f"radar_{frame_time:%Y%m%dT%H%MZ}.nc"

    if out_nc.exists() and out_nc.stat().st_size > 0:
        print(f"  [ข้าม] {out_nc.name} มีอยู่แล้ว")
        return out_nc

    unique_tiles = sorted(set(zip(tile_x.ravel(), tile_y.ravel())))

    def fetch_tile(coord):
        tx, ty = coord
        url = f"{host}{path}/256/{zoom}/{tx}/{ty}/2/0_0.png"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content)).convert("RGBA")
                return coord, np.array(img)
        except Exception:
            pass
        return coord, None

    tiles = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for coord, arr in pool.map(fetch_tile, unique_tiles):
            if arr is not None:
                tiles[coord] = arr

    if not tiles:
        print(f"  [ข้อผิดพลาด] ไม่สามารถดาวน์โหลด tiles สำหรับเวลา {frame_time:%H:%M} UTC ได้")
        return None

    # สุ่มตัวอย่างลงบนกริดพิกัดดาวเทียม
    dbz_grid = np.full(sat_lat.shape, np.nan, dtype=np.float32)

    for coord, arr in tiles.items():
        mask = (tile_x == coord[0]) & (tile_y == coord[1])
        if not np.any(mask):
            continue
        sample_px = px[mask]
        sample_py = py[mask]
        rgba = arr[sample_py, sample_px]

        # พิกเซลที่มีฝน: ค่า Alpha > 30
        rain_mask = rgba[:, 3] > 30
        if np.any(rain_mask):
            rgb_rain = rgba[rain_mask, :3].astype(np.float32)
            dist, idx = tree.query(rgb_rain)
            valid_color = dist < 45.0  # ป้องกันสีรอยต่อผิดเพี้ยน

            assigned = np.full(len(rgba), np.nan, dtype=np.float32)
            rain_indices = np.where(rain_mask)[0][valid_color]
            assigned[rain_indices] = palette_dbz[idx[valid_color]]

            dbz_grid[mask] = assigned

    finite = np.isfinite(dbz_grid).sum()
    max_dbz = np.nanmax(dbz_grid) if finite > 0 else 0.0

    # บันทึกไฟล์ NetCDF
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"dbz": (("y", "x"), dbz_grid)},
        attrs={"time": frame_time.isoformat(), "source": "RainViewer Thailand Radar Composite"}
    )
    try:
        ds.to_netcdf(out_nc, engine="h5netcdf")
    except Exception:
        ds.to_netcdf(out_nc)

    print(f"  -> บันทึกสำเร็จ: {out_nc.name}  (มีฝน {finite:,} พิกเซล, สูงสุด {max_dbz:.1f} dBZ)")
    return out_nc


def main() -> None:
    parser = argparse.ArgumentParser(description="ดาวน์โหลดเรดาร์ไทยจริงจาก RainViewer API สำหรับ convrain")
    parser.add_argument("--latest", action="store_true", help="ดาวน์โหลดเฉพาะเฟรมล่าสุด 1 เฟรม (default)")
    parser.add_argument("--all", action="store_true", help="ดาวน์โหลดทุกเฟรมที่มีในระบบ (~13 เฟรมย้อนหลัง 2 ชม.)")
    parser.add_argument("--count", type=int, default=None, help="ระบุจำนวนเฟรมที่ต้องการดาวน์โหลดย้อนหลัง")
    parser.add_argument("--radar-dir", default="data/radar", help="โฟลเดอร์เก็บไฟล์ radar NetCDF")
    parser.add_argument("--scans-dir", default="data/scans", help="โฟลเดอร์ scan ดาวเทียม (สำหรับดึงพิกัด)")
    parser.add_argument("--palette-path", default="data/rainviewer_colors.csv", help="ไฟล์แคชตารางเทียบสี")
    parser.add_argument("--zoom", type=int, default=6, help="ระดับ Zoom ของแผนที่ Web Mercator (default: 6)")

    args = parser.parse_args()

    # 1. โหลดพิกัดดาวเทียม
    scans_dir = Path(args.scans_dir)
    scan_files = sorted(scans_dir.glob("scan_*.nc"))
    if not scan_files:
        print(f"[ข้อผิดพลาด] ไม่พบไฟล์ scan ใน {scans_dir} กรุณาเตรียมไฟล์ scan ดาวเทียมก่อน", file=sys.stderr)
        sys.exit(1)

    print(f"โหลดกริดพิกัดดาวเทียมจาก {scan_files[0].name}...")
    with xr.open_dataset(scan_files[0], engine="h5netcdf") as sds:
        sat_lat = sds["lat"].values.astype(np.float64)
        sat_lon = sds["lon"].values.astype(np.float64)

    # 2. คำนวณความสัมพันธ์พิกัดดาวเทียม -> Web Mercator Tile
    zoom = args.zoom
    n = 2.0 ** zoom
    x_g = (sat_lon + 180.0) / 360.0 * n
    lat_rad = np.deg2rad(sat_lat)
    y_g = (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n

    tile_x = np.floor(x_g).astype(int)
    tile_y = np.floor(y_g).astype(int)
    px = np.clip(np.floor((x_g - tile_x) * 256).astype(int), 0, 255)
    py = np.clip(np.floor((y_g - tile_y) * 256).astype(int), 0, 255)

    # 3. โหลดตารางเทียบสีเรดาร์
    tree, palette_dbz = load_or_fetch_color_palette(Path(args.palette_path))

    # 4. เรียกดูรายการเฟรมเรดาร์ที่มีจาก API
    print("กำลังตรวจสอบเฟรมเรดาร์ล่าสุดจาก RainViewer API...")
    resp = requests.get(RAINVIEWER_MAPS_URL, timeout=10)
    resp.raise_for_status()
    meta = resp.json()
    host = meta["host"]
    past_frames = meta.get("radar", {}).get("past", [])

    if not past_frames:
        print("[ข้อผิดพลาด] ไม่พบรายการเฟรมเรดาร์ในระบบ RainViewer", file=sys.stderr)
        sys.exit(1)

    if args.all:
        selected_frames = past_frames
    elif args.count:
        selected_frames = past_frames[-args.count:]
    else:
        # Default: latest
        selected_frames = [past_frames[-1]]

    print(f"พบข้อมูลเรดาร์พร้อมดึงจำนวน {len(selected_frames)} เฟรม (ช่วง {datetime.fromtimestamp(selected_frames[0]['time'], tz=timezone.utc):%H:%M} ถึง {datetime.fromtimestamp(selected_frames[-1]['time'], tz=timezone.utc):%H:%M} UTC):")

    out_dir = Path(args.radar_dir)
    success = 0
    for frame in selected_frames:
        res = process_frame(frame, host, zoom, sat_lat, sat_lon, tile_x, tile_y,
                            px, py, tree, palette_dbz, out_dir)
        if res:
            success += 1

    print(f"\nดาวน์โหลดและแปลงข้อมูลเสร็จสิ้นทั้งหมด {success}/{len(selected_frames)} เฟรม เก็บไว้ที่ {out_dir}")


if __name__ == "__main__":
    main()

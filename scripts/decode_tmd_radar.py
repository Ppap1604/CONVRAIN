"""
ถอดภาพเรดาร์ TMD พิษณุโลก (PPI รัศมี 240 km, JPEG ทุก 15 นาที) → dBZ บนกริดดาวเทียมไทย 0.02° ของ data/Thai
ใช้ตรวจผล / ปรับเทียบเท่านั้น (ไม่ใช่ข้อมูลเข้าตอนรันจริง)

    python scripts/decode_tmd_radar.py --jpg-dir data/Thai/ground/radar_phs/raw_jpeg/20261005 \
        --palette data/Thai/ground/radar_phs/raw_jpeg/PHS_palette.json --out runs/radar_decoded
    → radar_<วัน>.npz: dbz[t, y, x] (NaN = นอกพื้นที่วัด, 0 = วัดได้แต่ไม่มีฝน), time_vals, lat, lon, valid

วิธี:
  - เทียบสีในปริภูมิ CIELAB (ใกล้เคียงการรับรู้สีของคน ทนต่อสีเพี้ยนจากการบีบอัด JPEG ได้ดีกว่าระยะ RGB):
    พิกเซลเป็นเสียงสะท้อนเมื่อห่างสีของแถบ dBZ ที่ใกล้ที่สุด ≤ 20 และห่างจาก "พื้นหลังคงที่" ≥ 16
  - พื้นหลังคงที่ = แผนที่ภูมิประเทศ / เส้นเขต / ตัวอักษรของภาพ (มัธยฐานรายพิกเซลของภาพที่มีเสียงสะท้อนน้อย)
    ไฟล์ PHS_background_rgb.npz ข้าง palette; ไม่มี → สร้างจากภาพที่ส่งเข้ามา (มัธยฐานของ 30% ที่มีสีน้อยที่สุด)
  - เรขาคณิต azimuthal equidistant รอบสถานี (16.7765°N 100.2161°E, จุดศูนย์กลางภาพ 436.01, 391.29 px, 0.66011 km/px);
    ใช้ระยะ 10–230 km; ตัดพิกเซลที่เป็นเสียงสะท้อนใน > 60 % ของทุกภาพ (clutter)
ตรวจแล้วนับพิกเซลเสียงสะท้อนได้ใกล้เคียงบันทึกของคลังภาพเอง (อัตราส่วน ~1.0)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CX, CY, KPX = 436.01, 391.29, 0.66011
LAT0, LON0 = 16.7765, 100.2161
PLOT = (75, 0, 800, 757)                 # กรอบพื้นที่แผนที่ในภาพ (x0, y0, x1, y1)
PAL_TOL, BG_MIN, STATIC = 20.0, 16.0, 0.60
LAT, LON, RES = (14.4, 19.2), (97.6, 102.8), 0.02


def srgb_to_lab(a):
    """sRGB (0–255) → CIELAB (D65)"""
    a = np.asarray(a, np.float32) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def grid_pixels():
    """กริดดาวเทียมไทย → พิกัดพิกเซลในภาพเรดาร์ + พื้นที่ที่ใช้ได้"""
    ny, nx = int(round((LAT[1] - LAT[0]) / RES)), int(round((LON[1] - LON[0]) / RES))
    lat = LAT[1] - RES * (np.arange(ny) + 0.5)                          # แถว 0 = เหนือสุด
    lon = LON[0] + RES * (np.arange(nx) + 0.5)
    la, lo = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")
    p0, l0 = np.radians(LAT0), np.radians(LON0)
    c = np.arccos(np.clip(np.sin(p0) * np.sin(la) + np.cos(p0) * np.cos(la) * np.cos(lo - l0), -1, 1))
    k = np.where(c > 0, c / np.sin(np.maximum(c, 1e-12)), 1.0)
    E = 6371.0088 * k * np.cos(la) * np.sin(lo - l0)
    N = 6371.0088 * k * (np.cos(p0) * np.sin(la) - np.sin(p0) * np.cos(la) * np.cos(lo - l0))
    x, y = np.rint(CX + E / KPX).astype(int), np.rint(CY - N / KPX).astype(int)
    rng = np.hypot(E, N)
    ok = (rng >= 10) & (rng <= 230) & (x >= PLOT[0]) & (x < PLOT[2]) & (y >= PLOT[1]) & (y < PLOT[3])
    return lat, lon, np.clip(x, 0, 799), np.clip(y, 0, 799), ok


def load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), np.uint8)


def background(files, bg_file: Path | None):
    if bg_file is not None and bg_file.exists():
        return np.load(bg_file)["rgb"].astype(np.float32)
    stack = np.stack([load_rgb(f) for f in files])
    colour = (stack.max(-1).astype(int) - stack.min(-1)).mean((1, 2))      # ภาพที่มีสีน้อย = เสียงสะท้อนน้อย
    quiet = stack[colour <= np.quantile(colour, 0.30)]
    return np.median(quiet, 0).astype(np.float32)


def decode(rgb_img, pal_lab, dbz, bg_lab, yy, xx):
    lab = srgb_to_lab(rgb_img[yy, xx].astype(np.float32))
    dp = np.sqrt(((lab[:, :, None, :] - pal_lab[None, None]) ** 2).sum(-1))
    dbg = np.sqrt(((lab - bg_lab) ** 2).sum(-1))
    echo = (dp.min(-1) <= PAL_TOL) & (dbg >= BG_MIN)
    return np.where(echo, dbz[dp.argmin(-1)], np.nan).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpg-dir", required=True)
    ap.add_argument("--palette", required=True)
    ap.add_argument("--background", default=None, help="ไฟล์พื้นหลัง (ค่าเริ่มต้น: PHS_background_rgb.npz ข้าง palette)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = json.loads(Path(args.palette).read_text())
    rgb, dbz = np.array(p["rgb"], np.float32), np.array(p["dbz"], np.float32)
    lat, lon, x, y, ok = grid_pixels()
    files = sorted(Path(args.jpg_dir).glob("PHS_*Z.jpg"))
    bg_file = Path(args.background) if args.background else Path(args.palette).with_name("PHS_background_rgb.npz")
    bg_lab = srgb_to_lab(background(files, bg_file)[y, x])
    pal_lab = srgb_to_lab(rgb)
    imgs, times = [], []
    for f in files:
        imgs.append(decode(load_rgb(f), pal_lab, dbz, bg_lab, y, x))
        s = f.stem.split("_")                                              # PHS_YYYYMMDD_HHMMZ
        times.append(np.datetime64(f"{s[1][:4]}-{s[1][4:6]}-{s[1][6:]}T{s[2][:2]}:{s[2][2:4]}"))
    A = np.stack(imgs)
    valid = ok & ~(np.isfinite(A).mean(0) > STATIC)
    A = np.where(valid[None], np.nan_to_num(A, nan=0.0), np.nan).astype(np.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    day = str(times[0])[:10].replace("-", "")
    np.savez_compressed(out / f"radar_{day}.npz", dbz=A, time_vals=np.array(times), lat=lat.astype(np.float32),
                        lon=lon.astype(np.float32), valid=valid)
    print(f"{len(files)} ภาพ → {out / f'radar_{day}.npz'} (พื้นที่วัดได้ {int(valid.sum())} ช่อง)")


if __name__ == "__main__":
    main()

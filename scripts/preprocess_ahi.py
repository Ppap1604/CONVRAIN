"""
Phase 0: แปลงไฟล์ Himawari HSD ดิบ → NetCDF ที่ crop ตามโดเมนแล้ว (1 ไฟล์ต่อ scan)

ไฟล์ HSD ตั้งชื่อแบบ HS_H09_YYYYMMDD_HHMM_Bnn_<area>_R20_Snnnn.DAT(.bz2)
    area = FLDK (เต็มดวง) หรือ JP01 (พื้นที่ญี่ปุ่น) ตาม satellite.area ใน config
สคริปต์จัดกลุ่มไฟล์ตามเวลา scan แล้วอ่านด้วย Satpy (ต้อง pip install satpy)

รัน:
    python scripts/preprocess_ahi.py --hsd-dir /data/hsd --out-dir /data/scans
    python scripts/preprocess_ahi.py --hsd-dir data/Japan/satellite/hsd --out-dir runs/jp/scans --config config/japan.yaml
ผลลัพธ์: /data/scans/scan_YYYYMMDDTHHMMZ.nc
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import _common  # noqa: F401  (เพิ่ม path ของ package)
from convrain.config import load_config
from convrain.ingest import read_ahi_hsd, save_scan_netcdf

HSD_RE = re.compile(r"HS_H0\d_(\d{8})_(\d{4})_(B\d{2})_([A-Z]{2}[A-Z0-9]{2})_")


def hsd_glob(cfg: dict | None = None) -> str:
    """รูปแบบชื่อไฟล์ HSD ของชนิดภาพใน config (ค่าเริ่มต้น FLDK)"""
    return f"HS_H0*_{(cfg or {}).get('satellite', {}).get('area', 'FLDK')}*"


def group_hsd(files, area: str = "FLDK"):
    """จัดกลุ่มไฟล์ segment ตามเวลา scan: {datetime: {band: [files]}} (เฉพาะชนิดภาพ area)"""
    groups: dict[datetime, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        m = HSD_RE.search(f.name)
        if m and m.group(4) == area:
            t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            groups[t][m.group(3)].append(f)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsd-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bands = cfg["satellite"]["bands"]

    groups = group_hsd(sorted(Path(args.hsd_dir).rglob(hsd_glob(cfg))), cfg["satellite"].get("area", "FLDK"))
    for t, by_band in sorted(groups.items()):
        dest = out / f"scan_{t:%Y%m%dT%H%MZ}.nc"
        if dest.exists() and not args.overwrite:
            continue
        missing = [b for b in bands if b not in by_band]
        if missing:
            print(f"{t:%Y-%m-%d %H:%M} ข้าม: ไม่มี band {missing}")
            continue
        day = [b for b in cfg["satellite"].get("day_bands", []) if b in by_band]          # ช่องกลางวัน (ถ้ามี)
        files = [f for b in list(bands) + day for f in by_band[b]]
        scan = read_ahi_hsd(files, cfg)
        save_scan_netcdf(scan, dest)
        print(f"{t:%Y-%m-%d %H:%M} -> {dest.name}  QC={'pass' if scan.quality_ok else 'fail'}")


if __name__ == "__main__":
    main()

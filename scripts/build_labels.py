"""
Phase 3: สร้าง label ต่อ track จากผล replay ที่ใส่เรดาร์ (หัวข้อ 10.2)

Input : โฟลเดอร์ผล replay.py (timeseries/ + tracks.parquet) ที่รันด้วย --radar-dir
Output: labels.parquet (ต่อ track), labeled.parquet (ตาราง object-scan + label)
        label_check.csv  สุ่ม 100 track ไว้ตรวจด้วยตา (หัวข้อ 15, Phase 3 ข้อ 4)

รัน:
    python scripts/build_labels.py --run-dir runs/2025
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _common  # noqa: F401
from convrain.config import load_config
from convrain.labels import attach_labels, build_track_labels
from convrain.output import read_timeseries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-coverage", type=float, default=0.5, help="สัดส่วน pixel ที่เรดาร์ครอบคลุมขั้นต่ำ")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(args.run_dir)

    ts = read_timeseries(run)
    if "radar_max_dbz" not in ts:
        raise SystemExit("ไม่มีคอลัมน์ radar_max_dbz: ต้องรัน replay.py พร้อม --radar-dir ก่อน")
    tracks = pd.read_parquet(run / "tracks.parquet")
    labels = build_track_labels(ts, tracks, cfg, min_coverage=args.min_coverage)
    df = attach_labels(ts, labels)

    labels.to_parquet(run / "labels.parquet", index=False)
    df.to_parquet(run / "labeled.parquet", index=False)
    labels.sample(min(100, len(labels)), random_state=0).to_csv(run / "label_check.csv", index=False)

    n = len(labels)
    print(f"track ทั้งหมด {n:,}")
    print(f"  มีเรดาร์ครอบคลุม     : {labels['radar_ok'].sum():,}")
    print(f"  มี onset             : {labels['t_onset'].notna().sum():,}")
    print(f"  onset ตั้งแต่เกิด    : {labels['onset_at_birth'].sum():,} (ตัดออกจากชุด onset)")
    print(f"  censored             : {labels['censor_time'].notna().sum():,}")
    print(f"ตาราง object-scan ที่มี label: {len(df):,} แถว -> {run / 'labeled.parquet'}")


if __name__ == "__main__":
    main()

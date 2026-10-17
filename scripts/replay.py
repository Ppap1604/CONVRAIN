"""
Phase 2–3 และ 6: replay ข้อมูลย้อนหลังทีละ scan (เหมือน real-time)

Input
    --scans-dir : NetCDF จาก preprocess_ahi.py (scan_YYYYMMDDTHHMMZ.nc)
    --radar-dir : (ทางเลือก ใช้สร้าง label) NetCDF ของเรดาร์ที่ regrid ลง grid ดาวเทียมแล้ว
                  ชื่อไฟล์ radar_YYYYMMDDTHHMMZ.nc มีตัวแปร "dbz" (y, x)
                  ใช้ convrain.labels.regrid_to_satellite ในการ regrid
    --models-dir: (ทางเลือก) rain_lut.npz / timing_params.json (ไม่ใส่ = ค่าที่ติดมากับโค้ด)
    --points    : (ทางเลือก) CSV จุด (key, lat, lon) ที่ต้องการเวลาฝนรายจุด
    --collect-timing-rows : เก็บแถวรายจุด (ตัวแปรครบ) ไว้ปรับเทียบด้วย calibrate_timing.py

Output (--out)
    json/, geojson/, timeseries/   ผลต่อ scan (หัวข้อ 9.5)
    points/                        เวลาฝนเริ่ม / หนักสุด / หยุด รายจุด (ถ้าใส่ --points)
    timing_rows.parquet            แถวสำหรับปรับเทียบ (ถ้าใส่ --collect-timing-rows)
    tracks.parquet                 track ที่ปิดแล้ว (ใช้ทำ censor ตอนสร้าง label)
    lut_pairs.npz                  คู่ (BT, BTD, rain rate) สำหรับ build_lut.py (ถ้าใส่ --collect-lut)
    timings.csv                    เวลาประมวลผลแต่ละขั้น (Gate ของ Phase 2)

รัน:
    python scripts/replay.py --scans-dir /data/scans --points gauges.csv --out runs/2025 --collect-timing-rows
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _common
from convrain.config import load_config
from convrain.experiment import replay_scans
from convrain.ingest import load_scan_netcdf
from convrain.labels import dbz_to_rainrate
from convrain.output import write_scan_outputs
from convrain.pipeline import ConvRainPipeline


def load_radar(path: Path) -> np.ndarray:
    import xarray as xr

    try:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            return ds["dbz"].values.astype(np.float32)
    except Exception:
        with xr.open_dataset(path) as ds:
            return ds["dbz"].values.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans-dir", required=True)
    ap.add_argument("--radar-dir", default=None)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--points", default=None, help="CSV (key, lat, lon) สำหรับเวลาฝนรายจุด")
    ap.add_argument("--collect-lut", action="store_true")
    ap.add_argument("--collect-timing-rows", action="store_true", help="เก็บแถวรายจุดไว้ปรับเทียบส่วนทำนายเวลา")
    ap.add_argument("--no-json", action="store_true", help="ไม่เขียน JSON/GeoJSON ต่อ scan (เร็วขึ้นตอนสร้างชุด train)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lut, meta = _common.load_models(args.models_dir, cfg)
    if args.points:
        cfg["timing"]["points_file"] = args.points
    pipe = ConvRainPipeline(cfg, rain_lut=lut, collect_lut_pairs=args.collect_lut,
                            collect_timing_rows=args.collect_timing_rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scans = _common.files_by_time(args.scans_dir, "scan_*.nc")
    radar = _common.files_by_time(args.radar_dir, "radar_*.nc") if args.radar_dir else []

    def items():
        prev_t = None
        for t, path in scans:
            scan = load_scan_netcdf(path, cfg)
            frames = rr = None
            if radar:
                # เรดาร์ทุก frame ในช่วง (scan ก่อน, scan นี้]
                lo = prev_t or t
                sel = [(rt, rp) for rt, rp in radar if (lo < rt <= t) or (prev_t is None and rt == t)]
                frames = [(rt, load_radar(rp)) for rt, rp in sel]
                at_t = [f for rt, f in frames if rt == t]
                rr = np.where(at_t[0] >= 15, dbz_to_rainrate(at_t[0]), 0.0) if at_t else None
            prev_t = t
            yield scan, frames, rr

    def on_result(res):
        if res.skipped:
            print(f"{res.time:%Y-%m-%d %H:%M} ข้าม ({res.errors})")
            return
        write_scan_outputs(out, res.document, res.footprints, res.timeseries, write_json=not args.no_json,
                           points=res.points, outlook=res.outlook, rainmap=res.rainmap)
        if res.errors:
            print(f"{res.time:%Y-%m-%d %H:%M} validation: {res.errors[:3]}")

    r = replay_scans(pipe, items(), on_result)
    r["tracks"].to_parquet(out / "tracks.parquet", index=False)
    r["timings"].to_csv(out / "timings.csv", index=False)
    if args.collect_timing_rows and pipe.timing.rows_log:
        import pandas as pd
        pd.concat(pipe.timing.rows_log, ignore_index=True).to_parquet(out / "timing_rows.parquet", index=False)
    if args.collect_lut and pipe.lut_pairs:
        np.savez_compressed(out / "lut_pairs.npz",
                            bt=np.concatenate([p[0] for p in pipe.lut_pairs]),
                            btd=np.concatenate([p[1] for p in pipe.lut_pairs]),
                            rr=np.concatenate([p[2] for p in pipe.lut_pairs]))
    print(f"เสร็จ {len(scans)} scan, object-scan {len(r['timeseries']):,} แถว")
    print("เวลาเฉลี่ยต่อ scan (ms):", r["timings"].mean().round(1).to_dict())


if __name__ == "__main__":
    main()

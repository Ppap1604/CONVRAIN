"""
replay ไฟล์ภาพดาวเทียมแบบ npz (อุณหภูมิสว่างที่ถอดจาก HSD แล้ว ตัดเฉพาะโดเมน) ทีละ scan เหมือน real-time

รูปแบบไฟล์ (ดู data/): Bnn (t, y, x) อุณหภูมิสว่าง K, lat (y), lon (x), SAZ (y, x) มุมเซนิธดาวเทียม, time_vals (t)
    ไทย   th_<วัน>.npz  : B08 … B16 ภาพเต็มดวงทุก 10 นาที
    ญี่ปุ่น jp_<วัน>.npz : B07 … B16 + R_B04, R_B05 (ค่าสะท้อน 0–1, กลางวัน) ภาพพื้นที่ญี่ปุ่น (ภาพย่อยแรกของทุก 10 นาที)
แบนด์ที่ไฟล์ไม่มีจะเป็น NaN (ตัวแปรที่ใช้แบนด์นั้นเป็น NaN); ไม่มี R_B04 / R_B05 / B07 = แผนที่ฝนใช้โหมดกลางคืนตลอด

Output (--out) เหมือน replay.py: json/, geojson/, timeseries/, points/, outlook/, tracks.parquet, timings.csv
                                 (+ timing_rows.parquet ถ้าใส่ --collect-timing-rows)
รัน:
    python scripts/replay_npz.py --npz data/Thai/satellite/th_20261005.npz --config config/thailand_phs.yaml \
        --points runs/th_ground/points.csv --out runs/th_20261005 --collect-timing-rows
    python scripts/replay_npz.py --npz data/Japan/satellite/jp_20230726.npz --config config/japan.yaml \
        --points runs/jp_ground/points.csv --out runs/jp_20230726 --collect-timing-rows
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _common
from convrain.config import load_config
from convrain.experiment import replay_scans
from convrain.ingest import Scan
from convrain.output import write_scan_outputs
from convrain.pipeline import ConvRainPipeline


def npz_scans(path: Path, bands: list[str]):
    """ไฟล์ npz → Scan ทีละภาพ (เรียงเวลา)"""
    z = np.load(path)
    lat, lon = np.meshgrid(z["lat"].astype(np.float32), z["lon"].astype(np.float32), indexing="ij")
    saz = z["SAZ"].astype(np.float32)
    times = pd.to_datetime(z["time_vals"]).floor("10min")
    cube = {b: z[b] for b in z.files if b.startswith("B")}
    refl = {b[2:]: z[b] for b in z.files if b.startswith("R_")}          # ค่าสะท้อนช่องกลางวัน (สัดส่วน 0–1)
    for i in range(len(times)):
        bt = {b: np.asarray(cube[b][i], dtype=np.float32).copy() for b in bands if b in cube}
        rf = {b: np.asarray(v[i], dtype=np.float32) for b, v in refl.items() if np.isfinite(v[i]).any()}
        yield Scan(time=times[i].to_pydatetime(), bt=bt, refl=rf, lat=lat, lon=lon, sat_zenith=saz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, nargs="+", help="ไฟล์ภาพ (หลายวันเรียงตามเวลาได้)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--points", default=None, help="CSV (key, lat, lon) สำหรับเวลาฝนรายจุด")
    ap.add_argument("--collect-timing-rows", action="store_true")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--max-scans", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    lut, _ = _common.load_models(args.models_dir, cfg)
    if args.points:
        cfg["timing"]["points_file"] = args.points
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bands = sorted(set(cfg["satellite"]["bands"]) | set(cfg["satellite"].get("day_bands", [])) | {"B13", "B15"})
    rows, tracks, timings = [], [], []
    for path in args.npz:
        # แต่ละไฟล์ = วันใหม่ เริ่มติดตามใหม่ (ข้อมูลข้ามคืนขาดอยู่แล้ว)
        pipe = ConvRainPipeline(cfg, rain_lut=lut, collect_timing_rows=args.collect_timing_rows)
        items = npz_scans(Path(path), bands)
        if args.max_scans:
            items = (s for k, s in enumerate(items) if k < args.max_scans)

        def on_result(res):
            if not res.skipped:
                write_scan_outputs(out, res.document, res.footprints, res.timeseries, write_json=not args.no_json,
                                   points=res.points, outlook=res.outlook, rainmap=res.rainmap)
                if res.errors:
                    print(f"{res.time:%Y-%m-%d %H:%M} validation: {res.errors[:3]}")

        r = replay_scans(pipe, items, on_result)
        rows.extend(pipe.timing.rows_log)
        tracks.append(r["tracks"])
        timings.append(r["timings"])
        print(f"{Path(path).name}: object-scan {len(r['timeseries']):,} แถว, เวลาฝนรายจุด {len(r['points']):,} แถว, "
              f"ms/scan {r['timings']['total'].median():.0f}")
    pd.concat(tracks, ignore_index=True).to_parquet(out / "tracks.parquet", index=False)
    pd.concat(timings, ignore_index=True).to_csv(out / "timings.csv", index=False)
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(out / "timing_rows.parquet", index=False)


if __name__ == "__main__":
    main()

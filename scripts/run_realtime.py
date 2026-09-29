"""
Phase 7: รันแบบ real-time (หัวข้อ 17)

ทำงานเป็น process เดียว ตรวจโฟลเดอร์ที่ไฟล์ HSD ใหม่เข้ามา (event-driven แบบ polling)
    1. หา scan ที่ไฟล์ครบทุก band และยังไม่ได้ประมวลผล
    2. อ่าน + crop ด้วย Satpy (ขั้น 0) → pipeline.step() (ขั้น 1–6)
    3. เขียน output JSON/GeoJSON/Parquet
    4. บันทึก state ของ tracker ลงดิสก์ทุก scan → restart แล้วต่อได้โดยไม่เสีย track
    5. บันทึก monitoring: เวลาประมวลผล, จำนวน object, ค่าเฉลี่ย feature (ตรวจ data drift)

รัน:
    python scripts/run_realtime.py --watch-dir /data/incoming --out /data/nowcast \
        --models-dir models/v1 --state /data/nowcast/state.pkl
ใช้ --once เพื่อประมวลผลสิ่งที่ค้างอยู่ครั้งเดียวแล้วจบ (เหมาะกับ cron ทุก 10 นาที)
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

import _common
from convrain.config import load_config
from convrain.ingest import read_ahi_hsd
from convrain.output import write_scan_outputs
from convrain.pipeline import ConvRainPipeline
from preprocess_ahi import group_hsd

MONITOR_FIELDS = ["scan_time", "processed_at", "n_objects", "n_warnings", "total_ms", "n_errors",
                  "mean_min_bt", "mean_btd_wv_cold", "mean_ttd_cold", "latency_s"]


def cleanup_old_files(out_dir: Path, keep_days: int) -> None:
    if keep_days <= 0:
        return
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=keep_days)
    for p in out_dir.rglob("*"):
        if p.is_file() and p.name not in ("state.pkl", "monitor.csv"):
            try:
                if datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) < cutoff:
                    p.unlink()
            except Exception:
                pass
    for p in sorted(out_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            try:
                p.rmdir()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--state", default=None, help="ไฟล์ state (default: <out>/state.pkl)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--poll-s", type=int, default=30)
    ap.add_argument("--segments", type=int, default=0,
                    help="จำนวน segment ต่อ band ที่ต้องมีครบก่อนประมวลผล (0 = ไม่ตรวจ)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--keep-days", type=int, default=3, help="จำนวนวันที่เก็บไฟล์ output ไว้ (0 = ไม่ลบ)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    onset, peak, stop, lut, meta = _common.load_models(args.models_dir)
    if meta.get("warn_threshold") is not None:
        cfg["onset"]["warn_threshold"] = meta["warn_threshold"]
    pipe = ConvRainPipeline(cfg, onset, peak, stop, lut)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = Path(args.state) if args.state else out / "state.pkl"
    if state.exists():
        try:
            pipe.load_state(state)
            print(f"กู้ state แล้ว: scan ล่าสุด {pipe.prev_time}, track ที่เปิดอยู่ {len(pipe.tracker.tracks)}")
        except Exception as e:
            print(f"โหลด state ไม่สำเร็จ ({e}) จะเริ่ม state ใหม่")

    monitor = out / "monitor.csv"
    if not monitor.exists():
        with open(monitor, "w", newline="") as f:
            csv.writer(f).writerow(MONITOR_FIELDS)

    bands = cfg["satellite"]["bands"]
    while True:
        groups = group_hsd(sorted(Path(args.watch_dir).rglob("HS_H0*_FLDK*")))
        pending = []
        for t, by_band in sorted(groups.items()):
            if pipe.prev_time is not None and t <= pipe.prev_time:
                continue
            complete = all(b in by_band for b in bands) and \
                (args.segments == 0 or all(len(by_band[b]) >= args.segments for b in bands))
            if complete:
                pending.append((t, [f for b in bands for f in by_band[b]]))

        for t, files in pending:
            try:
                scan = read_ahi_hsd(files, cfg)
                res = pipe.step(scan)
            except Exception as exc:  # ไม่ให้ scan เสียหนึ่งไฟล์หยุดทั้งระบบ
                print(f"{t:%Y-%m-%d %H:%M} ผิดพลาด: {exc}")
                continue
            if not res.skipped:
                write_scan_outputs(out, res.document, res.footprints, res.timeseries)
            pipe.save_state(state)

            ts = res.timeseries
            now = datetime.now(timezone.utc)
            row = [t.isoformat(), now.isoformat(),
                   0 if ts is None else len(ts),
                   0 if ts is None or "warning" not in ts else int(ts["warning"].sum()),
                   res.timings_ms.get("total", np.nan), len(res.errors),
                   np.nan if ts is None or not len(ts) else round(ts["min_bt"].mean(), 2),
                   np.nan if ts is None or not len(ts) else round(ts["btd_wv_cold"].mean(), 2),
                   np.nan if ts is None or not len(ts) else round(ts["ttd_cold"].mean(), 2),
                   round((now - t).total_seconds(), 1)]
            with open(monitor, "a", newline="") as f:
                csv.writer(f).writerow(row)
            status = "ข้าม (QC)" if res.skipped else f"{row[2]} object, เตือน {row[3]}, {row[4]} ms"
            print(f"{t:%Y-%m-%d %H:%M}  {status}  latency {row[-1]} s  errors {res.errors[:2]}")

        if pending:
            cleanup_old_files(out, args.keep_days)

        if args.once:
            break
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()

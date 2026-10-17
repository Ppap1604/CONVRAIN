"""
ปรับเทียบค่าคงที่ของส่วนทำนายเวลา (ฝนเริ่ม / หนักสุด / หยุด) ด้วยข้อมูลฝนที่จุด — offline เท่านั้น

Input
    --rows : timing_rows.parquet จาก replay.py --points <จุด> --collect-timing-rows (หลายโฟลเดอร์ได้)
    --rain : CSV ฝนที่จุด คอลัมน์ key, time (UTC) และ rain_mm_10min หรือ dbz
             เรดาร์ (dbz, ทุก 10–15 นาที): แปลงเป็นมม. / 10 นาทีด้วย Marshall–Palmer แล้วจัดเป็นช่วง 10 นาที
             เกณฑ์ฝนตกของเรดาร์ตั้งด้วย --wet-dbz (ไม่ตั้ง = 30 dBZ ≈ 3 มม./ชม.; ค่าคงที่ไทยที่ติดมาใช้ 10.4 dBZ)
             มาตรวัดฝน (rain_mm_10min): ป้ายเวลาเป็น "ปลายช่วง" ตั้ง --time-label end (ค่าเริ่มต้น mid)
             ข้อมูลรายชั่วโมงหยาบเกินไปสำหรับนิยามเหตุการณ์ 10 นาที
Output (--out)
    timing_params.json   ค่าคงที่ใหม่ (replay / run_realtime ใช้อัตโนมัติเมื่อส่ง --models-dir โฟลเดอร์นี้)
    calibration.json     จำนวนแถว / คู่ / วัน ต่อเหตุการณ์

รัน:
    python scripts/calibrate_timing.py --rows runs/sep/timing_rows.parquet --rain radar_points.csv --out models/th1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _common
from convrain.scope.calibrate import (event_rows, fit_event_models, radar_mm10, ten_minute_series,
                                      training_rows)


def load_rain(path: str, time_label: str, wet_dbz: float | None = None) -> dict:
    """ฝนที่จุด → {key: ช่วง 10 นาที (มม.)}; wet_dbz = เกณฑ์ "ฝนตก" ของเรดาร์ (ค่าเริ่มต้น 0.5 มม. / 10 นาที ≈ 30 dBZ)"""
    R = pd.read_csv(path)
    R["time"] = pd.to_datetime(R["time"], utc=True)
    if "rain_mm_10min" in R:
        val = R.rain_mm_10min.astype(float)
        shift = {"mid": 0, "end": -5, "start": 5}[time_label]
        R["time"] = R.time + pd.Timedelta(minutes=shift)
    elif "dbz" in R:
        val = pd.Series(radar_mm10(R.dbz), index=R.index)
        if wet_dbz is not None:                     # ปรับมาตราส่วนให้ "≥ 0.5 มม." ⇔ "≥ wet_dbz" (ลำดับเวลาไม่เปลี่ยน)
            val = val * 0.5 / float(radar_mm10([wet_dbz])[0]) * (1 + 1e-9)
    else:
        raise SystemExit("ไฟล์ฝนต้องมีคอลัมน์ rain_mm_10min หรือ dbz")
    R["val"] = val
    return {str(k): ten_minute_series(g.time, g.val) for k, g in R.groupby("key")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--rain", required=True)
    ap.add_argument("--time-label", default="mid", choices=["mid", "end", "start"])
    ap.add_argument("--wet-dbz", type=float, default=None, help="เรดาร์: เกณฑ์ฝนตก (dBZ) เช่น 10.4 = เรดาร์เห็นฝน")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = pd.concat([pd.read_parquet(p) for p in args.rows], ignore_index=True)
    rows["station"] = rows.station.astype(str)
    T = training_rows(rows, load_rain(args.rain, args.time_label, args.wet_dbz))
    if T.empty:
        raise SystemExit("ไม่มีคู่ (เมฆ, จุด) ที่มีฝนถึงเกณฑ์ — ตรวจว่า key ของจุดตรงกันและช่วงเวลาซ้อนกัน")
    params = fit_event_models(T)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / _common.TIMING_PARAMS).write_text(json.dumps(params, indent=1), encoding="utf-8")
    day = pd.to_datetime(T.t * 60, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    rep = {ev: {"rows": int(len(I := event_rows(T.assign(day=day), ev))), "pairs": int(I.groupby(["cell_id", "station"]).ngroups),
                "days": int(I.day.nunique())} for ev in ("onset", "peak", "stop")}
    (out / "calibration.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    print(f"ค่าคงที่ → {out / _common.TIMING_PARAMS}")


if __name__ == "__main__":
    main()

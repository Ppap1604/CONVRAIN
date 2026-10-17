"""
ปรับเทียบคำเตือน 3 ระดับ + เวลาฝนเริ่ม / หยุดพร้อมความมั่นใจ ด้วยข้อมูลฝนที่จุด — offline เท่านั้น

Input
    --npz    ไฟล์ภาพรายวัน (รูปแบบเดียวกับ data/*/satellite) หลายวัน — ยิ่งหลายวันยิ่งดี (แนะนำ ≥ 20 วันที่มีฝน)
    --points CSV key, lat, lon ของจุดที่มีข้อมูลฝน (มาตรวัดฝน หรือจุดเสมือนจากเรดาร์: scripts/ground_to_points.py)
    --rain   CSV ฝนที่จุด คอลัมน์ key, time (UTC) และ rain_mm_10min หรือ dbz
             มาตรวัดฝน 10 นาที: ฝนตก = ≥ 0.5 มม. / 10 นาที (--wet-mm); ป้ายเวลาปลายช่วง → --time-label end
             เรดาร์: ฝนตก = ≥ --wet-dbz (ค่าเริ่มต้น 30; ค่าคงที่ไทยที่ติดมากับโปรแกรมใช้ 10.4 = "เรดาร์เห็นฝน")
    --config ภูมิภาค (เกณฑ์อุณหภูมิ, parallax, ชุดแผนที่ฝน)
Output (--out)
    outlook_params.json  ค่าคงที่ (run_realtime / replay ใช้อัตโนมัติเมื่อส่ง --models-dir โฟลเดอร์นี้)
    outlook_eval.json    ผลบนวันที่โมเดลไม่เคยเห็น (แบ่งวันเป็น --folds กลุ่ม): คำเตือน POD / FAR / CSI,
                         ความแม่นของเวลาที่บอก, ความมั่นใจตรงกับความจริงแค่ไหน
รัน:
    python scripts/ground_to_points.py radar --npz data/Thai/ground/radar_phs/decoded/radar_*.npz --step 8 --out runs/th_ground
    python scripts/calibrate_outlook.py --npz data/Thai/satellite/th_*.npz --config config/thailand_phs.yaml \
        --points runs/th_ground/points.csv --rain runs/th_ground/rain_points.csv --wet-dbz 10.4 --out models/th_outlook
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

import _common  # noqa: F401  (ตั้ง sys.path)
from convrain.config import load_config
from convrain.scope import params
from convrain.scope.outlook_fit import collect_rows, fit_outlook, label_rows, rain_series
from replay_npz import npz_scans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--rain", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--wet-mm", type=float, default=0.5)
    ap.add_argument("--wet-dbz", type=float, default=30.0)
    ap.add_argument("--time-label", default="mid", choices=["mid", "end", "start"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg["timing"]
    rc = json.loads(Path(tc["rainmap_file"]).read_text(encoding="utf-8")) if tc.get("rainmap_file") else \
        params.load("rainmap", tc.get("constants", params.DEFAULT_REGION))
    pts = pd.read_csv(args.points)[["key", "lat", "lon"]]
    pts["key"] = pts.key.astype(str)
    bands = sorted(set(cfg["satellite"]["bands"]) | set(cfg["satellite"].get("day_bands", [])) | {"B13", "B15"})
    files = sorted({f for p in args.npz for f in glob.glob(p)})
    rows = []
    for f in files:
        R = collect_rows(npz_scans(Path(f), bands), pts, rc, tc["parallax_top_k"])
        rows.append(R.assign(day=Path(f).stem.split("_")[-1]))
        print(f"  {Path(f).name}: {len(R):,} แถว", flush=True)
    rows = pd.concat(rows, ignore_index=True)

    rain = pd.read_csv(args.rain)
    rain["key"] = rain.key.astype(str)
    rain["time"] = pd.to_datetime(rain.time, utc=True)
    if "rain_mm_10min" in rain:
        col, wet = "rain_mm_10min", args.wet_mm
        rain["time"] = rain.time + pd.Timedelta(minutes={"mid": 0, "end": -5, "start": 5}[args.time_label])
    elif "dbz" in rain:
        col, wet = "dbz", args.wet_dbz
    else:
        raise SystemExit("ไฟล์ฝนต้องมีคอลัมน์ rain_mm_10min หรือ dbz")
    L = label_rows(rows, rain_series(rain, col), wet)
    print(f"ตัวอย่างฝนเริ่ม {int(L.onset_ok.sum()):,}  ฝนหยุด {int(L.stop_ok.sum()):,}", flush=True)
    const, rep = fit_outlook(L, k=args.folds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / _common.OUTLOOK_PARAMS).write_text(json.dumps({**const, "stop_requires": "map"}), encoding="utf-8")
    rep.update({"days": int(rows.day.nunique()), "points": int(rows.key.nunique()), "wet_threshold": wet,
                "wet_variable": col})
    (out / "outlook_eval.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    o, s = rep["onset"], rep["stop"]
    print("ฝนเริ่ม: " + " | ".join(f"{k} CSI {v['CSI']:.3f} (POD {v['POD']:.2f} FAR {v['FAR']:.2f})" for k, v in o["levels"].items())
          + f" | IBS skill {o['IBS_skill']:.3f}")
    print(f"ฝนหยุด: IBS skill {s['IBS_skill']:.3f}; ความมั่นใจ ≥ 0.7 ถูก {s['statement']['confidence_gate']['0.7']['verified']}")
    print(f"ค่าคงที่ → {out / _common.OUTLOOK_PARAMS}")


if __name__ == "__main__":
    main()

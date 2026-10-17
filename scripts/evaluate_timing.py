"""
ประเมินส่วนทำนายเวลาแบบเว้นทีละวัน (leave-one-day-out) เทียบนาฬิกาตายตัว (t0 + ค่าคงที่) — offline เท่านั้น

Input เหมือน calibrate_timing.py (--rows, --rain); วันที่ทดสอบไม่ถูกใช้ปรับเทียบในรอบของมันเลย
Output (--out)/timing_eval.json + ตารางบนจอ: MAE, % ภายใน ±5 / 10 / 15 / 30 นาที และความครอบคลุมช่วง 50 / 80 %
แยกตามระยะเวลาจริงก่อนเหตุการณ์ (0–10, 10–30, 30–60, 60–120 นาที)
--builtin : ประเมินค่าคงที่ที่ติดมากับโค้ด (ไม่ปรับเทียบใหม่) แทน — เลือกชุดด้วย --region thailand | japan

รัน:
    python scripts/evaluate_timing.py --rows runs/sep/timing_rows.parquet --rain radar_points.csv --out reports/th1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from calibrate_timing import load_rain
from convrain.scope import load_event_models
from convrain.scope.calibrate import event_rows, fit_event_models, scores, training_rows
from convrain.scope.clock import QCOLS, FixedClock
from convrain.scope.hazard import from_dict

EVENTS = ("onset", "peak", "stop")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--rain", required=True)
    ap.add_argument("--time-label", default="mid", choices=["mid", "end", "start"])
    ap.add_argument("--wet-dbz", type=float, default=None, help="เรดาร์: เกณฑ์ฝนตก (dBZ)")
    ap.add_argument("--builtin", action="store_true", help="ใช้ค่าคงที่ที่ติดมากับโค้ด ไม่ปรับเทียบใหม่ต่อรอบ")
    ap.add_argument("--region", default="thailand", choices=["thailand", "japan"], help="ชุดค่าคงที่ของ --builtin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = pd.concat([pd.read_parquet(p) for p in args.rows], ignore_index=True)
    rows["station"] = rows.station.astype(str)
    T = training_rows(rows, load_rain(args.rain, args.time_label, args.wet_dbz))
    T["day"] = pd.to_datetime(T.t * 60, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    builtin = load_event_models(region=args.region) if args.builtin else None
    one_day = T.day.nunique() == 1
    if one_day and not builtin:
        raise SystemExit("มีข้อมูลวันเดียว: เว้นทีละวันไม่ได้ — ใช้ --builtin หรือเพิ่มวัน")
    if one_day:
        print("  มีข้อมูลวันเดียว: นาฬิกาตายตัวใช้ค่าคงที่จากวันเดียวกัน (in-sample, เข้าข้างนาฬิกาตายตัว)")
    res = {"days": sorted(T.day.unique()), "builtin": args.builtin, "fixed_clock_in_sample": bool(one_day),
           "events": {}}
    for ev in EVENTS:
        I = event_rows(T, ev)
        Qm = pd.DataFrame(index=I.index, columns=QCOLS, dtype=float)
        Qf = Qm.copy()
        for d in sorted(I.day.unique()):
            tr, te = (T if one_day else T[T.day != d]), I[I.day == d]
            try:
                m = builtin[ev] if builtin else from_dict(fit_event_models(tr)[ev])
            except ValueError as e:                       # วันฝึกน้อยเกินไป
                print(f"  {ev} {d}: ข้าม ({e})")
                continue
            Qm.loc[te.index] = m.quantiles(te).to_numpy()
            Qf.loc[te.index] = FixedClock.fit(event_rows(tr, ev), ev).quantiles(te).to_numpy()
        ok = Qm.q50.notna()
        I, Qm, Qf = I[ok], Qm[ok], Qf[ok]
        obs, lead = I[ev].to_numpy(float), (I[ev] - I.t).to_numpy(float)
        res["events"][ev] = {"rows": int(len(I)), "pairs": int(I.groupby(["cell_id", "station"]).ngroups),
                             "model": scores(obs, Qm, lead), "fixed_clock": scores(obs, Qf, lead)}
        for name in ("model", "fixed_clock"):
            print(f"  {ev:5s} {name:11s} " + " | ".join(
                f"{k}: n {v['n']} MAE {v['MAE']:5.1f} ±15 {v['w15'] * 100:3.0f}% cov {v['cov50'] * 100:3.0f}/{v['cov80'] * 100:3.0f}"
                for k, v in res["events"][ev][name].items()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing_eval.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"→ {out / 'timing_eval.json'}")


if __name__ == "__main__":
    main()

"""
Demo ตั้งแต่ต้นจนจบบนข้อมูลจำลอง (ไม่ต้องมีข้อมูลจริง)

    Phase 2–3  replay pipeline ทีละ scan → timeseries + tracks + เวลาฝนรายจุด + แถวสำหรับปรับเทียบ
    Phase 4–5  rain rate LUT + ปรับเทียบส่วนทำนายเวลาด้วยฝนจากเรดาร์จำลองที่จุด (วัน train)
    Phase 6    ประเมินเวลาฝนเริ่ม / หนักสุด / หยุด บนวัน test: ค่าที่ติดมากับโค้ด / ค่าที่ปรับเทียบใหม่ / นาฬิกาตายตัว
    Phase 7    รัน pipeline แบบ real-time ด้วยค่าที่ปรับเทียบ แล้วเขียน output JSON/GeoJSON/Parquet/points

รัน:  python scripts/demo_synthetic.py --out outputs/demo
หมายเหตุ: ตัวเลขจากข้อมูลจำลองไม่ได้บอกความแม่นยำกับข้อมูลจริง (ใช้ตรวจว่าทุกขั้นทำงานร่วมกันได้)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convrain.config import load_config  # noqa: E402
from convrain.experiment import fit_lut, replay_scans  # noqa: E402
from convrain.ingest import save_scan_netcdf  # noqa: E402
from convrain.output import write_scan_outputs  # noqa: E402
from convrain.pipeline import ConvRainPipeline  # noqa: E402
from convrain.scope import load_event_models  # noqa: E402
from convrain.scope.calibrate import event_rows, fit_event_models, radar_mm10, scores, ten_minute_series, \
    training_rows  # noqa: E402
from convrain.scope.clock import FixedClock  # noqa: E402
from convrain.scope.hazard import from_dict  # noqa: E402
from convrain.synthetic import SyntheticDay  # noqa: E402

STEP_PX = 6          # เกจเสมือนทุก 6 พิกเซล


def make_days(n: int, seed0: int, start: datetime):
    """วันจำลอง: เริ่ม 05:00 UTC (12:00 น. เวลาไทย) ช่วงที่ฝน convective บ่ายเกิดบ่อย"""
    return [SyntheticDay(seed=seed0 + i, start=start + timedelta(days=i), n_scans=36) for i in range(n)]


def gauges(day: SyntheticDay) -> pd.DataFrame:
    r, c = np.mgrid[STEP_PX // 2:day.ny:STEP_PX, STEP_PX // 2:day.nx:STEP_PX]
    r, c = r.ravel(), c.ravel()
    return pd.DataFrame({"key": [f"g{a}_{b}" for a, b in zip(r, c)], "lat": day.lat[r, c], "lon": day.lon[r, c],
                         "r": r, "c": c})


def rain_at_gauges(day: SyntheticDay, G: pd.DataFrame) -> dict:
    """ฝนจากเรดาร์จำลองที่เกจ (ทุก 5 นาที) → มม. / 10 นาที รายช่วง"""
    rec = {}
    for _, frames, _ in day.scans():
        for t, dbz in frames:
            rec.setdefault(t, dbz[G.r.to_numpy(), G.c.to_numpy()])
    times = sorted(rec)
    V = np.array([rec[t] for t in times])
    return {k: ten_minute_series(times, radar_mm10(V[:, i])) for i, k in enumerate(G.key)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--days", type=int, default=12, help="จำนวนวันจำลอง (2/3 train, 1/3 test)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    days = make_days(args.days, 100, datetime(2026, 5, 1, 5, 0, tzinfo=timezone.utc))
    G = gauges(days[0])
    G[["key", "lat", "lon"]].to_csv(out / "gauges.csv", index=False)
    cfg = load_config(overrides={"domain": days[0].domain(), "timing": {"points_file": str(out / "gauges.csv")}})
    n_train = int(round(2 / 3 * len(days)))

    # ================= Phase 2–3: replay =================
    print(f"[Phase 2–3] replay {len(days)} วันจำลอง ...")
    T_all, pairs, timings = [], [], []
    for i, day in enumerate(days):
        pipe = ConvRainPipeline(cfg, collect_lut_pairs=i < n_train, collect_timing_rows=True)
        r = replay_scans(pipe, day.scans())
        pairs.extend(pipe.lut_pairs)
        timings.append(r["timings"])
        if r["errors"]:
            print("  validation errors:", r["errors"][:3])
        if pipe.timing.rows_log:
            T = training_rows(pd.concat(pipe.timing.rows_log, ignore_index=True), rain_at_gauges(day, G))
            T_all.append(T.assign(day=i))
    T = pd.concat(T_all, ignore_index=True)
    print(f"  แถวรายจุดที่มีฝนถึงเกณฑ์: {len(T):,} ({T.groupby(['cell_id', 'station']).ngroups:,} คู่)")
    print("  เวลาเฉลี่ยต่อ scan (ms):", pd.concat(timings).mean().round(1).to_dict())

    # ================= Phase 4–5: LUT + ปรับเทียบ =================
    print("[Phase 4–5] rain rate LUT + ปรับเทียบส่วนทำนายเวลา (วัน train) ...")
    lut = fit_lut(pairs)
    lut.save(out / "rain_lut.npz")
    train, test = T[T.day < n_train], T[T.day >= n_train]
    try:
        params = fit_event_models(train)
        (out / "timing_params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
        calibrated = {ev: from_dict(params[ev]) for ev in params}
    except ValueError as e:
        print(f"  ปรับเทียบไม่ได้ ({e}) → ใช้ค่าที่ติดมากับโค้ด")
        calibrated = None

    # ================= Phase 6: ประเมินบน test =================
    print("[Phase 6] ประเมินบนวัน test (MAE นาที, % ภายใน ±15 นาที) ...")
    builtin = load_event_models()
    for ev in ("onset", "peak", "stop"):
        I = event_rows(test, ev)
        if not len(I):
            continue
        obs, lead = I[ev].to_numpy(float), (I[ev] - I.t).to_numpy(float)
        cands = {"ค่าที่ติดมากับโค้ด": builtin[ev], "นาฬิกาตายตัว": FixedClock.fit(event_rows(train, ev), ev)}
        if calibrated:
            cands["ปรับเทียบใหม่"] = calibrated[ev]
        for name, m in cands.items():
            s = scores(obs, m.quantiles(I), lead)["all"]
            print(f"  {ev:5s} {name:18s} n {s['n']:5d}  MAE {s['MAE']:5.1f}  ±15 {s['w15'] * 100:3.0f}%  "
                  f"cov50/80 {s['cov50'] * 100:3.0f}/{s['cov80'] * 100:3.0f}")

    # ================= Phase 7: real-time =================
    print("[Phase 7] รัน pipeline บนวัน test วันแรก แล้วเขียน output ...")
    if calibrated:
        cfg["timing"]["params_file"] = str(out / "timing_params.json")
    pipe = ConvRainPipeline(cfg, rain_lut=lut)
    test_day = days[n_train]
    n_err, last_doc = 0, None
    for scan, _, _ in test_day.scans():
        r = pipe.step(scan)
        write_scan_outputs(out / "realtime", r.document, r.footprints, r.timeseries, points=r.points, outlook=r.outlook, rainmap=r.rainmap)
        n_err += len(r.errors)
        if any(o["onset"]["p_within_30min"] is not None for o in r.document["objects"]):
            last_doc = r.document
    print(f"  เขียน output ไว้ที่ {out / 'realtime'}  validation errors = {n_err}")

    # ส่งออกข้อมูลจำลองของวันนี้เป็นไฟล์ แบบเดียวกับข้อมูลจริงหลัง Phase 0 (ลอง replay.py / plot_tracks.py ได้ทันที)
    (out / "scans").mkdir(exist_ok=True)
    (out / "radar").mkdir(exist_ok=True)
    for scan, frames, _ in test_day.scans():
        save_scan_netcdf(scan, out / "scans" / f"scan_{scan.time:%Y%m%dT%H%MZ}.nc")
        for rt, dbz in frames:
            xr.Dataset({"dbz": (("y", "x"), dbz)}).to_netcdf(out / "radar" / f"radar_{rt:%Y%m%dT%H%MZ}.nc", engine="h5netcdf")
    (out / "domain.yaml").write_text(yaml.safe_dump({"domain": test_day.domain()}))
    print(f"  ส่งออกข้อมูลจำลองของวันนี้ไว้ที่ {out / 'scans'} และ {out / 'radar'}")
    if last_doc:
        ex = next(o for o in last_doc["objects"] if o["onset"]["p_within_30min"] is not None)
        print("  ตัวอย่าง object:")
        print(json.dumps(ex, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""
Demo ตั้งแต่ต้นจนจบบนข้อมูลจำลอง (ไม่ต้องมีข้อมูลจริง)

ลำดับเหมือนแผนพัฒนาในเอกสาร หัวข้อ 13–17
    Phase 2–3  replay pipeline ทีละ scan + เก็บสถิติเรดาร์ใน object → timeseries + tracks
    Phase 3    สร้าง label (t_onset, t_peak, t_stop, censor)
    Phase 4–5  train onset/peak/stop hazard model + rain rate LUT + เลือก threshold
    Phase 6    ประเมินบน test เทียบ baseline + bootstrap CI + แยกกลุ่ม
    Phase 7    รัน pipeline แบบ real-time พร้อมโมเดล แล้วเขียน output JSON/GeoJSON/Parquet

รัน:  python scripts/demo_synthetic.py --out outputs/demo
หมายเหตุ: ตัวเลขความแม่นยำจากข้อมูลจำลองไม่ได้บอกความแม่นยำกับข้อมูลจริง
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convrain.config import load_config  # noqa: E402
from convrain.ingest import save_scan_netcdf  # noqa: E402
from convrain.evaluate import bootstrap_ci, evaluate_all, object_scores, reliability_table, stratified  # noqa: E402
from convrain.experiment import fit_lut, predict_table, replay_scans, train_models  # noqa: E402
from convrain.labels import attach_labels, build_track_labels  # noqa: E402
from convrain.models import ClimatologyBaseline, ExtrapolationBaseline, MB06InterestFields  # noqa: E402
from convrain.output import write_scan_outputs  # noqa: E402
from convrain.pipeline import ConvRainPipeline  # noqa: E402
from convrain.synthetic import SyntheticDay  # noqa: E402


def make_days(n: int, seed0: int, start: datetime):
    """วันจำลอง: เริ่ม 05:00 UTC (12:00 น. เวลาไทย) ช่วงที่ฝน convective บ่ายเกิดบ่อย"""
    return [SyntheticDay(seed=seed0 + i, start=start + timedelta(days=i)) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--days", type=int, default=24, help="จำนวนวันจำลองทั้งหมด (แบ่ง 60/20/20)")
    ap.add_argument("--estimator", default="lightgbm", choices=["lightgbm", "logistic"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    start = datetime(2026, 5, 1, 5, 0, tzinfo=timezone.utc)
    days = make_days(args.days, 100, start)
    cfg = load_config(overrides={"domain": days[0].domain()})

    # ================= Phase 2–3: replay + label =================
    print(f"[Phase 2–3] replay {len(days)} วันจำลอง ...")
    ts_all, tr_all, pairs, timings = [], [], [], []
    n_train_days = int(round(0.6 * len(days)))
    for i, day in enumerate(days):
        pipe = ConvRainPipeline(cfg, collect_lut_pairs=i < n_train_days)  # เก็บคู่ LUT เฉพาะวัน train
        r = replay_scans(pipe, day.scans())
        ts_all.append(r["timeseries"])
        tr_all.append(r["tracks"])
        pairs.extend(pipe.lut_pairs)
        timings.append(r["timings"])
        if r["errors"]:
            print("  validation errors:", r["errors"][:3])
    ts = pd.concat(ts_all, ignore_index=True)
    tracks = pd.concat(tr_all, ignore_index=True)
    tim = pd.concat(timings, ignore_index=True)
    print(f"  object-scan rows: {len(ts):,}  tracks: {ts.object_id.nunique():,}")
    print("  เวลาเฉลี่ยต่อ scan (ms):", tim.mean().round(1).to_dict())

    labels = build_track_labels(ts, tracks, cfg)
    df = attach_labels(ts, labels)
    n_on = labels["t_onset"].notna().sum()
    print(f"  track ที่มี onset: {n_on} (ตั้งแต่เกิด {labels['onset_at_birth'].sum()}), "
          f"censored: {labels['censor_time'].notna().sum()}")
    ts.to_parquet(out / "timeseries_all.parquet", index=False)
    labels.to_parquet(out / "labels.parquet", index=False)

    # ================= แบ่งชุดตามวัน =================
    day_of = pd.to_datetime(df["time"], utc=True).dt.strftime("%Y-%m-%d")
    all_days = sorted(day_of.unique())
    n_va = int(round(0.2 * len(all_days)))
    tr_days, va_days = all_days[:n_train_days], all_days[n_train_days:n_train_days + n_va]
    train, val = df[day_of.isin(tr_days)], df[day_of.isin(va_days)]
    test = df[~day_of.isin(tr_days + va_days)]
    print(f"  แบ่งวัน train/val/test = {len(tr_days)}/{len(va_days)}/{len(all_days) - len(tr_days) - len(va_days)}")

    # ================= Phase 4–5: train =================
    print(f"[Phase 4–5] train hazard models ({args.estimator}) + LUT ...")
    lut = fit_lut(pairs)
    models = train_models(train, val, cfg, estimator=args.estimator)
    thr = models["warn_threshold"]
    print(f"  warn threshold (CSI สูงสุดบน val) = {thr:.2f}, CSI val = {models['val_csi']:.3f}")
    print("  feature สำคัญ 8 อันดับแรก:", list(models["onset"].feature_importance().head(8).index))
    for name in ("onset", "peak", "stop"):
        if models[name] is not None:
            models[name].save(out / f"model_{name}.joblib")
    lut.save(out / "rain_lut.npz")

    # ================= Phase 6: evaluate บน test =================
    print("[Phase 6] ประเมินบน test เทียบ baseline ...")
    mb06 = MB06InterestFields()
    extrap = ExtrapolationBaseline().fit(train)
    clim = ClimatologyBaseline().fit(train)
    candidates = {"hazard_model": models["onset"], "B2_MB06": mb06, "B2_extrapolation": extrap,
                  "B3_climatology": clim}
    rows = []
    preds = {}
    for name, m in candidates.items():
        p = predict_table(m, test, cfg)
        preds[name] = p
        t = thr if name == "hazard_model" else 0.5
        r = evaluate_all(p, threshold=t)
        r["model"] = name
        rows.append(r)
    res = pd.DataFrame(rows).set_index("model")
    cols = ["POD@30", "FAR@30", "CSI@30", "onset_MAE_min@30", "window_coverage@30",
            "lead_median_min@30", "Brier@30", "CSI@60"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res[cols].round(3))
    res.to_csv(out / "test_metrics.csv")

    pm = preds["hazard_model"]
    csi, lo, hi = bootstrap_ci(pm, lambda d: object_scores(d, "p30", 30, thr)["CSI"], n=200)
    print(f"  CSI@30 hazard model = {csi:.3f}  95% CI [{lo:.3f}, {hi:.3f}] (bootstrap ตามวัน)")
    print("  reliability (p30):")
    print(reliability_table(pm, "p30", 30).round(3).to_string(index=False))
    # ตัวอย่างการแยกผลตามกลุ่ม: onset แบบ warm rain (BT แกนเย็น ณ onset > 253 K) vs cold rain
    first_on = (pm.assign(_t=pd.to_datetime(pm["time"], utc=True)).sort_values("_t")
                .groupby("object_id")["min_cold_bt_track"].last())
    pm["onset_type"] = pm["object_id"].map(lambda o: "warm" if first_on.get(o, 0) > 253 else "cold")
    print("  แยกตามชนิด onset (warm rain / cold rain):")
    print(stratified(pm, "onset_type", thr)[["onset_type", "POD@30", "FAR@30", "CSI@30"]]
          .round(3).to_string(index=False))

    # ================= Phase 7: real-time พร้อมโมเดล =================
    print("[Phase 7] รัน pipeline พร้อมโมเดลบนวัน test วันแรก แล้วเขียน output ...")
    cfg_rt = load_config(overrides={"domain": days[0].domain(), "onset": {"warn_threshold": thr},
                                    "output": {"model_version": models["onset"].version}})
    pipe = ConvRainPipeline(cfg_rt, onset_model=models["onset"], peak_model=models["peak"],
                            stop_model=models["stop"], rain_lut=lut)
    test_day = days[len(tr_days) + len(va_days)]
    n_err = 0
    last_doc = None
    for scan, _, _ in test_day.scans():
        r = pipe.step(scan)
        write_scan_outputs(out / "realtime", r.document, r.footprints, r.timeseries)
        n_err += len(r.errors)
        if any(o["onset"]["p_within_30min"] is not None for o in r.document["objects"]):
            last_doc = r.document
    print(f"  เขียน output ไว้ที่ {out / 'realtime'}  validation errors = {n_err}")

    # ส่งออกข้อมูลจำลองของวันนี้เป็นไฟล์ แบบเดียวกับข้อมูลจริงหลัง Phase 0
    # ใช้ลองสคริปต์ replay.py / build_labels.py / plot_tracks.py ได้ทันที
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

"""
Phase 6: ประเมินบนชุด test เทียบ baseline (หัวข้อ 10.4 และ 17)

Input : labeled.parquet (ชุดที่มีทั้ง train และ test), โฟลเดอร์โมเดลจาก train.py
        ชุด test = วันหลัง --test-start, baseline ที่ต้อง fit ใช้วันก่อน train-end ใน meta.json
Output: metrics.csv, reliability.csv, stratified_<group>.csv, bootstrap.json

รัน:
    python scripts/evaluate.py --labeled runs/all/labeled.parquet --models-dir models/v1 \
        --test-start 2025-09-01 --out reports/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _common
from convrain.config import load_config
from convrain.evaluate import bootstrap_ci, evaluate_all, object_scores, reliability_table, stratified
from convrain.experiment import predict_table
from convrain.models import ClimatologyBaseline, ExtrapolationBaseline, MB06InterestFields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, nargs="+")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--test-start", required=True, help="วันแรกของชุด test (YYYY-MM-DD)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config(args.config)
    onset, _, _, _, meta = _common.load_models(args.models_dir)
    thr = float(meta.get("warn_threshold", cfg["onset"]["warn_threshold"]))
    df = pd.concat([pd.read_parquet(p) for p in args.labeled], ignore_index=True)
    day = pd.to_datetime(df["time"], utc=True).dt.strftime("%Y-%m-%d")
    train = df[day <= meta.get("train_end", "0000-00-00")]
    test = df[day >= args.test_start]
    print(f"test {len(test):,} แถว, {test.object_id.nunique():,} track, threshold = {thr:.2f}")

    candidates = {"hazard_model": (onset, thr), "B2_MB06": (MB06InterestFields(), 0.5)}
    if len(train):
        candidates["B2_extrapolation"] = (ExtrapolationBaseline().fit(train), 0.5)
        candidates["B3_climatology"] = (ClimatologyBaseline().fit(train), 0.5)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, preds = [], {}
    for name, (m, t) in candidates.items():
        p = predict_table(m, test, cfg)
        preds[name] = p
        r = evaluate_all(p, t)
        r["model"] = name
        rows.append(r)
    res = pd.DataFrame(rows).set_index("model")
    res.to_csv(out / "metrics.csv")
    cols = ["POD@30", "FAR@30", "CSI@30", "onset_MAE_min@30", "window_coverage@30", "lead_median_min@30",
            "Brier@30", "POD@60", "FAR@60", "CSI@60"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res[cols].round(3))

    # bootstrap CI ของ CSI แบบสุ่มตามวัน (Gate ของ Phase 6: CI ไม่ทับกับ baseline)
    boot = {}
    for name, (m, t) in candidates.items():
        boot[name] = bootstrap_ci(preds[name], lambda d, t=t: object_scores(d, "p30", 30, t)["CSI"], n=args.n_boot)
        print(f"CSI@30 {name:18s} {boot[name][0]:.3f}  95% CI [{boot[name][1]:.3f}, {boot[name][2]:.3f}]")
    (out / "bootstrap.json").write_text(json.dumps(boot, indent=2))

    pm = preds["hazard_model"]
    reliability_table(pm, "p30", 30).to_csv(out / "reliability.csv", index=False)
    # แยกกลุ่ม: กลางวัน/กลางคืน, บก/ทะเล (หัวข้อ 10.4)
    pm["surface"] = np.where(pm["land_frac"] >= 0.5, "land", "sea")
    for g in ("is_day", "surface"):
        if pm[g].notna().any():
            stratified(pm, g, thr).to_csv(out / f"stratified_{g}.csv", index=False)
    print(f"บันทึกผลไว้ที่ {out}")


if __name__ == "__main__":
    main()

"""
Phase 4–5: train โมเดล onset / peak / stop (หัวข้อ 8 และ 16)

Input : labeled.parquet จาก build_labels.py (รวมหลายไฟล์ได้)
        แบ่งชุดตามวันที่ด้วย --train-end และ --val-end (ชุด test = หลัง val-end ห้ามแตะจนถึง Phase 6)
Output (--out)
        model_onset.joblib, model_peak.joblib, model_stop.joblib (ถ้ามี event พอ)
        meta.json            threshold คำเตือน, ช่วงวันที่, จำนวนแถว
        importance.csv       feature importance ของโมเดล onset
        (คัดลอก rain_lut.npz จาก build_lut.py มาไว้ในโฟลเดอร์เดียวกัน)

รัน:
    python scripts/train.py --labeled runs/2024/labeled.parquet runs/2025/labeled.parquet \
        --train-end 2025-06-30 --val-end 2025-08-31 --out models/v1 --estimator lightgbm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _common  # noqa: F401
from convrain.config import load_config
from convrain.evaluate import evaluate_all
from convrain.experiment import predict_table, train_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, nargs="+")
    ap.add_argument("--train-end", required=True, help="วันสุดท้ายของชุด train (YYYY-MM-DD)")
    ap.add_argument("--val-end", required=True, help="วันสุดท้ายของชุด validation (YYYY-MM-DD)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--estimator", default="lightgbm", choices=["lightgbm", "logistic"])
    ap.add_argument("--balance", action="store_true", help="ใช้ class weight (แล้ว calibrate ด้วย validation)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.concat([pd.read_parquet(p) for p in args.labeled], ignore_index=True)
    day = pd.to_datetime(df["time"], utc=True).dt.strftime("%Y-%m-%d")
    train = df[day <= args.train_end]
    if args.train_end == args.val_end:
        val = train.copy()
    else:
        val = df[(day > args.train_end) & (day <= args.val_end)]
    print(f"train {len(train):,} แถว ({train.object_id.nunique():,} track), "
          f"val {len(val):,} แถว ({val.object_id.nunique():,} track)")

    models = train_models(train, val, cfg, estimator=args.estimator, balance=args.balance)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("onset", "peak", "stop"):
        if models[name] is not None:
            models[name].save(out / f"model_{name}.joblib")
            print(f"บันทึก model_{name}.joblib")
        else:
            print(f"ไม่ได้ train {name}: event น้อยเกินไป")

    models["onset"].feature_importance().to_csv(out / "importance.csv", header=["importance"])
    val_metrics = evaluate_all(predict_table(models["onset"], val, cfg), models["warn_threshold"])
    meta = {
        "estimator": args.estimator, "balance": args.balance,
        "train_end": args.train_end, "val_end": args.val_end,
        "warn_threshold": models["warn_threshold"], "val_csi": models["val_csi"],
        "n_train_rows": int(len(train)), "n_val_rows": int(len(val)),
        "features": models["onset"].features, "version": models["onset"].version,
        "val_metrics": {k: (None if pd.isna(v) else float(v)) for k, v in val_metrics.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"warn threshold = {models['warn_threshold']:.2f}  (CSI@30 บน val = {models['val_csi']:.3f})")
    print("feature สำคัญ 10 อันดับแรก:")
    print(models["onset"].feature_importance().head(10).to_string())


if __name__ == "__main__":
    main()

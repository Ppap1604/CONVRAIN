"""
Phase 5: สร้าง rain rate lookup table จากคู่ข้อมูล AHI–เรดาร์ (หัวข้อ 9.1)

Input : lut_pairs.npz จาก replay.py --collect-lut (ควรใช้เฉพาะวันในชุด train)
Output: rain_lut.npz

รัน:
    python scripts/build_lut.py --pairs runs/train/lut_pairs.npz --out models/rain_lut.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _common  # noqa: F401
from convrain.intensity import RainRateLUT, auto_estimator_mmh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, nargs="+", help="ไฟล์ lut_pairs.npz (หลายไฟล์ได้)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--quantile", type=float, default=0.5)
    args = ap.parse_args()

    zs = [np.load(p) for p in args.pairs]
    bt = np.concatenate([z["bt"] for z in zs])
    btd = np.concatenate([z["btd"] for z in zs])
    rr = np.concatenate([z["rr"] for z in zs])
    lut = RainRateLUT(min_count=args.min_count).fit(bt, btd, rr, quantile=args.quantile)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lut.save(args.out)

    # สรุปเทียบกับ prior (auto-estimator) ต่อช่วง BT
    print(f"คู่ข้อมูล {bt.size:,} pixel -> {args.out}")
    print(" BT(K)   LUT(mm/h)  prior(mm/h)   n")
    for lo in range(190, 290, 10):
        sel = (bt >= lo) & (bt < lo + 10)
        c = lo + 5
        print(f" {lo}-{lo + 10}  {lut.predict(np.array([c]))[0]:9.2f}  {auto_estimator_mmh(c):10.2f}  {sel.sum():6d}")


if __name__ == "__main__":
    main()

"""
Phase 2: ภาพซ้อน track บนภาพ IR เพื่อตรวจ tracking ด้วยตา (หัวข้อ 15)

Input : --scans-dir (NetCDF), --run-dir (ผลที่มี geojson/ จาก replay.py หรือ run_realtime.py)
Output: PNG ต่อ scan และ animation GIF ใน <run-dir>/plots/

สีของขอบ object ตามสถานะ: developing=ฟ้า, raining=เขียว, peak=แดง, decaying=ส้ม
ป้ายกำกับ: 5 หลักท้ายของ object ID และ P(onset ≤ 30 นาที) ถ้ามี

รัน:
    python scripts/plot_tracks.py --scans-dir /data/scans --run-dir runs/case01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _common  # noqa: E402
from convrain.config import load_config  # noqa: E402
from convrain.ingest import load_scan_netcdf  # noqa: E402
from convrain.output import scan_stamp  # noqa: E402

COLORS = {"developing": "#2f7ed8", "raining": "#1a9850", "peak": "#d73027", "decaying": "#fc8d59"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--gif", action="store_true", help="รวมเป็น animation GIF")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(args.run_dir)
    plots = run / "plots"
    plots.mkdir(exist_ok=True)

    frames = []
    for t, path in _common.files_by_time(args.scans_dir, "scan_*.nc"):
        stamp = scan_stamp(t)
        pts = run / "geojson" / f"objects_{stamp}.geojson"
        fps = run / "geojson" / f"footprints_{stamp}.geojson"
        if not pts.exists():
            continue
        scan = load_scan_netcdf(path, cfg)
        objs = {f["properties"]["object_id"]: f for f in json.loads(pts.read_text())["features"]}
        polys = json.loads(fps.read_text())["features"] if fps.exists() else []

        fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
        # pcolormesh ใช้ lat/lon จริงของแต่ละ pixel (grid ของดาวเทียมไม่ใช่ตารางปกติ)
        im = ax.pcolormesh(scan.lon, scan.lat, scan.ir, cmap="gray_r", vmin=190, vmax=300, shading="auto")
        for poly in polys:
            oid = poly["properties"]["object_id"]
            status = objs.get(oid, {}).get("properties", {}).get("status", "developing")
            ring = poly["geometry"]["coordinates"][0]
            ax.plot([p[0] for p in ring], [p[1] for p in ring], color=COLORS.get(status, "k"), lw=1.2)
        for oid, f in objs.items():
            p = f["properties"]
            if p["status"] == "dissipated":
                continue
            lon, lat = f["geometry"]["coordinates"]
            p30 = p["onset"]["p_within_30min"]
            label = oid[-5:] + (f"\nP30={p30:.2f}" if p30 is not None else "")
            ax.text(lon, lat, label, fontsize=7, color=COLORS.get(p["status"], "k"),
                    ha="center", va="center", weight="bold")
        # ใช้ข้อความภาษาอังกฤษในภาพ เพราะฟอนต์ default ของ matplotlib ไม่มีอักษรไทย
        ax.set_title(f"{t:%Y-%m-%d %H:%M} UTC  (outlines after parallax correction)", fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, shrink=0.7, label="BT 10.4 µm (K)")
        png = plots / f"tracks_{stamp}.png"
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        frames.append(png)

    print(f"สร้างภาพ {len(frames)} ภาพใน {plots}")
    if args.gif and frames:
        import imageio.v2 as imageio

        imgs = [imageio.imread(p) for p in frames]
        imageio.mimsave(plots / "tracks.gif", imgs, duration=0.6, loop=0)
        print(f"GIF: {plots / 'tracks.gif'}")


if __name__ == "__main__":
    main()

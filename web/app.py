"""
FastAPI Server สำหรับ Convrain Web Platform
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web.data_loader import ConvrainDataLoader
from web.point_nowcast import analyze_point, compute_trajectories_and_impacts

app = FastAPI(
    title="Convrain Nowcast Platform API",
    description="API สำหรับแสดงผลการตรวจจับและพยากรณ์เวลาเริ่มตกของฝน convective และระบุพิกัด GPS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DataLoader จะถูก initialize ตอนเริ่ม server หรือใช้ default
loader = ConvrainDataLoader()

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Convrain Web Platform API is running. UI not found in static/."}


@app.get("/api/sources")
def get_source_info():
    """ดูสถานะแหล่งข้อมูลที่ระบบกำลังอ่านอยู่"""
    return loader.get_source_info()


@app.get("/api/datasets")
def get_datasets():
    """รายการชุดข้อมูลที่มีในระบบ (ข้อมูลจริง vs ข้อมูลจำลอง)"""
    current_id = "real" if "real_nowcast" in str(loader.base_dir) else "simulation"
    return {
        "current": current_id,
        "datasets": [
            {
                "id": "real",
                "name": "🛰️ ข้อมูลจริง Himawari-9 (28 ก.ย. 2026)",
                "path": "runs/real_nowcast",
                "is_real": True,
            },
            {
                "id": "simulation",
                "name": "🧪 ข้อมูลจำลอง 30 สแกน (Simulation)",
                "path": "runs/2025/realtime",
                "is_real": False,
            },
        ],
    }


@app.post("/api/datasets/{dataset_id}")
def switch_dataset(dataset_id: str):
    """สลับชุดข้อมูลที่แสดงผลบนเว็บ"""
    if not loader.set_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "status": "ok",
        "current": dataset_id,
        "scan_count": len(loader.list_scans()),
    }


@app.get("/api/scans")
def list_scans():
    """รายการ scan ทั้งหมดในระบบ"""
    scans = loader.list_scans()
    return {"total": len(scans), "scans": scans}


@app.get("/api/scans/latest")
def get_latest_scan():
    """ดึงข้อมูล scan ล่าสุด (JSON objects + Footprints GeoJSON)"""
    scans = loader.list_scans()
    if not scans:
        raise HTTPException(status_code=404, detail="ไม่พบข้อมูล scan ในระบบ")
    latest_stamp = scans[-1]["stamp"]
    return get_scan_by_stamp(latest_stamp)


@app.get("/api/scans/{stamp}")
def get_scan_by_stamp(stamp: str):
    """ดึงข้อมูล scan ตามรหัส timestamp เช่น 20260520T0720Z"""
    doc, footprints = loader.get_scan(stamp)
    if not doc:
        raise HTTPException(status_code=404, detail=f"ไม่พบข้อมูล scan: {stamp}")

    # รวบรวมข้อมูล summary
    objects = doc.get("objects", [])
    developing_count = sum(1 for o in objects if o.get("status") == "developing")
    raining_count = sum(1 for o in objects if o.get("status") in ["raining", "peak"])
    warning_count = sum(
        1 for o in objects 
        if (o.get("status") == "developing" and (o.get("onset", {}).get("p_within_30min") or 0) >= 0.5)
        or o.get("status") == "raining"
    )

    traj_and_impacts = compute_trajectories_and_impacts(doc, footprints)

    return {
        "stamp": stamp,
        "scan_time": doc.get("scan_time"),
        "model_version": doc.get("model_version"),
        "summary": {
            "total_objects": len(objects),
            "developing_count": developing_count,
            "raining_count": raining_count,
            "warning_count": warning_count,
            "impact_zones_count": len(traj_and_impacts["impact_zones"]["features"]),
        },
        "doc": doc,
        "footprints": footprints,
        "trajectories": traj_and_impacts["trajectories"],
        "impact_zones": traj_and_impacts["impact_zones"],
    }


@app.get("/api/nowcast/point")
def get_point_nowcast(
    lat: float,
    lon: float,
    stamp: Optional[str] = None,
):
    """
    วิเคราะห์และพยากรณ์การเกิดฝน ณ พิกัด GPS หรือพิกัดที่ระบุ
    คืนค่าสถานะ: ฝนตกอยู่แล้ว / เมฆกำลังพัฒนาตรงนี้ / เมฆกำลังเคลื่อนเข้ามา / ปลอดภัย
    พร้อมช่วงเวลา onset (ETA window), โอกาสเกิดฝน, และความแรง
    """
    if not isinstance(stamp, str):
        stamp = None
    doc, footprints = loader.get_scan(stamp)
    if not doc:
        raise HTTPException(status_code=404, detail="ไม่มีข้อมูล scan สำหรับวิเคราะห์")

    result = analyze_point(lat=lat, lon=lon, scan_doc=doc, footprints_fc=footprints)
    return result


@app.get("/api/objects/{track_id}/history")
def get_object_history(track_id: str):
    """ประวัติของ object ข้าม scan สำหรับพล็อตกราฟ"""
    history = loader.get_object_history(track_id)
    return {
        "track_id": track_id,
        "total_scans": len(history),
        "history": history,
    }


@app.get("/api/monitor")
def get_monitor():
    """ข้อมูลประสิทธิภาพและสุขภาพระบบ (latency, drift)"""
    return loader.get_monitor_stats()


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="Convrain Web Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default=None, help="โฟลเดอร์ output ของ convrain")
    args = parser.parse_args()

    global loader
    if args.data_dir:
        loader = ConvrainDataLoader(args.data_dir)

    print(f"Starting Convrain Web Platform on http://localhost:{args.port}")
    print(f"Data directory: {loader.base_dir.resolve()}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

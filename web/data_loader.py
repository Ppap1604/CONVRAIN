"""
ตัวจัดการโหลดและค้นหาข้อมูล output ของ Convrain
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

CANDIDATE_DIRS = [
    Path("nowcast"),
    Path("runs/2025/realtime"),
    Path("runs/test_real"),
    Path("examples"),
]


class ConvrainDataLoader:
    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir) if base_dir else self._find_best_dir()
        self._cache_json: dict[str, dict] = {}
        self._cache_geojson: dict[str, dict] = {}

    def _find_best_dir(self) -> Path:
        for p in CANDIDATE_DIRS:
            if p.exists() and (p / "json").exists() and any((p / "json").glob("*.json")):
                return p
        # ถ้าไม่มี ให้ลองหาไฟล์ sample ใน examples
        if Path("examples").exists():
            return Path("examples")
        return Path("nowcast")

    def get_source_info(self) -> dict[str, Any]:
        return {
            "current_dir": str(self.base_dir),
            "exists": self.base_dir.exists(),
            "scan_count": len(self.list_scans()),
        }

    def list_scans(self) -> list[dict[str, Any]]:
        """รายการ scans ทั้งหมด เรียงตามเวลาเก่าไปใหม่"""
        scans = []
        json_dir = self.base_dir / "json"
        
        # ค้นหาไฟล์ objects_*.json
        files = list(json_dir.glob("objects_*.json")) if json_dir.exists() else []
        if not files and self.base_dir == Path("examples"):
            files = list(self.base_dir.glob("objects_*.json"))

        for f in files:
            m = re.search(r"objects_(\d{8}T\d{4}Z)", f.name)
            stamp = m.group(1) if m else f.stem.replace("objects_", "")
            try:
                # แปลง stamp เป็น ISO format
                if "T" in stamp and stamp.endswith("Z"):
                    dt = datetime.strptime(stamp, "%Y%m%dT%H%MZ")
                    iso_time = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    iso_time = stamp
            except Exception:
                iso_time = stamp

            scans.append({
                "stamp": stamp,
                "scan_time": iso_time,
                "json_file": f.name,
                "has_footprint": self._has_footprint(stamp),
            })

        scans.sort(key=lambda x: x["stamp"])
        return scans

    def _has_footprint(self, stamp: str) -> bool:
        geo_dir = self.base_dir / "geojson"
        if not geo_dir.exists():
            geo_dir = self.base_dir
        return (geo_dir / f"footprints_{stamp}.geojson").exists() or (geo_dir / f"footprints_sample.geojson").exists()

    def get_scan(self, stamp: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """ดึงข้อมูล scan JSON และ Footprints GeoJSON ตาม stamp (None = ล่าสุด)"""
        scans = self.list_scans()
        if not scans:
            # fallback ถ้าไม่มีไฟล์เลย
            return None, None

        if stamp is None:
            stamp = scans[-1]["stamp"]

        # 1. โหลด JSON
        doc = self._cache_json.get(stamp)
        if doc is None:
            json_path = self.base_dir / "json" / f"objects_{stamp}.json"
            if not json_path.exists() and (self.base_dir / f"objects_{stamp}.json").exists():
                json_path = self.base_dir / f"objects_{stamp}.json"
            if not json_path.exists() and (self.base_dir / "objects_sample.json").exists():
                json_path = self.base_dir / "objects_sample.json"

            if json_path.exists():
                with open(json_path, encoding="utf-8") as f:
                    doc = json.load(f)
                    self._cache_json[stamp] = doc

        # 2. โหลด Footprint GeoJSON
        fc = self._cache_geojson.get(stamp)
        if fc is None:
            geo_path = self.base_dir / "geojson" / f"footprints_{stamp}.geojson"
            if not geo_path.exists() and (self.base_dir / f"footprints_{stamp}.geojson").exists():
                geo_path = self.base_dir / f"footprints_{stamp}.geojson"
            if not geo_path.exists() and (self.base_dir / "footprints_sample.geojson").exists():
                geo_path = self.base_dir / "footprints_sample.geojson"

            if geo_path.exists():
                with open(geo_path, encoding="utf-8") as f:
                    fc = json.load(f)
                    self._cache_geojson[stamp] = fc

        return doc, fc

    def get_object_history(self, object_id: str) -> list[dict[str, Any]]:
        """ดึงประวัติของ object ข้ามหลาย scan เพื่อสร้างกราฟ Time-Series"""
        history = []
        for scan in self.list_scans():
            doc, _ = self.get_scan(scan["stamp"])
            if not doc:
                continue
            for obj in doc.get("objects", []):
                if obj["object_id"] == object_id:
                    history.append({
                        "stamp": scan["stamp"],
                        "scan_time": doc.get("scan_time"),
                        "status": obj.get("status"),
                        "min_bt": obj.get("cloud_top", {}).get("min_bt_10p4"),
                        "cooling_rate": obj.get("cloud_top", {}).get("cooling_rate_10min_K"),
                        "rain_rate": obj.get("intensity", {}).get("rain_rate_mm_hr"),
                        "p30": obj.get("onset", {}).get("p_within_30min"),
                        "p60": obj.get("onset", {}).get("p_within_60min"),
                    })
        return history

    def get_monitor_stats(self) -> dict[str, Any]:
        """อ่านค่า monitor.csv ถ้ามี"""
        mon_path = self.base_dir / "monitor.csv"
        if not mon_path.exists():
            mon_path = Path("nowcast/monitor.csv")

        if mon_path.exists():
            rows = []
            try:
                with open(mon_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        rows.append(r)
                if rows:
                    latest = rows[-1]
                    return {
                        "available": True,
                        "latest": latest,
                        "total_records": len(rows),
                    }
            except Exception:
                pass

        # Fallback สถิติจำลอง
        return {
            "available": False,
            "latest": {
                "scan_time": "2026-05-20T07:20:00Z",
                "latency_s": "45.2",
                "total_ms": "68",
                "n_objects": "7",
                "n_warnings": "2",
            }
        }

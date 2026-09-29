"""
รูปแบบ output ของระบบ  (เอกสาร หัวข้อ 9.5)

Output หลักเป็นแบบ object-centric time series: ทุก scan ปล่อยรายการวัตถุเมฆที่กำลังติดตาม
    1. JSON ต่อ scan (โครงสร้างตามตัวอย่างในเอกสาร) + GeoJSON FeatureCollection สำหรับ GIS
    2. Footprint polygon ของแต่ละ object (GeoJSON ต่อ scan)
    3. Time series ต่อ object (Parquet) หนึ่งแถวต่อ object ต่อ scan
    4. Validation ของ output ก่อนปล่อย (JSON Schema + กฎข้าม field)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

STATUSES = ["developing", "raining", "peak", "decaying", "dissipated"]


def iso(t: datetime | None) -> str | None:
    """เวลาเป็นสตริง ISO 8601 แบบ UTC ลงท้ายด้วย Z"""
    if t is None or (isinstance(t, float) and np.isnan(t)) or pd.isna(t):
        return None
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x, nd: int = 2):
    """แปลงตัวเลขให้ JSON ได้ (NaN → null)"""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(xf) else round(xf, nd)


# =====================================================================
# [9.5] JSON Schema ของ output ต่อ scan
# =====================================================================
_NUM_OR_NULL = {"type": ["number", "null"]}
_TIME_OR_NULL = {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"}

OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["scan_time", "model_version", "objects"],
    "properties": {
        "scan_time": {"type": "string"},
        "model_version": {"type": "string"},
        "objects": {"type": "array", "items": {
            "type": "object",
            "required": ["object_id", "age_min", "status", "geometry", "cloud_top", "onset", "intensity", "motion"],
            "properties": {
                "object_id": {"type": "string"},
                "parent_id": {"type": ["string", "null"]},
                "age_min": {"type": "number", "minimum": 0},
                "status": {"enum": STATUSES},
                "geometry": {"type": "object", "required": ["type", "coordinates"],
                             "properties": {"type": {"const": "Point"},
                                            "coordinates": {"type": "array", "items": {"type": "number"},
                                                            "minItems": 2, "maxItems": 2}}},
                "geometry_parallax_corrected": {"type": "boolean"},
                "footprint_polygon": {"type": ["string", "null"]},
                "cloud_top": {"type": "object", "properties": {
                    "min_bt_10p4": _NUM_OR_NULL, "cooling_rate_10min_K": _NUM_OR_NULL,
                    "cloud_top_height_km": _NUM_OR_NULL}},
                "onset": {"type": "object", "properties": {
                    "p_within_30min": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "p_within_60min": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "eta_median": _TIME_OR_NULL,
                    "eta_window": {"type": ["array", "null"], "items": _TIME_OR_NULL}}},
                "intensity": {"type": "object", "properties": {
                    "rain_rate_mm_hr": {"type": ["number", "null"], "minimum": 0},
                    "peak_eta": _TIME_OR_NULL, "stop_eta": _TIME_OR_NULL}},
                "motion": {"type": "object", "properties": {
                    "speed_ms": _NUM_OR_NULL, "direction_deg": _NUM_OR_NULL}},
            },
        }},
    },
}


def validate_output(doc: dict, cfg: dict | None = None) -> list[str]:
    """
    ตรวจ output ก่อนปล่อยทุก scan คืน list ข้อผิดพลาด (ว่าง = ผ่าน)
        - โครงสร้างตาม JSON Schema
        - 0 ≤ p30 ≤ p60 ≤ 1
        - eta_window ครอบ eta_median
        - พิกัดอยู่ในโดเมน (เผื่อขอบ 1°)
        - status สอดคล้องกับค่า null (developing ต้องไม่มี rain_rate)
    """
    import jsonschema

    errors: list[str] = []
    v = jsonschema.Draft202012Validator(OUTPUT_SCHEMA)
    errors += [f"schema: {e.message} @ {list(e.path)}" for e in v.iter_errors(doc)]
    if errors:
        return errors

    dom = (cfg or {}).get("domain")
    for o in doc["objects"]:
        oid = o["object_id"]
        on = o["onset"]
        p30, p60 = on.get("p_within_30min"), on.get("p_within_60min")
        if p30 is not None and p60 is not None and p30 > p60 + 1e-6:
            errors.append(f"{oid}: p30 > p60")
        med, win = on.get("eta_median"), on.get("eta_window")
        if med and win and all(win) and not (win[0] <= med <= win[1]):
            errors.append(f"{oid}: eta_window ไม่ครอบ eta_median")
        lon, lat = o["geometry"]["coordinates"]
        if dom and not (dom["lat_min"] - 1 <= lat <= dom["lat_max"] + 1 and dom["lon_min"] - 1 <= lon <= dom["lon_max"] + 1):
            errors.append(f"{oid}: พิกัดอยู่นอกโดเมน ({lat:.2f}, {lon:.2f})")
        if o["status"] == "developing" and o["intensity"].get("rain_rate_mm_hr") is not None:
            errors.append(f"{oid}: status developing แต่มี rain_rate")
    return errors


# =====================================================================
# [9.5] สร้าง record ของ object และเอกสารต่อ scan
# =====================================================================
def object_record(track, feats: dict, px: dict, onset: dict | None, intensity: dict | None,
                  footprint_ref: str | None, parallax_on: bool) -> dict:
    """
    track     : Track (ขั้น 3)
    feats     : feature ของ scan นี้ (ขั้น 4)
    px        : ผล parallax (lat, lon, cth_km)
    onset     : {"p30","p60","eta_median","eta_lo","eta_hi"} หรือ None ถ้าไม่ได้ประเมิน
    intensity : {"rain_rate","peak_eta","stop_eta"} หรือ None
    """
    on = onset or {}
    it = intensity or {}
    lo, hi = on.get("eta_lo"), on.get("eta_hi")
    return {
        "object_id": track.track_id,
        "parent_id": track.parent_id,
        "age_min": round(track.age_min, 1),
        "status": track.status if track.active else "dissipated",
        "geometry": {"type": "Point", "coordinates": [round(px["lon"], 4), round(px["lat"], 4)]},
        "geometry_parallax_corrected": bool(parallax_on),
        "footprint_polygon": footprint_ref,
        "cloud_top": {
            "min_bt_10p4": _num(feats.get("min_bt"), 1),
            "cooling_rate_10min_K": _num(feats.get("cooling_rate_kf"), 2),
            "cloud_top_height_km": _num(px.get("cth_km"), 1),
        },
        "onset": {
            "p_within_30min": _num(on.get("p30"), 3),
            "p_within_60min": _num(on.get("p60"), 3),
            "eta_median": iso(on.get("eta_median")),
            "eta_window": [iso(lo), iso(hi)] if lo is not None and hi is not None else None,
        },
        "intensity": {
            "rain_rate_mm_hr": _num(it.get("rain_rate"), 1),
            "peak_eta": iso(it.get("peak_eta")),
            "stop_eta": iso(it.get("stop_eta")),
        },
        "motion": {"speed_ms": _num(feats.get("speed_ms"), 1), "direction_deg": _num(feats.get("direction_deg"), 0)},
    }


def scan_document(scan_time: datetime, objects: list[dict], model_version: str) -> dict:
    return {"scan_time": iso(scan_time), "model_version": model_version, "objects": objects}


def to_feature_collection(doc: dict) -> dict:
    """แปลงเอกสารต่อ scan เป็น GeoJSON FeatureCollection มาตรฐาน (เปิดใน QGIS ได้ทันที)"""
    feats = []
    for o in doc["objects"]:
        props = {k: v for k, v in o.items() if k != "geometry"}
        props["scan_time"] = doc["scan_time"]
        feats.append({"type": "Feature", "geometry": o["geometry"], "properties": props})
    return {"type": "FeatureCollection", "features": feats}


# =====================================================================
# [9.5] footprint polygon ของ object (หลังแก้ parallax)
# =====================================================================
def footprint_polygon(rows: np.ndarray, cols: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                      drow: float = 0.0, dcol: float = 0.0, max_points: int = 60) -> list[list[float]] | None:
    """ขอบของ object เป็น polygon [lon, lat] (ใช้ contour ของ mask ใน bounding box)"""
    from skimage.measure import approximate_polygon, find_contours

    if rows.size == 0:
        return None
    r0, c0 = rows.min() - 1, cols.min() - 1
    mask = np.zeros((rows.max() - r0 + 2, cols.max() - c0 + 2), dtype=float)
    mask[rows - r0, cols - c0] = 1.0
    contours = find_contours(mask, 0.5)
    if not contours:
        return None
    ring = max(contours, key=len)
    tol = 0.5
    while len(ring) > max_points and tol < 10:
        ring = approximate_polygon(ring, tolerance=tol)
        tol *= 2
    rr = ring[:, 0] + r0 + drow
    cc = ring[:, 1] + c0 + dcol
    la = ndimage.map_coordinates(lat, [rr, cc], order=1, mode="nearest")
    lo = ndimage.map_coordinates(lon, [rr, cc], order=1, mode="nearest")
    coords = [[round(float(x), 4), round(float(y), 4)] for x, y in zip(lo, la)]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


# =====================================================================
# เขียนไฟล์
# =====================================================================
def scan_stamp(t: datetime) -> str:
    """ชื่อไฟล์ตามเวลา scan เช่น 20260925T0720Z"""
    ts = pd.Timestamp(t)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y%m%dT%H%MZ")


def write_scan_outputs(out_dir: str | Path, doc: dict, footprints: dict[str, list] | None,
                       timeseries: pd.DataFrame | None, write_json: bool = True) -> dict[str, Path]:
    """
    โครงสร้างโฟลเดอร์
        out_dir/json/objects_<stamp>.json            เอกสารต่อ scan (หัวข้อ 9.5)
        out_dir/geojson/objects_<stamp>.geojson       FeatureCollection ของจุด
        out_dir/geojson/footprints_<stamp>.geojson    polygon ของแต่ละ object
        out_dir/timeseries/date=YYYY-MM-DD/scan_<HHMM>.parquet
    """
    out = Path(out_dir)
    t = pd.Timestamp(doc["scan_time"])
    stamp = scan_stamp(t)
    paths: dict[str, Path] = {}

    if write_json:  # ตอนสร้างชุด train ปิดได้ เขียนเฉพาะ Parquet ให้เร็วขึ้น
        paths.update(_write_json_outputs(out, doc, footprints, stamp))

    if timeseries is not None and len(timeseries):
        d = out / "timeseries" / f"date={t:%Y-%m-%d}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"scan_{t:%H%M}.parquet"
        timeseries.to_parquet(p, index=False)
        paths["parquet"] = p
    return paths


def _write_json_outputs(out: Path, doc: dict, footprints, stamp: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    (out / "json").mkdir(parents=True, exist_ok=True)
    p = out / "json" / f"objects_{stamp}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["json"] = p

    (out / "geojson").mkdir(parents=True, exist_ok=True)
    p = out / "geojson" / f"objects_{stamp}.geojson"
    p.write_text(json.dumps(to_feature_collection(doc), ensure_ascii=False), encoding="utf-8")
    paths["geojson"] = p

    features = []
    if footprints:
        features = [
            {"type": "Feature", "id": oid, "properties": {"object_id": oid, "scan_time": doc["scan_time"]},
             "geometry": {"type": "Polygon", "coordinates": [ring]}}
            for oid, ring in footprints.items() if ring
        ]
    fc = {"type": "FeatureCollection", "features": features}
    p = out / "geojson" / f"footprints_{stamp}.geojson"
    p.write_text(json.dumps(fc), encoding="utf-8")
    paths["footprints"] = p

    return paths


def read_timeseries(out_dir: str | Path) -> pd.DataFrame:
    """อ่านตาราง object-scan ทั้งหมดที่ pipeline เขียนไว้ (ใช้ตอนสร้าง label และ train)"""
    root = Path(out_dir) / "timeseries"
    files = sorted(root.glob("date=*/scan_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)

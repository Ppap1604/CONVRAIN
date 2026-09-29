"""
การวิเคราะห์การเกิดฝน ณ จุดพิกัด (Point-based Rain Onset Nowcast)
รองรับทั้ง GPS จากอุปกรณ์ หรือพิกัดที่ผู้ใช้ป้อน/คลิกบนแผนที่
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Any

from shapely.geometry import Point, shape, mapping
from shapely.affinity import translate


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """คำนวณระยะทางบนผิวโลก (กม.) ระหว่าง 2 พิกัด"""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """ทิศทางเข็มทิศ (0-360 องศา) จากจุด 1 ไปยังจุด 2"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return round(deg, 1)


def compass_direction_thai(deg: float) -> str:
    """แปลงองศาเข็มทิศเป็นชื่อทิศภาษาไทย"""
    directions = [
        (22.5, "เหนือ"),
        (67.5, "ตะวันออกเฉียงเหนือ"),
        (112.5, "ตะวันออก"),
        (157.5, "ตะวันออกเฉียงใต้"),
        (202.5, "ใต้"),
        (247.5, "ตะวันตกเฉียงใต้"),
        (292.5, "ตะวันตก"),
        (337.5, "ตะวันตกเฉียงเหนือ"),
        (360.0, "เหนือ"),
    ]
    deg = deg % 360.0
    for threshold, name in directions:
        if deg <= threshold:
            return name
    return "เหนือ"


def parse_iso(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None


def format_thai_time(dt: datetime | None) -> str:
    """แปลงเวลาเป็นเวลาไทย (UTC+7) รูปแบบ HH:MM น."""
    if dt is None:
        return "-"
    # แปลงเป็น UTC+7 ถ้ายังเป็น UTC
    if dt.tzinfo == timezone.utc or (dt.tzinfo is not None and dt.utcoffset() == timedelta(0)):
        dt_th = dt.astimezone(timezone(timedelta(hours=7)))
    elif dt.tzinfo is None:
        dt_th = dt + timedelta(hours=7)
    else:
        dt_th = dt.astimezone(timezone(timedelta(hours=7)))
    return dt_th.strftime("%H:%M น.")


def analyze_point(lat: float, lon: float, scan_doc: dict[str, Any], footprints_fc: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    วิเคราะห์การตกของฝน ณ พิกัด (lat, lon)
    1. ตรวจสอบว่าพิกัดอยู่ข้างใน Footprint ของเมฆปัจจุบันหรือไม่
    2. ถ้าไม่ได้อยู่ข้างใน ตรวจสอบว่ามีกลุ่มเมฆใดกำลังเคลื่อนที่ (Advection) เข้ามาใน 60 นาทีหรือไม่
    3. คำนวณช่วงเวลา onset, ความน่าจะเป็น, และความแรงฝน
    """
    user_point = Point(lon, lat)
    scan_time_str = scan_doc.get("scan_time")
    scan_time = parse_iso(scan_time_str) or datetime.now(timezone.utc)
    objects = scan_doc.get("objects", [])

    # แผนผัง object_id -> dict
    obj_by_id = {o["object_id"]: o for o in objects}

    # แปลง footprint features เป็น shapely geometry
    footprint_geoms: dict[str, Any] = {}
    if footprints_fc and "features" in footprints_fc:
        for feat in footprints_fc["features"]:
            oid = feat.get("id") or feat.get("properties", {}).get("object_id")
            if oid and "geometry" in feat and feat["geometry"]:
                try:
                    footprint_geoms[oid] = shape(feat["geometry"])
                except Exception:
                    pass

    # =========================================================================
    # กรณีที่ 1: พิกัดผู้ใช้อยู่ภายใน Footprint ปัจจุบัน (Overhead)
    # =========================================================================
    for oid, geom in footprint_geoms.items():
        if geom.contains(user_point):
            obj = obj_by_id.get(oid)
            if not obj:
                continue

            status = obj.get("status", "unknown")
            onset = obj.get("onset", {})
            intensity = obj.get("intensity", {})
            cloud_top = obj.get("cloud_top", {})
            rain_rate = intensity.get("rain_rate_mm_hr")

            if status in ["raining", "peak"]:
                peak_time = parse_iso(intensity.get("peak_eta"))
                stop_time = parse_iso(intensity.get("stop_eta"))
                rate_text = f"{rain_rate:.1f} มม./ชม." if rain_rate is not None else "ปานกลาง"
                summary = f"ฝนกำลังตก ณ ตำแหน่งนี้ ({rate_text})"
                if stop_time:
                    summary += f" คาดว่าจะหยุดตกประมาณ {format_thai_time(stop_time)}"

                return {
                    "query_location": {"lat": lat, "lon": lon},
                    "scan_time": scan_time_str,
                    "scan_time_th": format_thai_time(scan_time),
                    "status_code": "currently_raining",
                    "status_label": "ฝนกำลังตก ณ ตำแหน่งนี้",
                    "alert_level": "danger",  # danger, warning, notice, safe
                    "summary_th": summary,
                    "target_object": obj,
                    "metrics": {
                        "rain_rate_mm_hr": rain_rate,
                        "p_within_30min": 1.0,
                        "p_within_60min": 1.0,
                        "eta_median_th": format_thai_time(scan_time),
                        "eta_window_th": [format_thai_time(scan_time), format_thai_time(stop_time)] if stop_time else None,
                        "peak_eta_th": format_thai_time(peak_time),
                        "stop_eta_th": format_thai_time(stop_time),
                        "cloud_top_bt_k": cloud_top.get("min_bt_10p4"),
                        "cooling_rate": cloud_top.get("cooling_rate_10min_K"),
                    },
                    "advection": None,
                }

            elif status == "developing":
                p30 = onset.get("p_within_30min")
                p60 = onset.get("p_within_60min")
                eta_med = parse_iso(onset.get("eta_median"))
                win = onset.get("eta_window")
                win_th = [format_thai_time(parse_iso(w)) for w in win] if win and all(win) else None

                p30_pct = int(p30 * 100) if p30 is not None else 0
                summary = f"กลุ่มเมฆฝนกำลังพัฒนาตัวตรงตำแหน่งนี้ (โอกาสตกใน 30 นาที: {p30_pct}%)"
                if win_th:
                    summary += f" คาดว่าฝนจะเริ่มตกช่วง {win_th[0]} - {win_th[1]}"

                return {
                    "query_location": {"lat": lat, "lon": lon},
                    "scan_time": scan_time_str,
                    "scan_time_th": format_thai_time(scan_time),
                    "status_code": "developing_overhead",
                    "status_label": "เมฆกำลังพัฒนาตัวตรงนี้ (ฝนจ่อตก)",
                    "alert_level": "warning" if (p30 or 0) > 0.4 else "notice",
                    "summary_th": summary,
                    "target_object": obj,
                    "metrics": {
                        "rain_rate_mm_hr": None,
                        "p_within_30min": p30,
                        "p_within_60min": p60,
                        "eta_median_th": format_thai_time(eta_med),
                        "eta_window_th": win_th,
                        "peak_eta_th": None,
                        "stop_eta_th": None,
                        "cloud_top_bt_k": cloud_top.get("min_bt_10p4"),
                        "cooling_rate": cloud_top.get("cooling_rate_10min_K"),
                    },
                    "advection": None,
                }

            else:
                return {
                    "query_location": {"lat": lat, "lon": lon},
                    "scan_time": scan_time_str,
                    "scan_time_th": format_thai_time(scan_time),
                    "status_code": "decaying_overhead",
                    "status_label": "กลุ่มเมฆฝนกำลังสลายตัว",
                    "alert_level": "notice",
                    "summary_th": "กลุ่มเมฆฝนผ่านพ้นไปแล้วหรือกำลังสลายตัว โอกาสฝนตกซ้ำต่ำ",
                    "target_object": obj,
                    "metrics": {
                        "rain_rate_mm_hr": rain_rate,
                        "p_within_30min": 0.1,
                        "p_within_60min": 0.1,
                        "eta_median_th": None,
                        "eta_window_th": None,
                        "peak_eta_th": None,
                        "stop_eta_th": None,
                        "cloud_top_bt_k": cloud_top.get("min_bt_10p4"),
                        "cooling_rate": cloud_top.get("cooling_rate_10min_K"),
                    },
                    "advection": None,
                }

    # =========================================================================
    # กรณีที่ 2: พิกัดไม่อยู่ใน Footprint ปัจจุบัน -> ตรวจสอบการเคลื่อนตัว (Advection)
    # =========================================================================
    approaching_matches = []
    nearest_cell = None
    min_dist_km = float("inf")

    time_steps_min = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    for obj in objects:
        oid = obj["object_id"]
        c_lon, c_lat = obj["geometry"]["coordinates"]
        dist_km = haversine_distance_km(lat, lon, c_lat, c_lon)

        if dist_km < min_dist_km:
            min_dist_km = dist_km
            nearest_cell = {
                "object": obj,
                "distance_km": round(dist_km, 1),
                "bearing_to_user": bearing_deg(c_lat, c_lon, lat, lon),
                "bearing_from_user": bearing_deg(lat, lon, c_lat, c_lon),
            }

        # ถ้าอยู่ห่างเกิน 60 กม. โอกาสเคลื่อนมาถึงใน 1 ชม. น้อยมาก ข้ามเพื่อความเร็ว
        if dist_km > 65.0:
            continue

        motion = obj.get("motion", {})
        speed_ms = motion.get("speed_ms") or 0.0
        dir_deg = motion.get("direction_deg") or 0.0

        if speed_ms < 0.5:
            # เมฆเกือบอยู่กับที่
            continue

        geom = footprint_geoms.get(oid)
        # ถ้าไม่มี footprint ให้สร้างวงกลม buffer รอบจุดศูนย์กลางเมฆ (~4 กม.)
        if geom is None:
            geom = Point(c_lon, c_lat).buffer(0.04)

        # จำลองการเคลื่อนตัวทีละ 5 นาที
        for dt_min in time_steps_min:
            # คำนวณระยะการเคลื่อนที่ (เมตร)
            shift_m = speed_ms * (dt_min * 60.0)
            # แปลงเป็นพิกัดภูมิศาสตร์
            # dir_deg: 0 = N, 90 = E, 180 = S, 270 = W
            d_lat = (shift_m * math.cos(math.radians(dir_deg))) / 111139.0
            d_lon = (shift_m * math.sin(math.radians(dir_deg))) / (111139.0 * math.cos(math.radians(c_lat)))

            advected_geom = translate(geom, xoff=d_lon, yoff=d_lat)

            # ตรวจสอบว่าพิกัดผู้ใช้ตกใน Footprint ที่เคลื่อนมาถึงหรือไม่
            # ให้ขอบเขตเผื่อการขยายตัว (buffer 0.015 องศา ~ 1.5 กม.)
            if advected_geom.buffer(0.015).contains(user_point):
                approaching_matches.append({
                    "object": obj,
                    "arrival_min": dt_min,
                    "distance_km": round(dist_km, 1),
                    "bearing_from_object": bearing_deg(c_lat, c_lon, lat, lon),
                    "bearing_from_user": bearing_deg(lat, lon, c_lat, c_lon),
                })
                break  # เจอเวลาเร็วสุดที่กระทบแล้ว ไม่ต้องตรวจเวลาถัดไปของ object นี้

    # ถ้ามีกลุ่มเมฆกำลังเคลื่อนเข้ามา
    if approaching_matches:
        # เลือกกลุ่มที่จะมาถึงเร็วที่สุด
        approaching_matches.sort(key=lambda x: x["arrival_min"])
        best = approaching_matches[0]
        obj = best["object"]
        arr_min = best["arrival_min"]
        dist_km = best["distance_km"]
        bearing = best["bearing_from_user"]
        direction_name = compass_direction_thai(bearing)

        arrival_time = scan_time + timedelta(minutes=arr_min)
        window_lo = arrival_time - timedelta(minutes=5)
        window_hi = arrival_time + timedelta(minutes=8)

        status = obj.get("status")
        onset = obj.get("onset", {})
        intensity = obj.get("intensity", {})
        rain_rate = intensity.get("rain_rate_mm_hr") or 2.0
        p30 = onset.get("p_within_30min") or (0.9 if arr_min <= 30 else 0.5)
        p60 = onset.get("p_within_60min") or 0.95

        p_impact = int(p30 * 100) if arr_min <= 30 else int(p60 * 100)

        if status in ["raining", "peak"]:
            summary = (
                f"มีกลุ่มเมฆฝนกำลังเคลื่อนเข้ามาหาจาก{direction_name} (ห่าง {dist_km} กม.) "
                f"คาดว่าฝนจะเริ่มตกในอีก ~{arr_min} นาที (ช่วง {format_thai_time(window_lo)} - {format_thai_time(window_hi)})"
            )
        else:
            summary = (
                f"มีกลุ่มเมฆที่กำลังพัฒนาตัวเคลื่อนเข้ามาจาก{direction_name} (ห่าง {dist_km} กม.) "
                f"มีโอกาสเกิดฝน {p_impact}% คาดว่าจะมาถึงช่วง {format_thai_time(window_lo)} - {format_thai_time(window_hi)}"
            )

        return {
            "query_location": {"lat": lat, "lon": lon},
            "scan_time": scan_time_str,
            "scan_time_th": format_thai_time(scan_time),
            "status_code": "approaching",
            "status_label": f"กลุ่มเมฆกำลังเคลื่อนเข้ามา (อีก ~{arr_min} นาที)",
            "alert_level": "warning" if arr_min <= 30 else "notice",
            "summary_th": summary,
            "target_object": obj,
            "metrics": {
                "rain_rate_mm_hr": rain_rate if status in ["raining", "peak"] else None,
                "p_within_30min": p30,
                "p_within_60min": p60,
                "eta_median_th": format_thai_time(arrival_time),
                "eta_window_th": [format_thai_time(window_lo), format_thai_time(window_hi)],
                "arrival_minutes": arr_min,
                "distance_km": dist_km,
                "cloud_direction_thai": direction_name,
                "cloud_speed_kmh": round((obj.get("motion", {}).get("speed_ms") or 0.0) * 3.6, 1),
            },
            "advection": {
                "origin_lat": obj["geometry"]["coordinates"][1],
                "origin_lon": obj["geometry"]["coordinates"][0],
                "bearing_from_user": bearing,
                "distance_km": dist_km,
                "arrival_min": arr_min,
            },
        }

    # =========================================================================
    # กรณีที่ 3: ไม่มีเมฆที่จะเคลื่อนมาถึงใน 60 นาที (ปลอดภัย)
    # =========================================================================
    near_text = ""
    if nearest_cell:
        n_dist = nearest_cell["distance_km"]
        n_dir = compass_direction_thai(nearest_cell["bearing_from_user"])
        near_text = f" (กลุ่มเมฆที่ใกล้ที่สุดอยู่ทาง{n_dir} ห่างออกไป {n_dist} กม.)"

    return {
        "query_location": {"lat": lat, "lon": lon},
        "scan_time": scan_time_str,
        "scan_time_th": format_thai_time(scan_time),
        "status_code": "safe_no_rain",
        "status_label": "ไม่มีฝนในระยะ 60 นาทีข้างหน้า",
        "alert_level": "safe",
        "summary_th": f"สภาพอากาศ ณ จุดนี้ยังไม่มีแนวโน้มฝนตกใน 60 นาทีข้างหน้า{near_text}",
        "target_object": nearest_cell["object"] if nearest_cell else None,
        "metrics": {
            "rain_rate_mm_hr": 0.0,
            "p_within_30min": 0.0,
            "p_within_60min": 0.05,
            "eta_median_th": None,
            "eta_window_th": None,
            "nearest_cloud_distance_km": nearest_cell["distance_km"] if nearest_cell else None,
        },
        "advection": None,
    }


def compute_trajectories_and_impacts(scan_doc: dict[str, Any], footprints_fc: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    คำนวณเวกเตอร์วิถีการเคลื่อนที่ (Trajectory Vectors)
    และพื้นที่คาดการณ์ที่ฝนจะตกลงบนแผนที่ (Projected Rain Landfall / Impact Zones)
    """
    scan_time_str = scan_doc.get("scan_time")
    scan_time = parse_iso(scan_time_str) or datetime.now(timezone.utc)
    objects = scan_doc.get("objects", [])

    footprint_geoms: dict[str, Any] = {}
    if footprints_fc and "features" in footprints_fc:
        for feat in footprints_fc["features"]:
            oid = feat.get("id") or feat.get("properties", {}).get("object_id")
            if oid and "geometry" in feat and feat["geometry"]:
                try:
                    footprint_geoms[oid] = shape(feat["geometry"])
                except Exception:
                    pass

    trajectories = []
    impact_features = []

    time_ticks = [15, 30, 45, 60]

    for obj in objects:
        oid = obj["object_id"]
        status = obj.get("status", "unknown")
        c_lon, c_lat = obj["geometry"]["coordinates"]
        motion = obj.get("motion", {})
        speed_ms = motion.get("speed_ms") or 0.0
        dir_deg = motion.get("direction_deg") or 0.0
        onset = obj.get("onset", {})
        intensity = obj.get("intensity", {})

        # คำนวณเส้นทางข้างหน้า 60 นาที (ทุก 15 นาที)
        path = [[c_lon, c_lat]]
        waypoints = []

        if speed_ms > 0.2:
            for t_min in time_ticks:
                dist_m = speed_ms * (t_min * 60.0)
                d_lat = (dist_m * math.cos(math.radians(dir_deg))) / 111139.0
                d_lon = (dist_m * math.sin(math.radians(dir_deg))) / (111139.0 * math.cos(math.radians(c_lat)))
                pt_lat = round(c_lat + d_lat, 4)
                pt_lon = round(c_lon + d_lon, 4)
                path.append([pt_lon, pt_lat])
                waypoints.append({
                    "min": t_min,
                    "lat": pt_lat,
                    "lon": pt_lon,
                    "time_th": format_thai_time(scan_time + timedelta(minutes=t_min))
                })

        trajectories.append({
            "object_id": oid,
            "status": status,
            "speed_ms": speed_ms,
            "speed_kmh": round(speed_ms * 3.6, 1),
            "direction_deg": dir_deg,
            "origin": [c_lon, c_lat],
            "path": path,
            "waypoints": waypoints
        })

        # คำนวณ Projected Landfall / Impact Zone
        # กรณี 1: Developing เมฆที่กำลังก่อตัว และมี eta_median
        geom = footprint_geoms.get(oid)
        if geom is None and speed_ms > 0.2:
            geom = Point(c_lon, c_lat).buffer(0.035)

        if geom is not None and speed_ms > 0.2:
            eta_median_str = onset.get("eta_median")
            eta_dt = parse_iso(eta_median_str)
            
            project_min = None
            impact_type = "onset"

            if status == "developing" and eta_dt:
                delta_s = (eta_dt - scan_time).total_seconds()
                if 60 <= delta_s <= 5400: # 1 ถึง 90 นาทีข้างหน้า
                    project_min = delta_s / 60.0
                    impact_type = "onset_landfall"
            elif status in ["raining", "peak"]:
                # สำหรับฝนที่กำลังตก คำนวณจุดกระทบในอนาคตอีก 20 นาที (Peak corridor)
                project_min = 20.0
                impact_type = "peak_corridor"

            if project_min is not None:
                shift_m = speed_ms * (project_min * 60.0)
                d_lat = (shift_m * math.cos(math.radians(dir_deg))) / 111139.0
                d_lon = (shift_m * math.sin(math.radians(dir_deg))) / (111139.0 * math.cos(math.radians(c_lat)))

                impact_geom = translate(geom, xoff=d_lon, yoff=d_lat)
                target_center_lat = round(c_lat + d_lat, 4)
                target_center_lon = round(c_lon + d_lon, 4)

                eta_time = scan_time + timedelta(minutes=project_min)
                win_lo = eta_time - timedelta(minutes=5)
                win_hi = eta_time + timedelta(minutes=5)

                impact_features.append({
                    "type": "Feature",
                    "id": f"impact_{oid}",
                    "properties": {
                        "object_id": oid,
                        "status": status,
                        "impact_type": impact_type,
                        "project_min": round(project_min, 1),
                        "eta_th": format_thai_time(eta_time),
                        "eta_window_th": [format_thai_time(win_lo), format_thai_time(win_hi)],
                        "target_center": [target_center_lon, target_center_lat],
                        "p_within_30min": onset.get("p_within_30min"),
                        "rain_rate_mm_hr": intensity.get("rain_rate_mm_hr"),
                        "speed_kmh": round(speed_ms * 3.6, 1),
                        "direction_deg": dir_deg,
                    },
                    "geometry": mapping(impact_geom)
                })

    impact_fc = {
        "type": "FeatureCollection",
        "features": impact_features
    }

    return {
        "trajectories": trajectories,
        "impact_zones": impact_fc
    }


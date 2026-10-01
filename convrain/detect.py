"""
ขั้น 1: Detection แบบ multi-threshold  (เอกสาร หัวข้อ 4)

แนวคิด
    - จับเมฆตั้งแต่ยังเป็น cumulus ที่ยอดเมฆอุ่น (T1 ≈ 273 K) ไม่ต้องรอถึง 235 K
    - ใช้ threshold ซ้อนหลายระดับ (273/253/235/220 K) สร้างโครงสร้างแบบต้นไม้
      เมฆก้อนใหญ่ที่ระดับอุ่นอาจมีหลาย "แกน" ที่ระดับเย็น → แยกเป็นหลาย object
    - แกนของแต่ละกิ่ง = component ที่ลึกที่สุดที่ยังมีพื้นที่ ≥ A_min
    - แบ่ง pixel ที่เหลือของก้อนใหญ่ให้แต่ละแกนด้วย watershed บนภาพ BT

ความซับซ้อน O(N) ต่อ scan (N = จำนวน pixel)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from .ingest import Scan

# 8-connectivity: pixel ที่แตะกันทแยงถือว่าต่อกัน
STRUCTURE_8 = np.ones((3, 3), dtype=bool)


@dataclass
class DetectedObject:
    """วัตถุเมฆ 1 ก้อนที่ตรวจพบใน scan เดียว (ยังไม่มี track ID)"""

    label: int                 # หมายเลขใน label image ของ scan นี้
    rows: np.ndarray           # ตำแหน่งแถวของ pixel ใน object
    cols: np.ndarray           # ตำแหน่งคอลัมน์ของ pixel ใน object
    area_px: int
    area_km2: float
    centroid_rc: tuple[float, float]  # centroid ถ่วงน้ำหนักด้วยความเย็น (row, col)
    lat: float
    lon: float
    min_bt: float
    cold_mean_bt: float        # ค่าเฉลี่ยของ pixel ที่เย็นที่สุด 10%
    mean_bt: float
    core_level_k: float        # threshold ที่ลึกที่สุดที่แกนของ object นี้ผ่าน
    touches_edge: bool = False
    # [CST / Adler-Negri] แกนพาความร้อน (Convective Updraft Core)
    core_centroid_rc: tuple[float, float] = (0.0, 0.0)
    core_lat: float = 0.0
    core_lon: float = 0.0
    # [Parallax Correction] พิกัดบนพื้นดินจริง (Ground-True Coordinates)
    lat_pc: float = 0.0
    lon_pc: float = 0.0
    core_lat_pc: float = 0.0
    core_lon_pc: float = 0.0
    cth_km: float = 0.0
    parallax_shift_km: float = 0.0
    # [Multispectral Microphysics] จุลฟิสิกส์เมฆและสัญญาณพายุรุนแรง
    has_overshooting: bool = False
    ot_pixel_count: int = 0
    core_thermal_depth: float = 0.0
    core_gradient_k_per_km: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class Detections:
    labels: np.ndarray                 # label image (0 = ไม่ใช่ object)
    objects: list[DetectedObject]


# =====================================================================
# [ขั้น 1.1] mask เมฆที่มีศักยภาพ + ตัด cirrus บาง
# =====================================================================
def candidate_mask(scan: Scan, cfg: dict) -> np.ndarray:
    """
    pixel ที่ BT(10.4) < T1 และไม่ใช่ cirrus บาง
    ใช้หลัก Multi-spectral Band Math (Inoue 1987; Mecikalski & Bedka 2006):
    1. Split Window BTD(10.4 - 12.4): cirrus บางมี BTD_SW สูง (> ~3 K) ร่วมกับ BT เย็น
    2. Tri-Spectral TTD: ถ้า TTD < -2.0 K ร่วมกับ BTD_SW สูง ยืนยันว่าเป็น cirrus โปร่งแสงไม่มีฝน
    """
    d = cfg["detect"]
    t1 = max(d["thresholds_k"])
    mask = scan.ir < t1
    btd_sw = scan.btd_sw
    if np.isfinite(btd_sw).any():
        cirrus = (btd_sw > d["thin_cirrus_btd_sw"]) & (scan.ir < d["thin_cirrus_bt_max"])
        ttd = scan.ttd
        if np.isfinite(ttd).any():
            cirrus |= (btd_sw > 2.0) & (ttd < -2.0) & (scan.ir < 265.0)
        mask &= ~cirrus
    return mask


# =====================================================================
# [ขั้น 1.2–1.3] หาแกนของแต่ละกิ่งในต้นไม้ของ threshold
# =====================================================================
def _find_cores(region: np.ndarray, ir: np.ndarray, levels: list[float], min_area: int,
                level_idx: int = 0) -> list[tuple[np.ndarray, float]]:
    """
    ค้นแบบ recursive: ถ้า component ที่ระดับถัดไป (เย็นกว่า) มีพื้นที่ ≥ min_area
    ให้ลงไปหาแกนในแต่ละ component นั้นต่อ ถ้าไม่มีเลย component ปัจจุบันคือแกน
    คืนค่า list ของ (mask ของแกน, ระดับ threshold ของแกน)
    """
    if level_idx + 1 >= len(levels):
        return [(region, levels[level_idx])]
    colder = region & (ir < levels[level_idx + 1])
    lab, n = ndimage.label(colder, structure=STRUCTURE_8)
    if n == 0:
        return [(region, levels[level_idx])]
    sizes = ndimage.sum_labels(colder, lab, index=np.arange(1, n + 1))
    valid = [i + 1 for i, s in enumerate(sizes) if s >= min_area]
    if not valid:
        return [(region, levels[level_idx])]
    cores: list[tuple[np.ndarray, float]] = []
    for k in valid:
        cores.extend(_find_cores(lab == k, ir, levels, min_area, level_idx + 1))
    return cores


# =====================================================================
# [ขั้น 1] ฟังก์ชันหลัก
# =====================================================================
def detect_objects(scan: Scan, cfg: dict) -> Detections:
    """
    ตรวจจับวัตถุเมฆใน scan

    ขั้นตอน
        1. mask = BT < T1 และไม่ใช่ cirrus บาง
        2. connected components ที่ระดับ T1 = ก้อนเมฆชั้นนอก
        3. ในแต่ละก้อนหาแกนตามต้นไม้ threshold (ลึกสุดที่พื้นที่ ≥ A_min)
        4. ถ้ามีหลายแกน แบ่ง pixel ของก้อนด้วย watershed บน BT
        5. ตัด object ที่เล็กเกินไป / แตะขอบโดเมน แล้วคำนวณคุณสมบัติพื้นฐาน
    """
    d = cfg["detect"]
    levels = sorted(d["thresholds_k"], reverse=True)  # เรียงจากอุ่น → เย็น
    min_area = int(d["min_area_px"])
    ir = scan.ir
    ny, nx = ir.shape

    mask = candidate_mask(scan, cfg)
    outer, n_outer = ndimage.label(mask, structure=STRUCTURE_8)
    labels = np.zeros((ny, nx), dtype=np.int32)
    core_level: dict[int, float] = {}
    next_label = 1

    for i, sl in enumerate(ndimage.find_objects(outer), start=1):
        if sl is None:
            continue
        region = outer[sl] == i
        if region.sum() < min_area:
            continue
        sub_ir = ir[sl]
        cores = _find_cores(region, sub_ir, levels, min_area)

        if len(cores) == 1:
            # แกนเดียว: ทั้งก้อนเป็น object เดียว
            labels[sl][region] = next_label
            core_level[next_label] = cores[0][1]
            next_label += 1
            continue

        # หลายแกน: watershed บน BT โดยใช้แกนเป็น marker (ภาพ BT เย็น = แอ่ง)
        markers = np.zeros(region.shape, dtype=np.int32)
        for core_mask, lvl in cores:
            markers[core_mask] = next_label
            core_level[next_label] = lvl
            next_label += 1
        seg = watershed(sub_ir, markers=markers, mask=region)
        labels[sl][region] = seg[region]

    # [Level Set Method] ปรับขอบเขตวัตถุให้แนบสนิทกับ thermal gradient สูงสุด
    if d.get("refine_contours", False):
        labels = _refine_labels_active_contour(labels, ir, int(d.get("active_contour_iter", 5)))

    objects = _measure(labels, scan, cfg, core_level)
    # เรียงหมายเลข label ใหม่ให้ต่อเนื่องหลังตัด object ทิ้ง
    new_labels = np.zeros_like(labels)
    for k, obj in enumerate(objects, start=1):
        new_labels[obj.rows, obj.cols] = k
        obj.label = k
    return Detections(labels=new_labels, objects=objects)



def _refine_labels_active_contour(labels: np.ndarray, ir: np.ndarray, num_iter: int = 5) -> np.ndarray:
    """
    ปรับเส้นรอบรูปของก้อนเมฆด้วย Morphological Geodesic Active Contours (Level Set)
    เพื่อให้ขอบเขตยึดติดกับ thermal gradient ที่ชันที่สุดของยอดเมฆ
    """
    try:
        from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
    except ImportError:
        return labels

    refined = np.copy(labels)
    ny, nx = labels.shape
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        sub_mask = labels[sl] == lab
        if sub_mask.sum() < 6:
            continue
        # ขยายกรอบ 3 pixel ในภาพรวม เพื่อให้มีขอบคำนวณ numerical gradient ได้อย่างเสถียร
        r0 = max(0, sl[0].start - 3)
        r1 = min(ny, sl[0].stop + 3)
        c0 = max(0, sl[1].start - 3)
        c1 = min(nx, sl[1].stop + 3)
        if (r1 - r0) < 6 or (c1 - c0) < 6:
            continue

        patch_mask = labels[r0:r1, c0:c1] == lab
        patch_ir = ir[r0:r1, c0:c1]
        try:
            g = inverse_gaussian_gradient(patch_ir, alpha=100.0, sigma=1.0)
            evolved = morphological_geodesic_active_contour(
                g, num_iter=num_iter, init_level_set=patch_mask, balloon=0, smoothing=1
            )
            evolved_bool = (evolved > 0)
            if evolved_bool.sum() >= 4:
                patch = refined[r0:r1, c0:c1]
                mask_to_clear = patch == lab
                patch[mask_to_clear] = 0
                empty_or_self = (patch == 0) | mask_to_clear
                patch[evolved_bool & empty_or_self] = lab
                refined[r0:r1, c0:c1] = patch
        except Exception:
            continue
    return refined


# =====================================================================
# [ขั้น 1.5] คุณสมบัติพื้นฐานของแต่ละ object + ระบุตำแหน่งแกนพาความร้อนและ Parallax
# =====================================================================
def _measure(labels: np.ndarray, scan: Scan, cfg: dict, core_level: dict[int, float]) -> list[DetectedObject]:
    from .parallax import cloud_top_height_km, parallax_correct, shift_km

    d = cfg["detect"]
    min_area = int(d["min_area_px"])
    cold_frac = cfg["features"]["cold_fraction"]
    t1 = max(d["thresholds_k"])
    ir = scan.ir
    ny, nx = ir.shape
    area_map = scan.pixel_area_km2 if scan.pixel_area_km2 is not None else np.full(ir.shape, 4.0)

    sat_cfg = cfg.get("satellite", {"sub_lon": 140.7, "orbit_radius_km": 42164.0})
    sub_lon = sat_cfg.get("sub_lon", 140.7)
    orbit_r = sat_cfg.get("orbit_radius_km", 42164.0)
    parallax_on = cfg.get("parallax", {}).get("enabled", True)

    objects: list[DetectedObject] = []
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        local = labels[sl] == lab
        rr, cc = np.nonzero(local)
        rows = (rr + sl[0].start).astype(np.int32)
        cols = (cc + sl[1].start).astype(np.int32)
        if rows.size < min_area:
            continue
        touches = bool(rows.min() == 0 or cols.min() == 0 or rows.max() == ny - 1 or cols.max() == nx - 1)
        if touches and d["drop_edge_objects"]:
            continue

        bt = ir[rows, cols]
        # 1. Centroid รวมของก้อนเมฆ ถ่วงด้วยความเย็น (T1 − BT)
        w = np.clip(t1 - bt, 0.1, None)
        w_sum = float(w.sum())
        r0 = float(np.sum(rows * w) / w_sum)
        c0 = float(np.sum(cols * w) / w_sum)
        coords = np.array([[r0], [c0]])
        lat = float(ndimage.map_coordinates(scan.lat, coords, order=1, mode="nearest")[0])
        lon = float(ndimage.map_coordinates(scan.lon, coords, order=1, mode="nearest")[0])

        min_bt_val = float(bt.min())
        n_cold = max(1, int(np.ceil(cold_frac * bt.size)))
        cold_mean = float(np.mean(np.partition(bt, n_cold - 1)[:n_cold]))
        mean_bt_val = float(bt.mean())

        # 2. [CST / Adler-Negri & RDT-CW] แกนพาความร้อน (Convective Updraft Core)
        # ค้นหาพิกเซลที่อยู่ในโดมแกนพาความร้อน (ใกล้เคียง min_bt หรืออยู่ในกลุ่มเย็นสุด 10%)
        core_threshold = min(min_bt_val + 3.0, cold_mean + 0.5)
        core_mask_local = bt <= core_threshold
        if not core_mask_local.any():
            core_mask_local = bt <= (min_bt_val + 0.1)

        core_rows = rows[core_mask_local]
        core_cols = cols[core_mask_local]
        core_bt = bt[core_mask_local]
        core_w = np.clip(core_threshold + 0.5 - core_bt, 0.1, None)
        core_w_sum = float(core_w.sum())
        core_r0 = float(np.sum(core_rows * core_w) / core_w_sum)
        core_c0 = float(np.sum(core_cols * core_w) / core_w_sum)
        core_coords = np.array([[core_r0], [core_c0]])
        core_lat = float(ndimage.map_coordinates(scan.lat, core_coords, order=1, mode="nearest")[0])
        core_lon = float(ndimage.map_coordinates(scan.lon, core_coords, order=1, mode="nearest")[0])

        core_thermal_depth = float(mean_bt_val - min_bt_val)

        # 3. [Parallax Correction ทันทีที่ขั้นตรวจจับ]
        cth_km = float(cloud_top_height_km(cold_mean))
        if parallax_on:
            lat_pc, lon_pc = parallax_correct(lat, lon, cth_km, sub_lon, orbit_r)
            lat_pc, lon_pc = float(lat_pc), float(lon_pc)
            core_lat_pc, core_lon_pc = parallax_correct(core_lat, core_lon, cth_km, sub_lon, orbit_r)
            core_lat_pc, core_lon_pc = float(core_lat_pc), float(core_lon_pc)
            p_shift = float(shift_km(lat, lon, lat_pc, lon_pc))
        else:
            lat_pc, lon_pc = lat, lon
            core_lat_pc, core_lon_pc = core_lat, core_lon
            p_shift = 0.0

        # 4. [Multispectral Overshooting Top (OT) Diagnostic]
        # Mecikalski & Bedka (2006) / Setvak et al. (2010): BTD(6.2 - 10.4) >= -0.5 K
        ot_count = 0
        btd_wv_map = scan.btd_wv
        if np.isfinite(btd_wv_map).any():
            core_wv = btd_wv_map[core_rows, core_cols]
            ot_count = int(np.sum(core_wv >= -0.5))
        has_ot = ot_count > 0

        # 5. [Core Thermal Gradient] ความชันอุณหภูมิรอบแกนพาความร้อน (K/km)
        pix_km = float(np.sqrt(np.mean(area_map[core_rows, core_cols]))) if core_rows.size else 2.0
        core_grad = float(core_thermal_depth / max(pix_km * np.sqrt(core_rows.size), 1.0))

        objects.append(DetectedObject(
            label=lab, rows=rows, cols=cols, area_px=int(rows.size),
            area_km2=float(area_map[rows, cols].sum()), centroid_rc=(r0, c0),
            lat=lat, lon=lon, min_bt=min_bt_val, cold_mean_bt=cold_mean,
            mean_bt=mean_bt_val, core_level_k=float(core_level.get(lab, t1)),
            touches_edge=touches,
            core_centroid_rc=(core_r0, core_c0), core_lat=core_lat, core_lon=core_lon,
            lat_pc=lat_pc, lon_pc=lon_pc, core_lat_pc=core_lat_pc, core_lon_pc=core_lon_pc,
            cth_km=cth_km, parallax_shift_km=p_shift,
            has_overshooting=has_ot, ot_pixel_count=ot_count,
            core_thermal_depth=core_thermal_depth, core_gradient_k_per_km=core_grad,
        ))
    return objects

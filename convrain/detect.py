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
    cirrus บาง: BTD split window สูง (> ~3 K) ร่วมกับ BT เย็น
    (เมฆ convective หนาทึบมี BTD_SW ใกล้ 0 จึงไม่ถูกตัด)
    """
    d = cfg["detect"]
    t1 = max(d["thresholds_k"])
    mask = scan.ir < t1
    btd_sw = scan.btd_sw
    if np.isfinite(btd_sw).any():
        cirrus = (btd_sw > d["thin_cirrus_btd_sw"]) & (scan.ir < d["thin_cirrus_bt_max"])
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

    objects = _measure(labels, scan, cfg, core_level)
    # เรียงหมายเลข label ใหม่ให้ต่อเนื่องหลังตัด object ทิ้ง
    new_labels = np.zeros_like(labels)
    for k, obj in enumerate(objects, start=1):
        new_labels[obj.rows, obj.cols] = k
        obj.label = k
    return Detections(labels=new_labels, objects=objects)


# =====================================================================
# [ขั้น 1.5] คุณสมบัติพื้นฐานของแต่ละ object
# =====================================================================
def _measure(labels: np.ndarray, scan: Scan, cfg: dict, core_level: dict[int, float]) -> list[DetectedObject]:
    d = cfg["detect"]
    min_area = int(d["min_area_px"])
    cold_frac = cfg["features"]["cold_fraction"]
    t1 = max(d["thresholds_k"])
    ir = scan.ir
    ny, nx = ir.shape
    area_map = scan.pixel_area_km2 if scan.pixel_area_km2 is not None else np.full(ir.shape, 4.0)

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
        # centroid ถ่วงด้วยความเย็น (T1 − BT) เพื่อให้เข้าใกล้แกนเมฆ
        w = np.clip(t1 - bt, 0.1, None)
        r0 = float(np.sum(rows * w) / w.sum())
        c0 = float(np.sum(cols * w) / w.sum())
        coords = np.array([[r0], [c0]])
        lat = float(ndimage.map_coordinates(scan.lat, coords, order=1, mode="nearest")[0])
        lon = float(ndimage.map_coordinates(scan.lon, coords, order=1, mode="nearest")[0])

        n_cold = max(1, int(np.ceil(cold_frac * bt.size)))
        cold_mean = float(np.mean(np.partition(bt, n_cold - 1)[:n_cold]))

        objects.append(DetectedObject(
            label=lab, rows=rows, cols=cols, area_px=int(rows.size),
            area_km2=float(area_map[rows, cols].sum()), centroid_rc=(r0, c0),
            lat=lat, lon=lon, min_bt=float(bt.min()), cold_mean_bt=cold_mean,
            mean_bt=float(bt.mean()), core_level_k=float(core_level.get(lab, t1)),
            touches_edge=touches,
        ))
    return objects

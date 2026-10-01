"""
การคาดการณ์รูปร่างและขนาดของเมฆในอนาคต (Shape & Size Morphing)
ด้วยระเบียบวิธี Semi-Lagrangian Advection + Diffusion Equation

หลักการ (Fluid Dynamics):
    1. นำหน้ากากรูปทรงของเมฆ (Binary/Continuous Mask) ณ เวลาปัจจุบัน
    2. คำนวณวิถีย้อนกลับ (Backward Trajectory) ตามสนามลม (u_px, v_px)
    3. ประมาณค่าการเคลื่อนที่แบบ Semi-Lagrangian บนกริด
    4. ใส่ Diffusion เพื่อจำลองการแพร่ขยาย/การสลายตัวของขอบเมฆตามธรรมชาติ
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours

from .detect import DetectedObject
from .motion import MotionField


def advect_mask(
    mask: np.ndarray,
    u_px: np.ndarray,
    v_px: np.ndarray,
    dt_min: float,
    base_dt_min: float = 10.0,
    diffusion_sigma: float = 0.5,
    threshold: float = 0.35,
) -> np.ndarray:
    """
    เลื่อนรูปทรงของหน้ากากเมฆไปข้างหน้า dt_min นาที ด้วย Semi-Lagrangian advection
    พร้อม diffusion จำลองการขยายตัวของขอบเมฆ
    """
    scale = float(dt_min / base_dt_min)
    ny, nx = mask.shape
    y, x = np.mgrid[0:ny, 0:nx]

    # ย้อนรอยต้นทางตามสนามความเร็วลม
    src_y = y - v_px * scale
    src_x = x - u_px * scale

    # Sample ค่าความหนาแน่นเมฆจากจุดต้นทาง
    advected = ndimage.map_coordinates(mask.astype(np.float32), [src_y, src_x], order=1, mode="constant", cval=0.0)

    # Diffusion: ขอบเมฆจะค่อยๆ แผ่ขยายออกตามกาลเวลา (scale แปรตาม sqrt(t))
    if diffusion_sigma > 0:
        sig = float(diffusion_sigma * np.sqrt(max(scale, 0.1)))
        advected = ndimage.gaussian_filter(advected, sigma=sig)

    return advected >= threshold


def forecast_object_footprint(
    obj: DetectedObject,
    motion: MotionField | None,
    lat: np.ndarray,
    lon: np.ndarray,
    lead_time_min: float,
    pad_px: int = 30,
    diffusion_sigma: float = 0.5,
    threshold: float = 0.35,
    max_points: int = 60,
) -> list[list[float]] | None:
    """
    พยากรณ์รูปทรง polygon [lon, lat] ของ object ในอีก lead_time_min นาทีข้างหน้า
    โดยตัดกรอบเฉพาะ bounding box เพื่อความรวดเร็วบน CPU
    """
    if obj.rows.size == 0:
        return None

    ny, nx = lat.shape
    # คำนวณ displacement โดยประมาณเพื่อขยายกรอบตัด
    if motion is not None:
        dv, du = motion.displacement(obj.rows, obj.cols, lead_time_min)
        mean_dv, mean_du = float(np.mean(dv)), float(np.mean(du))
    else:
        mean_dv, mean_du = 0.0, 0.0

    r_min = max(0, int(np.floor(min(obj.rows.min(), obj.rows.min() + mean_dv))) - pad_px)
    r_max = min(ny, int(np.ceil(max(obj.rows.max(), obj.rows.max() + mean_dv))) + pad_px)
    c_min = max(0, int(np.floor(min(obj.cols.min(), obj.cols.min() + mean_du))) - pad_px)
    c_max = min(nx, int(np.ceil(max(obj.cols.max(), obj.cols.max() + mean_du))) + pad_px)

    sub_h = r_max - r_min
    sub_w = c_max - c_min
    if sub_h <= 0 or sub_w <= 0:
        return None

    # สร้าง mask เริ่มต้นใน sub-grid
    sub_mask = np.zeros((sub_h, sub_w), dtype=np.float32)
    sub_mask[obj.rows - r_min, obj.cols - c_min] = 1.0

    if motion is not None:
        sub_u = motion.u_px[r_min:r_max, c_min:c_max]
        sub_v = motion.v_px[r_min:r_max, c_min:c_max]
        base_dt = motion.dt_min
    else:
        sub_u = np.zeros((sub_h, sub_w), dtype=np.float32)
        sub_v = np.zeros((sub_h, sub_w), dtype=np.float32)
        base_dt = 10.0

    future_mask = advect_mask(
        sub_mask, sub_u, sub_v, dt_min=lead_time_min, base_dt_min=base_dt,
        diffusion_sigma=diffusion_sigma, threshold=threshold
    )

    # สกัด contours ของรูปทรงในอนาคต
    padded = np.pad(future_mask, 1, mode="constant")
    contours = find_contours(padded.astype(float), 0.5)
    if not contours:
        return None

    ring = max(contours, key=len) - 1.0  # ปรับชดเชย padding
    tol = 0.5
    while len(ring) > max_points and tol < 10.0:
        ring = approximate_polygon(ring, tolerance=tol)
        tol *= 2.0

    rr = np.clip(ring[:, 0] + r_min, 0, ny - 1)
    cc = np.clip(ring[:, 1] + c_min, 0, nx - 1)

    la = ndimage.map_coordinates(lat, [rr, cc], order=1, mode="nearest")
    lo = ndimage.map_coordinates(lon, [rr, cc], order=1, mode="nearest")
    coords = [[round(float(x), 4), round(float(y), 4)] for x, y in zip(lo, la)]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def forecast_footprints_series(
    obj: DetectedObject,
    motion: MotionField | None,
    lat: np.ndarray,
    lon: np.ndarray,
    lead_times_min: list[int] = [15, 30, 45, 60],
) -> dict[int, list[list[float]] | None]:
    """
    พยากรณ์ชุด polygon สำหรับทุกช่วงเวลาเตือนภัย (เช่น +15, +30, +45, +60 นาที)
    """
    forecasts = {}
    for lt in lead_times_min:
        forecasts[lt] = forecast_object_footprint(obj, motion, lat, lon, lead_time_min=float(lt))
    return forecasts

"""
ขั้น 2: Motion field เดียว ด้วย optical flow แบบเบา  (เอกสาร หัวข้อ 5)

คำนวณการเคลื่อนที่ของเมฆ (u, v) ครั้งเดียวต่อ scan แล้วใช้ซ้ำใน
    - tracking (advect object ก่อนจับคู่)      ขั้น 3
    - cooling rate แบบ Lagrangian               ขั้น 4
    - extrapolate ตำแหน่งฝนในอนาคต             ขั้น 6
แทนการใช้ทั้ง TOOCAN และ OCTANE ใน workflow เดิม

วิธีหลัก: DIS optical flow (Kroeger et al. 2016) ของ OpenCV บนภาพที่ downsample
วิธีสำรอง: phase correlation หา global shift (ใช้เมื่อไม่มี OpenCV)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class MotionField:
    """
    u_px, v_px : การเคลื่อนที่ (pixel) ในช่วงเวลา dt_min ของแต่ละ pixel
                 ความหมาย: สิ่งที่อยู่ที่ (row, col) ในภาพก่อนหน้า
                 จะไปอยู่ที่ (row + v, col + u) ในภาพปัจจุบัน
    """

    u_px: np.ndarray
    v_px: np.ndarray
    dt_min: float
    pixel_km: float

    def displacement(self, rows: np.ndarray, cols: np.ndarray, dt_min: float | None = None):
        """การเคลื่อนที่ (dv, du) ที่ตำแหน่ง pixel ที่ระบุ ปรับตามช่วงเวลาที่ต้องการ"""
        scale = 1.0 if dt_min is None else dt_min / self.dt_min
        return self.v_px[rows, cols] * scale, self.u_px[rows, cols] * scale

    def speed_direction(self, rows: np.ndarray, cols: np.ndarray) -> tuple[float, float]:
        """
        ความเร็ว (m/s) และทิศทางที่ object เคลื่อนที่ไป (องศา, 0 = เหนือ, 90 = ตะวันออก)
        หมายเหตุ: แถวของภาพเพิ่มขึ้นไปทางใต้ จึงกลับเครื่องหมาย v
        """
        u = float(np.mean(self.u_px[rows, cols]))
        v = float(np.mean(self.v_px[rows, cols]))
        dist_m = np.hypot(u, v) * self.pixel_km * 1000.0
        speed = dist_m / (self.dt_min * 60.0)
        direction = (np.degrees(np.arctan2(u, -v)) + 360.0) % 360.0
        return speed, direction


# =====================================================================
# [ขั้น 2.1] แปลง BT เป็นภาพ 8-bit และ downsample
# =====================================================================
def _to_uint8(ir: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    """เมฆเย็น = สว่าง เพื่อให้ texture ของเมฆเด่นในภาพ"""
    lo, hi = clip
    scaled = (hi - np.clip(ir, lo, hi)) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    ny, nx = img.shape
    return cv2.resize(img, (max(1, nx // factor), max(1, ny // factor)), interpolation=cv2.INTER_AREA)


# =====================================================================
# [ขั้น 2.3] smooth แบบ normalized convolution (ถ่วงด้วย mask ของเมฆ)
# =====================================================================
def _normalized_smooth(field: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """
    ค่าเฉลี่ยถ่วงน้ำหนักแบบ Gaussian: ใช้เฉพาะ flow ในบริเวณที่มีเมฆ (texture ให้ติดตามได้)
    บริเวณท้องฟ้าโปร่งได้ค่าจากเมฆรอบข้าง ถ้าไม่มีเมฆใกล้เลยใช้ค่าเฉลี่ยทั้งโดเมน
    """
    num = ndimage.gaussian_filter(field * weight, sigma)
    den = ndimage.gaussian_filter(weight, sigma)
    wide_num = ndimage.gaussian_filter(field * weight, sigma * 6)
    wide_den = ndimage.gaussian_filter(weight, sigma * 6)
    global_mean = float((field * weight).sum() / weight.sum()) if weight.sum() > 0 else 0.0
    out = np.where(den > 0.05, num / np.maximum(den, 1e-6),
                   np.where(wide_den > 0.01, wide_num / np.maximum(wide_den, 1e-6), global_mean))
    return out.astype(np.float32)


class MotionEstimator:
    """เก็บ state ของ motion field ก่อนหน้าไว้ทำ temporal smoothing (ข้อ 2.5)"""

    def __init__(self, cfg: dict):
        self.cfg = cfg["motion"]
        self.t1 = max(cfg["detect"]["thresholds_k"])
        self.prev: MotionField | None = None
        self._dis = None
        if self.cfg["method"] == "dis" and cv2 is not None:
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    # =================================================================
    # [ขั้น 2] ฟังก์ชันหลัก
    # =================================================================
    def estimate(self, prev_ir: np.ndarray, curr_ir: np.ndarray, dt_min: float,
                 pixel_km: float | None = None) -> MotionField:
        c = self.cfg
        pixel_km = float(pixel_km or c["pixel_size_km"])

        method = c.get("method", "dis")
        if method == "mcc" and cv2 is not None:
            u, v = self._mcc_flow(prev_ir, curr_ir, dt_min, pixel_km)
        elif self._dis is not None:
            u, v = self._dis_flow(prev_ir, curr_ir, dt_min, pixel_km)
        else:
            u, v = self._phase_shift(prev_ir, curr_ir)

        # [ขั้น 2.5] temporal smoothing กับ field ของ scan ก่อน (ลด jitter)
        a = float(c["temporal_alpha"])
        if self.prev is not None and self.prev.u_px.shape == u.shape:
            scale = dt_min / self.prev.dt_min
            u = a * u + (1 - a) * self.prev.u_px * scale
            v = a * v + (1 - a) * self.prev.v_px * scale

        self.prev = MotionField(u.astype(np.float32), v.astype(np.float32), dt_min, pixel_km)
        return self.prev

    def reset(self) -> None:
        self.prev = None

    # -----------------------------------------------------------------
    # [ขั้น 2.2–2.4] DIS optical flow + QC + smoothing
    # -----------------------------------------------------------------
    def _dis_flow(self, prev_ir, curr_ir, dt_min, pixel_km):
        c = self.cfg
        ds = int(c["downsample"])
        ny, nx = curr_ir.shape
        a = _downsample(_to_uint8(prev_ir, c["bt_clip"]), ds)
        b = _downsample(_to_uint8(curr_ir, c["bt_clip"]), ds)
        flow = self._dis.calc(a, b, None)  # (h, w, 2): dx, dy ในหน่วย pixel ของภาพย่อ
        u_s, v_s = flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)

        # [ขั้น 2.4] QC: ความเร็วเกินจริง หรือไม่มีเมฆ → น้ำหนัก 0
        cloudy = _downsample(((prev_ir < self.t1) | (curr_ir < self.t1)).astype(np.uint8) * 255, ds) > 127
        max_px = c["max_speed_ms"] * dt_min * 60.0 / (pixel_km * 1000.0) / ds
        ok = cloudy & (np.hypot(u_s, v_s) <= max_px)
        w = ok.astype(np.float32)

        # [ขั้น 2.3] smooth แล้ว upsample กลับ (คูณ ds เพื่อเปลี่ยนหน่วยเป็น pixel เต็ม)
        sigma = float(c["smooth_sigma_px"])
        u_s = _normalized_smooth(u_s, w, sigma)
        v_s = _normalized_smooth(v_s, w, sigma)
        u = cv2.resize(u_s, (nx, ny), interpolation=cv2.INTER_LINEAR) * ds
        v = cv2.resize(v_s, (nx, ny), interpolation=cv2.INTER_LINEAR) * ds
        return u, v

    # -----------------------------------------------------------------
    # Maximum Cross-Correlation (MCC) with sub-pixel peak fitting
    # -----------------------------------------------------------------
    def _mcc_flow(self, prev_ir, curr_ir, dt_min, pixel_km):
        c = self.cfg
        ds = int(c.get("downsample", 2))
        ny, nx = curr_ir.shape
        a = _downsample(_to_uint8(prev_ir, c["bt_clip"]), ds)
        b = _downsample(_to_uint8(curr_ir, c["bt_clip"]), ds)
        h, w = a.shape

        tile_size = int(c.get("mcc_tile_size", 24))
        stride = int(c.get("mcc_stride", 12))
        search_pad = int(c.get("mcc_search_pad", 14))

        u_grid = np.zeros((h, w), dtype=np.float32)
        v_grid = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        cloudy = _downsample(((prev_ir < self.t1) | (curr_ir < self.t1)).astype(np.uint8) * 255, ds) > 127
        max_px = c["max_speed_ms"] * dt_min * 60.0 / (pixel_km * 1000.0) / ds

        half = tile_size // 2
        for y in range(half, h - half, stride):
            for x in range(half, w - half, stride):
                if not cloudy[y - half:y + half, x - half:x + half].any():
                    continue

                patch = a[y - half:y + half, x - half:x + half]
                sy0 = max(0, y - half - search_pad)
                sy1 = min(h, y + half + search_pad)
                sx0 = max(0, x - half - search_pad)
                sx1 = min(w, x + half + search_pad)

                search_area = b[sy0:sy1, sx0:sx1]
                if search_area.shape[0] < patch.shape[0] or search_area.shape[1] < patch.shape[1]:
                    continue

                res = cv2.matchTemplate(search_area, patch, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val < float(c.get("mcc_min_corr", 0.35)):
                    continue

                px, py = max_loc
                dx = float((sx0 + px) - (x - half))
                dy = float((sy0 + py) - (y - half))

                # Sub-pixel quadratic interpolation
                if 0 < px < res.shape[1] - 1:
                    denom_x = 2 * (2 * res[py, px] - res[py, px + 1] - res[py, px - 1])
                    if abs(denom_x) > 1e-5:
                        dx += (res[py, px + 1] - res[py, px - 1]) / denom_x
                if 0 < py < res.shape[0] - 1:
                    denom_y = 2 * (2 * res[py, px] - res[py + 1, px] - res[py - 1, px])
                    if abs(denom_y) > 1e-5:
                        dy += (res[py + 1, px] - res[py - 1, px]) / denom_y

                if np.hypot(dx, dy) <= max_px:
                    u_grid[y - half:y + half, x - half:x + half] += dx
                    v_grid[y - half:y + half, x - half:x + half] += dy
                    weight[y - half:y + half, x - half:x + half] += max_val

        mask_nz = weight > 0
        u_grid[mask_nz] /= weight[mask_nz]
        v_grid[mask_nz] /= weight[mask_nz]

        sigma = float(c["smooth_sigma_px"])
        u_s = _normalized_smooth(u_grid, (weight > 0).astype(np.float32), sigma)
        v_s = _normalized_smooth(v_grid, (weight > 0).astype(np.float32), sigma)

        u = cv2.resize(u_s, (nx, ny), interpolation=cv2.INTER_LINEAR) * ds
        v = cv2.resize(v_s, (nx, ny), interpolation=cv2.INTER_LINEAR) * ds
        return u, v

    # -----------------------------------------------------------------
    # วิธีสำรอง: global shift ด้วย phase correlation
    # -----------------------------------------------------------------
    @staticmethod
    def _phase_shift(prev_ir, curr_ir):
        from skimage.registration import phase_cross_correlation

        shift, _, _ = phase_cross_correlation(prev_ir, curr_ir, upsample_factor=4)
        # shift คือค่าที่ต้องเลื่อน curr ให้ตรง prev → การเคลื่อนที่จริงคือค่าลบ
        dv, du = -shift[0], -shift[1]
        return np.full(curr_ir.shape, du, np.float32), np.full(curr_ir.shape, dv, np.float32)

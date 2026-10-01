"""
Unit tests สำหรับระบบเพิ่มความแม่นยำสูงสุด (Maximum Accuracy Enhancements):
1. Active Contours (Level Set) ใน detect.py
2. Maximum Cross-Correlation (MCC) ใน motion.py
3. Kinematic EKF ใน track.py
4. Semi-Lagrangian Advection + Diffusion ใน morph.py
"""
import numpy as np
import pytest
from datetime import datetime, timezone

from convrain.config import load_config
from convrain.detect import detect_objects
from convrain.ingest import Scan
from convrain.motion import MotionEstimator
from convrain.track import KinematicEKF, Tracker
from convrain.morph import advect_mask, forecast_object_footprint


def make_dummy_scan(time: datetime, shape=(60, 60)) -> Scan:
    ny, nx = shape
    ir = np.full((ny, nx), 290.0, dtype=np.float32)
    # ก้อนเมฆเย็นตรงกลาง (220 K)
    ir[20:40, 20:40] = 220.0
    ir[25:35, 25:35] = 205.0

    btd_sw = np.zeros((ny, nx), dtype=np.float32)
    btd_wv = np.zeros((ny, nx), dtype=np.float32)
    lat, lon = np.meshgrid(np.linspace(10, 15, ny), np.linspace(100, 105, nx), indexing="ij")

    bt = {"B13": ir, "B15": ir, "B08": ir}
    return Scan(time=time, lat=lat.astype(np.float32), lon=lon.astype(np.float32), bt=bt)


def test_active_contour_detection():
    cfg = load_config()
    cfg["detect"]["refine_contours"] = True
    cfg["detect"]["active_contour_iter"] = 3

    scan = make_dummy_scan(datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc))
    det = detect_objects(scan, cfg)
    assert len(det.objects) >= 1
    obj = det.objects[0]
    assert obj.area_px > 0
    assert obj.min_bt < 230.0


def test_mcc_motion_estimator():
    cfg = load_config()
    cfg["motion"]["method"] = "mcc"
    cfg["motion"]["mcc_tile_size"] = 16
    cfg["motion"]["mcc_stride"] = 8
    cfg["motion"]["mcc_search_pad"] = 8

    estimator = MotionEstimator(cfg)
    ny, nx = 80, 80
    prev_ir = np.full((ny, nx), 290.0, dtype=np.float32)
    curr_ir = np.full((ny, nx), 290.0, dtype=np.float32)

    # เลื่อนเมฆไปทางขวา 4 pixel และลงล่าง 2 pixel
    prev_ir[30:50, 30:50] = 210.0
    curr_ir[32:52, 34:54] = 210.0

    motion = estimator.estimate(prev_ir, curr_ir, dt_min=10.0)
    assert motion.u_px.shape == (ny, nx)
    assert motion.v_px.shape == (ny, nx)

    # ความเร็วควรชี้ไปทางขวา (u > 0) และลงล่าง (v > 0)
    u_cloud = motion.u_px[32:52, 34:54].mean()
    v_cloud = motion.v_px[32:52, 34:54].mean()
    assert u_cloud > 0.5
    assert v_cloud > 0.2


def test_kinematic_ekf():
    ekf = KinematicEKF(r0=10.0, c0=20.0, q_pos=0.1, r_pos=0.5)
    assert ekf.pos == (10.0, 20.0)

    # เคลื่อนที่ไปเรื่อยๆ: r เพิ่มทีละ 2, c เพิ่มทีละ 3
    for step in range(1, 4):
        ekf.update(10.0 + 2.0 * step, 20.0 + 3.0 * step, dt=1.0)

    pos = ekf.pos
    assert abs(pos[0] - 16.0) < 1.0
    assert abs(pos[1] - 29.0) < 1.0
    vel = ekf.vel
    assert vel[0] > 1.0
    assert vel[1] > 1.5


def test_tracker_kinematic_matching():
    cfg = load_config()
    cfg["track"]["use_kinematic_cost"] = True
    tracker = Tracker(cfg)

    t0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    scan0 = make_dummy_scan(t0)
    det0 = detect_objects(scan0, cfg)

    up0 = tracker.update(det0, motion=None, time=t0, dt_min=10.0)
    assert len(up0.observed) == 1
    t_id = up0.observed[0].track_id

    # Scan 1: เมฆขยับไปเล็กน้อย
    t1 = datetime(2026, 10, 1, 12, 10, tzinfo=timezone.utc)
    scan1 = make_dummy_scan(t1)
    det1 = detect_objects(scan1, cfg)

    up1 = tracker.update(det1, motion=None, time=t1, dt_min=10.0)
    assert len(up1.observed) == 1
    assert up1.observed[0].track_id == t_id  # จับคู่ได้สำเร็จ


def test_semi_lagrangian_advection_and_morph():
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True

    u = np.full((60, 60), 4.0, dtype=np.float32)
    v = np.full((60, 60), -2.0, dtype=np.float32)

    advected = advect_mask(mask, u, v, dt_min=10.0, base_dt_min=10.0, diffusion_sigma=0.5)
    assert advected.shape == (60, 60)
    assert advected.sum() > 0

    # ตรวจสอบว่า centroid เคลื่อนที่ไปตามทิศทางลม (u>0 ขวา, v<0 ขึ้น)
    cy0, cx0 = np.mean(np.nonzero(mask), axis=1)
    cy1, cx1 = np.mean(np.nonzero(advected), axis=1)
    assert cx1 > cx0 + 2.0
    assert cy1 < cy0 - 1.0


def test_convective_core_and_parallax_localization():
    cfg = load_config()
    cfg["detect"]["refine_contours"] = False
    cfg["parallax"]["enabled"] = True

    ny, nx = 80, 80
    ir = np.full((ny, nx), 290.0, dtype=np.float32)
    # ก้อนเมฆขนาดใหญ่ 30x30 แต่อสมมาตร:
    # แผ่นเมฆทั่ง (Anvil) อุณหภูมิ 230 K อยู่ช่วง (20:50, 20:50)
    ir[20:50, 20:50] = 230.0
    # แกนลมพัดขึ้นรุนแรง (Updraft Core) อุณหภูมิ 198 K กระจุกอยู่มุมบนซ้าย (22:28, 22:28)
    ir[22:28, 22:28] = 198.0

    btd_sw = np.zeros((ny, nx), dtype=np.float32)
    # จำลอง Overshooting Top: BTD_WV >= 0 ตรงบริเวณแกน
    btd_wv = np.full((ny, nx), -15.0, dtype=np.float32)
    btd_wv[23:27, 23:27] = 1.5

    lat, lon = np.meshgrid(np.linspace(13, 15, ny), np.linspace(100, 102, nx), indexing="ij")
    bt = {"B13": ir, "B15": ir, "B08": ir + btd_wv}

    scan = Scan(time=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
                lat=lat.astype(np.float32), lon=lon.astype(np.float32), bt=bt)

    det = detect_objects(scan, cfg)
    assert len(det.objects) == 1
    obj = det.objects[0]

    # 1. ทดสอบว่า Centroid รวม กับ Core Centroid แยกจากกันตามหลัก CST (Adler & Negri 1988)
    # Centroid รวมจะถูกดึงไปทางกึ่งกลางของ Anvil (~35, ~35)
    # แต่ Core Centroid จะอยู่ชิดมุมบนซ้ายตรง Updraft Core (~25, ~25)
    r_all, c_all = obj.centroid_rc
    r_core, c_core = obj.core_centroid_rc
    assert r_core < r_all - 3.0, f"Core r ({r_core}) should be significantly colder/higher than all r ({r_all})"
    assert c_core < c_all - 3.0, f"Core c ({c_core}) should be significantly colder/higher than all c ({c_all})"

    # 2. ทดสอบ Parallax Ground Coordinate (ต้องมี shift และ cth_km สมจริง)
    assert obj.cth_km > 12.0  # เมฆ 198K ต้องสูงกว่า 12 km
    assert obj.parallax_shift_km > 10.0  # การเลื่อนต้องมากกว่า 10 km
    assert abs(obj.lat_pc - obj.lat) > 0.01
    assert abs(obj.lon_pc - obj.lon) > 0.05

    # 3. ทดสอบ Overshooting Top Detection
    assert obj.has_overshooting is True
    assert obj.ot_pixel_count >= 10
    assert obj.core_thermal_depth > 25.0

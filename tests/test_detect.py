"""ขั้น 1: detection แบบ multi-threshold"""
from conftest import blob, make_scan

import numpy as np

from convrain.detect import detect_objects


def test_two_separate_clouds(cfg, t0):
    bt = np.minimum(blob((80, 80), 20, 20, 5, 230), blob((80, 80), 55, 55, 5, 250))
    det = detect_objects(make_scan(bt, t0, cfg), cfg)
    assert len(det.objects) == 2
    mins = sorted(o.min_bt for o in det.objects)
    assert abs(mins[0] - 230) < 1 and abs(mins[1] - 250) < 1


def test_two_cores_in_one_warm_region_are_split(cfg, t0):
    # สองแกนเย็น (220 K) อยู่ใกล้กันจนต่อกันที่ระดับ 273 K → ต้องได้ 2 object จาก watershed
    bt = np.minimum(blob((80, 80), 40, 32, 8, 215), blob((80, 80), 40, 46, 8, 215))
    scan = make_scan(bt, t0, cfg)
    assert (scan.ir[40, 32:47] < 273).all()  # ยืนยันว่าต่อกันที่ระดับอุ่น
    det = detect_objects(scan, cfg)
    assert len(det.objects) == 2
    assert det.labels[40, 32] != det.labels[40, 46]


def test_deep_and_moderate_core_are_split(cfg, t0):
    # แกนหนึ่งลึกถึง 220 K อีกแกนแค่ 250 K → ยังแยกเป็น 2 object (เซลล์ใหม่ข้างพายุเดิม)
    bt = np.minimum(blob((80, 80), 40, 28, 7, 210), blob((80, 80), 40, 46, 6, 248))
    det = detect_objects(make_scan(bt, t0, cfg), cfg)
    assert len(det.objects) == 2


def test_small_blob_dropped(cfg, t0):
    bt = blob((60, 60), 30, 30, 0.8, 260)
    det = detect_objects(make_scan(bt, t0, cfg), cfg)
    assert all(o.area_px >= cfg["detect"]["min_area_px"] for o in det.objects)


def test_edge_object_dropped(cfg, t0):
    bt = blob((60, 60), 1, 30, 5, 230)
    det = detect_objects(make_scan(bt, t0, cfg), cfg)
    assert len(det.objects) == 0

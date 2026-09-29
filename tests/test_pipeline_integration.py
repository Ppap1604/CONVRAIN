"""ระดับ 2–3: Pipeline Integration & State Continuity tests"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from conftest import blob, make_scan
from convrain.detect import detect_objects
from convrain.pipeline import ConvRainPipeline
from convrain.synthetic import SyntheticDay
from convrain.track import Tracker


def test_missing_scan_gap_handling(cfg, t0):
    """ทดสอบกรณีสแกนดาวเทียมหายไป (เช่น เว้นช่วง 30 นาที แทนที่จะเป็น 10 นาที)"""
    pipe = ConvRainPipeline(cfg)

    # Scan 1: t0
    bt1 = blob((80, 80), 40, 40, 6, 225)
    scan1 = make_scan(bt1, t0, cfg)
    res1 = pipe.step(scan1)
    assert len(res1.document["objects"]) >= 1
    t1_id = res1.document["objects"][0]["object_id"]

    # Scan 2: t0 + 30 นาที (เว้นไป 2 ช่วงสแกน) ขยับเล็กน้อย
    t2 = t0 + timedelta(minutes=30)
    bt2 = blob((80, 80), 41, 42, 6, 220)
    scan2 = make_scan(bt2, t2, cfg)
    res2 = pipe.step(scan2)

    # ตรวจสอบว่าระบบไม่ crash และคำนวณเวลาต่อเนื่องได้
    assert res2.errors == []
    assert pipe.prev_time == t2
    # ตรวจสอบว่า timings ถูกคำนวณสมบูรณ์
    assert res2.timings_ms["total"] > 0


def test_state_persistence_with_active_tracks(tmp_path, cfg):
    """ทดสอบการ save_state และ load_state ขณะมี active tracks หลากหลายสถานะ"""
    day = SyntheticDay(seed=42, start=datetime(2026, 6, 1, 5, 0, tzinfo=timezone.utc), n_scans=8)
    pipe = ConvRainPipeline(cfg)
    scans = list(day.scans())

    # รันไป 5 สแกน
    for scan, frames, rr in scans[:5]:
        r = pipe.step(scan, radar_frames=frames, radar_rainrate=rr)
        assert r.errors == []

    active_ids_orig = set(pipe.tracker.tracks.keys())
    assert len(active_ids_orig) > 0

    state_path = tmp_path / "pipeline_state.pkl"
    pipe.save_state(state_path)

    # กู้คืนใน instance ใหม่
    pipe_restored = ConvRainPipeline(cfg)
    pipe_restored.load_state(state_path)

    assert set(pipe_restored.tracker.tracks.keys()) == active_ids_orig
    assert pipe_restored.prev_time == pipe.prev_time

    # รันสแกนถัดไปบน instance ที่กู้มา
    r_next = pipe_restored.step(scans[5][0])
    assert r_next.errors == []
    assert len(r_next.document["objects"]) > 0

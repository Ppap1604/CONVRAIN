"""ทั้ง pipeline + output validation"""
from datetime import datetime, timezone

import numpy as np

from convrain.config import load_config
from convrain.output import validate_output
from convrain.pipeline import ConvRainPipeline
from convrain.synthetic import SyntheticDay


def good_doc():
    return {"scan_time": "2026-09-25T07:20:00Z", "model_version": "t", "objects": [{
        "object_id": "o1", "parent_id": None, "age_min": 40, "status": "developing",
        "geometry": {"type": "Point", "coordinates": [100.5, 13.7]}, "geometry_parallax_corrected": True,
        "footprint_polygon": None,
        "cloud_top": {"min_bt_10p4": 231.4, "cooling_rate_10min_K": -3.8, "cloud_top_height_km": 11.2},
        "onset": {"p_within_30min": 0.71, "p_within_60min": 0.86, "eta_median": "2026-09-25T07:45:00Z",
                  "eta_window": ["2026-09-25T07:35:00Z", "2026-09-25T07:55:00Z"]},
        "intensity": {"rain_rate_mm_hr": None, "peak_eta": None, "stop_eta": None},
        "motion": {"speed_ms": 6.2, "direction_deg": 210}}]}


def test_output_validation_rules(cfg):
    cfg = {**cfg, "domain": {"lat_min": 4, "lat_max": 22, "lon_min": 96, "lon_max": 107}}
    assert validate_output(good_doc(), cfg) == []
    d = good_doc()
    d["objects"][0]["onset"]["p_within_30min"] = 0.9
    d["objects"][0]["onset"]["p_within_60min"] = 0.5
    assert any("p30 > p60" in e for e in validate_output(d, cfg))
    d = good_doc()
    d["objects"][0]["intensity"]["rain_rate_mm_hr"] = 5.0
    assert any("developing" in e for e in validate_output(d, cfg))
    d = good_doc()
    d["objects"][0]["onset"]["eta_window"] = ["2026-09-25T07:50:00Z", "2026-09-25T07:55:00Z"]
    assert any("eta_window" in e for e in validate_output(d, cfg))
    d = good_doc()
    del d["objects"][0]["status"]
    assert validate_output(d, cfg)


def test_pipeline_runs_and_restores_state(tmp_path):
    day = SyntheticDay(seed=7, start=datetime(2026, 5, 3, 5, 0, tzinfo=timezone.utc), n_scans=14)
    cfg = load_config(overrides={"domain": day.domain()})
    pipe = ConvRainPipeline(cfg)
    scans = list(day.scans())
    n_rows = 0
    for scan, frames, rr in scans[:8]:
        r = pipe.step(scan, radar_frames=frames, radar_rainrate=rr)
        assert r.errors == []
        n_rows += 0 if r.timeseries is None else len(r.timeseries)
    assert n_rows > 0
    pipe.save_state(tmp_path / "state.pkl")
    ids_before = set(pipe.tracker.tracks)

    # เริ่ม process ใหม่แล้วกู้ state: track เดิมต้องต่อได้
    pipe2 = ConvRainPipeline(cfg)
    pipe2.load_state(tmp_path / "state.pkl")
    r = pipe2.step(scans[8][0])
    assert r.errors == []
    still = {o["object_id"] for o in r.document["objects"]}
    assert ids_before & still, "track เดิมควรต่อเนื่องหลังกู้ state"
    assert np.isfinite(r.timings_ms["total"])

"""ระดับ 6: End-to-End Workflow & Pipeline Lifecycle tests (Phase 0–7)"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from convrain.config import load_config
from convrain.evaluate import evaluate_all
from convrain.experiment import fit_lut, predict_table, replay_scans, train_models
from convrain.labels import attach_labels, build_track_labels
from convrain.models import ClimatologyBaseline, ExtrapolationBaseline, MB06InterestFields
from convrain.output import validate_output, write_scan_outputs
from convrain.pipeline import ConvRainPipeline
from convrain.synthetic import SyntheticDay


def test_e2e_lifecycle_phase_0_to_7(tmp_path):
    """
    ทดสอบวงรอบการทำงานเต็มระบบ ตั้งแต่ Replay -> Label -> Train -> Eval -> Real-time Output
    """
    out_dir = tmp_path / "e2e_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. จำลองข้อมูล 3 วัน (วันละ 8 scans): train, val, test
    start = datetime(2026, 5, 1, 5, 0, tzinfo=timezone.utc)
    days = [SyntheticDay(seed=100 + i, start=start + timedelta(days=i), n_scans=8) for i in range(3)]
    cfg = load_config(overrides={"domain": days[0].domain()})

    # 2. Phase 2–3: Replay scans
    ts_all, tr_all, pairs = [], [], []
    for i, day in enumerate(days):
        # เก็บคู่ LUT เฉพาะวัน train (วันแรก)
        pipe = ConvRainPipeline(cfg, collect_lut_pairs=(i == 0))
        r = replay_scans(pipe, day.scans())
        assert r["errors"] == []
        if len(r["timeseries"]):
            ts_all.append(r["timeseries"])
        if len(r["tracks"]):
            tr_all.append(r["tracks"])
        pairs.extend(pipe.lut_pairs)

    ts = pd.concat(ts_all, ignore_index=True)
    tracks = pd.concat(tr_all, ignore_index=True)
    assert len(ts) > 0
    assert len(tracks) > 0

    # 3. Phase 3: Build labels
    labels = build_track_labels(ts, tracks, cfg)
    df = attach_labels(ts, labels)
    assert "t_onset" in df.columns
    assert "censor_time" in df.columns

    # 4. Phase 4–5: Fit LUT & Train models
    day_strs = pd.to_datetime(df["time"], utc=True).dt.strftime("%Y-%m-%d").unique()
    train_df = df[df["time"].dt.strftime("%Y-%m-%d") == day_strs[0]]
    val_df = df[df["time"].dt.strftime("%Y-%m-%d") == day_strs[1]]
    test_df = df[df["time"].dt.strftime("%Y-%m-%d") == day_strs[2]]

    lut = fit_lut(pairs, min_count=2)
    assert lut.calibrated

    models = train_models(train_df, val_df, cfg, estimator="logistic")
    assert models["onset"] is not None

    # บันทึกโมเดล
    models["onset"].save(out_dir / "model_onset.joblib")
    lut.save(out_dir / "rain_lut.npz")
    assert (out_dir / "model_onset.joblib").exists()
    assert (out_dir / "rain_lut.npz").exists()

    # 5. Phase 6: Evaluate บน Test เทียบกับ Baseline
    p_hazard = predict_table(models["onset"], test_df, cfg)
    res_hazard = evaluate_all(p_hazard, threshold=models["warn_threshold"])
    assert "POD@30" in res_hazard
    assert "CSI@30" in res_hazard

    # เทียบกับ MB06 Baseline
    mb06 = MB06InterestFields()
    p_mb06 = predict_table(mb06, test_df, cfg)
    res_mb06 = evaluate_all(p_mb06, threshold=0.5)
    assert "CSI@30" in res_mb06

    # 6. Phase 7: Real-time simulation พร้อมโมเดล และเขียน Output
    cfg_rt = {**cfg, "onset": {**cfg["onset"], "warn_threshold": models["warn_threshold"]}}
    pipe_rt = ConvRainPipeline(cfg_rt, onset_model=models["onset"], rain_lut=lut)

    rt_out = out_dir / "nowcast"
    test_day = days[2]
    n_written = 0

    for scan, _, _ in test_day.scans():
        step_res = pipe_rt.step(scan)
        assert step_res.errors == []
        write_scan_outputs(rt_out, step_res.document, step_res.footprints, step_res.timeseries)
        n_written += 1

    assert n_written == 8

    # 7. ตรวจสอบไฟล์ Output ทั้งหมด
    json_files = list((rt_out / "json").glob("objects_*.json"))
    geojson_files = list((rt_out / "geojson").glob("objects_*.geojson"))
    footprint_files = list((rt_out / "geojson").glob("footprints_*.geojson"))
    parquet_files = list((rt_out / "timeseries").rglob("*.parquet"))

    assert len(json_files) == 8, "ต้องมี JSON ต่อ scan ครบ 8 ไฟล์"
    assert len(geojson_files) == 8, "ต้องมี GeoJSON ต่อ scan ครบ 8 ไฟล์"
    assert len(footprint_files) > 0, "ต้องมี Footprints geojson อย่างน้อย 1 ไฟล์"
    assert len(parquet_files) > 0, "ต้องมี Parquet timeseries"

    # ตรวจสอบความถูกต้องของ schema ใน JSON ล่าสุด
    sample_json = json.loads(json_files[-1].read_text(encoding="utf-8"))
    val_errors = validate_output(sample_json, cfg_rt)
    assert val_errors == [], f"JSON Schema validation failed: {val_errors}"

    # ตรวจสอบการอ่าน Parquet
    df_parquet = pd.read_parquet(parquet_files[0])
    assert "object_id" in df_parquet.columns
    assert "cold_bt" in df_parquet.columns

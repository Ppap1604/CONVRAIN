"""
Pipeline หลัก: รวมขั้น 0–6 เป็น step() ต่อ scan  (เอกสาร หัวข้อ 2)

    [Scan ใหม่ทุก 10 นาที]
      ขั้น 0  preprocessing (geometry + QC)          ingest.py
      ขั้น 1  detection                               detect.py
      ขั้น 2  motion field                            motion.py
      ขั้น 3  tracking + Kalman                       track.py
      ขั้น 4  features                                features.py
      ขั้น 5  prefilter + onset hazard model          models/hazard.py
      ขั้น 6  parallax, rain rate, lifecycle, peak/stop   parallax.py, intensity.py
      Output  JSON / GeoJSON / Parquet + validation   output.py

หลักการสำคัญ (หัวข้อ 18.1): ทุกขั้นรับ state + scan ใหม่แล้วคืน state ใหม่
โค้ดเดียวกันใช้ทั้งตอน replay ข้อมูลย้อนหลังและตอน real-time → ผล offline = online
"""
from __future__ import annotations

import pickle
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .detect import detect_objects
from .features import compute_features
from .ingest import Scan, prepare_geometry, quality_check
from .intensity import RainRateLUT, object_rain, stop_eta_fallback, update_status
from .labels import lut_pairs, object_radar_stats
from .models.hazard import HazardModel
from .motion import MotionEstimator
from .output import footprint_polygon, object_record, scan_document, scan_stamp, validate_output
from .parallax import correct_object
from .track import Track, Tracker

HISTORY_KEEP = 8  # เก็บ history ต่อ track ย้อนหลัง 8 scan (พอสำหรับแนวโน้ม 30 นาที)


@dataclass
class ScanResult:
    time: datetime
    skipped: bool = False
    document: dict | None = None                 # JSON ต่อ scan (หัวข้อ 9.5)
    footprints: dict = field(default_factory=dict)
    timeseries: pd.DataFrame | None = None       # แถว object-scan ของ scan นี้
    closed_tracks: pd.DataFrame | None = None    # track ที่ปิดใน scan นี้ (ใช้ทำ censor ตอนสร้าง label)
    timings_ms: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


class ConvRainPipeline:
    def __init__(self, cfg: dict, onset_model: HazardModel | None = None,
                 peak_model: HazardModel | None = None, stop_model: HazardModel | None = None,
                 rain_lut: RainRateLUT | None = None, collect_lut_pairs: bool = False):
        self.cfg = cfg
        self.onset_model = onset_model
        self.peak_model = peak_model
        self.stop_model = stop_model
        self.lut = rain_lut or RainRateLUT()   # ยังไม่ calibrate = ใช้ prior auto-estimator
        self.motion = MotionEstimator(cfg)
        self.tracker = Tracker(cfg)
        self.prev_ir: np.ndarray | None = None
        self.prev_time: datetime | None = None
        self.collect_lut_pairs = collect_lut_pairs
        self.lut_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    @property
    def model_version(self) -> str:
        parts = [m.version for m in (self.onset_model, self.peak_model, self.stop_model) if m is not None]
        return "+".join(parts) if parts else self.cfg["output"]["model_version"]

    # =================================================================
    # ฟังก์ชันหลัก: ประมวลผล 1 scan
    # =================================================================
    def step(self, scan: Scan, radar_frames: list[tuple[datetime, np.ndarray]] | None = None,
             radar_rainrate: np.ndarray | None = None) -> ScanResult:
        """
        scan           : ข้อมูลดาวเทียม 1 scan (ขั้น 0)
        radar_frames   : (ทางเลือก, เฉพาะตอนสร้าง label) เรดาร์ dBZ บน grid ดาวเทียม
                         ทุก frame ในช่วง (scan ก่อน, scan นี้]
        radar_rainrate : (ทางเลือก) rain rate ของเรดาร์ ณ เวลา scan ใช้เก็บคู่ข้อมูลทำ LUT
        """
        cfg = self.cfg
        tm: dict[str, float] = {}
        t0 = _time.perf_counter()

        # ---------------- [ขั้น 0] geometry + QC ----------------
        prepare_geometry(scan, cfg)
        quality_check(scan)
        if not scan.quality_ok:
            # scan เสีย: ข้ามไปทั้ง scan tracker จะขยาย Δt อัตโนมัติใน scan ถัดไป
            return ScanResult(time=scan.time, skipped=True, errors=["scan ไม่ผ่าน QC"])
        dt_min = float(cfg["satellite"]["scan_interval_min"])
        if self.prev_time is not None:
            dt_min = (scan.time - self.prev_time).total_seconds() / 60.0
            if dt_min > cfg["ingest"]["max_gap_min"] or dt_min <= 0:
                # ข้อมูลขาดนานเกินไป: ปิด track ทั้งหมดแล้วเริ่มใหม่
                self.tracker.reset()
                self.motion.reset()
                self.prev_ir = None
                dt_min = float(cfg["satellite"]["scan_interval_min"])
        tm["preprocess"] = _time.perf_counter() - t0

        # ---------------- [ขั้น 1] detection ----------------
        t = _time.perf_counter()
        det = detect_objects(scan, cfg)
        tm["detect"] = _time.perf_counter() - t

        # ---------------- [ขั้น 2] motion field ----------------
        t = _time.perf_counter()
        motion = None
        if self.prev_ir is not None:
            pixel_km = float(np.sqrt(np.nanmean(scan.pixel_area_km2)))
            motion = self.motion.estimate(self.prev_ir, scan.ir, dt_min, pixel_km)
        tm["motion"] = _time.perf_counter() - t

        # ---------------- [ขั้น 3] tracking ----------------
        t = _time.perf_counter()
        upd = self.tracker.update(det, motion, scan.time, dt_min)
        tm["track"] = _time.perf_counter() - t

        # ---------------- [ขั้น 4] features ----------------
        t = _time.perf_counter()
        feats = [compute_features(tr, scan, motion, cfg) for tr in upd.observed]
        for tr in upd.observed:
            del tr.history[:-HISTORY_KEEP]
        fdf = pd.DataFrame(feats)
        tm["features"] = _time.perf_counter() - t

        # ---------------- [ขั้น 5] prefilter + onset model ----------------
        t = _time.perf_counter()
        onset_pred = self._predict_onset(upd.observed, fdf)
        tm["onset_model"] = _time.perf_counter() - t

        # ---------------- [ขั้น 6] parallax, rain, lifecycle, peak/stop ----------------
        t = _time.perf_counter()
        records, footprints, extra_rows = self._stage6(scan, upd.observed, feats, onset_pred, radar_frames,
                                                       radar_rainrate)
        # track ที่ปิดใน scan นี้ → ส่งออกเป็น status dissipated
        closed_rows = []
        for tr in upd.closed:
            closed_rows.append({"object_id": tr.track_id, "parent_id": tr.parent_id, "birth_time": tr.birth_time,
                                "end_time": tr.last_time, "end_reason": tr.end_reason,
                                "merged_into": tr.merged_into, "n_scans": tr.n_scans})
            if tr.history:
                last = tr.history[-1]
                px = {"lat": last.get("lat_pc", last["lat"]), "lon": last.get("lon_pc", last["lon"]),
                      "cth_km": last.get("cth_km")}
                records.append(object_record(tr, last, px, None, None, None, cfg["parallax"]["enabled"]))
        tm["stage6"] = _time.perf_counter() - t

        # ---------------- Output + validation ----------------
        doc = scan_document(scan.time, records, self.model_version)
        errors = validate_output(doc, cfg)
        ts = fdf.copy()
        if len(ts):
            ts = ts.merge(pd.DataFrame(extra_rows), on="object_id", how="left")
            ts.insert(0, "scan_time", scan.time)

        self.prev_ir, self.prev_time = scan.ir.copy(), scan.time
        tm["total"] = _time.perf_counter() - t0
        return ScanResult(time=scan.time, document=doc, footprints=footprints, timeseries=ts,
                          closed_tracks=pd.DataFrame(closed_rows) if closed_rows else None,
                          timings_ms={k: round(v * 1000, 2) for k, v in tm.items()}, errors=errors)

    # -----------------------------------------------------------------
    # [ขั้น 5] prefilter + hazard model
    # -----------------------------------------------------------------
    def _predict_onset(self, tracks: list[Track], fdf: pd.DataFrame) -> dict[str, dict]:
        """
        [8.1] ส่งเข้าโมเดลเฉพาะ track ที่ยัง developing และผ่านเงื่อนไขหลวม ๆ
        [8.2] ได้ P30, P60, ETA ± ช่วง แปลงเวลาเป็น datetime
        """
        oc = self.cfg["onset"]
        if self.onset_model is None or not len(fdf):
            return {}
        ok = np.array([tr.status == "developing" for tr in tracks]) \
            & (fdf["min_bt"].values < oc["prefilter_max_min_bt"]) \
            & (fdf["n_scans"].values >= oc["prefilter_min_age_scans"])
        if not ok.any():
            return {}
        rows = fdf[ok]
        pred = self.onset_model.predict(rows, quantiles=tuple(oc["eta_quantiles"]))
        out = {}
        for tr, (_, r) in zip([tr for tr, k in zip(tracks, ok) if k], pred.iterrows()):
            now = tr.last_time

            def at(m):
                return now + timedelta(minutes=float(m)) if np.isfinite(m) else None

            out[tr.track_id] = {"p30": r["p30"], "p60": r["p60"], "eta_median": at(r["eta_median_min"]),
                                "eta_lo": at(r["eta_lo_min"]), "eta_hi": at(r["eta_hi_min"]),
                                "eta_median_min": r["eta_median_min"], "eta_lo_min": r["eta_lo_min"],
                                "eta_hi_min": r["eta_hi_min"]}
            if out[tr.track_id]["eta_median"] is not None:
                tr.expected_onset = out[tr.track_id]["eta_median"]
        return out

    # -----------------------------------------------------------------
    # [ขั้น 6] parallax, rain rate, lifecycle, peak/stop + output record
    # -----------------------------------------------------------------
    def _stage6(self, scan, tracks, feats, onset_pred, radar_frames, radar_rainrate):
        cfg = self.cfg
        warn = cfg["onset"]["warn_threshold"]
        records, footprints, extra = [], {}, []
        stamp = scan_stamp(scan.time)

        for tr, f in zip(tracks, feats):
            obj = tr.obj
            # [9.4] parallax correction ของ object
            px = correct_object(scan, obj.lat, obj.lon, obj.cold_mean_bt, *obj.centroid_rc, cfg)
            f.update({"lat_pc": px["lat"], "lon_pc": px["lon"], "cth_km": px["cth_km"],
                      "parallax_shift_km": px["shift_km"]})

            on = onset_pred.get(tr.track_id)
            # [9.1] rain rate: คำนวณเฉพาะ object ที่มีโอกาส (ลด compute ตามหลักการในหัวข้อ 9)
            need_rain = (tr.status != "developing" or (on is not None and on["p30"] >= warn)
                         or self.onset_model is None or obj.min_bt < 253.0)
            rain = object_rain(tr, scan, self.lut, cfg) if need_rain else \
                {"rain_rate_max_mmh": 0.0, "rain_rate_mean_mmh": 0.0, "rain_area_km2": 0.0}

            # [9.2–9.3] lifecycle
            status = update_status(tr, f, rain, scan.time, cfg, onset_eta=tr.expected_onset)

            # peak / stop ETA: ใช้ hazard model ถ้ามี ไม่มีใช้ fallback
            peak_eta = stop_eta = None
            one = pd.DataFrame([f])
            if status == "raining" and self.peak_model is not None:
                m = self.peak_model.predict(one)["eta_median_min"].iloc[0]
                peak_eta = scan.time + timedelta(minutes=float(m)) if np.isfinite(m) else None
            if status in ("raining", "peak", "decaying"):
                if self.stop_model is not None:
                    m = self.stop_model.predict(one)["eta_median_min"].iloc[0]
                    stop_eta = scan.time + timedelta(minutes=float(m)) if np.isfinite(m) else None
                elif status == "decaying":
                    stop_eta = stop_eta_fallback(f, self.lut, scan.time, cfg)

            intensity = None
            if status != "developing":
                intensity = {"rain_rate": rain["rain_rate_max_mmh"], "peak_eta": peak_eta, "stop_eta": stop_eta}

            # footprint polygon (หลังแก้ parallax)
            ref = None
            if cfg["output"]["write_footprints"]:
                ring = footprint_polygon(obj.rows, obj.cols, scan.lat, scan.lon, px["drow"], px["dcol"])
                if ring:
                    footprints[tr.track_id] = ring
                    ref = f"footprints_{stamp}.geojson#{tr.track_id}"

            records.append(object_record(tr, f, px, on if status == "developing" else None, intensity, ref,
                                         cfg["parallax"]["enabled"]))

            # แถวเสริมของตาราง object-scan
            row = {"object_id": tr.track_id, "status": status, **rain,
                   "lat_pc": px["lat"], "lon_pc": px["lon"], "cth_km": px["cth_km"],
                   "p30": on["p30"] if on else np.nan, "p60": on["p60"] if on else np.nan,
                   "eta_median_min": on["eta_median_min"] if on else np.nan,
                   "eta_lo_min": on["eta_lo_min"] if on else np.nan,
                   "eta_hi_min": on["eta_hi_min"] if on else np.nan,
                   "warning": bool(on is not None and on["p30"] >= warn)}

            # label จากเรดาร์ (เฉพาะตอน replay เพื่อ train, หัวข้อ 10.2)
            if radar_frames is not None:
                row.update(object_radar_stats(obj.rows, obj.cols, px["drow"], px["dcol"], radar_frames,
                                              cfg["labels"]["onset_dbz"], cfg["labels"]["buffer_px"]))
            if self.collect_lut_pairs and radar_rainrate is not None:
                self.lut_pairs.append(lut_pairs(scan.ir, scan.btd_wv, obj.rows, obj.cols,
                                                px["drow"], px["dcol"], radar_rainrate))
            extra.append(row)
        return records, footprints, extra

    # =================================================================
    # ปิดการ replay: track ที่ยังเปิดอยู่ถือว่า censored (end_of_data)
    # =================================================================
    def finalize(self) -> pd.DataFrame:
        rows = [{"object_id": t.track_id, "parent_id": t.parent_id, "birth_time": t.birth_time,
                 "end_time": t.last_time, "end_reason": "end_of_data", "merged_into": None,
                 "n_scans": t.n_scans} for t in self.tracker.tracks.values()]
        return pd.DataFrame(rows)

    # =================================================================
    # บันทึก / กู้คืน state (ใช้ตอน real-time เพื่อ restart ได้โดยไม่เสีย track, หัวข้อ 17 Phase 7)
    # =================================================================
    def save_state(self, path: str | Path) -> None:
        state = {"tracker": self.tracker, "motion_prev": self.motion.prev,
                 "prev_ir": self.prev_ir, "prev_time": self.prev_time}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.tracker = state["tracker"]
        self.motion.prev = state["motion_prev"]
        self.prev_ir, self.prev_time = state["prev_ir"], state["prev_time"]

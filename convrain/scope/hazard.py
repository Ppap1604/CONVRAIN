"""
LightHazard — แบบจำลองอัตราเกิดเหตุการณ์แบบช่วงเวลา (discrete-time hazard) ที่เบา (8–11 ค่าคงที่ + นาฬิกา Weibull)

แบ่งอนาคตเป็นช่วง 10 นาที k = 1 … 18 (3 ชม.):
    h_k = P(เกิดในช่วง k | ยังไม่เกิด) = logistic( a0 + a1·ln k + Σ β_j·z_j + Σ γ_j·z_j·ln k )
    F(10k นาที) = 1 − Π_{i ≤ k} (1 − h_i)        → ควอนไทล์ = เวลาที่ F ถึง q (เทียบเชิงเส้นในช่วง)
  z     ตัวแปรที่รู้ ณ เวลาออก t (มาตรฐานด้วยค่าเฉลี่ย / SD ของข้อมูลปรับเทียบ, ค่าหาย → มัธยฐาน)
        clock_prior = ln(เวลาที่นาฬิกา Weibull ทายว่าเหลือ + 10 นาที) = นาฬิกาวงจรชีวิตเมฆเป็นจุดตั้งต้น
  γ     ผลที่เปลี่ยนตามระยะ k: สัญญาณของคอลัมน์อากาศมีผลแรงเมื่อเหตุการณ์ใกล้ และจางลงเมื่อไกล
เกิน 3 ชม. ใช้หางของนาฬิกา Weibull (วงจรชีวิตเมฆ)

ตัวแปรของแต่ละเหตุการณ์ (ฟิสิกส์ต่างกัน) อยู่ใน EVENT_FORMS
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .clock import QCOLS, QS, FixedClock, WeibullClock

K_STEPS = 18
STEP_MIN = 10.0

RAIN_MAP = ["rain_now", "rain_near", "rain_arrival"]
EVENT_FORMS = {
    # ฝนเริ่ม: คอลัมน์เหนือจุดหนาขึ้น (BTD ลด) และเย็นลง → ยอดเมฆลึก / แกนเข้ามาเหนือจุด
    #          + แผนที่ฝนตอนนี้: ฝนอยู่ใกล้แค่ไหน และฝนฝั่งต้นลมจะมาถึงในกี่นาที
    "onset": {"z": ["clock_prior", "st_btd", "d_b13_10", "d_btd_10"] + RAIN_MAP, "tv": ["st_btd", "d_b13_10"]},
    # ฝนหนักสุด: ยอดเมฆลึกที่สุดอยู่ใกล้จุด (กระแสอากาศขึ้นแรงสุดอยู่เหนือจุด) + แผนที่ฝนตอนนี้
    "peak": {"z": ["clock_prior", "st_btd", "d_b13_10", "depth_6"] + RAIN_MAP, "tv": ["st_btd", "depth_6"]},
    # ฝนหยุด: คอลัมน์อุ่นขึ้นและบางลง (แกนผ่านไป / สลาย)
    "stop": {"z": ["clock_prior", "st_btd", "warm_20", "d_btd_10"], "tv": ["warm_20", "d_btd_10"]},
}


class _Logit:
    """L2 logistic (คอลัมน์ 1 อยู่ใน X): Σ logloss + ½‖w‖² / C"""

    def __init__(self, C=1.0, w=None):
        self.C, self.w = C, (None if w is None else np.asarray(w, dtype=float))

    def fit(self, X, y):
        def f(w):
            z = X @ w
            return (np.sum(np.logaddexp(0, z) - y * z) + 0.5 * w @ w / self.C,
                    X.T @ (expit(z) - y) + w / self.C)
        self.w = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def p(self, X):
        return expit(X @ self.w)


class LightHazard:
    def __init__(self, prior: WeibullClock, z: list, tv: list):
        if not set(tv) <= set(z):
            raise ValueError("ตัวแปร tv ต้องอยู่ใน z")
        self.prior, self.z, self.tv = prior, list(z), list(tv)

    @classmethod
    def for_event(cls, prior: WeibullClock, event: str) -> "LightHazard":
        return cls(prior, EVENT_FORMS[event]["z"], EVENT_FORMS[event]["tv"])

    # ---------- ตัวแปร
    def _X(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = rows.reindex(columns=[c for c in self.z if c != "clock_prior"]).astype(float)
        if "clock_prior" in self.z:
            q50 = self.prior.quantiles(rows).q50.to_numpy()
            X.insert(self.z.index("clock_prior"), "clock_prior", np.log(np.maximum(q50 - rows.t.to_numpy(float), 0) + 10))
        return X[self.z]

    def _zscore(self, X: pd.DataFrame, fit=False) -> np.ndarray:
        if fit:
            self.med = X.median().fillna(0.0)                       # ตัวแปรที่ไม่มีค่าเลย → 0 (ไม่มีผล)
            self.mu = X.fillna(self.med).mean()
            self.sd = X.fillna(self.med).std().replace(0, 1).fillna(1)
        return ((X.fillna(self.med) - self.mu) / self.sd).to_numpy()

    def _design(self, Z, k):
        lk = np.log(k)[:, None]
        idx = [self.z.index(c) for c in self.tv]
        return np.hstack([np.ones((len(k), 1)), lk, Z, Z[:, idx] * lk])

    # ---------- ฟิต / ทำนาย
    def fit(self, rows: pd.DataFrame, event: str, C: float = 1.0) -> "LightHazard":
        """rows: แถวที่ t < เวลาเหตุการณ์; แตกเป็น (แถว, ช่วง k) จนถึงช่วงที่เกิด หรือ 18 ช่วง"""
        Z = self._zscore(self._X(rows), fit=True)
        kstar = np.ceil((rows[event] - rows.t).to_numpy(float) / STEP_MIN).astype(int)
        n_k = np.minimum(kstar, K_STEPS)
        ii = np.repeat(np.arange(len(rows)), n_k)
        kk = np.concatenate([np.arange(1, m + 1) for m in n_k]) if len(n_k) else np.array([], int)
        self.m = _Logit(C).fit(self._design(Z[ii], kk), (kk == kstar[ii]).astype(float))
        return self

    def cdf_edges(self, rows: pd.DataFrame) -> np.ndarray:
        """F ที่ขอบช่วง 0, 10, …, 180 นาทีหลัง t (แถว × 19)"""
        Z = self._zscore(self._X(rows))
        n = len(rows)
        H = np.column_stack([self.m.p(self._design(Z, np.full(n, k))) for k in range(1, K_STEPS + 1)])
        return np.hstack([np.zeros((n, 1)), 1 - np.cumprod(1 - H, axis=1)])

    def quantiles(self, rows: pd.DataFrame) -> pd.DataFrame:
        Fc, n = self.cdf_edges(rows), len(rows)
        t = rows.t.to_numpy(float)
        tail = self.prior.quantiles(rows)
        out = {}
        for q, c in zip(QS, QCOLS):
            j = (Fc < q).sum(axis=1)                                  # ขอบแรกที่ F ≥ q
            jj = np.clip(j, 1, K_STEPS)
            f0, f1 = Fc[np.arange(n), jj - 1], Fc[np.arange(n), jj]
            v = t + STEP_MIN * (jj - 1) + STEP_MIN * (q - f0) / np.maximum(f1 - f0, 1e-9)
            out[c] = np.where(j > K_STEPS, np.maximum(t + STEP_MIN * K_STEPS, tail[c].to_numpy()), v)
        return pd.DataFrame(out, index=rows.index)

    def prob_within(self, rows: pd.DataFrame, minutes: float) -> np.ndarray:
        Fc = self.cdf_edges(rows)
        x = np.clip(minutes / STEP_MIN, 0, K_STEPS)
        i = int(np.floor(x))
        return Fc[:, i] if i == K_STEPS else Fc[:, i] + (x - i) * (Fc[:, i + 1] - Fc[:, i])

    # ---------- ค่าคงที่ ↔ dict
    def to_dict(self) -> dict:
        return {"model": "hazard", "z": self.z, "tv": self.tv, "w": self.m.w.tolist(), "med": self.med.tolist(),
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "prior": self.prior.to_dict()}

    @classmethod
    def from_dict(cls, P: dict) -> "LightHazard":
        h = cls(WeibullClock.from_dict(P["prior"]), P["z"], P["tv"])
        h.med, h.mu, h.sd = (pd.Series(P[k], index=P["z"]) for k in ("med", "mu", "sd"))
        h.m = _Logit(w=P["w"])
        return h


class Switch:
    """ใช้โมเดลใกล้ (near) เมื่อมันเองทายว่าเหตุการณ์อยู่ห่าง ≤ minutes นาที ไม่งั้นใช้โมเดลไกล (far)
    ตัดสินจากคำทำนายของตัวเอง ไม่ใช่เวลาจริง จึงใช้ได้ตอนรันจริง"""

    def __init__(self, near, far, minutes: float = 30.0):
        self.near, self.far, self.minutes = near, far, minutes

    def to_dict(self) -> dict:
        return {"model": "switch", "near": self.near.to_dict(), "far": self.far.to_dict(), "minutes": self.minutes}

    def _use_near(self, rows) -> np.ndarray:
        return (self.near.quantiles(rows).q50 - rows.t).to_numpy() <= self.minutes

    def quantiles(self, rows: pd.DataFrame) -> pd.DataFrame:
        qn, qf = self.near.quantiles(rows), self.far.quantiles(rows)
        use = (qn.q50 - rows.t).to_numpy() <= self.minutes
        return pd.DataFrame(np.where(use[:, None], qn.to_numpy(), qf.to_numpy()), index=rows.index, columns=QCOLS)

    def prob_within(self, rows: pd.DataFrame, minutes: float) -> np.ndarray:
        return np.where(self._use_near(rows), self.near.prob_within(rows, minutes), self.far.prob_within(rows, minutes))


def from_dict(P: dict):
    """สร้างโมเดลจาก dict ค่าคงที่ (ชนิดดูจาก P["model"])"""
    kind = P["model"]
    if kind == "weibull":
        return WeibullClock.from_dict(P)
    if kind == "hazard":
        return LightHazard.from_dict(P)
    if kind == "switch":
        return Switch(from_dict(P["near"]), from_dict(P["far"]), P.get("minutes", 30.0))
    if kind == "F":
        return FixedClock(P["offsets"])
    raise ValueError(f"ไม่รู้จักโมเดล {kind}")

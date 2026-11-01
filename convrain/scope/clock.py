"""
นาฬิกาเวลาเหตุการณ์ (ฝนเริ่ม / หนักสุด / หยุด) ที่จุดบนพื้น นับจาก t0 ของสายตระกูลเมฆ

ตารางเข้า: 1 แถว = (สายตระกูล, จุด, เวลาออก t); เวลาเป็นนาที (หน่วยเดียวกันทั้งตาราง)
  t    เวลาออกคำทำนาย                    b    t0 ของสายตระกูล (นาฬิกาเริ่มเมื่อแกนลึกปรากฏ)
  d2   ระยะ Mahalanobis² จากจุดถึงวงรีแกน (ตัดที่ 16)      btd  B13 − B15 ของพิกเซลเหนือจุด (= st_btd)
ผล: ควอนไทล์ q10, q25, q50, q75, q90 ของเวลาเหตุการณ์ (นาที)

FixedClock (นาฬิกาตายตัว): เหตุการณ์ = t0 + ค่าคงที่ ไม่ก่อน t — ใช้เป็นตัวเทียบตอนประเมิน
WeibullClock:
    u = เวลาเหตุการณ์ − t0 + 60 นาที  ~  Weibull(k, λ),   ln λ = a + b1·z(d2) + b2·z(btd)
    เมฆอายุ s = t − t0 + 60 แล้วยังไม่เกิด → ควอนไทล์แบบมีเงื่อนไข  u_q = λ·((s/λ)^k − ln(1 − q))^(1/k)
    k > 1 = อัตราเกิดเพิ่มตามอายุ (เมฆ "แก่" ลง) → ยิ่งเมฆแก่ ยิ่งทายว่าใกล้; 4 ค่าคงที่ + ตัวขยายช่วง c
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

QS = (0.10, 0.25, 0.50, 0.75, 0.90)
QCOLS = [f"q{int(q * 100)}" for q in QS]
SHIFT = 60.0                     # ให้เหตุการณ์ที่เกิดก่อน t0 (ไม่เกิน 60 นาที) ยังมี u > 0


def widen(Q: pd.DataFrame, c: float) -> pd.DataFrame:
    """ขยาย / หดช่วงรอบมัธยฐานด้วยตัวคูณ c (มัธยฐานไม่เปลี่ยน) — ปรับให้ช่วง 50 / 80 % ครอบคลุมตามชื่อ"""
    return pd.DataFrame({k: Q.q50 + c * (Q[k] - Q.q50) for k in QCOLS}, index=Q.index)


class FixedClock:
    def __init__(self, offsets: dict):
        self.offsets = offsets                       # q10 … q90 → นาทีหลัง t0

    @classmethod
    def fit(cls, rows: pd.DataFrame, event: str) -> "FixedClock":
        p = rows.drop_duplicates(["cell_id", "station", "b"])
        r = (p[event] - p.b).to_numpy(float)
        return cls({c: float(np.quantile(r, q)) for q, c in zip(QS, QCOLS)})

    def to_dict(self) -> dict:
        return {"model": "F", "offsets": self.offsets}

    def quantiles(self, rows: pd.DataFrame) -> pd.DataFrame:
        t, b = rows.t.to_numpy(float), rows.b.to_numpy(float)
        return pd.DataFrame({c: np.maximum(t, b + v) for c, v in self.offsets.items()}, index=rows.index)


def _nll(theta, u, s, Z):
    k = np.exp(theta[0])
    lam = np.exp(theta[1] + Z @ theta[2:])
    zu, zs = u / lam, s / lam
    return -np.sum(np.log(k) - np.log(lam) + (k - 1) * np.log(zu) - zu ** k + zs ** k)


class WeibullClock:
    covs = ("d2", "btd")

    def __init__(self, theta, mu, sd, btd_median: float, c: float = 1.0):
        self.theta, self.mu, self.sd = (np.asarray(v, dtype=float) for v in (theta, mu, sd))
        self.btd_median, self.c = float(btd_median), float(c)

    @classmethod
    def from_dict(cls, P: dict) -> "WeibullClock":
        return cls(P["theta"], P["mu"], P["sd"], P["btd_median"], P.get("c", 1.0))

    def to_dict(self) -> dict:
        return {"model": "weibull", "theta": self.theta.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "btd_median": self.btd_median, "c": self.c}

    def _z(self, rows: pd.DataFrame) -> np.ndarray:
        X = np.c_[rows.d2.to_numpy(float), rows.btd.fillna(self.btd_median).to_numpy(float)]
        return (X - self.mu) / self.sd

    def _k_lam(self, rows):
        return np.exp(self.theta[0]), np.exp(self.theta[1] + self._z(rows) @ self.theta[2:])

    @classmethod
    def fit(cls, rows: pd.DataFrame, event: str, c: float = 1.0) -> "WeibullClock":
        """MLE บนแถวที่ยังไม่เกิดเหตุการณ์ (t < event); แต่ละแถวมีเงื่อนไขว่าเมฆอยู่รอดถึงอายุ s (left truncation)"""
        u = (rows[event] - rows.b + SHIFT).to_numpy(float)
        s = (rows.t - rows.b + SHIFT).to_numpy(float)
        med = float(np.nanmedian(rows.btd))
        X = np.c_[rows.d2.to_numpy(float), rows.btd.fillna(med).to_numpy(float)]
        mu, sd = X.mean(0), X.std(0)
        th0 = np.r_[np.log(1.5), np.log(np.median(u)), np.zeros(2)]
        th = minimize(_nll, th0, args=(u, s, (X - mu) / sd), method="L-BFGS-B").x
        return cls(th, mu, sd, med, c)

    def quantiles(self, rows: pd.DataFrame) -> pd.DataFrame:
        k, lam = self._k_lam(rows)
        s = (rows.t - rows.b + SHIFT).to_numpy(float)
        b = rows.b.to_numpy(float)
        Q = pd.DataFrame({c: b - SHIFT + lam * ((s / lam) ** k - np.log(1 - q)) ** (1 / k)
                          for q, c in zip(QS, QCOLS)}, index=rows.index)
        return widen(Q, self.c)

    def prob_within(self, rows: pd.DataFrame, minutes: float) -> np.ndarray:
        """P(เหตุการณ์เกิดภายใน minutes นาทีหลัง t | ยังไม่เกิดถึง t)"""
        k, lam = self._k_lam(rows)
        s = (rows.t - rows.b + SHIFT).to_numpy(float)
        return 1.0 - np.exp((s / lam) ** k - ((s + minutes) / lam) ** k)


def from_dict(P: dict):
    """สร้างโมเดลเวลาจาก dict ค่าคงที่ (ชนิดดูจาก P["model"])"""
    kind = P["model"]
    if kind == "weibull":
        return WeibullClock.from_dict(P)
    if kind == "F":
        return FixedClock(P["offsets"])
    raise ValueError(f"ไม่รู้จักโมเดล {kind}")


class Logit:
    """L2 logistic (คอลัมน์ 1 อยู่ใน X): Σ logloss + ½‖w‖² / C"""

    def __init__(self, C=1.0, w=None):
        self.C, self.w = C, (None if w is None else np.asarray(w, dtype=float))

    def fit(self, X, y):
        from scipy.special import expit

        def f(w):
            z = X @ w
            return (np.sum(np.logaddexp(0, z) - y * z) + 0.5 * w @ w / self.C,
                    X.T @ (expit(z) - y) + w / self.C)
        self.w = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
        return self

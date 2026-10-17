"""
ค่าคงที่ที่ติดมากับโปรแกรม (โฟลเดอร์ constants/) — เลือกชุดตามภูมิภาคด้วย timing.constants ใน config

    timing_<ภูมิภาค>.json    โมเดลเวลาฝนเริ่ม / หนักสุด / หยุด ของเมฆที่ติดตาม
    rainmap_<ภูมิภาค>.json   แผนที่ฝนตอนนี้ (โหมดกลางวัน / กลางคืน)
    outlook_<ภูมิภาค>.json   คำเตือน 3 ระดับ + เวลาฝนเริ่ม / หยุดพร้อมความมั่นใจ ของทุกจุด

  thailand  ปรับเทียบด้วยเรดาร์ตรวจอากาศของไทย (ภาพทุก 15 นาที) เหนือภาคเหนือตอนล่าง / ภาคกลางตอนบน
  japan     ปรับเทียบด้วยมาตรวัดฝน 10 นาทีของญี่ปุ่น (AMeDAS) เหนือภาคคันโต / คันไซ ฤดูร้อน
ข้อมูลฝนใช้ปรับเทียบเท่านั้น ตอนรันจริงโปรแกรมอ่านแค่ภาพดาวเทียม

    onset  hazard รายช่วง: คอลัมน์อากาศเหนือจุดหนาขึ้น / เย็นลง + แผนที่ฝนตอนนี้รอบจุดและฝนที่กำลังเคลื่อนเข้ามา
    peak   hazard รายช่วง: ยอดเมฆลึกสุดใกล้จุด + แผนที่ฝนตอนนี้
    stop   ตัวสลับ: hazard รายช่วง (คอลัมน์อุ่นขึ้น / บางลง) เมื่อมันทายว่าเหลือ ≤ 30 นาที ไม่งั้นนาฬิกา Weibull
"""
from __future__ import annotations

import json
from pathlib import Path

CONSTANTS = Path(__file__).resolve().parent / "constants"
REGIONS = ("thailand", "japan")
DEFAULT_REGION = "thailand"


def constants_path(kind: str, region: str = DEFAULT_REGION) -> Path:
    p = CONSTANTS / f"{kind}_{region}.json"
    if not p.exists():
        raise FileNotFoundError(f"ไม่มีค่าคงที่ {p.name} (ภูมิภาคที่มี: {', '.join(REGIONS)})")
    return p


def load(kind: str, region: str = DEFAULT_REGION) -> dict:
    return json.loads(constants_path(kind, region).read_text(encoding="utf-8"))


EVENT_MODELS = load("timing", DEFAULT_REGION)

"""
ส่วนทำนายเวลาฝนเริ่ม / หนักสุด / หยุด (ขั้น 5 ของ pipeline) — อ่าน README.md ในโฟลเดอร์นี้

    timing.py     ScopeTiming: ตัวที่ pipeline เรียกทุก scan
    lineage.py    สายตระกูลเมฆ + นาฬิกา t0 จาก track ของขั้น 3
    columns.py    ค่าคอลัมน์อากาศเหนือจุดบนพื้น
    clock.py      นาฬิกา: WeibullClock (โมเดลเวลาฝน), FixedClock (ตัวเทียบ), from_dict
    calibrate.py  ปรับเทียบค่าคงที่ใหม่จากข้อมูลฝนที่จุด (เรดาร์ / มาตรวัดฝน) — ใช้ offline เท่านั้น
    params.py     ค่าคงที่ที่ติดมากับโค้ด
    geometry.py   parallax, ระยะ, วงรีแกน, พิกเซลเหนือจุด
"""
from .timing import EVENTS, ScopeTiming, TimingResult, load_event_models  # noqa: F401

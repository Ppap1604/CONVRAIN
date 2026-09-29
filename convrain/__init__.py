"""
convrain: ระบบตรวจจับและพยากรณ์เวลาเริ่มตกของฝน convective จากดาวเทียม Himawari อย่างเดียว

โครงสร้างโมดูลตาม workflow (เอกสาร หัวข้อ 2 และ 18.1):
    ingest     ขั้น 0  รับข้อมูล, crop, BT/BTD
    detect     ขั้น 1  multi-threshold detection
    motion     ขั้น 2  optical flow (motion field เดียว)
    track      ขั้น 3  online tracking + Kalman filter
    features   ขั้น 4  interest fields ต่อ object
    models     ขั้น 5  hazard model สำหรับ onset / peak / stop + baseline
    intensity  ขั้น 6  rain rate LUT, lifecycle status
    parallax   ขั้น 6  parallax correction
    output     หัวข้อ 9.5  JSON/GeoJSON/Parquet + validation
    labels     หัวข้อ 10  สร้าง label จากเรดาร์
    evaluate   หัวข้อ 10.4 metric และ bootstrap
    pipeline   รวมทุกขั้นเป็น step() ต่อ scan
"""

__version__ = "0.1.0"

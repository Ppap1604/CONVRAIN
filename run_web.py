"""
สคริปต์รัน Convrain Web Platform
รัน:
    python run_web.py
    python run_web.py --port 8000 --data-dir runs/2025/realtime
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# เพิ่ม root directory ลงใน sys.path
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# ปรับ stdout/stderr ให้รองรับ UTF-8 บน Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import uvicorn
from web.app import app
import web.app as web_app
from web.data_loader import ConvrainDataLoader


def main():
    parser = argparse.ArgumentParser(description="เปิดเว็บเซิร์ฟเวอร์ Convrain Nowcast Web Platform")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--data-dir", default=None, help="โฟลเดอร์ output ของ convrain (เช่น runs/2025/realtime หรือ nowcast)")
    args = parser.parse_args()

    # ตั้งค่า data loader
    if args.data_dir:
        web_app.loader = ConvrainDataLoader(args.data_dir)
    else:
        web_app.loader = ConvrainDataLoader()

    print("=" * 65)
    print("⛈️  CONVRAIN NOWCAST WEB PLATFORM")
    print("=" * 65)
    print(f"📡 ข้อมูล Output กำลังอ่านจาก: {web_app.loader.base_dir.resolve()}")
    print(f"📊 จำนวนรอบสแกนที่พบ: {len(web_app.loader.list_scans())} scans")
    print(f"🌐 เปิดใช้งานเว็บไซต์ที่: http://localhost:{args.port}")
    print(f"📍 รองรับ GPS, ปักหมุดบนแผนที่, และพยากรณ์เวลาฝนเริ่มตก (ETA Window)")
    print("=" * 65)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

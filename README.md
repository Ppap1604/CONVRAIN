# convrain

โค้ดของระบบตรวจจับและพยากรณ์เวลาเริ่มตกของฝน convective จากดาวเทียม Himawari อย่างเดียว
เขียนตามเอกสาร **Satellite-only Convective Rain Onset Workflow** (ขั้น 0–6 = หัวข้อ 3–9, แผนพัฒนา = หัวข้อ 13–18)

> **สถานะ:** โค้ดทุกส่วนทดสอบแล้วกับ **ข้อมูลจำลอง** เท่านั้น (unit test 23 ข้อ + demo ตั้งแต่ต้นจนจบ)
> ส่วนอ่านไฟล์ HSD จริงด้วย Satpy (`read_ahi_hsd`) ยังไม่ได้ทดสอบกับไฟล์จริง
> ค่า threshold ทั้งหมดใน `config/default.yaml` เป็นค่าตั้งต้น ต้องจูนใหม่ด้วยข้อมูลไทย

---

## 1. โครงสร้าง

```
convrain/
├── config/default.yaml        ค่าตั้งค่าทุกขั้น (แต่ละการทดลองเขียนไฟล์ทับเฉพาะค่าที่ต่าง)
├── convrain/
│   ├── ingest.py              ขั้น 0  อ่าน AHI, crop, BT/BTD, geometry, QC        (หัวข้อ 3)
│   ├── detect.py              ขั้น 1  multi-threshold + watershed                (หัวข้อ 4)
│   ├── motion.py              ขั้น 2  DIS optical flow (motion field เดียว)      (หัวข้อ 5)
│   ├── track.py               ขั้น 3  online tracking + Kalman filter            (หัวข้อ 6)
│   ├── features.py            ขั้น 4  interest fields + แนวโน้ม Lagrangian         (หัวข้อ 7)
│   ├── models/hazard.py       ขั้น 5  discrete-time hazard (onset / peak / stop)  (หัวข้อ 8)
│   ├── models/baseline.py             baseline: MB06, extrapolation, climatology  (หัวข้อ 10.4)
│   ├── intensity.py           ขั้น 6  rain rate LUT + สถานะ lifecycle            (หัวข้อ 9.1–9.3)
│   ├── parallax.py            ขั้น 6  parallax correction                        (หัวข้อ 9.4)
│   ├── output.py                      JSON / GeoJSON / Parquet + validation       (หัวข้อ 9.5)
│   ├── labels.py                      label จากเรดาร์                              (หัวข้อ 10)
│   ├── evaluate.py                    POD/FAR/CSI, timing, Brier, bootstrap        (หัวข้อ 10.4)
│   ├── experiment.py                  replay, แบ่งข้อมูล, train, predict           (หัวข้อ 15–17)
│   ├── pipeline.py                    รวมขั้น 0–6 เป็น step() ต่อ scan
│   └── synthetic.py                   ข้อมูลจำลองสำหรับทดสอบ
├── scripts/                   คำสั่งตาม Phase ของแผนพัฒนา (ดูข้อ 4)
├── notebooks/01_eda_phase3.ipynb   EDA + Gate ของ Phase 3
├── examples/                  ตัวอย่าง output จริงจากข้อมูลจำลอง (JSON, footprints, animation track)
└── tests/                     unit test
```

ในโค้ดทุกไฟล์มีหัวข้อกำกับแบบ `# [ขั้น 3.2] ...` ตรงกับขั้นย่อยในเอกสาร

## 2. ติดตั้ง

```bash
conda env create -f environment.yml && conda activate convrain
# หรือ
python -m venv convrain_env
source convrain_env/bin/activate
pip install -r requirements.txt
```

## 3. ลองรันทันที (ข้อมูลจำลอง ไม่ต้องมีข้อมูลจริง)

```bash
python -m pytest -q tests                       # unit test
python scripts/demo_synthetic.py --out outputs/demo
```

demo ทำครบ Phase 2–7: replay 24 วันจำลอง → สร้าง label → train → ประเมินเทียบ baseline → รันแบบ real-time และเขียน output
จากนั้นส่งออกข้อมูลจำลอง 1 วันเป็นไฟล์ (`outputs/demo/scans`, `outputs/demo/radar`) ไว้ลองสคริปต์ของข้อมูลจริงได้

**ตัวเลขความแม่นยำจากข้อมูลจำลองไม่ได้บอกความแม่นยำกับข้อมูลจริง** ข้อมูลจำลองใช้เพื่อตรวจว่าโค้ดทุกขั้นทำงานร่วมกันได้เท่านั้น

## 4. ขั้นตอนกับข้อมูลจริง (ตามแผนพัฒนา)

| Phase | คำสั่ง | ผลลัพธ์ |
|---|---|---|
| 0 | `python scripts/preprocess_ahi.py --hsd-dir RAW --out-dir data/scans` | NetCDF ที่ crop แล้ว 1 ไฟล์ต่อ scan |
| 0 | regrid เรดาร์ลง grid ดาวเทียมด้วย `convrain.labels.regrid_to_satellite` แล้วบันทึกเป็น `radar_YYYYMMDDTHHMMZ.nc` (ตัวแปร `dbz`) | ไฟล์เรดาร์ต่อ frame |
| 2–3 | `python scripts/replay.py --scans-dir data/scans --radar-dir data/radar --out runs/2025 --collect-lut --no-json` | ตาราง object-scan + tracks + คู่ข้อมูล LUT |
| 2 | `python scripts/plot_tracks.py --scans-dir data/scans --run-dir runs/case01 --gif` | ภาพซ้อน track ไว้ตรวจด้วยตา (ต้อง replay โดยไม่ใส่ `--no-json`) |
| 3 | `python scripts/build_labels.py --run-dir runs/2025` | `labeled.parquet`, `label_check.csv` |
| 3 | เปิด `notebooks/01_eda_phase3.ipynb` | Gate: AUC ของ feature เดี่ยว > 0.6 |
| 5 | `python scripts/build_lut.py --pairs runs/train/lut_pairs.npz --out models/v1/rain_lut.npz` | rain rate LUT (ใช้เฉพาะวันในชุด train) |
| 4–5 | `python scripts/train.py --labeled runs/2025/labeled.parquet --train-end 2025-07-31 --val-end 2025-08-31 --out models/v1` | โมเดล onset/peak/stop + threshold คำเตือน |
| 6 | `python scripts/evaluate.py --labeled runs/2025/labeled.parquet --models-dir models/v1 --test-start 2025-09-01 --out reports/v1` | metric เทียบ baseline + bootstrap CI + แยกกลุ่ม |
| 7 | `python scripts/run_realtime.py --watch-dir INCOMING --out nowcast --models-dir models/v1` | output ทุก 10 นาที + `monitor.csv` + state สำหรับ restart |
| 7 | `python run_web.py --port 8000` | เว็บไซต์ Interactive Nowcast + ระบุพิกัด GPS เพื่อดูเวลาฝนตก |

ทุกสคริปต์รับ `--config my.yaml` ที่ระบุเฉพาะค่าที่ต่างจาก `config/default.yaml`

## 5. Output (หัวข้อ 9.5)

```
nowcast/
├── json/objects_20260925T0720Z.json          เอกสารต่อ scan ตามตัวอย่างในเอกสาร
├── geojson/objects_20260925T0720Z.geojson     FeatureCollection ของจุด (เปิดใน QGIS)
├── geojson/footprints_20260925T0720Z.geojson  polygon ของแต่ละ object (แก้ parallax แล้ว)
└── timeseries/date=2026-09-25/scan_0720.parquet   หนึ่งแถวต่อ object ต่อ scan
```

ทุก scan ผ่านการตรวจด้วย JSON Schema และกฎข้าม field ก่อนเขียน (`output.validate_output`)
`footprint_polygon` อ้างอิงไฟล์ footprints ของ scan เดียวกัน (`footprints_<stamp>.geojson#<object_id>`)

## 6. จุดที่โค้ดต่างจากเอกสาร หรือต้องตัดสินใจเพิ่ม

1. **Hazard model แบบ direct multi-step:** ใช้ feature ณ เวลาพยากรณ์และเพิ่ม feature `k` (ช่วงที่ 1–6) แทนการ extrapolate feature ไปอนาคตด้วย Kalman (หัวข้อ 8.2) เพราะง่ายกว่าและไม่สะสม error จากการ extrapolate
2. **ช่วง ETA กว้างอย่างน้อย ±5 นาที** (`onset.eta_min_halfwidth_min`) เพราะ scan ทุก 10 นาทีทำให้ความแม่นของเวลามีเพดาน (หัวข้อ 12 ข้อ 2)
3. **Kalman filter ใช้ BT เฉลี่ยของ pixel ที่เย็นที่สุด 10%** แทน min BT เดี่ยว เพื่อลดผลของ noise ของ pixel เดียว
4. **Rain rate ก่อน calibrate ใช้ GOES auto-estimator** (Vicente et al. 1998) เป็น prior ซึ่ง calibrate สำหรับสหรัฐฯ ต้องแทนด้วย LUT จากข้อมูลไทย
5. **Threshold ของ baseline MB06** ในไฟล์ `models/baseline.py` ควรตรวจกับตาราง interest field ในเปเปอร์ต้นฉบับก่อนใช้รายงานผล
6. **Z-R สำหรับแปลงเรดาร์เป็น rain rate** ใช้ค่า convective ทั่วไป (a=300, b=1.4) ควรปรับด้วย rain gauge ของไทย
7. **Land mask และความสูงภูมิประเทศ** ต้องเตรียมไฟล์เอง แล้วตั้ง `ingest.static_file` ถ้าไม่ตั้ง feature เหล่านี้จะเป็น NaN (LightGBM รับได้)
8. **Peak และ stop:** ถ้า event ในชุด train น้อยกว่า 10 ครั้ง จะไม่ train โมเดล และใช้กฎ fallback แทน (stop จากการอุ่นขึ้นของยอดเมฆ)

## 7. เวลาประมวลผล

บนข้อมูลจำลอง 160×160 pixel ใช้ ~40–70 ms ต่อ scan บน CPU core เดียว (ดู `timings.csv`)
โดเมนไทยจริง (~0.5 ล้าน pixel) คาดว่าอยู่ในระดับวินาทีต่อ scan ตามงบในหัวข้อ 11 แต่ต้อง benchmark กับข้อมูลจริง

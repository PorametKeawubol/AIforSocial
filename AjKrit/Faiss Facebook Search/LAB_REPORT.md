# รายงานผลทดลอง: FAISS Model Benchmark & Metric Conversion

## การตั้งค่าทดลอง

- Query: `ที่ทำการออฟฟิศอยู่ที่ไหน?`
- Corpus: ข้อความภาษาไทย 5 รายการตามใบงาน
- Models: WangchanBERTa, PhayaThaiBERT และ Multilingual-MPNet
- FAISS indexes: `IndexFlatIP` และ `IndexFlatL2`
- Vector dimension: 768 ทุกโมเดล
- การเตรียม embedding: แปลงเป็น `float32` และทำ L2 normalization ให้ทั้ง corpus และ query
- เครื่องที่รัน: CPU

ข้อควรระวัง: `IndexFlatL2.search()` คืนค่า **squared L2 distance**; โปรแกรมจึง
คำนวณรากที่สองก่อนบันทึกในคอลัมน์ L2 distance.

## ตารางผลลัพธ์

| Model              | Matched text                                                      | Cosine (actual) | L2 distance (actual) | Cosine (calc. from L2) | L2 (calc. from cosine) |
| ------------------ | ----------------------------------------------------------------- | --------------: | -------------------: | ---------------------: | ---------------------: |
| WangchanBERTa      | สำนักงานใหญ่ของคุณตั้งอยู่ที่ไหน? |        0.888745 |             0.471710 |               0.888745 |               0.471710 |
| PhayaThaiBERT      | สำนักงานใหญ่ของคุณตั้งอยู่ที่ไหน? |        0.915454 |             0.411209 |               0.915454 |               0.411208 |
| Multilingual-MPNet | สำนักงานใหญ่ของคุณตั้งอยู่ที่ไหน? |        0.846029 |             0.554926 |               0.846028 |               0.554926 |

ทั้ง `IndexFlatIP` และ `IndexFlatL2` เลือกเอกสาร index 0 เหมือนกันทุกโมเดล
ซึ่งจัดอยู่ใน category `location`.

## คำตอบคำถามวิเคราะห์

1. **ค่า cosine ตรงกับค่าที่คำนวณจาก L2 หรือไม่?**

   ตรงกันภายในความคลาดเคลื่อนแบบ floating point. ค่าคลาดเคลื่อน cosine มากที่สุด
   คือ `2.09e-7` และ L2 มากที่สุดคือ `3.76e-7`. เพราะทุก embedding ถูกทำให้มี
   ความยาวเท่ากับ 1 จึงใช้ความสัมพันธ์
   `cosine = 1 - (L2² / 2)` ได้โดยตรง. ความต่างระดับ `1e-7` เกิดจากการคำนวณ
   `float32` ใน FAISS และการถอดรากที่สอง ไม่ใช่ความต่างเชิงความหมายของผลค้นหา.
2. **โมเดลใดให้ similarity สูงสุดและจับคู่ได้ดีที่สุด?**

   สำหรับ query นี้ **PhayaThaiBERT** สูงสุดที่ `0.915454` และจับคู่ข้อความ
   `สำนักงานใหญ่ของคุณตั้งอยู่ที่ไหน?` ได้ถูกต้อง. WangchanBERTa ตามมาที่
   `0.888745` และ Multilingual-MPNet ที่ `0.846029`; อย่างไรก็ดีทั้งสามโมเดลคืน
   category ที่ถูกต้อง. ข้อสรุปว่า PhayaThaiBERT เหมาะที่สุดจึงใช้ได้เฉพาะชุดทดสอบ
   นี้—หากต้องคัดเลือกใช้งานจริงควรวัดหลาย query ที่มี ground truth.
3. **เหตุใด L2 normalization จึงสำคัญ?**

   เมื่อ `||u|| = ||v|| = 1`, จะได้
   `||u-v||² = ||u||² + ||v||² - 2u·v = 2 - 2cosine`.
   ดังนั้น L2 distance และ cosine มีลำดับผลค้นหาเดียวกันและแปลงกลับกันได้พอดี.
   หากไม่ normalize ความยาวของเวกเตอร์จะมีผลต่อ inner product และ L2 distance;
   สูตรแปลงนี้จึงใช้ไม่ได้โดยตรง.

## ไฟล์หลัก

- `benchmark_faiss.py` — โค้ดทดลองที่รันได้
- `faiss_model_comparison_results.csv` — ผลละเอียดสำหรับเปิดใน spreadsheet
- `faiss_model_comparison_results.json` — ผลเต็มความละเอียดและ metadata

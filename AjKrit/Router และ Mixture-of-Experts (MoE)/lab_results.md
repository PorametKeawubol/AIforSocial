# ผลการทดลอง Router และ Mixture-of-Experts (MoE)

## วิธีรัน

เปิด terminal ในโฟลเดอร์นี้แล้วใช้:

```bash
.venv/bin/python code.py
```

ติดตั้งใหม่ได้ด้วย `python3 -m venv .venv` แล้ว `.venv/bin/python -m pip install -r requirements.txt`
ใน VS Code เลือก Python interpreter เป็น `.venv/bin/python` ของโฟลเดอร์นี้

## Task 1: Gating weights

| กรณี | h_q | Entity | Relation | Subgraph | Query Type |
|---|---|---:|---:|---:|---|
| 1.1 Simple | [0.9, 0.1, 0.1, 0.2] | 0.385057 | 0.338117 | 0.276827 | Simple (1-hop) |
| 1.2 Complex | [0.1, 0.8, 0.9, 0.8] | 0.105380 | 0.168607 | 0.726014 | Complex (Multi-hop) |
| 1.3 Custom | [1.0, 0.0, 0.0, 0.0] | 0.431906 | 0.353615 | 0.214478 | Simple (1-hop) |

Router คำนวณ logits = W @ h_q แล้วใช้ softmax ทำให้น้ำหนักรวมเป็น 1
ตัวอย่าง Simple ได้ logits [0.73, 0.60, 0.40] ส่วน Complex ได้ [0.09, 0.56, 2.02]
เวกเตอร์ Complex ให้น้ำหนักมิติที่ 2–4 สูง ซึ่งสอดคล้องกับแถว Subgraph ของ W จึงเลือก Subgraph มากขึ้น
Custom เน้นมิติแรก จึงให้น้ำหนัก Entity สูงที่สุด
เกณฑ์จำแนก query คือ Subgraph > 0.4 ซึ่งเป็นคนละเกณฑ์กับการเลือก Expert > 0.3

## Task 2: Sparse gating

ส่วนสำคัญใน `process_query`:

```python
if sparse and weight <= 0.3:
    continue
output = retrieve(h_q)
```

ตรวจน้ำหนักก่อนเรียก Expert และใช้ > 0.3 แบบเคร่งครัด (เท่ากับ 0.3 ไม่เรียก)
โหมด `sparse=False` จำลองโค้ดเดิม: เรียกครบสาม Expert ก่อนเลือก context ด้วย > 0.2
เรียก `process_query(query, h_q)` จะใช้ sparse เป็นค่าเริ่มต้น

| กรณี | เรียกแบบเดิม | เรียกแบบ Sparse | Expert ที่เรียก | จำนวนการเรียกลดลง |
|---|---:|---:|---|---:|
| Simple | 3 | 2 | Entity, Relation | 33.3% |
| Complex | 3 | 1 | Subgraph | 66.7% |
| Custom | 3 | 2 | Entity, Relation | 33.3% |

ลดงาน retrieval ของ Expert ที่ถูกข้าม แต่เปอร์เซ็นต์ข้างต้นเป็นจำนวนการเรียก ไม่ใช่เวลา latency หรือค่าใช้จ่ายที่วัดได้
เวลาและต้นทุนจริงขึ้นกับแต่ละ Expert, overhead ของ Router และรูปแบบการรัน
ตัวอย่างนี้เป็นผลค้นหาคงที่ขนาดเล็ก จึงไม่เหมาะสำหรับสรุปการเพิ่มความเร็วของระบบจริง

## Task 3: Retrieval noise

| Query | Context แบบเดิม | Context แบบ Sparse | ข้อสังเกตเรื่อง Noise |
|---|---|---|---|
| Simple: Aspirin รักษาอะไร? | Entities + treats relation + เส้นทางผ่าน COX-2 | Entities + Aspirin-treats->Pain | แบบเดิมมีเส้นทางการอักเสบเกินคำถาม Sparse ตัดเส้นทางนี้ออก แต่ยังมีชื่อ COX-2 ที่ไม่จำเป็น |
| Complex: กลไกการลดอักเสบผ่าน COX-2 | Aspirin-inhibits->COX-2, COX-2-causes->Inflammation | เหมือนแบบเดิม | เส้นทางสอดคล้องกับคำถามในกราฟจำลอง จึงมี noise ต่ำเชิงคุณภาพ และไม่มีความแตกต่างของ prompt ระหว่างสองโหมด |

Multi-hop ไม่จำเป็นต้องมี noise มากกว่า Simple เสมอไป ขึ้นกับว่าบริบทตรงคำถามหรือไม่
สำหรับกราฟขนาดใหญ่ อาจลด noise ด้วยการจำกัดจำนวน hop, เลือกเส้นทางที่เชื่อมเอนทิตีในคำถาม,
จัดอันดับความสัมพันธ์ตามคำถาม และตัดข้อมูลซ้ำ โดยระวังไม่ตัดโหนดกลางที่จำเป็นต่อการเชื่อมเหตุผล
ระดับ noise ในที่นี้เป็นการวิเคราะห์เชิงคุณภาพ ยังไม่มีชุดข้อมูลอ้างอิงหรือ metric เชิงตัวเลข

## ขอบเขตของตัวอย่าง

เวกเตอร์กำหนดเอง ไม่ได้ encode จากข้อความ; W ไม่ได้ผ่านการฝึก; Experts คืนค่าคงที่และยังไม่ใช้ query_emb
การเปลี่ยนข้อความโดยใช้เวกเตอร์เดิมจึงไม่เปลี่ยนการ routing หรือผล retrieval
ขั้นตอนสุดท้ายสร้าง prompt เท่านั้น ยังไม่เรียก LLM และไม่ได้ทำ vector fusion หรือ Steiner Tree จริง
ปรับกราฟจาก Graph เป็น DiGraph เพื่อรักษาทิศทาง subject → object ตอนแสดงความสัมพันธ์
ข้อมูล Aspirin/COX-2 เป็นข้อมูลจำลองตามใบงานสำหรับฝึกโค้ด

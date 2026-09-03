# ใบงานปฏิบัติ: Low-VRAM LLM Engineering

> รายงานฉบับนี้อ้างอิงผลที่รันใหม่จาก `strict_lab_runner.py` ใน
> `strict_lab_results.json` วันที่ 2 กันยายน 2026. ใช้ model, prompt,
> `num_gpu=99`, `num_thread=4`, `num_ctx`, embedding model และ
> `RETRIEVAL_K=3` ตามใบงาน ไม่ใช้ guardrail หรือ prompt เพิ่มเติมในผลชุดนี้.

## สภาพแวดล้อม

- Ollama `0.30.10`
- GPU: AMD Radeon RX 7600S ผ่าน ROCm, VRAM 8 GiB
- เนื่องจากเครื่องที่ทดลองมี VRAM 8 GiB ผลวัด VRAM จึงเป็นค่าของเครื่องนี้;
  ส่วนคำแนะนำสำหรับ GPU 2GB เป็นการวิเคราะห์จากผลดังกล่าว

## Activity 1 — เปรียบเทียบ Micro-LLM

คำถามเดียวกันทุกโมเดล: `Explain the difference between HTTP/2 and HTTP/3.`

| Model | Parameter | Response Quality | Speed | ความเห็น |
| ----- | --------- | ---------------- | ----- | -------- |
| Qwen2.5 | 0.5B | ต่ำ: อธิบายหลายจุดที่ไม่ตรงกับข้อเท็จจริง เช่น TLS เป็น transport layer และ feature ที่ไม่มีจริง | 5.48 s, 169.7 tok/s | สร้าง token เร็วที่สุด แต่คำตอบยาวจึงเวลา total ไม่เร็วสุด |
| Llama 3.2 | 1B | ต่ำ: มี factual error ในคำอธิบาย HTTP/2/HTTP/3 | 7.08 s, 133.5 tok/s | ช้าที่สุดในรอบนี้ |
| SmolLM2 | 1.7B | ดีที่สุดในสามตัว: ระบุว่า HTTP/3 สร้างบน QUIC/UDP ได้ถูกต้อง แม้รายละเอียดบางส่วนยังควรตรวจทาน | 4.46 s, 97.0 tok/s | เวลาตอบรวมเร็วที่สุด เพราะสร้างคำตอบสั้นกว่า |

**Q1. Model ใดตอบเร็วที่สุด?**

หากวัดเวลาตอบรวม SmolLM2 1.7B เร็วที่สุด (`4.46` วินาที) ในรอบนี้; หากวัด
ความเร็วสร้าง token Qwen2.5 0.5B เร็วที่สุด (`169.7 tok/s`). เวลา total ยัง
ขึ้นกับจำนวน token ที่ model สร้าง ไม่ใช่ขนาด model เพียงอย่างเดียว.

**Q2. Model ใดให้คำตอบดีที่สุด?**

SmolLM2 1.7B ให้คำตอบดีที่สุดสำหรับ prompt นี้ เพราะอธิบายแกนสำคัญว่า HTTP/3
ใช้ QUIC แทน TCP ได้ถูกต้องกว่าอีกสองโมเดล.

**Q3. Model ขนาดใหญ่ขึ้นให้ผลดีขึ้นเสมอหรือไม่?**

ไม่เสมอไป. ในรอบนี้ SmolLM2 1.7B คุณภาพดีกว่า แต่ Llama 1B ใช้เวลานานกว่า
Qwen 0.5B และยังมี factual error. ขนาด model เป็นเพียงปัจจัยหนึ่ง.

## Activity 2 — VRAM Measurement

ทดลอง `llama3.2:1b` ด้วยคำถาม `Explain Retrieval-Augmented Generation.`

| สถานะ | VRAM Used |
| ----- | --------- |
| ก่อน Run LLM | 473.8 MiB |
| ขณะ Model Loaded | 2114.3 MiB |
| ขณะ Generate | 2114.3 MiB (peak) |
| หลัง `/bye` | 473.8 MiB |

**Q4. การ Run LLM ทำให้ VRAM เพิ่มขึ้นหรือไม่?**

เพิ่มขึ้น โดยจาก 473.8 MiB เป็น 2114.3 MiB หรือเพิ่มประมาณ 1641 MiB ขณะ model
ถูกโหลดและ generate.

**Q5. Model ใช้ VRAM ทั้งหมดหรือไม่?**

ไม่ทั้งหมด: GPU เครื่องทดลองมี 8 GiB แต่ Llama 3.2 1B ใช้ peak ประมาณ 2.11 GiB.

**Q6. ถ้า Model มีขนาดใหญ่กว่า VRAM จะเกิดอะไรขึ้น?**

Ollama/llama.cpp อาจ offload layer ที่เหลือไป CPU/RAM ทำให้ inference ช้าลง;
หาก RAM หรือทรัพยากรไม่พอ ก็อาจโหลด model ไม่สำเร็จ.

## Step 9 — LangChain + Ollama

รัน [app_lab.py](app_lab.py) ด้วยค่าตามใบงาน:

```python
ChatOllama(model="llama3.2:1b", num_gpu=99, num_ctx=1024, num_thread=4)
```

โปรแกรมเชื่อมต่อ LangChain กับ Ollama สำเร็จและสร้างคำตอบได้. คำตอบมีรายละเอียด
ทางเทคนิคที่ควรตรวจทาน ซึ่งสอดคล้องกับผล Activity 1 ว่า Llama 1B ไม่ควรใช้ตอบ
ข้อเท็จจริงสำคัญโดยไม่มีแหล่งอ้างอิง.

## Activity 4 — ทดลอง Parameter

Prompt: `Explain HTTP/3 in simple terms.` โดยใช้ `ChatOllama` และค่าตามใบงาน
ทุกตัว ยกเว้น `num_ctx` ที่เปลี่ยนตามตาราง.

| `num_ctx` | Response Time | VRAM | Quality |
| --------- | ------------- | ---- | ------- |
| 512 | 4.86 s | 1998.3 MiB | ตอบได้เป็นภาษาง่าย แต่มีรายละเอียด protocol ที่คลาดเคลื่อน |
| 1024 | 4.70 s | 2014.3 MiB | ตอบได้เป็นภาษาง่าย แต่มีรายละเอียด protocol ที่คลาดเคลื่อน |
| 2048 | 4.32 s | 2048.3 MiB | ตอบได้เป็นภาษาง่าย แต่มีรายละเอียด HTTP/1.1/packet ที่คลาดเคลื่อน |

**Q11. เมื่อเพิ่ม Context Size เกิดอะไรขึ้นกับ Memory?**

VRAM เพิ่มตาม context size: 1998.3 MiB → 2014.3 MiB → 2048.3 MiB เพราะ KV
cache ต้องกันพื้นที่ให้ token ใน context มากขึ้น. เวลาตอบไม่เพิ่มเป็นเส้นตรงใน
รอบนี้ เพราะ prompt สั้นและความต่างจากการ load model มีผลมาก.

**Q12. สำหรับ GPU 2GB ควรใช้ Context Size ประมาณเท่าใด?**

ควรเริ่มที่ `256–512` พร้อม model ที่เล็ก/quantized กว่า. ผลนี้แสดงว่า Llama
3.2 1B ใช้มากกว่า 2 GiB แม้ `num_ctx=512` บน runtime นี้ จึงไม่เหมาะกับการ
offload ทั้งหมดบน GPU 2GB.

## Activity 5 — ทดลอง Local RAG

ตั้งค่า `RETRIEVAL_K = 3` ตามใบงาน, embedding
`all-MiniLM-L6-v2`, ChromaDB แบบ in-memory และ Llama 3.2 1B.

`knowledge.txt` ใช้ข้อความตามใบงานทุกประโยค โดยแทน `<br><br>` ด้วยบรรทัดว่าง.
จึงได้ 6 documents และใช้ splitter ตามใบงาน (`split("\n\n")`). ผลคือระบบ
**ร้องขอ K=3 และดึงได้ K=3** ซึ่งเป็น 3 documents ที่ใกล้ที่สุดจากทั้งหมด 6.

| Question | Retrieved Docs | Answer Correct? |
| -------- | -------------- | --------------- |
| HTTP/3 ใช้อะไร? | HTTP/3→QUIC; HTTP/2→TCP; HTTP/3 ลด HOL blocking | ไม่ใช่ — ตอบ TCP ทั้งที่ doc อันดับหนึ่งระบุ QUIC |
| HTTP/2 vs HTTP/3 | HTTP/3 ลด HOL blocking; HTTP/3→QUIC; HTTP/2→TCP | ไม่ทั้งหมด — ตอบเชิงทั่วไป แต่ไม่ระบุ TCP/QUIC |
| QUIC คืออะไร? | HTTP/3→QUIC; QUIC→UDP; QUIC encrypted/faster setup | ถูกบางส่วน — ระบุว่า HTTP/3 ใช้ QUIC แต่คำอธิบายไม่ตรง context ทั้งหมด |

## Activity 6 — LLM vs RAG

คำถาม: `According to the knowledge base, what transport protocol does HTTP/3 use?`

| Metric | LLM | RAG |
| ------ | --- | --- |
| Correctness | ไม่ถูกต้อง — ตอบ TCP | ถูกต้อง — ตอบ QUIC |
| Response Time | 1.84 s | 1.70 s |
| Specific Knowledge | ไม่มี context จาก knowledge base | ได้ context 3 docs และระบุชัดว่า HTTP/3 uses QUIC |
| Hallucination | มี: ตอบ TCP ผิด | ไม่พบในคำถามนี้ |

**Q13. ทำไม RAG จึงช่วยให้ Model ขนาด 1B ตอบคำถามเฉพาะ Domain ได้ดีขึ้น?**

RAG ดึงหลักฐานเฉพาะ domain มาใส่ใน prompt จึงลดการพึ่งพาความจำเดิมของ model.
ผลคำถามนี้แสดงว่า LLM เดี่ยวตอบ TCP ผิด แต่ RAG ที่ดึง 3 documents ตอบ QUIC
ถูกต้อง. อย่างไรก็ดี Activity 5 แสดงว่า generator 1B ยังผิดได้ในคำถามอื่น จึงควร
เพิ่ม citation, output schema หรือ answer verifier ในงานจริง.

## Part 7 — Engineering Challenge

รัน query router ตาม prompt ในใบงานโดยไม่มี heuristic หรือ guardrail เพิ่มเติม:

| Query | Router result | ข้อสังเกต |
| ----- | ------------- | -------- |
| Coffee poem | `KNOWLEDGE` | **ผิด route**; ถูกส่งไป RAG และตอบว่าไม่ช่วยคำขอนี้ |
| HTTP/3 transport protocol | `KNOWLEDGE` | route ถูกต้องไป RAG แต่ Llama 1B ตอบ TCP ผิด |

รอบนี้ router เลือก knowledge query ถูกต้อง แต่จัด coffee poem ผิดเป็น knowledge.
router ขนาด 0.5B อาจให้ผลไม่คงที่เมื่อเปลี่ยน prompt หรือรอบรัน; หากพัฒนาต่อใน
ระบบจริงสามารถเพิ่ม rule-based validation หรือ few-shot examples ได้ แต่ไม่ได้ใช้
ในผลตามใบงานฉบับนี้.

## ไฟล์ผลรัน

- [strict_lab_runner.py](strict_lab_runner.py) — runner ที่ยึดโจทย์
- [strict_lab_results.json](strict_lab_results.json) — คำตอบดิบ, timing และ VRAM จากรอบที่รายงาน
- [app_lab.py](app_lab.py) — Step 9 ตามรูปแบบโค้ดในใบงาน

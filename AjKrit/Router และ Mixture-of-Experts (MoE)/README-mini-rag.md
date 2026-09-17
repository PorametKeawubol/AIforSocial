# ใบงานใหม่: Minimal RAG Routing

เทียบจาก `ใบงานปฏิบัติการ_ การสร้างระบบ RAG ขั้นสูงด้วย MoE — ขั้นที่ 2.md` ในเครื่อง โดยยังตรวจเนื้อหาล่าสุดจาก Google Docs และ PDF บน LMS ไม่ได้

## ต้องเพิ่มอะไรจากงานเดิม

| ประเด็น | งานเดิม (`step2.py` / รายงาน Advanced) | ใบงานใหม่ / `mini_rag.py` |
| --- | --- | --- |
| Router | น้ำหนัก Entity / Relation / Subgraph จาก Step 1 | `MiniRouter` เลือก graph_db / vector_db / direct_llm |
| Schema | ผลการกรองและ verification | Pydantic `RouteDecision` จำกัดชื่อ route และมี target_key |
| Data Prep | ตัวอย่าง Aspirin / COX-2 และเวกเตอร์จำลอง | Graph mock ของ PaymentAPI และเอกสาร deployment.yaml ใน FAISS |
| Pipeline | ContextRefiner, FactVerifier, iterative retrieval | เลือก route แล้วคืน context ตามแหล่งข้อมูล |
| Test Suite | แยก `test_step2.py` | อยู่ใน `mini_rag.py` พร้อม Data Prep และ Router |
| หลักฐานส่งงาน | Threshold / hallucination / retry | ผล query, expected route, actual route, context และ pass/fail |

เพิ่มไฟล์ใหม่สำหรับงาน routing เพราะเป็นคนละการทดลองกับรายงาน Advanced เดิม หากต้องส่งตามใบงานใหม่นี้ ให้แนบโค้ดและผล routing เพิ่ม ไม่ใช้ผล threshold/verification เป็นหลักฐานแทน

## วิธีรัน

รันจากโฟลเดอร์นี้ ใช้ Python ใน virtual environment ที่มีอยู่:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python mini_rag.py --test
.venv/bin/python mini_rag.py --output mini_rag_results.json
```

หากนำไปใช้แยกไฟล์ ติดตั้งเพียง `numpy faiss-cpu pydantic` แล้วรัน `python mini_rag.py` ได้ ไม่ต้องมี `code.py` หรือ `step2.py`

ชุดทดสอบรวมคำถาม Mini Test Matrix 3 ข้อ และคำถามจากตัวอย่าง `test_queries.json` อีก 3 ข้อ โดยตรวจทั้ง route และ context; `--test` ตรวจ schema, input, keyword precedence และระยะห่าง squared L2 ของ FAISS เพิ่มด้วย เมื่อทดสอบไม่ผ่านโปรแกรมจะออกด้วย exit code 1

## ขอบเขตตามตัวอย่าง minimal

- Graph ใช้ dictionary จำลอง ไม่ได้เชื่อม Neo4j หรือทำ multi-hop traversal จริง
- FAISS ค้นจริง แต่เอกสารมีเพียงหนึ่ง chunk และใช้เวกเตอร์คงที่ 4 มิติ จึงไม่ได้วัดคุณภาพ semantic search
- direct_llm คืน placeholder ตาม Test Matrix ไม่ได้เรียกโมเดลหรือสร้างโค้ด Python ตอบคำถามจริง
- Router ใช้ keyword ตามใบงาน โดย graph มีลำดับก่อน vector และ target ของ graph ถูกกำหนดเป็น PaymentAPI; เช่นคำถามคู่มือที่มีคำว่า Pod ก็อาจไป graph ได้
- คำถาม OOMKilled จาก JSON ทดสอบการเลือก vector route แต่ context ยังเป็นข้อความ deployment.yaml จาก mini dataset ไม่ใช่คู่มือแก้ OOMKilled ฉบับเต็ม

ท้ายใบงานมีข้อมูลขยาย `graph_data.cypher`, `devops_manual.md`, `test_queries.json` สำหรับฝึกต่อกับ Neo4j และคู่มือหลายหัวข้อ ซึ่งอยู่นอกตัวอย่าง minimal ข้างต้น หากต้องใช้ชุดขยายจริง ต้องเพิ่มการนำเข้ากราฟ, chunking/embedding ของคู่มือ และ retrieval ที่เหมาะสมก่อนอ้างว่าได้ทดสอบ root cause หรือ semantic search แล้ว โค้ดนี้ไม่รันคำสั่งล้างกราฟที่อยู่ในเอกสาร

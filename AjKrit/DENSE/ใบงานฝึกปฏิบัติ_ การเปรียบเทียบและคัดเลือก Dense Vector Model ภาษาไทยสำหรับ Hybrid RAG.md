# ใบงานฝึกปฏิบัติ: การเปรียบเทียบและคัดเลือก Dense Vector Model ภาษาไทยสำหรับ Hybrid RAG

ใบงานนี้ออกแบบมาเพื่อทดสอบและประเมินประสิทธิภาพของ Dense Embedding Models ภาษาไทยตัวอื่น ๆ นอกเหนือจาก intfloat/multilingual-e5-small โดยใช้วัดผลเปรียบเทียบกับ Ground Truth ในชุดข้อมูลเดิม

วัตถุประสงค์

* ทดลองรัน Dense Model หลายสถาปัตยกรรม (Architecture) บนโจทย์ภาษาไทย  
* เปรียบเทียบความสามารถในการจับความหมายบริบท (Semantic) และรหัสเฉพาะ (Exact Code)  
    
* คัดเลือก Dense Model ที่มีประสิทธิภาพสูงสุดเพื่อนำไปใช้รวมคะแนน RRF กับ BM25

รายชื่อ Candidate Models สำหรับภาษาไทย

| ชื่อ Model (HuggingFace) | Dimension | Prefix Requirement | จุดเด่น / ลักษณะเฉพาะ |
| :---- | :---- | :---- | :---- |
| intfloat/multilingual-e5-small | 384 | ต้องใส่ passage: / query:   | Baseline เดิม น้ำหนักเบา ประมวลผลไว  |
| intfloat/multilingual-e5-large | 1024 | ต้องใส่ passage: / query: | โมเดล E5 ขนาดใหญ่ ความแม่นยำสูงขึ้น |
| BAAI/bge-m3 | 1024 | ไม่ต้องใส่ Prefix | SOTA Multilingual รองรับบริบทความยาวสูง |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 768 | ไม่ต้องใส่ Prefix | เสถียรในงาน Semantic Similarity หลายภาษา |

โค้ดทดลองเปรียบเทียบหลาย Dense Model (Python)

ตารางบันทึกผลการประเมินคุณภาพ

| Query | Ground Truth | E5-Small (Baseline) | E5-Large | BGE-M3 | MPNet-Multi | Model ที่ค้นหาได้แม่นยำที่สุด |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Q1 (Synonym/Context) | Doc 1  |  |  |  |  |  |
| Q2 (Exact Code Match) | Doc 3  |  |  |  |  |  |
| Q3 (Hybrid Keyword & Intent) | Doc 2  |  |  |  |  |  |

โจทย์และคำถามท้ายใบงาน

* คำถามที่ 1 (Ranking Quality): โมเดลใดสามารถดึงเอกสาร Ground Truth ของทั้ง 3 Queries ขึ้นมาติด อันดับ 1 (Rank 1\) ได้มากที่สุด?  
*   
* คำถามที่ 2 (Exact Term Retention): ในโจทย์ Q2 ที่มีรหัสเฉพาะ D-9902 มี Dense Model ตัวใดแสดงผลลัพธ์ได้ดีขึ้นกว่า E5-Small Baseline หรือไม่?  
*   
* คำถามที่ 3 (Production Selection): หากพิจารณาปัจจัยด้านความเร็ว (Inference Speed) ร่วมกับความแม่นยำ คุณจะเลือก Model ใดไปใช้คู่กับ BM25 ในระบบ Hybrid RAG พร้อมระบุเหตุผลประกอบ


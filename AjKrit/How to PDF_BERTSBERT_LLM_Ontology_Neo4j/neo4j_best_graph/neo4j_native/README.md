# กราฟจาก Neo4j จริง

นำเข้าฐานข้อมูล Neo4j 5.26.30 สำเร็จที่ http://localhost:7475/browser/ (Bolt localhost:7688) แล้วอ่านกลับตรวจ ID โหนดและทุก property ของความสัมพันธ์ตรงกับต้นฉบับ

- กราฟทั้งหมด 266 โหนด / 879 ความสัมพันธ์
- เฉพาะเนื้อหา 207 โหนด / 263 ความสัมพันธ์
- เปิด `index.html` หรือ `Neo4j_Native_Graph_Report.pdf` เพื่อดูกราฟจาก Browser โดยตรง
- `*.svg`, `*.png` มาจากเมนู Download ของ Neo4j Browser ไม่ได้วาดใหม่
- `*_browser.png` เป็นภาพหน้าจอจริง พร้อม query และ Results overview
- `database_export.json` อ่านกลับจาก Neo4j และ `verification.json` คือผลตรวจ
- พื้นหลังภาพที่ Browser ส่งออกโปร่งใส; HTML/PDF แสดงบนพื้นขาว
- ภาพแต่ละมุมมองเป็นกราฟย่อย; `05_full_semantic` เป็นกราฟเนื้อหาครบ 207 โหนด / 263 ความสัมพันธ์

โหนดทั้งหมดใช้ label `PlantGuide2019Reviewed` เพื่อค้นแยกจากงานอื่นที่มีอยู่แล้วในฐานข้อมูล ไม่มีการลบข้อมูลเดิม

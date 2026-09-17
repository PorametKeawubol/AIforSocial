# สีใน Neo4j Browser สำหรับกราฟ A/B

ฐานข้อมูลใหม่ไม่มี shared label เช่น `Task1Reviewed`, `FinalTask1` หรือ `SectionKG` แล้ว สีจึงแยกตาม domain label โดยตรง:

| Label | สี |
|---|---|
| SourceDocument | แดง `#F16667` |
| Chapter | ม่วงเข้ม `#6D5BD0` |
| Case | น้ำเงิน `#4C8EDA` |
| Document | เขียว `#8DCC93` |
| Evidence | ส้ม `#F79767` |
| Contact | ม่วง `#C990C0` |
| Organization | ฟ้า `#57C7E3` |
| PersonRole | เหลือง `#FFE081` |

เปิด <http://localhost:7474/browser/> และรีเฟรชหนึ่งครั้ง แล้วรัน query จาก `ab_views.cypher` Neo4j Browser จะจำสีของแต่ละ label ในโปรไฟล์ ถ้าต้องการกำหนดให้ตรงกับภาพตัวอย่าง ให้คลิก label ใน Results overview แล้วเลือกสีตามตาราง และตั้ง Caption ของ Case เป็น `display`, Entity เป็น `name`, SourceDocument เป็น `display`

ภาพยืนยันจาก Neo4j Browser อยู่ที่:

- `artifacts/final_ab/A/neo4j-focus.png`
- `artifacts/final_ab/B/neo4j-focus.png`

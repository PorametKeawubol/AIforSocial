# **🧪 Lab Sheet**

## **หัวข้อ: การสกัดใจความสำคัญจากข้อความภาษาไทยด้วย BERT (WangchanBERTa) บนเครื่อง Local Python**

---

### **🎯 วัตถุประสงค์**

* สร้าง Virtual Environment ด้วยรหัสนักศึกษา

* ติดตั้งไลบรารีที่จำเป็นในการใช้งานโมเดลภาษาไทย

* ใช้ BERT เพื่อดึงประโยคที่สำคัญจากข้อความ

* ประมวลผลข้อมูลด้วย PyTorch และแสดงผลผ่าน Console

---

## **🔹 Part 1: เตรียม Python Environment (ครั้งแรกเท่านั้น)**

### **📍 1\. ตรวจสอบว่าเครื่องมี Python ติดตั้งแล้ว**

bash  
คัดลอกแก้ไข  
`python --version`

หากไม่มี ให้ดาวน์โหลดจาก: [https://www.python.org/downloads/](https://www.python.org/downloads/)

---

## **🔹 Part 2: สร้าง Virtual Environment ด้วยรหัสนักศึกษา**

เช่น รหัสนักศึกษา: `6610110554`  
 ให้ตั้งชื่อ environment ว่า `env_6610110554`

### **🧑‍💻 Windows:**

bash  
คัดลอกแก้ไข  
`python -m venv env_6610110554`  
`env_6610110554\Scripts\activate`

### **🐧 macOS / Linux:**

bash  
คัดลอกแก้ไข  
`python3 -m venv env_6610110554`  
`source env_6610110554/bin/activate`

✅ หลังจาก activate แล้วจะเห็น `(env_6610110554)` ขึ้นต้นบรรทัด

Fix …. 3.12  \-\> py \-3.12 \-m venv `env_6610110554`

---

## **🔹 Part 3: ติดตั้งไลบรารีที่จำเป็น**

bash  
คัดลอกแก้ไข

`pip install sentence-transformers`

---

## **🔹 Part 4: สร้างไฟล์ Python สำหรับรัน**

สร้างไฟล์ชื่อ `thai_summary.py` และคัดลอกโค้ดด้านล่างไปใส่

python  
คัดลอกแก้ไข  
from sentence\_transformers import SentenceTransformer, util

\# โหลดโมเดล sentence-transformers ที่รองรับหลายภาษา (มีไทยด้วย)  
model \= SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

text \= """  
ประเทศไทยมีทรัพยากรธรรมชาติที่หลากหลาย ทั้งในด้านป่าไม้ แหล่งน้ำ และความหลากหลายทางชีวภาพ  
เศรษฐกิจของประเทศอาศัยการส่งออก การท่องเที่ยว และภาคการเกษตรเป็นหลัก  
ประเทศไทยกำลังเผชิญกับปัญหาสิ่งแวดล้อม เช่น มลพิษทางอากาศและการเปลี่ยนแปลงภูมิอากาศ  
รัฐบาลจึงส่งเสริมนโยบาย BCG เพื่อความยั่งยืนทางเศรษฐกิจและสิ่งแวดล้อม  
"""

\# ตัดข้อความเป็นประโยค (ง่ายๆ ด้วย split หรือใช้ pythainlp)  
sentences \= \[s.strip() for s in text.split('\\n') if s.strip()\]

\# สร้าง embeddings ของแต่ละประโยค  
embeddings \= model.encode(sentences, convert\_to\_tensor\=True)

\# คำนวณ centroid vector (เฉลี่ย embedding ทั้งหมด)  
centroid \= embeddings.mean(dim\=0, keepdim\=True)

\# คำนวณ cosine similarity ระหว่าง centroid กับแต่ละประโยค  
cos\_scores \= util.pytorch\_cos\_sim(centroid, embeddings)\[0\]

\# เลือกประโยค similarity สูงสุด 2 ประโยค  
top\_n \= 2  
top\_results \= cos\_scores.topk(k\=top\_n)

print("บทสรุป:")  
for score, idx in zip(top\_results\[0\], top\_results\[1\]):  
    print("-", sentences\[idx\])

---

## **🔹 Part 5: รันโปรแกรมเพื่อดูผลลัพธ์**

bash  
คัดลอกแก้ไข  
`python thai_summary.py`

### **🔍 ตัวอย่างผลลัพธ์:**

text  
บทสรุป:  
\- ประเทศไทยมีทรัพยากรธรรมชาติที่หลากหลาย ทั้งในด้านป่าไม้ แหล่งน้ำ และความหลากหลายทางชีวภาพ  
\- ประเทศไทยกำลังเผชิญกับปัญหาสิ่งแวดล้อม เช่น มลพิษทางอากาศและการเปลี่ยนแปลงภูมิอากาศ

---

## **📝 คำถามท้ายแล็บ (Lab Questions)**

1. โมเดลที่เราใช้คืออะไร และมีข้อดีอย่างไรในการสกัดใจความภาษาไทย?

2. หากโมเดลที่ดีสำหรับภาษาไทย เพื่อเปรียบเทียบผลลัพทธ์ ?

---

## **📦 ส่งข้อความ** 

* ส่งข้อความผลลัพธ์ และอธิบายใจความที่โปรแกรมสกัดมาได้


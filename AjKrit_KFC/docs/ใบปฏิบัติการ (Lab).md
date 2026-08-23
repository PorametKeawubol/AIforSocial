# **ใบปฏิบัติการ (Lab)**

## **การพัฒนา LINE Chatbot ด้วย NLP สำหรับการตรวจจับ Intent ของผู้ใช้งาน**

# **วัตถุประสงค์**

เมื่อจบการปฏิบัติการ นักศึกษาจะสามารถ

1. อธิบายหลักการของ Intent Classification  
2. เปรียบเทียบการค้นหาด้วย Keyword และ NLP  
3. ใช้ Sentence-BERT เพื่อค้นหา Intent  
4. เชื่อมต่อ NLP กับ LINE Chatbot  
5. พัฒนา Chatbot ที่เข้าใจภาษาธรรมชาติของผู้ใช้  
6. วิเคราะห์ค่าความคล้ายคลึง (Cosine Similarity)  
7. ประเมินประสิทธิภาพของ Intent Detection

---

# **สถานการณ์**

บริษัท KFC ต้องการสร้าง Chatbot สำหรับตอบคำถามลูกค้า  
ลูกค้าอาจพิมพ์ข้อความได้หลายรูปแบบ เช่น  
เมนู  
ขอเมนู  
มีอะไรขาย  
อยากกินไก่  
โปรโมชั่น  
ขอรายการอาหาร  
วันนี้มีอะไรอร่อย  
แนะนำเมนูหน่อย  
หากใช้  
if text \== "menu":  
Chatbot จะไม่สามารถเข้าใจข้อความอื่นได้  
ดังนั้นจึงต้องใช้ NLP เพื่อวิเคราะห์ความหมายของข้อความ  
---

# **ทฤษฎี**

## **Intent คืออะไร**

Intent คือ "ความต้องการของผู้ใช้งาน"  
ตัวอย่าง

| ข้อความ | Intent |
| ----- | ----- |
| เมนู | menu |
| ขอเมนูหน่อย | menu |
| อยากกินไก่ | menu |
| โปรโมชั่นวันนี้ | promotion |
| ร้านอยู่ไหน | location |
| สวัสดี | greeting |

---

# **วิธีการตรวจจับ Intent**

## **วิธีที่ 1 Keyword Matching**

หลักการ  
User

↓

ค้นหา Keyword

↓

ตรงหรือไม่  
ตัวอย่าง  
MENU\_KEYWORDS \= \[  
    "menu",  
    "เมนู",  
    "อาหาร",  
    "รายการอาหาร"  
\]

if any(word in text for word in MENU\_KEYWORDS):  
    print("MENU")  
ข้อดี

* ง่าย  
* เร็ว

ข้อเสีย

* รองรับคำได้น้อย

---

## **วิธีที่ 2 TF-IDF**

User

↓

TF-IDF Vector

↓

Cosine Similarity

↓

Intent  
เหมาะสำหรับ

* งานค้นหาเอกสาร  
* FAQ

ข้อเสีย  
ไม่เข้าใจความหมายของคำ  
เช่น  
อยากกินไก่

กับ  
เมนูอาหาร  
อาจมีคะแนนต่ำ  
---

## **วิธีที่ 3 Sentence-BERT**

หลักการ  
ข้อความ  
↓  
Sentence Transformer  
↓  
Embedding  
↓  
Cosine Similarity  
↓  
Intent  
Sentence-BERT จะเปลี่ยนประโยคเป็นเวกเตอร์  
เช่น  
เมนู  
↓  
\[0.15,0.71,0.82,...\]  
อยากกินไก่  
↓  
\[0.17,0.70,0.81,...\]  
เวกเตอร์จะอยู่ใกล้กัน  
---

# **ทำไมต้องใช้ Sentence-BERT**

ข้อความ  
เมนู  
อยากกินไก่  
มีอะไรขาย  
ทั้งสามข้อความมีความหมายเดียวกัน  
Sentence-BERT สามารถเข้าใจได้  
---

# **สถาปัตยกรรม**

LINE User  
      │  
      ▼  
Webhook  
      │  
      ▼  
Receive Text  
      │  
      ▼  
Sentence-BERT  
      │  
      ▼  
Cosine Similarity  
      │  
      ▼  
Intent  
      │  
      ▼  
Scraper  
      │  
      ▼  
Reply  
---

# **การติดตั้ง**

pip install sentence-transformers  
pip install torch  
pip install scikit-learn  
pip install numpy  
---

# **ตัวอย่างที่ 1 สร้าง Sentence-BERT**

from sentence\_transformers import SentenceTransformer

model \= SentenceTransformer(  
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  
)  
---

# **ตัวอย่างที่ 2**

กำหนด Intent  
menu\_examples \= \[  
    "เมนู",  
    "ขอเมนู",  
    "มีอะไรขาย",  
    "อาหาร",  
    "อยากกินไก่",  
    "แนะนำเมนู",  
    "kfc menu",  
    "โปรโมชั่นอาหาร"  
\]  
---

# **สร้าง Embedding**

embeddings \= model.encode(  
    menu\_examples,  
    convert\_to\_tensor=True  
)  
Embedding จะถูกสร้างเพียงครั้งเดียว  
---

# **รับข้อความผู้ใช้**

query \= "วันนี้มีอะไรอร่อย"  
---

# **แปลงเป็น Embedding**

query\_embedding \= model.encode(  
    query,  
    convert\_to\_tensor=True  
)  
---

# **คำนวณ Similarity**

from sentence\_transformers import util

scores \= util.cos\_sim(  
    query\_embedding,  
    embeddings  
)  
---

# **ค่าที่ได้**

เมนู               0.72

ขอเมนู             0.81

มีอะไรขาย          0.88

อาหาร             0.64

อยากกินไก่         0.76  
---

เลือกคะแนนสูงสุด  
best\_score \= scores.max().item()  
---

กำหนด Threshold  
if best\_score \>= 0.60:  
    print("MENU")  
---

# **ฟังก์ชันสำหรับตรวจจับ Intent**

from sentence\_transformers import SentenceTransformer, util

model \= SentenceTransformer(  
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  
)

menu\_examples \= \[  
    "เมนู",  
    "ขอเมนู",  
    "มีอะไรขาย",  
    "อยากกินไก่",  
    "รายการอาหาร",  
    "อาหาร"  
\]

menu\_embedding \= model.encode(  
    menu\_examples,  
    convert\_to\_tensor=True  
)

def detect\_menu(text):

    embedding \= model.encode(  
        text,  
        convert\_to\_tensor=True  
    )

    score \= util.cos\_sim(  
        embedding,  
        menu\_embedding  
    )

    max\_score \= score.max().item()

    return max\_score  
---

# **ทดลอง**

query \= "วันนี้มีอะไรอร่อย"

score \= detect\_menu(query)

print(score)  
ผลลัพธ์  
0.84  
---

# **นำไปใช้กับ LINE**

เดิม  
if event.message.text.lower()=="menu":  
เปลี่ยนเป็น  
score \= detect\_menu(event.message.text)

if score \>= 0.60:

    result \= fetch\_kfc\_menu(url)

else:

    reply \= "ขออภัย ไม่เข้าใจคำถาม"  
Chatbot จะเข้าใจ  
อยากกินไก่  
มีอะไรขาย  
เมนู  
รายการอาหาร  
ทั้งหมด  
---

# **ขยายหลาย Intent**

สร้างฐานข้อมูล Intent  
intent\_examples \= {

"menu":\[

"เมนู",

"อาหาร",

"รายการอาหาร",

"มีอะไรขาย"

\],

"promotion":\[

"โปรโมชั่น",

"ลดราคา",

"โปร",

"วันนี้ลดอะไร"

\],

"location":\[

"ร้านอยู่ไหน",

"สาขา",

"แผนที่",

"ใกล้ฉัน"

\],

"greeting":\[

"สวัสดี",

"hello",

"hi"

\]

}  
---

คำนวณ Embedding  
intent\_embedding \= {}

for intent,texts in intent\_examples.items():

    intent\_embedding\[intent\]=model.encode(  
        texts,  
        convert\_to\_tensor=True  
    )  
---

ตรวจจับ Intent  
best\_intent=None  
best\_score=0

for intent,embedding in intent\_embedding.items():

    score=util.cos\_sim(  
        query\_embedding,  
        embedding  
    ).max().item()

    if score\>best\_score:

        best\_score=score

        best\_intent=intent  
ผล  
menu

0.82  
---

# **เชื่อมต่อกับ LINE Bot**

User

↓

Webhook

↓

Intent Detection

↓

menu ?

↓

YES

↓

Scraper

↓

BeautifulSoup

↓

LINE Reply  
---

# **การทดลอง**

ให้นักศึกษาทดสอบข้อความต่อไปนี้

| ข้อความ | Intent ที่คาดหวัง |
| ----- | ----- |
| เมนู | menu |
| ขอเมนู | menu |
| อยากกินไก่ | menu |
| มีอะไรขาย | menu |
| โปรวันนี้ | promotion |
| ร้านอยู่ไหน | location |
| สวัสดี | greeting |
| ขอบคุณ | other |

---

# **งานที่ต้องส่ง**

1. อธิบายหลักการ Intent Detection  
2. เปรียบเทียบ Keyword Matching กับ Sentence-BERT  
3. พัฒนา LINE Chatbot ที่รองรับอย่างน้อย **4 Intent**  
4. แสดงค่า Cosine Similarity ของแต่ละ Intent  
5. ทดสอบด้วยข้อความอย่างน้อย **30 ข้อความ** (Intent ละ 7–8 ข้อความ)  
6. สรุปผลในรูปตาราง พร้อมวิเคราะห์กรณีที่ตรวจจับผิด

ตัวอย่างตารางผลการทดลอง

| ข้อความ | Intent จริง | Intent ที่ระบบทำนาย | Similarity | ถูก/ผิด |
| ----- | ----- | ----- | ----- | ----- |
| ขอเมนู | menu | menu | 0.93 | ✓ |
| อยากกินไก่ | menu | menu | 0.87 | ✓ |
| โปรวันนี้ | promotion | promotion | 0.90 | ✓ |
| ร้านใกล้ฉัน | location | location | 0.85 | ✓ |
| หิวมาก | menu | other | 0.42 | ✗ |

---

# ผลการทดลองของเรา: 6 Intent

การทดลองนี้ใช้โมเดล `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` และกำหนด Threshold เท่ากับ `0.70` โดยใช้ข้อความทดสอบที่ไม่ซ้ำกับตัวอย่างฝึกโดยตรง

| ข้อความ | Intent จริง | Intent ที่ระบบทำนาย | Similarity | ถูก/ผิด |
|---|---|---|---:|:---:|
| ขอรายการของกินหน่อย | menu | menu | 0.887 | ✓ |
| อยากรู้ว่ามีไก่อะไรบ้าง | menu | menu | 0.836 | ✓ |
| ช่วยบอกเมนูน่าลอง | menu | menu | 0.959 | ✓ |
| มีอาหารอะไรให้เลือก | menu | menu | 0.901 | ✓ |
| หิวมาก | menu | other | 0.650 | ✗ |
| วันนี้อยากกิน KFC มีอะไรบ้าง | menu | menu | 0.816 | ✓ |
| เปิดดูเมนูให้หน่อย | menu | menu | 0.896 | ✓ |
| ช่วงนี้ KFC มีข้อเสนออะไร | promotion | order | 0.888 | ✗ |
| มีราคาพิเศษไหม | promotion | promotion | 0.764 | ✓ |
| อยากรู้โปรไก่ทอด | promotion | promotion | 0.879 | ✓ |
| ตอนนี้มีแคมเปญอะไร | promotion | promotion | 0.736 | ✓ |
| ซื้อชุดไหนคุ้มสุด | promotion | other | 0.531 | ✗ |
| มีโปรสำหรับวันนี้หรือเปล่า | promotion | promotion | 0.924 | ✓ |
| มีดีลพิเศษช่วงนี้ไหม | promotion | other | 0.674 | ✗ |
| ช่วยบอกสาขาที่อยู่ใกล้บ้านหน่อย | location | location | 0.728 | ✓ |
| ไป KFC สาขาไหนดี | location | location | 0.929 | ✓ |
| แถวบ้านฉันมีร้านไหม | location | location | 0.807 | ✓ |
| อยากได้ที่อยู่ของร้าน | location | location | 0.798 | ✓ |
| ร้านไหนเดินทางสะดวก | location | location | 0.741 | ✓ |
| ขอทางไปร้าน KFC | location | location | 0.938 | ✓ |
| มี KFC ในห้างไหนบ้าง | location | location | 0.952 | ✓ |
| หวัดดีบอต | greeting | greeting | 0.935 | ✓ |
| เข้ามาทักทายครับ | greeting | greeting | 0.955 | ✓ |
| สวัสดีจ้า KFC | greeting | greeting | 0.941 | ✓ |
| ดีจังที่เจอกัน | greeting | greeting | 0.779 | ✓ |
| ขอเริ่มคุยด้วยนะ | greeting | other | 0.662 | ✗ |
| ฮัลโหล | greeting | greeting | 0.991 | ✓ |
| มีใครอยู่ไหม | greeting | other | 0.634 | ✗ |
| ช่วยบอกช่องทางสั่งซื้อ | order | order | 0.702 | ✓ |
| อยากให้มาส่งที่บ้าน | order | order | 0.837 | ✓ |
| สั่งผ่านมือถือได้ไหม | order | other | 0.491 | ✗ |
| รับออเดอร์กลับบ้านไหม | order | order | 0.897 | ✓ |
| จะซื้อไก่ต้องทำอย่างไร | order | menu | 0.838 | ✗ |
| ขอขั้นตอนการสั่งหน่อย | order | other | 0.584 | ✗ |
| อยากสั่งชุดอาหารตอนนี้ | order | order | 0.890 | ✓ |
| รับทราบครับ ขอบใจ | thanks | thanks | 0.977 | ✓ |
| ขอบคุณมากสำหรับรายละเอียด | thanks | thanks | 0.831 | ✓ |
| ขอบคุณที่ช่วยตอบ | thanks | thanks | 0.874 | ✓ |
| ช่วยได้เยอะเลย ขอบคุณนะ | thanks | thanks | 0.875 | ✓ |
| ขอบพระคุณสำหรับคำตอบ | thanks | thanks | 0.818 | ✓ |
| ขอบคุณที่อธิบายครับ | thanks | thanks | 0.707 | ✓ |
| ได้รับคำตอบแล้ว ขอบคุณครับ | thanks | thanks | 0.890 | ✓ |

## สรุปผลการทดลอง

| Intent | ทำนายถูก | จำนวนทั้งหมด | Accuracy |
|---|---:|---:|---:|
| menu | 6 | 7 | 85.71% |
| promotion | 4 | 7 | 57.14% |
| location | 7 | 7 | 100.00% |
| greeting | 5 | 7 | 71.43% |
| order | 4 | 7 | 57.14% |
| thanks | 7 | 7 | 100.00% |
| **รวม** | **33** | **42** | **78.57%** |

กรณีที่ตรวจจับผิด ได้แก่ `หิวมาก` ซึ่งควรอยู่ในกลุ่ม `menu` แต่คะแนนสูงสุดต่ำกว่า Threshold จึงถูกจัดเป็น `other`, ข้อความ `ช่วงนี้ KFC มีข้อเสนออะไร` ถูกจัดเป็น `order` เพราะคะแนนใกล้เคียงกัน และข้อความเกี่ยวกับการสั่งซื้อบางรายการถูกจัดเป็น `other` หรือ `menu` แสดงว่าควรเพิ่มตัวอย่างฝึกของ Intent `promotion` และ `order` ให้หลากหลายขึ้น

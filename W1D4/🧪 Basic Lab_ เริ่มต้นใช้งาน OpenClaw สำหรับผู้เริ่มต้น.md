# **🧪 Basic Lab: เริ่มต้นใช้งาน OpenClaw สำหรับผู้เริ่มต้น**

## **🎯 วัตถุประสงค์**

1. ติดตั้ง OpenClaw  
2. เชื่อมต่อกับ LLM Provider (OpenAI, Anthropic หรือ Ollama)  
3. ทดลองสั่งงานผ่าน Command Line  
4. สร้าง Task Automation เบื้องต้น  
5. เรียนรู้แนวคิด Agent และ Skill ของ OpenClaw

OpenClaw เป็น AI Agent Framework ที่ทำงานบนเครื่องของผู้ใช้และสามารถเชื่อมต่อกับ LLM ต่าง ๆ เพื่อทำงานอัตโนมัติ เช่น อ่านไฟล์ ค้นหาเอกสาร จัดการโฟลเดอร์ และสื่อสารผ่านแอปต่าง ๆ ได้ ([OpenClaw](https://open-claw.bot/docs/what-is-openclaw?utm_source=chatgpt.com))  
---

# **Lab 1: ติดตั้ง OpenClaw**

## **Windows  with Docker** 

### **1\. Clone the repository**

git clone https://github.com/openclaw/openclaw.git  
cd openclaw

### **2\. Build the Docker image**

From the repository root:

docker build \-t openclaw:local .

This creates a local image named `openclaw:local`.

### **3\. Run the container**

On Windows PowerShell:

docker run \-d \`  
 \--name openclaw \`  
 \-p 18789:18789 \`  
 \-v C:\\openclaw:/home/node/.openclaw \`  
 openclaw:local

Or in Command Prompt (`cmd.exe`):

docker run \-d ^  
 \--name openclaw ^  
 \-p 18789:18789 ^  
 \-v C:\\openclaw:/home/node/.openclaw ^  
 openclaw:local

หรือใช้วิธี Onboarding ที่ทีมพัฒนาแนะนำ  
openclaw onboard

ซึ่งจะช่วยตั้งค่า Gateway, Workspace และ AI Provider อัตโนมัติ ([GitHub](https://github.com/openclaw/openclaw?utm_source=chatgpt.com))  
---

# **Lab 2: เชื่อมต่อ Ollama**

## **ติดตั้ง Ollama**

ดาวน์โหลดจาก  
[Ollama Official Website](https://ollama.com/?utm_source=chatgpt.com)  
ดาวน์โหลดโมเดล  
ollama pull llama3.2

ทดสอบ  
ollama run llama3.2

---

## **ตั้งค่า OpenClaw ให้ใช้ Ollama**

ไฟล์ Config  
provider:  
  type: ollama

model:  
  name: llama3.2

ทดสอบ  
openclaw chat

ถาม  
สวัสดี ช่วยแนะนำ Data Engineering ให้หน่อย

OpenClaw รองรับ Ollama เป็น Local Model Provider ได้โดยตรง ([openclaw.ch](https://openclaw.ch/?utm_source=chatgpt.com))  
---

# **Lab 3: Chat กับ Agent**

เริ่ม Agent  
openclaw chat

ลองถาม  
สรุปไฟล์ README.md ในโฟลเดอร์นี้

หรือ  
อธิบายโค้ด Python ใน project นี้

Agent จะอ่านไฟล์และสร้างคำตอบจากข้อมูลภายในเครื่อง  
---

# **Lab 4: File Management Agent**

สร้างโฟลเดอร์ทดลอง  
mkdir lab\_files  
cd lab\_files

สร้างไฟล์  
echo "Artificial Intelligence" \> ai.txt  
echo "Data Engineering" \> data.txt

สั่งงาน Agent  
อ่านทุกไฟล์ในโฟลเดอร์นี้และสรุปเนื้อหา

หรือ  
รวมไฟล์ทั้งหมดเป็น report.txt

---

# **Lab 5: Code Generation**

สั่ง Agent  
สร้าง Python Script สำหรับอ่าน CSV และแสดงสถิติพื้นฐาน

ตัวอย่างผลลัพธ์  
import pandas as pd

df \= pd.read\_csv("data.csv")

print(df.describe())

---

# **Lab 6: Automation Task**

สั่ง Agent  
ค้นหาไฟล์ PDF ทั้งหมดในโฟลเดอร์ Downloads

หรือ  
สร้างรายการไฟล์ PDF พร้อมขนาดไฟล์

OpenClaw สามารถเข้าถึงระบบไฟล์และรันคำสั่งต่าง ๆ ได้ผ่าน Tool Integration ([OpenClaw](https://open-claw.bot/docs/what-is-openclaw?utm_source=chatgpt.com))  
---

# **Lab 7: Custom Skill**

สร้างโฟลเดอร์  
skills/weather

สร้างไฟล์  
SKILL.md

ตัวอย่าง  
\# Weather Assistant

ตอบคำถามเกี่ยวกับสภาพอากาศ

รีสตาร์ท Agent  
openclaw restart

Skill คือ Plugin ที่เพิ่มความสามารถให้ Agent เช่น Email, Calendar หรือ Web Search ([TechRadar](https://www.techradar.com/pro/what-are-openclaw-skills-a-detailed-guide?utm_source=chatgpt.com))  
---

# **Lab 8: Mini Project**

## **AI Research Assistant**

ให้ Agent ช่วยงานวิจัย  
Prompt:  
ค้นหาไฟล์ PDF ทั้งหมดในโฟลเดอร์ Research

สรุป Abstract

สร้างตารางสรุป

บันทึกผลลัพธ์เป็น summary.md

ผลลัพธ์ที่คาดหวัง  
Paper Name  
Authors  
Year  
Research Area  
Key Findings

---

# **คำถามท้าย Lab**

1. OpenClaw แตกต่างจาก ChatGPT อย่างไร?  
2. Skill คืออะไร?  
3. Local LLM มีข้อดีอะไรเมื่อเทียบกับ Cloud LLM?  
4. Agent สามารถเข้าถึงไฟล์ในเครื่องได้อย่างไร?  
5. หากต้องการสร้าง AI Research Assistant ควรใช้ Skill อะไรบ้าง?

---

# **Assignment**

ให้นักศึกษาสร้าง Agent ชื่อ **PSU Research Assistant**  
ความสามารถที่ต้องมี

* อ่านไฟล์ PDF งานวิจัย  
* สรุป Abstract  
* สร้าง Markdown Report  
* ใช้งานผ่าน Ollama (Local LLM)

ส่ง

* Screenshot การทำงาน  
* Configuration File  
* Prompt ที่ใช้  
* Report ที่ Agent สร้าง


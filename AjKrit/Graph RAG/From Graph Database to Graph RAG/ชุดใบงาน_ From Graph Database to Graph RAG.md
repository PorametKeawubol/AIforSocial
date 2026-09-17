# ชุดใบงาน: From Graph Database to Graph RAG

## ภาพรวม 

| Lab | Domain | คำศัพท์เฉพาะ | สิ่งที่ฝึก |
| ----- | ----- | ----- | ----- |
| Lab 1 | 🏥 Healthcare | Disease, Symptom, Treatment | Graph Database |
| Lab 2 | 🌾 Agriculture | Crop, Pest, Fertilizer | Knowledge Graph |
| Lab 3 | 🎓 Education | Course, CLO, PLO, Assessment | Ontology |
| Lab 4 | 💳 Finance | Customer, Transaction, Fraud | RDF \+ SPARQL |
| Lab 5 | 🌐 Cybersecurity | Attack, Vulnerability, IOC | Graph RAG |

### ข้อดีคือ นักศึกษาจะเห็นว่า โครงสร้างเดียวกันสามารถเปลี่ยน Domain ได้

### ---

# LAB 1 — Graph Database

## Domain: Healthcare

### เป้าหมาย

### สร้าง Graph Database สำหรับจัดเก็บความสัมพันธ์ระหว่าง

### Disease

### Symptom

### Treatment

### Medication

### Dataset

### ให้นักศึกษาใช้ข้อมูลตัวอย่าง:

### Disease        Symptom             		Treatment

### \------------------------------------------------

### Diabetes       	Frequent Urination  	Insulin Therapy

### Diabetes       Thirst              		Insulin Therapy

### Hypertension   Headache            		Antihypertensive

### Asthma         Wheezing            		Inhaler

### 

### Task 1 — ออกแบบ Graph

### ให้นักศึกษาวาด Graph

### Diabetes

###    │

###    ├── HAS\_SYMPTOM ──\> Frequent Urination

###    │

###    └── TREATED\_BY ──\> Insulin Therapy

### Task 2 — สร้าง Neo4j

### สร้าง Node:

### (:Disease)

### (:Symptom)

### (:Treatment)

### และ Relationship:

### HAS\_SYMPTOM

### TREATED\_BY

### Task 3 — Query

### ให้นักศึกษาเขียน Cypher เพื่อค้นหา:

1. ### Disease ทั้งหมด

2. ### Symptoms ของ Diabetes

3. ### Treatment ของ Asthma

4. ### Disease ที่มี Symptom \= Headache

### **ผลลัพธ์ นักศึกษาจะได้ Graph Database แรกของตัวเอง รูปภาพใน Neo4j** 

### ---

# LAB 2 — Knowledge Graph

## Domain: Smart Agriculture

### คำศัพท์เฉพาะ

### Crop

### Pest

### Disease

### Fertilizer

### Soil

### ตัวอย่าง:

### Rice

###  │

###  ├── AFFECTED\_BY → Rice Blast

###  │

###  ├── ATTACKED\_BY → Rice Stem Borer

###  │

###  └── GROWN\_IN → Clay Soil

### Task 1 — Extract Knowledge

### ให้ข้อมูลข้อความ:

> ### “Rice can be affected by Rice Blast. Rice Stem Borer is a common pest of rice. Nitrogen Fertilizer can improve rice growth.”

### ให้นักศึกษา Extract:

### Entity

### Relationship

### Entity

### เช่น

### Rice → AFFECTED\_BY → Rice Blast

### Rice → ATTACKED\_BY → Rice Stem Borer

### Rice → BENEFITS\_FROM → Nitrogen Fertilizer

### Task 2 — สร้าง Knowledge Graph

### สร้าง Graph:

###             Rice

###           /    |    \\

###          /     |     \\

###         ↓      ↓      ↓

### Rice Blast   Pest   Fertilizer

### Task 3 — Graph Question

### ตอบคำถามจาก Graph:

1. ### What pests affect Rice?

   

2. ### What diseases are associated with Rice?

   

3. ### What fertilizer is related to Rice?

   

### 

### จุดประสงค์

### Lab นี้ต้องการให้เห็นว่า

### Graph Database ≠ Knowledge Graph

### เพราะเรากำลังเปลี่ยนจาก “ข้อมูล” มาเป็น Knowledge \+ Semantic Relationships

### ---

# LAB 3 — Ontology

## Domain: Education

### Lab นี้ผมแนะนำเป็นตัวที่สำคัญที่สุด เพราะนักศึกษาจะเริ่มเข้าใจว่า Ontology ไม่ใช่แค่ Graph

### Domain Vocabulary

### Program

### Course

### PLO

### CLO

### Assessment

### Student

### ตัวอย่าง:

### Course

###    │

###    ├── HAS\_CLO → CLO1

###    │

###    ├── CONTRIBUTES\_TO → PLO1

###    │

###    └── ASSESSED\_BY → Midterm Exam

### Task 1 — กำหนด Classes

### ให้นักศึกษาสร้าง:

### Program

### Course

### PLO

### CLO

### Assessment

### Student

### Task 2 — กำหนด Relationships

### Program

###    └── HAS\_COURSE → Course

### 

### Course

###    ├── HAS\_CLO → CLO

###    ├── CONTRIBUTES\_TO → PLO

###    └── ASSESSED\_BY → Assessment

### Task 3 — กำหนด Constraints

### ตัวอย่าง:

### Course HAS\_CLO CLO

### Course CONTRIBUTES\_TO PLO

### Course ASSESSED\_BY Assessment

### ห้าม:

### Course HAS\_CLO Student

### Task 4 — สร้าง Ontology Diagram

### นักศึกษาต้องออกแบบ:

###          Program

###              │

###          HAS\_COURSE

###              ↓

###            Course

###           /  |   \\

###          /   |    \\

###         ↓    ↓     ↓

###       CLO   PLO  Assessment

### Task 5 — Mapping

### ให้ข้อมูลจริง:

### Course:

### Artificial Intelligence

### 

### CLO:

### CLO1 \- Explain AI concepts

### 

### PLO:

### PLO1 \- Knowledge

### 

### Assessment:

### Final Examination

### ให้นักศึกษานำข้อมูลเข้าสู่ Ontology

### Learning Outcome

### นักศึกษาควรตอบได้ว่า:

> ### Ontology ทำหน้าที่เป็นแบบจำลองความรู้ (Knowledge Model) ก่อนที่จะสร้าง Knowledge Graph

### ---

# LAB 4 — RDF \+ SPARQL

[https://docs.google.com/document/d/1uxVfe6sehu343yV52RaL3s\_p77kHQnmECiDufeT6UpM/edit?usp=sharing](https://docs.google.com/document/d/1uxVfe6sehu343yV52RaL3s_p77kHQnmECiDufeT6UpM/edit?usp=sharing) 

## Domain: Financial Fraud Detection

### Domain Vocabulary

### Customer

### Account

### Transaction

### Merchant

### Fraud

### ตัวอย่าง:

### Customer\_001

###       │

###       └── OWNS → Account\_001

###                      │

###                      └── MADE\_TRANSACTION

###                               ↓

###                          Transaction\_001

###                               │

###                               ↓

###                           Merchant\_X

### ---

## Task 1 — สร้าง RDF Triple

### ให้นักศึกษาแปลงข้อมูล:

### Customer\_001 owns Account\_001

### Account\_001 made Transaction\_001

### Transaction\_001 paid Merchant\_X

### Transaction\_001 hasAmount 25000

### เป็น:

### Subject          	Predicate          	Object

### \------------------------------------------------

### Customer\_001     	owns              	 Account\_001

### Account\_001      	madeTransaction    Transaction\_001

### Transaction\_001  	paidMerchant       	Merchant\_X

### Transaction\_001  	hasAmount         	 25000

### ---

## Task 2 — RDF/Turtle

### ให้นักศึกษาเขียน Turtle เช่น:

### :Customer\_001 :owns :Account\_001 .

### 

### :Account\_001 :madeTransaction

###               :Transaction\_001 .

### 

### :Transaction\_001 :paidMerchant :Merchant\_X .

### 

### :Transaction\_001 :hasAmount 25000 .

### 

### 

### ---


### **ใบงานปฏิบัติการ: การสร้างระบบ RAG ขั้นสูงด้วย MoE — ขั้นที่ 2**

Minimal Runnable Code รวมทั้ง Data Prep, Router Agent, และ Test Suite สำหรับทดสอบ 3 เส้นทาง (graph\_db, vector\_db, direct\_llm) ในไฟล์เดียว

Python  
from typing import Literal  
from pydantic import BaseModel, Field  
import numpy as np  
import faiss

\# \==========================================  
\# 1\. Pydantic Router Schema & Logic  
\# \==========================================  
class RouteDecision(BaseModel):  
    destination: Literal\["graph\_db", "vector\_db", "direct\_llm"\]  
    target\_key: str \= Field(default="")

class MiniRouter:  
    def route(self, query: str) \-\> RouteDecision:  
        q \= query.lower()  
        if any(k in q for k in \["dependency", "pod", "root cause", "paymentapi"\]):  
            return RouteDecision(destination="graph\_db", target\_key="PaymentAPI")  
        elif any(k in q for k in \["ขั้นตอน", "คู่มือ", "deployment.yaml", "setup"\]):  
            return RouteDecision(destination="vector\_db", target\_key=query)  
        return RouteDecision(destination="direct\_llm")

\# \==========================================  
\# 2\. Data Preparation (Mini Mock Data)  
\# \==========================================  
\# \[Graph Data\] จำลอง Triple จาก Neo4j  
neo4j\_mock\_data \= {  
    "PaymentAPI": \[  
        "PaymentAPI CALLS AuthService",  
        "AuthService DEPLOYED\_ON AuthPod-01"  
    \]  
}

\# \[FAISS Vector Data\]  
DIM \= 4  
faiss\_index \= faiss.IndexFlatL2(DIM)  
text\_chunks \= \["วิธีตั้งค่า deployment.yaml: ให้ระบุ memory limit อย่างน้อย 512Mi"\]

\# สร้าง Vector จำลองขนาด 4 dims แล้วใส่ลง FAISS  
doc\_vector \= np.array(\[\[0.1, 0.8, 0.2, 0.9\]\], dtype='float32')  
faiss\_index.add(doc\_vector)

\# \==========================================  
\# 3\. Pipeline Execution Function  
\# \==========================================  
def execute\_rag\_pipeline(user\_query: str) \-\> dict:  
    router \= MiniRouter()  
    decision \= router.route(user\_query)  
      
    if decision.destination \== "graph\_db":  
        retrieved\_data \= neo4j\_mock\_data.get(decision.target\_key, \[\])  
    elif decision.destination \== "vector\_db":  
        \# จำลอง Query Vector เพื่อค้นหาบน FAISS  
        query\_vector \= np.array(\[\[0.1, 0.7, 0.2, 0.8\]\], dtype='float32')  
        distances, indices \= faiss\_index.search(query\_vector, k=1)  
        retrieved\_data \= text\_chunks\[indices\[0\]\[0\]\]  
    else:  
        retrieved\_data \= "Response generated directly from Parametric Knowledge."

    return {  
        "query": user\_query,  
        "destination": decision.destination,  
        "retrieved\_context": retrieved\_data  
    }

\# \==========================================  
\# 4\. Small Test Suite (ชุดทดสอบ)  
\# \==========================================  
test\_queries \= \[  
    "PaymentAPI มี dependency เชื่อมกับระบบไหนบ้าง?",  \# Expected: graph\_db  
    "ขอขั้นตอนการตั้งค่า deployment.yaml หน่อยครับ",    \# Expected: vector\_db  
    "เขียนฟังก์ชัน Python ลบตัวเลขซ้ำใน List"          \# Expected: direct\_llm  
\]

if \_\_name\_\_ \== "\_\_main\_\_":  
    print("=== RUNNING MINI RAG TEST SUITE \===\\n")  
    for i, query in enumerate(test\_queries, 1):  
        res \= execute\_rag\_pipeline(query)  
        print(f"Test \#{i}: {res\['query'\]}")  
        print(f" \-\> Destination Chosen : \[{res\['destination'\].upper()}\]")  
        print(f" \-\> Context Retrieved  : {res\['retrieved\_context'\]}\\n")

**สรุปผลคาดการณ์การทดสอบ (Test Matrix)**

| \# | Test Query | Expected Route | Expected Output Context |
| :---- | :---- | :---- | :---- |
| **1** | PaymentAPI มี dependency... | **GRAPH\_DB** | \['PaymentAPI CALLS AuthService', 'AuthService DEPLOYED\_ON AuthPod-01'\] |
| **2** | ขอขั้นตอนการตั้งค่า deployment.yaml... | **VECTOR\_DB** | 'วิธีตั้งค่า deployment.yaml: ให้ระบุ memory limit อย่างน้อย 512Mi' |
| **3** | เขียนฟังก์ชัน Python... | **DIRECT\_LLM** | 'Response generated directly from Parametric Knowledge.' |

นี่คือชุดไฟล์และข้อมูลจำลอง 3 ไฟล์ตามโครงสร้างใบงาน สามารถคัดลอกไปบันทึกเป็นไฟล์เพื่อใช้ฝึกทำระบบ RAG Pipeline ได้ทันที

**1\. ไฟล์สำหรับ Neo4j (**graph\_data.cypher**)**

*นำไปรันใน Neo4j Browser หรือ Cypher Shell เพื่อเตรียมข้อมูล Graph DB*

Cypher  
// ล้างข้อมูลเดิมใน Graph  
MATCH (n) DETACH DELETE n;

// สร้าง Nodes  
CREATE (s1:Service {name: 'PaymentAPI', embedding: \[0.10, 0.85, 0.20, 0.75\]})  
CREATE (s2:Service {name: 'AuthService', embedding: \[0.15, 0.80, 0.30, 0.70\]})  
CREATE (s3:Service {name: 'OrderService', embedding: \[0.12, 0.82, 0.25, 0.72\]})

CREATE (p1:Pod {name: 'AuthPod-01', embedding: \[0.20, 0.75, 0.85, 0.20\]})  
CREATE (p2:Pod {name: 'OrderPod-01', embedding: \[0.18, 0.78, 0.80, 0.22\]})

CREATE (i1:Incident {name: 'MemoryLeak\_OOMKilled', embedding: \[0.25, 0.70, 0.90, 0.15\]})  
CREATE (i2:Incident {name: 'DatabaseConnectionTimeout', embedding: \[0.30, 0.65, 0.85, 0.20\]})

// สร้าง Dependencies และ Relationships  
CREATE (s1)-\[:CALLS\]-\>(s2)  
CREATE (s1)-\[:CALLS\]-\>(s3)  
CREATE (s2)-\[:DEPLOYED\_ON\]-\>(p1)  
CREATE (s3)-\[:DEPLOYED\_ON\]-\>(p2)  
CREATE (p1)-\[:HAS\_INCIDENT\]-\>(i1)  
CREATE (p2)-\[:HAS\_INCIDENT\]-\>(i2);

**2\. ไฟล์สำหรับ FAISS (**devops\_manual.md**)**

*เอกสารข้อความย่อย นำไปทำ Chunking และแปลงเป็น Embedding ลง FAISS*

Markdown  
\# DevOps & Infrastructure Standard Operating Procedures (SOP)

\#\# 1\. Kubernetes Deployment Procedure  
การสั่งทำ Deployment หรือ Rollout อัปเดต Microservice บน Production ให้ใช้คำสั่ง Helm:  
\`helm upgrade \--install \<service-name\> ./charts\`  
ข้อควรระวัง: ต้องทำการตรวจสอบ Health Check Endpoint (/healthz) เสมอก่อนสลับ Traffic

\#\# 2\. Memory Leak and OOMKilled Troubleshooting  
เมื่อ Pod โดนสั่งดับด้วยสถานะ OOMKilled (Exit Code 137):  
1\. ให้ตรวจสอบ Memory Limit ในไฟล์ deployment.yaml  
2\. ปรับเพิ่ม Memory Limit และ Request resources ขึ้น 50%  
3\. ใช้คำสั่ง \`kubectl logs \<pod-name\> \--previous\` เพื่อดู Stack Trace ก่อน Pod ดับ

\#\# 3\. SLA and Reliability Policy  
ระบบ PaymentAPI และ OrderService มีข้อตกลง SLA กำหนด Uptime ไม่ต่ำกว่า 99.95% ต่อเดือน  
หากเกิด Down time เกิน 5 นาที ต้องทำการเปิด Incident Severity-1 (Sev-1) ในระบบทันที

**3\. ไฟล์สำหรับ Router Agent (**test\_queries.json**)**

*ชุดทดสอบ Query สำหรับฝึกวัดประสิทธิภาพ Router และ Re-retrieval Loop*  
JSON  
\[  
  {  
    "query\_id": 1,  
    "user\_query": "ทำไม PaymentAPI ถึงทำงานล้มเหลว มันส่งผลกระทบไปที่ Pod ไหนบ้าง?",  
    "expected\_route": "graph\_db",  
    "target\_entity": "PaymentAPI",  
    "description": "ทดสอบ Multi-hop Traversal หา Root Cause บน Neo4j"  
  },  
  {  
    "query\_id": 2,  
    "user\_query": "หากเจอสถานะ OOMKilled มีขั้นตอนการแก้ไขปัญหาในคู่มืออย่างไร?",  
    "expected\_route": "vector\_db",  
    "target\_entity": "OOMKilled",  
    "description": "ทดสอบ Semantic Search ค้นหาขั้นตอนจากเอกสารบน FAISS"  
  },  
  {  
    "query\_id": 3,  
    "user\_query": "เขียนฟังก์ชัน Python สำหรับเช็คว่า List มีค่าว่างหรือไม่",  
    "expected\_route": "direct\_llm",  
    "target\_entity": null,  
    "description": "ทดสอบคำถามทั่วไปที่ LLM ตอบได้ทันทีโดยไม่ต้องดึง Database"  
  }  
\]


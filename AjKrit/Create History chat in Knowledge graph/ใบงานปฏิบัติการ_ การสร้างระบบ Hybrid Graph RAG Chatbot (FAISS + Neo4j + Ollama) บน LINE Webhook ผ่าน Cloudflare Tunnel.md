**ใบงานปฏิบัติการ: การสร้างระบบ Hybrid Graph RAG Chatbot (FAISS \+ Neo4j \+ Ollama) บน LINE Webhook ผ่าน Cloudflare Tunnel**

**1\. หลักการทำงานของ History Chat บน Webhook**

* **Extract User ID:** ดึงค่า `user_id = event.source.user_id` จาก Payload ที่ LINE ส่งมา  
* **Fetch History:** ดึงประวัติการสนทนาเดิมของ `user_id` นั้นๆ ออกมา (เช่น 4-6 ข้อความล่าสุด)  
* **Inject Context:** รวมประวัติ \+ Context จาก RAG \+ คำถามใหม่ ใส่ใน Prompt เดียวกัน  
* **Save History:** บันทึกข้อความใหม่ของผู้ใช้และคำตอบของ LLM กลับลงเซสชันของ `user_id`

**2\. ตัวอย่างการปรับโค้ด `webhook.py` (In-Memory Storage)**

สร้างโครงสร้าง Dictionary สำหรับเก็บข้อความย้อนหลังแยกตาม `user_id` พร้อมจำกัดจำนวน Turn เพื่อไม่ให้ Context ยาวเกินไป

Python  
import os  
from flask import Flask, request, abort  
from linebot import LineBotApi, WebhookHandler  
from linebot.exceptions import InvalidSignatureError  
from linebot.models import MessageEvent, TextMessage, TextSendMessage  
from dotenv import load\_dotenv  
from rag\_search import llm\_response  \# เรียกใช้ฟังก์ชัน RAG

load\_dotenv()

app \= Flask(\_\_name\_\_)  
line\_bot\_api \= LineBotApi(os.getenv("CHANNEL\_ACCESS\_TOKEN"))  
handler \= WebhookHandler(os.getenv("CHANNEL\_SECRET"))

\# In-Memory Storage: { user\_id: \[ {"role": "user/assistant", "text": "..."}, ... \] }  
user\_chat\_history \= {}  
MAX\_HISTORY\_TURNS \= 6  \# บันทึกสูงสุด 6 ข้อความ (ย้อนหลัง 3 คู่)

def get\_formatted\_history(user\_id):  
    history\_list \= user\_chat\_history.get(user\_id, \[\])  
    if not history\_list:  
        return "ไม่มีประวัติการสนทนาก่อนหน้า"  
      
    formatted \= \[\]  
    for msg in history\_list:  
        role \= "ผู้ใช้" if msg\["role"\] \== "user" else "ระบบ"  
        formatted.append(f"{role}: {msg\['text'\]}")  
    return "\\n".join(formatted)

def add\_to\_history(user\_id, role, text):  
    if user\_id not in user\_chat\_history:  
        user\_chat\_history\[user\_id\] \= \[\]  
      
    user\_chat\_history\[user\_id\].append({"role": role, "text": text})  
    \# ตัดข้อความเก่าออกหากเกินจำนวนที่กำหนด  
    if len(user\_chat\_history\[user\_id\]) \> MAX\_HISTORY\_TURNS:  
        user\_chat\_history\[user\_id\] \= user\_chat\_history\[user\_id\]\[-MAX\_HISTORY\_TURNS:\]

@app.route("/callback", methods=\["POST"\])  
def callback():  
    signature \= request.headers.get("X-Line-Signature")  
    body \= request.get\_data(as\_text=True)

    try:  
        handler.handle(body, signature)  
    except InvalidSignatureError:  
        abort(400)  
    return "OK"

@handler.add(MessageEvent, message=TextMessage)  
def handle\_message(event):  
    user\_id \= event.source.user\_id  
    user\_question \= event.message.text.strip()

    \# รองรับคำสั่งล้างประวัติการสนทนา  
    if user\_question \== "ล้างประวัติ":  
        user\_chat\_history.pop(user\_id, None)  
        line\_bot\_api.reply\_message(event.reply\_token, TextSendMessage(text="ล้างประวัติการสนทนาเรียบร้อยแล้วครับ"))  
        return  
      
    \# 1\. ดึงประวัติการสนทนาเดิม  
    history\_context \= get\_formatted\_history(user\_id)  
      
    \# 2\. ส่งคำถามและประวัติไปประมวลผลที่ LLM  
    try:  
        answer \= llm\_response(user\_question, chat\_history=history\_context)  
    except Exception as e:  
        print(f"Error: {e}")  
        answer \= "ขออภัยครับ ไม่สามารถประมวลผลคำถามได้ในขณะนี้"

    \# 3\. บันทึกคำถามและคำตอบใหม่ลง History  
    add\_to\_history(user\_id, "user", user\_question)  
    add\_to\_history(user\_id, "assistant", answer)

    \# 4\. ส่งคำตอบกลับไปยัง LINE  
    line\_bot\_api.reply\_message(  
        event.reply\_token,  
        TextSendMessage(text=answer)  
    )

if \_\_name\_\_ \== "\_\_main\_\_":  
    app.run(port=5000, debug=False)

**3\. การปรับสคริปต์ `rag_search.py` ให้รับ History Context**

เพิ่มพารามิเตอร์ `chat_history` เพื่อส่งประวัติบทสนทนาเข้า Prompt ของ LLaMA

Python  
def llm\_response(question, chat\_history=""):  
    retrieved\_docs \= search\_vector\_documents(question, k=TOP\_K)  
    graph\_context \= search\_graph\_context(question)

    vector\_context \= "\\n\\n".join(\[f"\[Doc {i+1}\]\\n{d.page\_content}" for i, d in enumerate(retrieved\_docs)\])

    prompt \= f"""คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร

\[ประวัติการสนทนาที่ผ่านมา\]:  
{chat\_history}

\[Vector Context\]:  
{vector\_context}

\[Graph Context\]:  
{graph\_context}

คำถามปัจจุบัน: {question}  
คำตอบ:"""

    response \= llm.invoke(prompt)  
    return response.content.strip()

| import os from langchain\_huggingface import HuggingFaceEmbeddings from langchain\_community.vectorstores import FAISS from langchain\_ollama import ChatOllama from langchain\_neo4j import Neo4jGraph from dotenv import load\_dotenv load\_dotenv() FAISS\_PATH \= "faiss\_index" EMBED\_MODEL \= "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" LLM\_MODEL \= "llama3.2:1b" TOP\_K \= 3 \# 1\. Load Stores & LLM embeddings \= HuggingFaceEmbeddings(     model\_name=EMBED\_MODEL,     model\_kwargs={'device': 'cpu'} ) vectorstore \= FAISS.load\_local(FAISS\_PATH, embeddings, allow\_dangerous\_deserialization=True) graph \= Neo4jGraph(     url=os.getenv("NEO4J\_URI"),     username=os.getenv("NEO4J\_USERNAME"),     password=os.getenv("NEO4J\_PASSWORD") ) llm \= ChatOllama(model=LLM\_MODEL, temperature=0) \# 2\. Retrieval Functions def search\_vector\_documents(question, k=TOP\_K):     return vectorstore.similarity\_search(question, k=k) def search\_graph\_context(question):     cypher \= """     MATCH (n)-\[r\]-\>(m)     WHERE any(word IN split(toLower($query), ' ') WHERE toLower(n.id) CONTAINS word OR toLower(m.id) CONTAINS word)     RETURN n.id AS source, type(r) AS rel, m.id AS target     LIMIT 10     """     try:         results \= graph.query(cypher, params={"query": question})         if not results:             return "ไม่พบข้อมูลความสัมพันธ์ใน Graph"         return "\\n".join(\[f"- ({r\['source'\]}) \--\[{r\['rel'\]}\]--\> ({r\['target'\]})" for r in results\])     except Exception:         return "ไม่สามารถดึงข้อมูล Graph ได้" \# 3\. Context-Aware Hybrid RAG Generation (รวม Chat History) def llm\_response(question: str, chat\_history: str \= "") \-\> str:     """     ฟังก์ชันประมวลผลคำตอบแบบ Hybrid RAG ร่วมกับประวัติการสนทนา (Chat History)     :param question: คำถามปัจจุบันจากผู้ใช้     :param chat\_history: ข้อความประวัติบทสนทนาย้อนหลัง (String Format)     :return: คำตอบจาก LLM     """     \# 1\. ดึงบริบทจาก Vector Store และ Knowledge Graph     retrieved\_docs \= search\_vector\_documents(question, k=TOP\_K)     graph\_context \= search\_graph\_context(question)     \# 2\. จัดรูปแบบ Vector Context     if retrieved\_docs:         vector\_context \= "\\n\\n".join(\[             f"\[Doc {i+1} | Pages: {d.metadata.get('pages', 'N/A')}\]\\n{d.page\_content}"             for i, d in enumerate(retrieved\_docs)         \])     else:         vector\_context \= "ไม่พบข้อมูลใน Vector Store"     \# 3\. จัดรูปแบบ Chat History กรณีที่ไม่มีประวัติส่งมา     history\_str \= chat\_history if chat\_history.strip() else "ไม่มีประวัติการสนทนาก่อนหน้า"     \# 4\. ออกแบบ Prompt ให้รองรับการเชื่อมโยงประวัติบทสนทนา     prompt \= f"""คุณเป็นผู้ช่วยตอบคำถามจากเอกสารที่มีความสามารถในการจดจำบทสนทนา กฎการตอบ: 1\. วิเคราะห์คำถามปัจจุบันร่วมกับ \[ประวัติการสนทนาที่ผ่านมา\] โดยเฉพาะคำถามที่มีคำสรรพนาม (เช่น "มัน", "สิ่งนี้", "ระบบดังกล่าว") 2\. ใช้ข้อมูลจาก \[Vector Context\] และ \[Graph Context\] เพื่อตอบคำถามให้ถูกต้องตรงตามเอกสาร 3\. หากบริบททั้งหมดไม่พอตอบคำถาม ให้ตอบว่า "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารครับ" 4\. ตอบเป็นภาษาไทย สั้น กระชับ สุภาพ และเข้าใจง่าย \[ประวัติการสนทนาที่ผ่านมา\]: {history\_str} \[Vector Context\]: {vector\_context} \[Graph Context\]: {graph\_context} คำถามปัจจุบัน: {question} คำตอบ:"""     \# 5\. เรียกใช้ LLM เพื่อสร้างคำตอบ     response \= llm.invoke(prompt)     return response.content.strip() \# 4\. ส่วนทดสอบการทำงานผ่าน Terminal if \_\_name\_\_ \== "\_\_main\_\_":     print("=" \* 60\)     print("TESTING RAG SEARCH WITH CHAT HISTORY")     print("=" \* 60\)     \# จำลองประวัติการสนทนาเดิม     mock\_history \= (         "ผู้ใช้: ระบบนี้ประกอบด้วยอะไรบ้าง?\\n"         "ระบบ: ระบบนี้ใช้ FAISS, Neo4j, Ollama และ Cloudflare Tunnel ครับ"     )     \# คำถามต่อเนื่องที่ต้องใช้ประวัติสนทนาช่วยเข้าใจ Context     mock\_question \= "แล้ว Neo4j ทำหน้าที่อะไรในระบบนี้?"     print(f"📜 \[Mock Chat History\]:\\n{mock\_history}\\n")     print(f"❓ \[Current Question\]: {mock\_question}\\n")     print("🤖 \[LLM Answer\]:")     print(llm\_response(mock\_question, chat\_history=mock\_history))  |
| :---- |

**ข้อควรระวังในการพัฒนาต่อยอด**

* **Memory Volatility:** การเก็บลง Dictionary ในโค้ดจะสูญหายทันทีเมื่อเซิร์ฟเวอร์ Restart  
* **Context Overflows:** การตั้ง `MAX_HISTORY_TURNS` สูงเกินไปจะทำให้ Token เต็ม Context Window ของ LLM และประมวลผลช้าลง


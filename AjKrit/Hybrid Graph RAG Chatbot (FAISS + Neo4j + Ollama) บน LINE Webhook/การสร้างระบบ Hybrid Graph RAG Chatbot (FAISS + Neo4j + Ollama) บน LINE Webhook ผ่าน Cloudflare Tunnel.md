# **ใบงานปฏิบัติการ: การสร้างระบบ Hybrid Graph RAG Chatbot (FAISS \+ Neo4j \+ Ollama) บน LINE Webhook ผ่าน Cloudflare Tunnel**

## **🎯 วัตถุประสงค์การเรียนรู้**

> 1. เพื่อศึกษาสถาปัตยกรรม **Hybrid Graph RAG** ที่ผสานการสืบค้นแบบความหมาย (FAISS) และโครงสร้างความสัมพันธ์ (Neo4j Knowledge Graph)  
> 2. เพื่อแยกโมดูลประมวลผล Indexing ออกจากระบบบริการ Real-time Webhook ช่วยลดภาระการทำงานของเซิร์ฟเวอร์  
> 3. เพื่อประยุกต์ใช้ **Cloudflare Tunnel (cloudflared)** ในการเชื่อมต่อ Webhook สำหรับ LINE Messaging API โดยไม่มีข้อจำกัดเรื่อง Session Timeout  
> 4. เพื่อวัดผลและเปรียบเทียบคุณภาพคำตอบของ LLM จากค่าการค้นหา Top-K ที่แตกต่างกัน

## **📌 โครงสร้างไฟล์โปรเจกต์ (Project Structure)**

Plaintext  
rag-hybrid-bot/  
│  
├── document.pdf             \# เอกสาร PDF ต้นฉบับ  
├── .env                     \# ไฟล์เก็บค่า Config และ API Keys  
├── build\_index.py           \# \[File 1\] สกัด PDF \-\> FAISS Vector \+ Neo4j Graph Index  
├── rag\_search.py            \# \[File 2\] โหลด Index \-\> Hybrid Search \-\> Ollama (LLaMA)  
└── webhook.py               \# \[File 3\] รับส่งข้อความ LINE Webhook (Flask)

[https://www.pidthong.org/img\_ebook/file/9-file-4786.pdf](https://www.pidthong.org/img_ebook/file/9-file-4786.pdf)  \-\>document.pdf   

## **💻 การเตรียมระบบและแพ็กเกจ (Prerequisites)**

### **1\. ติดตั้ง  Packages**

เปิด Terminal แล้วรันคำสั่งติดตั้ง Library ที่จำเป็น:

pip install \-U langchain langchain-community langchain-huggingface langchain-ollama langchain-neo4j langchain-experimental  
pip install faiss-cpu sentence-transformers pypdf neo4j  
pip install flask line-bot-sdk \-dotenv

### **2\. ติดตั้ง Cloudflare CLI (cloudflared)**

* **Windows (Winget):** winget install Cloudflare.cloudflared  
* **macOS (Homebrew):** brew install cloudflared  
* **Linux (Debian/Ubuntu):**

  curl \-L \--output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb  
  sudo dpkg \-i cloudflared.deb

### **3\. ดาวน์โหลด LLM Model**

เปิด Terminal แล้วดึงโมเดล LLaMA 3.2 ผ่าน Ollama:

ollama pull llama3.2:1b

4\. Neo4j Docker   
docker-compose.yml

| services:   neo4j:     container\_name: neo4j     image: neo4j:latest     ports:       \- 7474:7474       \- 7687:7687     environment:       \- NEO4J\_AUTH=neo4j/${NEO4J\_PASSWORD}       \- NEO4J\_PLUGINS=\["apoc"\]       \- NEO4J\_dbms\_security\_procedures\_unrestricted=apoc.\*       \- NEO4J\_dbms\_security\_procedures\_allowlist=apoc.\*       \- NEO4J\_apoc\_export\_file\_enabled=true       \- NEO4J\_apoc\_import\_file\_enabled=true       \- NEO4J\_apoc\_import\_file\_use\_\_neo4j\_\_config=true     volumes:       \- ./neo4j\_db/data:/data       \- ./neo4j\_db/logs:/logs       \- ./neo4j\_db/import:/var/lib/neo4j/import       \- ./neo4j\_db/plugins:/plugins |
| :---- |

## **📝 ขั้นตอนการปฏิบัติงาน (Step-by-Step Instructions)**

### **ขั้นตอนที่ 1: กำหนดค่าสภาพแวดล้อม (.env)**

สร้างไฟล์ชื่อ .env ในโฟลเดอร์หลักของโปรเจกต์:  
ข้อมูลโค้ด  
\# LINE Messaging API Configuration  
CHANNEL\_ACCESS\_TOKEN=YOUR\_LINE\_CHANNEL\_ACCESS\_TOKEN  
CHANNEL\_SECRET=YOUR\_LINE\_CHANNEL\_SECRET

\# Neo4j Database Configuration  
NEO4J\_URI=bolt://localhost:7687  
NEO4J\_USERNAME=neo4j  
NEO4J\_PASSWORD=your\_password

### **ขั้นตอนที่ 2: สคริปต์ประมวลผลเอกสารและสร้าง Index (build\_index.py)**

ไฟล์นี้ทำหน้าที่อ่าน PDF แบ่ง Chunk สร้าง Vector Index ลง FAISS และสกัด Entity/Relationship บันทึกลง Neo4j (รันเฉพาะเมื่อเอกสารเปลี่ยนแปลง)

Model I :  langchain\_neo4j import Neo4jGraph

| import osfrom pypdf import PdfReaderfrom langchain\_core.documents import Documentfrom langchain\_huggingface import HuggingFaceEmbeddingsfrom langchain\_community.vectorstores import FAISSfrom langchain\_ollama import ChatOllamafrom langchain\_neo4j import Neo4jGraphfrom langchain\_experimental.graph\_transformers import LLMGraphTransformerfrom dotenv import load\_dotenvload\_dotenv()PDF\_PATH \= "document.pdf"FAISS\_PATH \= "faiss\_index"EMBED\_MODEL \= "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"LLM\_MODEL \= "llama3.2:1b"\# 1\. Extract text from PDFdef extract\_lines\_from\_pdf(path):    reader \= PdfReader(path)    lines \= \[\]    for page\_no, page in enumerate(reader.pages, start=1):        text \= page.extract\_text()        if text:            for line in text.split("\\n"):                line \= line.strip()                if line:                    lines.append({"text": line, "page": page\_no})    return lines\# 2\. Chunking Logicdef chunk\_lines(lines, chunk\_size=5, overlap=2):    chunks \= \[\]    step \= chunk\_size \- overlap    for i in range(0, len(lines), step):        selected \= lines\[i:i \+ chunk\_size\]        if not selected:            continue        text \= "\\n".join(item\["text"\] for item in selected)        pages \= sorted(set(item\["page"\] for item in selected))        chunks.append(            Document(                page\_content=text,                metadata={"start\_line": i, "pages": pages}            )        )    return chunksif \_\_name\_\_ \== "\_\_main\_\_":    print("=" \* 60)    print("HYBRID INDEXING (FAISS \+ NEO4J GRAPH)")    print("=" \* 60)    \# อ่านไฟล์และทำ Chunking    lines \= extract\_lines\_from\_pdf(PDF\_PATH)    documents \= chunk\_lines(lines, chunk\_size=5, overlap=2)    print(f"Total Lines: {len(lines)} | Total Chunks: {len(documents)}")    \# Part A: FAISS Vector Indexing    print("\\n\[1/2\] Creating FAISS Index...")    embeddings \= HuggingFaceEmbeddings(model\_name=EMBED\_MODEL)    vectorstore \= FAISS.from\_documents(documents, embeddings)    os.makedirs(FAISS\_PATH, exist\_ok=True)    vectorstore.save\_local(FAISS\_PATH)    print(f"FAISS saved to: {FAISS\_PATH}")    \# Part B: Neo4j Knowledge Graph Indexing    print("\\n\[2/2\] Extracting Graph Entities to Neo4j...")    graph \= Neo4jGraph(        url=os.getenv("NEO4J\_URI"),        username=os.getenv("NEO4J\_USERNAME"),        password=os.getenv("NEO4J\_PASSWORD")    )        llm \= ChatOllama(model=LLM\_MODEL, temperature=0)    llm\_transformer \= LLMGraphTransformer(llm=llm)    graph\_docs \= llm\_transformer.convert\_to\_graph\_documents(documents)    graph.add\_graph\_documents(graph\_docs)    print("Graph indexing completed on Neo4j\!") |
| :---- |

Model II: neo4j import GraphDatabase \+ pythainlp.tag.thainer import ThaiNameTagger

| import os import re from dotenv import load\_dotenv from pypdf import PdfReader from langchain\_core.documents import Document from langchain\_huggingface import HuggingFaceEmbeddings from langchain\_community.vectorstores import FAISS from langchain\_community.graphs.graph\_document import GraphDocument, Node, Relationship from pythainlp.tag.thainer import ThaiNameTagger from neo4j import GraphDatabase load\_dotenv() PDF\_PATH \= "document.pdf" FAISS\_PATH \= "faiss\_index" EMBED\_MODEL \= "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \# Initialize Thai NER on CPU ner\_tagger \= ThaiNameTagger() \# Regex patterns for structured entity types PATTERNS \= {     "DATE": r"\\b(\\d{1,2}\\s+(?:ม\\.ค\\.|ก\\.พ\\.|มี\\.ค\\.|เม\\.ย\\.|พ\\.ค\\.|มิ\\.ย\\.|ก\\.ค\\.|ส\\.ค\\.|ก\\.ย\\.|ต\\.ค\\.|พ\\.ย\\.|ธ\\.ค\\.|มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)\\s+\\d{4})\\b",     "MONEY": r"\\b\\d+(?:,\\d{3})\*(?:\\.\\d+)?\\s\*(?:บาท|ล้านบาท)\\b",     "PERCENT": r"\\b\\d+(?:\\.\\d+)?\\s\*%\\b",     "SECTION": r"\\b(?:มาตรา|ข้อ)\\s\*\\d+\\b" } \# Target terms dictionary CUSTOM\_DICTIONARY \= {     "SYSTEM": \["Hybrid RAG", "FAISS", "Neo4j", "LINE Webhook", "Cloudflare Tunnel", "Ollama", "LLaMA"\],     "MODULE": \["build\_index.py", "rag\_search.py", "webhook.py"\] } \# 1\. Extract text from PDF def extract\_lines\_from\_pdf(path):     reader \= PdfReader(path)     lines \= \[\]     for page\_no, page in enumerate(reader.pages, start=1):         text \= page.extract\_text()         if text:             for line in text.split("\\n"):                 line \= line.strip()                 if line:                     lines.append({"text": line, "page": page\_no})     return lines \# 2\. Chunking Logic def chunk\_lines(lines, chunk\_size=5, overlap=2):     chunks \= \[\]     step \= chunk\_size \- overlap     for i in range(0, len(lines), step):         selected \= lines\[i:i \+ chunk\_size\]         if not selected:             continue         text \= "\\n".join(item\["text"\] for item in selected)         pages \= sorted(set(item\["page"\] for item in selected))         chunks.append(             Document(                 page\_content=text,                 metadata={"start\_line": i, "pages": pages}             )         )     return chunks \# 3\. Extraction Helpers (CPU Processing) def extract\_pattern\_nodes(text):     nodes \= \[\]     for label, pattern in PATTERNS.items():         matches \= re.findall(pattern, text)         for match in set(matches):             nodes.append({"id": match.strip(), "type": label})     return nodes def extract\_custom\_nodes(text):     nodes \= \[\]     for label, terms in CUSTOM\_DICTIONARY.items():         for term in terms:             if term in text:                 nodes.append({"id": term, "type": label})     return nodes def extract\_advanced\_thai\_graph(documents):     graph\_docs \= \[\]     for idx, doc in enumerate(documents, start=1):         text \= doc.page\_content         nodes\_dict \= {}         \# Pre-trained Thai NER (PERSON, LOCATION, ORGANIZATION, TIME)         raw\_ner \= ner\_tagger.get\_ner(text)         current\_entity \= ""         current\_label \= ""         for word, pos, tag in raw\_ner:             if tag \!= "O":                 label \= tag.split("-")\[-1\]                 if tag.startswith("B-"):                     if current\_entity:                         nodes\_dict\[current\_entity\] \= Node(id=current\_entity, type=current\_label)                     current\_entity \= word                     current\_label \= label                 elif tag.startswith("I-") and label \== current\_label:                     current\_entity \+= word             else:                 if current\_entity:                     nodes\_dict\[current\_entity\] \= Node(id=current\_entity, type=current\_label)                     current\_entity \= ""                     current\_label \= ""         \# Regex Entities         for p\_node in extract\_pattern\_nodes(text):             nodes\_dict\[p\_node\["id"\]\] \= Node(id=p\_node\["id"\], type=p\_node\["type"\])         \# Dictionary Entities         for c\_node in extract\_custom\_nodes(text):             nodes\_dict\[c\_node\["id"\]\] \= Node(id=c\_node\["id"\], type=c\_node\["type"\])         nodes \= list(nodes\_dict.values())         \# Build Relationships         relationships \= \[\]         for i in range(len(nodes)):             for j in range(i \+ 1, min(i \+ 3, len(nodes))):                 rel\_type \= "LOCATED\_IN" if nodes\[j\].type \== "LOCATION" else "ASSOCIATED\_WITH"                 relationships.append(                     Relationship(source=nodes\[i\], target=nodes\[j\], type=rel\_type)                 )         if nodes:             graph\_docs.append(                 GraphDocument(nodes=nodes, relationships=relationships, source=doc)             )         print(f"Extracted Chunk {idx}/{len(documents)} \-\> {len(nodes)} distinct nodes.")     return graph\_docs \# Cypher write transaction function for native session def write\_graph\_document(tx, graph\_doc):     \# Insert Nodes     for node in graph\_doc.nodes:         safe\_label \= re.sub(r'\[^a-zA-Z0-9\_\]', '', node.type) or "Entity"         query \= f"MERGE (n:\`{safe\_label}\` {{id: $id}})"         tx.run(query, id=node.id)     \# Insert Relationships     for rel in graph\_doc.relationships:         safe\_rel\_type \= re.sub(r'\[^a-zA-Z0-9\_\]', '', rel.type) or "RELATED\_TO"         safe\_source\_label \= re.sub(r'\[^a-zA-Z0-9\_\]', '', rel.source.type) or "Entity"         safe\_target\_label \= re.sub(r'\[^a-zA-Z0-9\_\]', '', rel.target.type) or "Entity"         query \= f"""         MERGE (source:\`{safe\_source\_label}\` {{id: $source\_id}})         MERGE (target:\`{safe\_target\_label}\` {{id: $target\_id}})         MERGE (source)-\[r:\`{safe\_rel\_type}\`\]-\>(target)         """         tx.run(             query,             source\_id=rel.source.id,             target\_id=rel.target.id         ) if \_\_name\_\_ \== "\_\_main\_\_":     print("=" \* 60\)     print("HYBRID INDEXING (FAISS \+ NATIVE NEO4J DRIVER)")     print("=" \* 60\)     \# Read and chunk PDF     lines \= extract\_lines\_from\_pdf(PDF\_PATH)     documents \= chunk\_lines(lines, chunk\_size=5, overlap=2)     print(f"Total Lines: {len(lines)} | Total Chunks: {len(documents)}")     \# Part A: FAISS Vector Indexing (Forced to CPU)     print("\\n\[1/2\] Creating FAISS Index...")     embeddings \= HuggingFaceEmbeddings(         model\_name=EMBED\_MODEL,         model\_kwargs={'device': 'cpu'}     )     vectorstore \= FAISS.from\_documents(documents, embeddings)     os.makedirs(FAISS\_PATH, exist\_ok=True)     vectorstore.save\_local(FAISS\_PATH)     print(f"FAISS saved to: {FAISS\_PATH}")     \# Part B: Neo4j Knowledge Graph Indexing (Native Driver Session)     print("\\n\[2/2\] Extracting Graph Entities to Neo4j via Native Driver...")          driver \= GraphDatabase.driver(         os.getenv("NEO4J\_URI"),         auth=(os.getenv("NEO4J\_USERNAME"), os.getenv("NEO4J\_PASSWORD"))     )     graph\_docs \= extract\_advanced\_thai\_graph(documents)     with driver.session() as session:         for idx, g\_doc in enumerate(graph\_docs, start=1):             session.execute\_write(write\_graph\_document, g\_doc)             print(f"Ingested Graph Document {idx}/{len(graph\_docs)} into Neo4j.")     driver.close()     print("Graph indexing completed on Neo4j\!")  |
| :---- |

### **ขั้นตอนที่ 3: สคริปต์สืบค้นข้อมูลและตอบคำถาม (rag\_search.py)**

ทำหน้าที่โหลด Index ดึงข้อมูลแบบ Hybrid Search (FAISS \+ Neo4j) แล้วส่ง Prompt ให้ LLaMA ประมวลผล

import os  
from langchain\_huggingface import HuggingFaceEmbeddings  
from langchain\_community.vectorstores import FAISS  
from langchain\_ollama import ChatOllama  
from langchain\_neo4j import Neo4jGraph  
from dotenv import load\_dotenv

load\_dotenv()

FAISS\_PATH \= "faiss\_index"  
EMBED\_MODEL \= "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  
LLM\_MODEL \= "llama3.2:1b"  
TOP\_K \= 3

\# 1\. Load Stores & LLM  
embeddings \= HuggingFaceEmbeddings(model\_name=EMBED\_MODEL)  
vectorstore \= FAISS.load\_local(FAISS\_PATH, embeddings, allow\_dangerous\_deserialization=True)

graph \= Neo4jGraph(  
    url=os.getenv("NEO4J\_URI"),  
    username=os.getenv("NEO4J\_USERNAME"),  
    password=os.getenv("NEO4J\_PASSWORD")  
)  
llm \= ChatOllama(model=LLM\_MODEL, temperature=0)

\# 2\. Retrieval Functions  
def search\_vector\_documents(question, k=TOP\_K):  
    return vectorstore.similarity\_search(question, k=k)

def search\_graph\_context(question):  
    cypher \= """  
    MATCH (n)-\[r\]-\>(m)  
    WHERE any(word IN split(toLower($query), ' ') WHERE toLower(n.id) CONTAINS word OR toLower(m.id) CONTAINS word)  
    RETURN n.id AS source, type(r) AS rel, m.id AS target  
    LIMIT 10  
    """  
    try:  
        results \= graph.query(cypher, params={"query": question})  
        if not results:  
            return "ไม่พบข้อมูลความสัมพันธ์ใน Graph"  
        return "\\n".join(\[f"- ({r\['source'\]}) \--\[{r\['rel'\]}\]--\> ({r\['target'\]})" for r in results\])  
    except Exception:  
        return "ไม่สามารถดึงข้อมูล Graph ได้"

\# 3\. Hybrid RAG Generation  
def llm\_response(question):  
    retrieved\_docs \= search\_vector\_documents(question, k=TOP\_K)  
    graph\_context \= search\_graph\_context(question)

    if not retrieved\_docs and graph\_context \== "ไม่พบข้อมูลความสัมพันธ์ใน Graph":  
        return "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารครับ"

    vector\_context \= "\\n\\n".join(\[f"\[Doc {i+1} | Pages: {d.metadata.get('pages')}\]\\n{d.page\_content}" for i, d in enumerate(retrieved\_docs)\])

    prompt \= f"""คุณเป็นผู้ช่วยตอบคำถามจากเอกสาร

กฎการตอบ:  
1\. ใช้ข้อมูลจาก \[Vector Context\] และ \[Graph Context\] เท่านั้น  
2\. หากบริบทไม่พอ ให้ตอบว่า "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"  
3\. ตอบเป็นภาษาไทย สั้น กระชับ และเข้าใจง่าย

\[Vector Context\]:  
{vector\_context}

\[Graph Context\]:  
{graph\_context}

คำถาม: {question}  
คำตอบ:"""

    response \= llm.invoke(prompt)  
    return response.content.strip()

if \_\_name\_\_ \== "\_\_main\_\_":  
    question \= input("Question: ")  
    print("\\n--- HYBRID RAG ANSWER \---")  
    print(llm\_response(question))

### **ขั้นตอนที่ 4: สคริปต์บริการ Webhook สำหรับ LINE (webhook.py)**

import os  
from flask import Flask, request, abort  
from linebot import LineBotApi, WebhookHandler  
from linebot.exceptions import InvalidSignatureError  
from linebot.models import MessageEvent, TextMessage, TextSendMessage  
from dotenv import load\_dotenv  
from rag\_search import llm\_response

load\_dotenv()

CHANNEL\_ACCESS\_TOKEN \= os.getenv("CHANNEL\_ACCESS\_TOKEN")  
CHANNEL\_SECRET \= os.getenv("CHANNEL\_SECRET")

line\_bot\_api \= LineBotApi(CHANNEL\_ACCESS\_TOKEN)  
handler \= WebhookHandler(CHANNEL\_SECRET)

app \= Flask(\_\_name\_\_)

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
    question \= event.message.text  
    print(f"Question: {question}")

    try:  
        answer \= llm\_response(question)  
    except Exception as e:  
        print(f"RAG Error: {e}")  
        answer \= "ขออภัยครับ ระบบไม่สามารถประมวลผลคำถามได้ในขณะนี้"

    line\_bot\_api.reply\_message(  
        event.reply\_token,  
        TextSendMessage(text=answer)  
    )

if \_\_name\_\_ \== "\_\_main\_\_":  
    app.run(port=5000, debug=False)

## **🚀 การเปิดใช้งานระบบและการทำ Tunneling**

> 1. ประมวลผล Indexing ครั้งแรก:

>     build\_index.py

> 2. เปิดเซิร์ฟเวอร์ Webhook\[cite: 1\]:

>     webhook.py

> 3. **เปิดใช้งาน Cloudflare Tunnel (ใน Terminal อีกหน้าต่าง):**

>    cloudflared tunnel \--url http://localhost:5000

> 4. **ตั้งค่า LINE Developers Console:**  
   * คัดลอก URL จาก Cloudflare (เช่น \[https://xxx.trycloudflare.com\](https://xxx.trycloudflare.com))  
   * เติม /callback ต่อท้าย เช่น \[https://xxx.trycloudflare.com/callback\](https://xxx.trycloudflare.com/callback)  
   * นำไปวางในช่อง **Webhook URL** บน LINE Developer Portal แล้วเปิดสวิตช์ **Use Webhook**


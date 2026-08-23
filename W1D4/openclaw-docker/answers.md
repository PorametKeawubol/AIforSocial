# คำถามท้าย Lab

1. OpenClaw แตกต่างจาก ChatGPT อย่างไร?
OpenClaw เป็น agent framework ที่รันบนเครื่องหรือ server ของผู้ใช้และเชื่อมกับ tools, filesystem, workspace และ messaging channels ได้ ส่วน ChatGPT เป็น chat application/model interface เป็นหลัก

2. Skill คืออะไร?
Skill คือส่วนขยายความสามารถของ agent เช่น อ่านไฟล์ ค้นเว็บ ใช้ messaging หรือทำ workflow เฉพาะทาง

3. Local LLM มีข้อดีอะไรเมื่อเทียบกับ Cloud LLM?
ข้อมูลอยู่ในเครื่อง ลดค่าใช้จ่ายต่อ request ใช้งานได้กับข้อมูล private และควบคุม runtime/model ได้เอง

4. Agent สามารถเข้าถึงไฟล์ในเครื่องได้อย่างไร?
ผ่าน workspace ที่ mount เข้า container เช่น `W1D4/openclaw-docker/workspace` ถูก mount เป็น `/home/node/.openclaw/workspace`

5. หากต้องการสร้าง AI Research Assistant ควรใช้ Skill อะไรบ้าง?
ควรมี file/PDF reading, Markdown report generation, search หรือ memory สำหรับค้นเอกสาร และ messaging skill สำหรับสั่งงานผ่าน Telegram


"""Bounded, expiring conversation memory; no graph or model dependency."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path


def conversation_key(source: dict) -> str | None:
    kind = source.get("type", "user")
    scope = source.get({"user": "userId", "group": "groupId", "room": "roomId"}.get(kind, ""))
    user = source.get("userId")
    if not scope or not user:
        return None
    return hashlib.sha256(f"{kind}:{scope}:{user}".encode()).hexdigest()


class ChatHistory:
    def __init__(self, path: Path, max_turns: int = 4, ttl_seconds: int = 86400):
        self.path = path
        self.max_turns = max(1, max_turns)
        self.ttl = max(1, ttl_seconds)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS conversations (key TEXT PRIMARY KEY, updated REAL, turns TEXT)")
        os.chmod(path, 0o600)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA secure_delete = ON")
        return connection

    def get(self, key: str) -> list[dict]:
        with self._connect() as db:
            db.execute("DELETE FROM conversations WHERE updated < ?", (time.time() - self.ttl,))
            row = db.execute("SELECT turns FROM conversations WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else []

    def append(self, key: str, question: str, answer: str, topic: str = ""):
        turns = (self.get(key) + [{"question": question[:1500], "answer": answer[:3000], "topic": topic[:250]}])[-self.max_turns:]
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO conversations VALUES (?, ?, ?)",
                       (key, time.time(), json.dumps(turns, ensure_ascii=False)))

    def clear(self, key: str):
        with self._connect() as db:
            db.execute("DELETE FROM conversations WHERE key = ?", (key,))


_FOLLOWUP = re.compile(
    r"^(?:(?:แล้ว|และ|ถ้าอย่างนั้น|กรณีนี้|เรื่องนี้|อันนี้|งั้น|เพิ่มเติม|ขอรายละเอียด)\s*)?"
    r"(?:ต้องเตรียม|เตรียมอะไร|ใช้เอกสาร|เอกสารอะไร|ไปที่ไหน|ไปไหน|แจ้งที่ไหน|ติดต่อที่ไหน|"
    r"ต้องแจ้ง|ต้องทำ|ทำยังไง|ทำอย่างไร|ทำไง|มีค่าใช้จ่าย|เสียเงิน|นานไหม|รอกี่|"
    r"กี่วัน|กี่ชั่วโมง|มีขั้นตอน|ขั้นตอนอะไร|ทำต่อ|มีสิทธิ|ขอรายละเอียด|ถ้าไม่มี|แล้วถ้า|"
    r"ได้ไหม|ทำได้ไหม|ทำไม|เพราะอะไร|หมายความว่า|อธิบายเพิ่ม|อธิบายสั้น)"
)
_NEW_TOPIC = ("คนหาย", "เพื่อนหาย", "ลักพาตัว", "รถหาย", "ขโมย", "เอกสารหาย", "บัตรหาย",
              "ถูกทำร้าย", "ยิงกัน", "ฉ้อโกง", "น้ำท่วม", "พายุ", "ใบขับขี่", "ประกันตัว", "ทวงหนี้")


def is_followup(question: str) -> bool:
    return len(question) <= 180 and bool(_FOLLOWUP.match(question)) and not any(word in question for word in _NEW_TOPIC)


def resolve_followup(question: str, history: list[dict]) -> str:
    """Expand elliptical questions only; explicit topic changes stay independent."""
    if not history or not is_followup(question):
        return question
    last = history[-1]
    topic = last.get("topic") or last.get("question", "")
    topic = re.sub(r"^\d+(?:\.\d+)*\s*", "", topic).strip()
    return f"กรณี{topic}: {question}" if topic else question

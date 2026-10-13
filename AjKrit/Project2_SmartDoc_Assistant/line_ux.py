"""LINE navigation and evidence-topic follow-ups (all question actions are text)."""
from __future__ import annotations

import json
import re
from pathlib import Path

CATEGORIES = {
    "ขั้นตอนและเอกสาร": ("1", "2"),
    "สิทธิและกฎหมาย": ("6", "7", "1.17", "1.18"),
    "ความปลอดภัย": ("3", "4"),
    "รถและจราจร": ("5", "7", "1.5", "2.9", "3.6"),
}
MENU_LABELS = ("ถาม Assistant", *CATEGORIES, "หัวข้อทั้งหมด")
HELP = ("สวัสดีครับ เลือกหมวดจากเมนู หรือพิมพ์คำถามได้เลย\n"
        "เช่น คนหายทำอย่างไร / เอกสารหายต้องเตรียมอะไร\n\n"
        "ถามต่อสั้น ๆ ได้ เช่น ต้องเตรียมอะไร หรือไปที่ไหน\n"
        "พิมพ์ ‘ล้างประวัติ’ เพื่อเริ่มเรื่องใหม่\n"
        "ระบบใช้คู่มือฉบับ พ.ศ. 2558 ไม่ใช่ช่องทางแจ้งเหตุฉุกเฉิน")


def message_action(label: str, text: str | None = None, max_label: int = 20) -> dict:
    return {"type": "message", "label": label[:max_label], "text": (text or label)[:300]}


def quick_reply(topic: str = "") -> dict:
    topic = re.sub(r"^\d+(?:\.\d+)*\s*", "", topic).strip()[:180]
    if topic:
        pairs = [("ต้องเตรียมอะไร", f"กรณี{topic} ต้องเตรียมหลักฐานหรือเอกสารอะไรบ้าง"),
                 ("ไปที่ไหน", f"กรณี{topic} ต้องไปแจ้งหรือติดต่อที่ไหน"),
                 ("ต้องแจ้งตำรวจไหม", f"กรณี{topic} ต้องแจ้งตำรวจหรือไม่")]
        if any(word in topic for word in ("กฎหมาย", "สิทธิ", "ประกัน")):
            pairs = [("มีสิทธิอะไรบ้าง", f"กรณี{topic} มีสิทธิอะไรบ้าง"),
                     ("ขั้นตอนเป็นอย่างไร", f"กรณี{topic} มีขั้นตอนอย่างไร"),
                     ("ติดต่อที่ไหน", f"กรณี{topic} ติดต่อหน่วยงานไหน")]
        elif topic.startswith(("การป้องกัน", "การปฏิบัติ", "ความรู้เกี่ยวกับการใช้", "ท่านจะขับ", "ความรู้เกี่ยวกับเครื่องหมาย")):
            pairs = [("ควรปฏิบัติอย่างไร", f"เรื่อง{topic} ควรปฏิบัติอย่างไร"),
                     ("ข้อควรระวัง", f"เรื่อง{topic} มีข้อควรระวังอะไรบ้าง"),
                     ("ติดต่อใครได้บ้าง", f"เรื่อง{topic} ติดต่อหน่วยงานใดได้บ้าง")]
    else:
        pairs = [("คนหาย", "คนหายต้องทำอย่างไร")]
    pairs += [("อธิบายสั้นๆ", f"เรื่อง{topic} อธิบายสั้นๆ หน่อย" if topic else "อธิบายสั้นๆ หน่อย")]
    pairs += [("หัวข้อทั้งหมด", "หัวข้อทั้งหมด"), ("ล้างประวัติ", "ล้างประวัติ")]
    return {"items": [{"type": "action", "action": message_action(label, text)} for label, text in pairs]}


def _bubble(title: str, actions: list[dict], subtitle: str = "แตะหัวข้อเพื่อถาม Assistant") -> dict:
    return {"type": "bubble", "size": "mega",
            "header": {"type": "box", "layout": "vertical", "backgroundColor": "#102840",
                       "contents": [{"type": "text", "text": title, "color": "#FFFFFF", "weight": "bold", "wrap": True}]},
            "body": {"type": "box", "layout": "vertical", "spacing": "md",
                     "contents": [{"type": "text", "text": subtitle, "size": "sm", "color": "#666666", "wrap": True}] +
                     [{"type": "button", "height": "sm", "style": "link", "action": action} for action in actions]}}


class LineUX:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir

    def topics(self, category: str) -> list[tuple[str, str]]:
        try:
            chunks = json.loads((self.index_dir / "chunks.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        found = {}
        for chunk in chunks:
            sid = chunk.get("section_id", "")
            title = chunk.get("section_title", "")
            if not sid or not title:
                continue
            match = any(sid == p or sid.startswith(p + ".") for p in CATEGORIES[category])
            if sid == "7":
                traffic = "จราจร" in title or "ช่องเดินรถ" in title
                if category == "รถและจราจร":
                    match = traffic
                elif category == "สิทธิและกฎหมาย":
                    match = not traffic
            if category == "ขั้นตอนและเอกสาร" and sid in ("1.17", "1.18", "1.5", "2.9"):
                match = False
            if category == "ความปลอดภัย" and sid == "3.6":
                match = False
            if match:
                found[(sid, title)] = (sid, title)
        return sorted(found.values(), key=lambda item: tuple(int(n) for n in item[0].split(".") if n.isdigit()))

    def navigation(self, text: str) -> list[dict] | None:
        if text == "ถาม Assistant":
            return [{"type": "text", "text": HELP, "quickReply": quick_reply()}]
        if text == "หัวข้อทั้งหมด":
            bubbles = [_bubble(category, [message_action(category)], "เลือกหมวดเพื่อดูหัวข้อย่อยทั้งหมด") for category in CATEGORIES]
        elif text in CATEGORIES:
            topics = self.topics(text)
            if not topics:
                return [{"type": "text", "text": "ยังโหลดรายการหัวข้อไม่ได้ กรุณาพิมพ์คำถามได้เลยครับ"}]
            bubbles = []
            for offset in range(0, len(topics), 5):
                actions = [message_action(re.sub(r"^\d+(?:\.\d+)*\s*", "", title),
                                          f"ขอขั้นตอนหรือรายละเอียดเรื่อง {title}", max_label=40) for _, title in topics[offset:offset + 5]]
                bubbles.append(_bubble(f"{text} ({offset // 5 + 1})", actions))
        else:
            return None
        return [{"type": "flex", "altText": text, "contents": {"type": "carousel", "contents": bubbles[:12]},
                 "quickReply": quick_reply()}]


def rich_menu_payload(width: int, height: int) -> dict:
    # Original artwork: decorative header 0..28%, buttons 28..92%.
    xs = [0, width // 3, 2 * width // 3, width]
    ys = [round(height * .28), round(height * .60), round(height * .92)]
    areas = []
    for index, label in enumerate(MENU_LABELS):
        row, col = divmod(index, 3)
        areas.append({"bounds": {"x": xs[col], "y": ys[row], "width": xs[col + 1] - xs[col], "height": ys[row + 1] - ys[row]},
                      "action": message_action(label)})
    return {"size": {"width": width, "height": height}, "selected": True,
            "name": "ThaiPolice RAG UX", "chatBarText": "เมนูช่วยเหลือ", "areas": areas}

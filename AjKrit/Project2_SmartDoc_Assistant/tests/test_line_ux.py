from __future__ import annotations

import base64
import hashlib
import hmac
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from assistant import SmartDocAssistant
from chat_history import ChatHistory, conversation_key, resolve_followup
from config import settings
from line_ux import CATEGORIES, LineUX, quick_reply, rich_menu_payload
from line_webhook import create_app, LINE_REPLY_URL, LINE_PUSH_URL
from llm import OllamaClient
from models import AnswerResult, DocumentChunk, RetrievalResult


class UXTests(unittest.TestCase):
    def test_short_explanation_quick_reply_uses_context(self):
        actions = [i["action"] for i in quick_reply()["items"]]
        self.assertNotIn("เอกสารหาย", [a["label"] for a in actions])
        short = next(a for a in actions if a["label"] == "อธิบายสั้นๆ")
        self.assertEqual(short["text"], "อธิบายสั้นๆ หน่อย")
        self.assertIn("คนหาย", resolve_followup(short["text"], [{"topic": "1.4 แจ้งความคนหาย"}]))
        contextual = next(i["action"] for i in quick_reply("1.4 แจ้งความคนหาย")["items"] if i["action"]["label"] == "อธิบายสั้นๆ")
        self.assertIn("แจ้งความคนหาย", contextual["text"])

    def test_acknowledgement_without_loading(self):
        for source_type in ("user", "group"):
            with self.subTest(source_type=source_type), tempfile.TemporaryDirectory() as directory:
                config = replace(settings, artifact_dir=Path(directory), line_channel_secret="test-secret", line_access_token="test-token")
                assistant = Mock()
                assistant.answer.return_value = AnswerResult(question="คนหาย", answer="คำตอบ", grounded=False)
                with patch("line_webhook._line_post", new_callable=AsyncMock) as send:
                    client = TestClient(create_app(assistant, config))
                    def post(text, number):
                        event = {"type": "message", "webhookEventId": str(number), "replyToken": "reply",
                                 "source": {"type": source_type, "userId": "user", **({"groupId": "group"} if source_type == "group" else {})},
                                 "message": {"type": "text", "text": text}}
                        body = json.dumps({"events": [event]}).encode()
                        signature = base64.b64encode(hmac.new(b"test-secret", body, hashlib.sha256).digest()).decode()
                        return client.post("/callback", content=body, headers={"x-line-signature": signature})
                    self.assertEqual(post("คนหาย", 1).status_code, 200)
                    urls = [c.args[0] for c in send.call_args_list]
                    self.assertEqual(urls, [LINE_REPLY_URL, LINE_PUSH_URL])
                    self.assertEqual(send.call_args_list[0].args[2]["messages"][0]["text"], config.line_ack_text)
                    self.assertEqual(assistant.answer.call_count, 1)
                    send.reset_mock()
                    self.assertEqual(post("สวัสดี", 2).status_code, 200)
                    self.assertEqual([c.args[0] for c in send.call_args_list], [LINE_REPLY_URL])
                    self.assertEqual(assistant.answer.call_count, 1)
                    send.reset_mock()
                    self.assertEqual(post("ล้างประวัติ", 3).status_code, 200)
                    self.assertEqual([c.args[0] for c in send.call_args_list], [LINE_PUSH_URL])

    def test_history_persists_bounds_expires_and_clears(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            memory = ChatHistory(path, max_turns=2, ttl_seconds=60)
            for n in range(3):
                memory.append("one", str(n), "answer", "คนหาย")
            self.assertEqual([t["question"] for t in ChatHistory(path).get("one")], ["1", "2"])
            self.assertEqual(memory.get("other"), [])
            with patch("chat_history.time.time", return_value=10**12):
                self.assertEqual(memory.get("one"), [])
            memory.append("one", "new", "answer")
            memory.clear("one")
            self.assertEqual(memory.get("one"), [])

    def test_group_context_is_per_member_and_scope(self):
        first = conversation_key({"type": "group", "groupId": "g", "userId": "u"})
        self.assertNotEqual(first, conversation_key({"type": "group", "groupId": "g", "userId": "v"}))
        self.assertNotEqual(first, conversation_key({"type": "user", "userId": "u"}))
        self.assertIsNone(conversation_key({"type": "group", "groupId": "g"}))

    def test_followup_and_new_topic(self):
        history = [{"question": "เพื่อนหาย", "topic": "1.4 แจ้งความคนหาย"}]
        self.assertIn("แจ้งความคนหาย", resolve_followup("ต้องเตรียมอะไร", history))
        self.assertEqual(resolve_followup("รถหายทำไง", history), "รถหายทำไง")
        self.assertEqual(resolve_followup("ต้องแจ้งรถหายที่ไหน", history), "ต้องแจ้งรถหายที่ไหน")
        self.assertEqual(resolve_followup("ไปที่ไหน", []), "ไปที่ไหน")

    def test_assistant_retrieves_expanded_followup(self):
        assistant = SmartDocAssistant()
        chunk = DocumentChunk("c", "ข้อความ", [25], ["12"], "1.4", "แจ้งความคนหาย")
        retriever = Mock()
        retriever.search.return_value = [RetrievalResult(chunk, dense_score=.8)]
        assistant._retriever = retriever
        result = assistant.answer("ไปที่ไหน", use_llm=False, chat_history=[{"topic": "แจ้งความคนหาย"}])
        self.assertIn("แจ้งความคนหาย", retriever.search.call_args.args[0])
        self.assertEqual(result.question, "ไปที่ไหน")

    def test_followup_without_memory_asks_for_topic(self):
        assistant = SmartDocAssistant()
        result = assistant.answer("ไปที่ไหน", chat_history=[])
        self.assertEqual(result.metrics["intent"], "clarify_topic")
        self.assertEqual(result.retrieval, [])

    def test_llm_history_is_context_not_citation_evidence(self):
        llm = OllamaClient("http://unused", "test")
        chunk = DocumentChunk("c", "แจ้งได้ตลอดเวลา", [25], ["12"], "1.4", "แจ้งความคนหาย")
        llm.chat = Mock(return_value=(' {"answer":"แจ้งได้ตลอดเวลา [S1]", "citations":["S1", "S99"]}', {}))
        result = llm.answer_from_context("คนหาย", [RetrievalResult(chunk, dense_score=.8)],
                                        chat_history=[{"question": "ก่อนหน้า", "answer": "ข้อมูลเก่า [S99]"}])
        messages = llm.chat.call_args.args[0]
        self.assertIn("ไม่ใช่หลักฐาน", messages[0]["content"])
        self.assertIn("HISTORY", messages[1]["content"])
        self.assertEqual(result.citations, ["S1"])

    def test_menu_matches_artwork_and_quick_reply_is_text(self):
        menu = rich_menu_payload(1527, 1030)
        self.assertEqual(len(menu["areas"]), 6)
        self.assertGreater(menu["areas"][0]["bounds"]["y"], 250)
        self.assertEqual(menu["areas"][2]["action"]["text"], "สิทธิและกฎหมาย")
        action = quick_reply("1.3 แจ้งความเอกสารสำคัญหาย")["items"][0]["action"]
        self.assertEqual(action["type"], "message")
        self.assertIn("เอกสารสำคัญหาย", action["text"])
        self.assertLessEqual(len(action["label"]), 20)

    def test_all_topics_are_navigable(self):
        ux = LineUX(settings.index_dir)
        for category in CATEGORIES:
            topics = ux.topics(category)
            self.assertTrue(topics)
            buttons = sum(len(b["body"]["contents"]) - 1 for b in ux.navigation(category)[0]["contents"]["contents"])
            self.assertEqual(buttons, len(topics))

    def test_webhook_navigation_rag_history_clear_and_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(settings, artifact_dir=Path(directory), line_channel_secret="test-secret", line_access_token="test-token")
            assistant = Mock()
            assistant.answer.return_value = AnswerResult(question="คนหาย", answer="คำตอบ", grounded=True,
                                                        sources=[{"section_title": "1.4 แจ้งความคนหาย"}])
            with patch("line_webhook._line_post", new_callable=AsyncMock) as send:
                client = TestClient(create_app(assistant, config))
                def post(text, number):
                    event = {"type": "message", "webhookEventId": str(number), "replyToken": "reply",
                             "source": {"type": "group", "groupId": "group", "userId": "user"},
                             "message": {"type": "text", "id": str(number), "text": text}}
                    body = json.dumps({"events": [event]}).encode()
                    signature = base64.b64encode(hmac.new(b"test-secret", body, hashlib.sha256).digest()).decode()
                    return client.post("/callback", content=body, headers={"x-line-signature": signature})
                self.assertEqual(post("หัวข้อทั้งหมด", 1).status_code, 200)
                assistant.answer.assert_not_called()
                self.assertEqual(send.call_args.args[2]["messages"][0]["type"], "flex")
                post("คนหาย", 2)
                self.assertEqual(send.call_args.args[2]["to"], "group")
                self.assertIn("quickReply", send.call_args.args[2]["messages"][-1])
                post("ไปที่ไหน", 3)
                self.assertEqual(len(assistant.answer.call_args.kwargs["chat_history"]), 1)
                calls = assistant.answer.call_count
                post("ไปที่ไหน", 3)
                self.assertEqual(assistant.answer.call_count, calls)
                post("ล้างประวัติ", 4)
                post("คนหาย", 5)
                self.assertEqual(assistant.answer.call_args.kwargs["chat_history"], [])
                self.assertEqual(client.post("/callback", content=b"{}", headers={"x-line-signature": "bad"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()

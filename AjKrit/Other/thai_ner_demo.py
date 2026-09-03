from __future__ import annotations

import torch
from pythainlp.corpus.common import thai_words
from pythainlp.tokenize import Tokenizer, sent_tokenize, word_tokenize
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForTokenClassification, AutoTokenizer


# --- 1. Load tokenizer and model ---
MODEL_NAME = "pythainlp/thainer-corpus-v2-base-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
model.eval()


# --- 2. Load custom tokenizer ---
custom_words = set(thai_words())
custom_words.update(["จำเลย", "โจทก์", "ศาลฎีกา", "เครื่องหมายการค้า"])
custom_tokenizer = Tokenizer(custom_words, engine="newmm")


# --- 3. Custom NER tags override ---
legal_dict = {
    "จำเลย": "B-LEGAL",
    "โจทก์": "B-LEGAL",
    "เครื่องหมายการค้า": "B-LEGAL",
    "ศาลฎีกา": "B-LEGAL",
}


# --- 4. Input text ---
text = """
ความผิดตาม พ.ร.บ.เครื่องหมายการค้า พ.ศ. ๒๕๓๔ มาตรา ๑๑๐ (๑) ผู้กระทำต้องรู้ว่าสินค้าที่นำเข้ามาในราชอาณาจักร จำหน่าย เสนอจำหน่าย หรือมีไว้เพื่อจำหน่ายนั้น มีเครื่องหมายการค้าปลอมหรือเลียนเครื่องหมายการค้าของบุคคลอื่นที่ได้จดทะเบียนแล้วในราชอาณาจักร

การกระทำที่จะเป็นความผิดตาม พ.ร.บ.เครื่องหมายการค้า พ.ศ. ๒๕๓๔ มาตรา ๑๑๐ (๑) ผู้กระทำต้องรู้ว่าสินค้าที่ตนนำเข้ามาในราชอาณาจักร จำหน่าย เสนอจำหน่าย หรือมีไว้เพื่อจำหน่าย เป็นสินค้าที่มีเครื่องหมายการค้าปลอมหรือเลียนเครื่องหมายการค้าของบุคคลอื่นที่ได้จดทะเบียนแล้วในราชอาณาจักร โจทก์จึงมีหน้าที่นำสืบให้เห็นว่าจำเลยรู้ว่าสินค้าของกลางที่จำเลยเสนอจำหน่ายและมีไว้เพื่อจำหน่ายเป็นสินค้าที่มีเครื่องหมายการค้าปลอมและเลียนเครื่องหมายการค้าของผู้เสียหายที่ได้จดทะเบียนแล้วในราชอาณาจักร

เมื่อพยานหลักฐานที่โจทก์นำสืบไม่ปรากฏข้อเท็จจริงจากทางนำสืบของโจทก์ว่า จำเลยรู้ว่าสินค้าของกลางเป็นสินค้าที่มีเครื่องหมายการค้าปลอมและเลียนเครื่องหมายการค้าของผู้เสียหายที่ได้จดทะเบียนแล้วในราชอาณาจักร กลับได้ความจากผู้รับมอบอำนาจช่วงผู้เสียหายว่า สินค้าของผู้เสียหายยังไม่มีการจำหน่ายในประเทศไทย คงมีจำหน่ายเฉพาะที่ประเทศสหรัฐอเมริกาและญี่ปุ่นเท่านั้น โดยผู้เสียหายยังไม่ได้มอบหมายให้ตัวแทนรายใดเป็นตัวแทนในการจำหน่ายสินค้าในประเทศไทย อันแสดงได้ว่าเครื่องหมายการค้าของผู้เสียหายยังไม่ได้เป็นที่แพร่หลายในประเทศไทย ซึ่งในส่วนนี้จำเลยนำสืบต่อสู้ว่า เครื่องหมายที่ติดบนสินค้าของกลางเป็นสัญลักษณ์ไม้กางเขน จำเลยเชื่อโดยสุจริตว่าเป็นเครื่องหมายที่บุคคลใดก็สามารถนำไปใช้ได้ สอดคล้องกับที่จำเลยให้การปฏิเสธในชั้นสอบสวนว่า จำเลยซื้อสินค้าของกลางมาเพื่อขายต่อ โดยไม่ทราบว่าเป็นสินค้าที่มีเครื่องหมายการค้าปลอมและเลียนเครื่องหมายการค้าของบุคคลอื่น และจำเลยยังมีบาทหลวง ท. เจ้าอาวาสวัด น. เบิกความว่า เครื่องหมายการค้าตามฟ้องเป็นรูปกางเขนของศาสนาคริสต์ โดยโจทก์ไม่ได้นำสืบให้เห็นเป็นอย่างอื่น ดังนั้น เมื่อจำเลยไม่ได้เป็นผู้ทำปลอมและเลียนเครื่องหมายการค้าด้วยตนเอง และการที่เครื่องหมายการค้าของผู้เสียหายยังไม่ได้เป็นที่แพร่หลายในประเทศไทย ประกอบกับการที่เครื่องหมายการค้าของผู้เสียหายมีลักษณะใกล้เคียงกับไม้กางเขนในศาสนาคริสต์ แม้เครื่องหมายการค้าของโจทก์จะได้รับการจดทะเบียนเครื่องหมายการค้าแล้ว ก็อาจเป็นไปได้ที่ประชาชนหรือผู้ขายสินค้าทั่วไปจะไม่รู้จักเครื่องหมายการค้าของผู้เสียหาย และอาจเข้าใจได้ว่าเครื่องหมายการค้าที่ปรากฏบนสินค้าของกลางเป็นไม้กางเขนในศาสนาคริสต์ซึ่งบุคคลทั่วไปสามารถใช้ได้ พยานหลักฐานโจทก์เท่าที่นำสืบมายังมีความสงสัยตามสมควรว่า จำเลยรู้หรือไม่ว่าสินค้าของกลางที่จำเลยเสนอจำหน่ายและมีไว้เพื่อจำหน่ายเป็นสินค้าที่มีเครื่องหมายการค้าปลอมและเลียนเครื่องหมายการค้าของผู้เสียหายที่ได้จดทะเบียนแล้วในราชอาณาจักร ต้องยกประโยชน์แห่งความสงสัยให้แก่จำเลย (คำพิพากษาศาลฎีกาที่ ๑๒๕/๒๕๖๗)

แผนกคดีทรัพย์สินทางปัญญาและการค้าระหว่างประเทศในศาลฎีกา

กันยายน ๒๕๖๗
""".strip()


# --- 5. Sentence split ---
sentences = sent_tokenize(text, engine="whitespace")


def ner_tag_sentence(sentence: str) -> list[tuple[str, str]]:
    # Tokenize sentence using PyThaiNLP tokenizer with placeholders
    tokens = custom_tokenizer.word_tokenize(sentence.replace(" ", "<_>"))

    # Tokenize using HF tokenizer for NER
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    word_ids = inputs.word_ids()

    # Convert ID to token and tag
    token_results: list[tuple[str, str]] = []
    seen_word_ids: set[int] = set()

    for word_id, pred_id in zip(word_ids, predictions):
        if word_id is None or word_id in seen_word_ids:
            continue

        token_text = tokens[word_id].replace("<_>", " ")
        label_text = model.config.id2label[pred_id]

        # Override tag with legal dictionary
        if token_text in legal_dict:
            label_text = f"{legal_dict[token_text]} (custom)"

        if token_text.strip():
            token_results.append((token_text, label_text))
            seen_word_ids.add(word_id)

    return token_results


# --- 6. Run NER on all sentences ---
all_ner: list[tuple[str, str]] = []
for sent in sentences:
    all_ner.extend(ner_tag_sentence(sent))


# --- 7. Print results ---
console = Console()
table = Table(title="NER Result", show_lines=True)
table.add_column("No.", justify="right", style="cyan", no_wrap=True)
table.add_column("Token", style="white")
table.add_column("Tag", style="green")

for index, (word, tag) in enumerate(all_ner, start=1):
    table.add_row(str(index), word, tag)

console.print()
console.print(table)
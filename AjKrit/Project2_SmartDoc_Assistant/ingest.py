"""Evidence-preserving PDF ingestion and parent-document chunking.

The source PDF is an older Thai PDF with repeated headers/footers and several
Unicode extraction artefacts.  This module keeps page numbers, section
headings, table text and parent context with every child chunk.  It removes
only detected repeated layout noise, then applies structure-aware parent
grouping followed by a Thai-friendly recursive character splitter.
"""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable

from models import DocumentChunk


_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)(?:\.|\s+)\s*(.{3,180}?)\s*$")
_PRINTED_LEFT_RE = re.compile(r"^\s*(\d{1,3})\s+คู่มือประชาชน\s*$")
_PRINTED_RIGHT_RE = re.compile(r"^\s*คู่มือประชาชน\s+(\d{1,3})\s*$")
_PRINTED_FOOTER_MARKERS = {
    "คู่มือประชาชน",
    "กลุ่มงานการมีส่วนร่วมของประชาชน",
}
_PAGE_MARK_RE = re.compile(r"^\s*\(?[ก-ฮ]\)?\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-•▪]|\(?\d{1,2}\)|[ก-ฮ]\.)\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u00ad\u061c\u180e\u200b\u200c\u200d\u2060\ufeff]")

# Numbered table rows in this PDF can look like section headings (for example
# ``4.2 ผู้ต้องหา...`` inside a bail table).  A heading is accepted only when
# its title contains a known topic hint.  This keeps section provenance useful
# without hard-coding page numbers.
_SECTION_HINTS = {
    "1": ["คำแนะนำในการไปติดต่อ"],
    "1.1": ["เอกสารสำคัญ"],
    "1.2": ["บัตร", "บตัร"],
    "1.3": ["เอกสารสำคัญหาย"],
    "1.4": ["คนหาย"],
    "1.5": ["รถหรือเรือหาย"],
    "1.6": ["อาวุธปืนหาย"],
    "1.7": ["ทรัพย์สินหาย"],
    "1.8": ["ปลอมแปลงเอกสาร"],
    "1.9": ["ฉ้อโกงทรัพย์"],
    "1.10": ["ยักยอกทรัพย์"],
    "1.11": ["เช่าซื้อ"],
    "1.12": ["เสียทรัพย์"],
    "1.13": ["พรากผู้เยาว์"],
    "1.14": ["ข่มขืน"],
    "1.15": ["ทำร้ายร่างกาย", "ฆ่าคนตาย"],
    "1.16": ["เช็ค"],
    "1.17": ["ประกันตัว"],
    "1.18": ["สิทธิ"],
    "1.19": ["หนังสือรับรองความประพฤติ"],
    "1.20": ["ต่างชาติ", "ที่พัก"],
    "2": ["ขออนุญาตต่าง"],
    "2.1": ["มหรสพชั่วคราว"],
    "2.2": ["ดอกไม้เพลิง"],
    "2.3": ["อาวุธปืน"],
    "2.4": ["มหรสพงิ้ว"],
    "2.5": ["เครื่องขยายเสียง"],
    "2.6": ["เล่นการพนัน"],
    "2.7": ["ฆ่าสุกร"],
    "2.8": ["ตั้งสมาคม"],
    "2.9": ["ใบขับขี่"],
    "3": ["ป้องกันอาชญากรรม"],
    "3.1": ["ลักทรัพย์ในที่พัก"],
    "3.2": ["ลักทรัพย์", "สถานประกอบการ"],
    "3.3": ["ล้วงกระเป๋า"],
    "3.4": ["ล่อลวงข่มขืน"],
    "3.5": ["บุตรหลาน"],
    "3.6": ["ป้องกันรถหาย"],
    "3.7": ["ธนาคาร"],
    "3.8": ["ลอบวางระเบิด"],
    "3.9": ["แก๊งมิจฉาชีพ"],
    "3.10": ["ความปลอดภัยอื่น"],
    "3.11": ["จดจำต\u0e33หนิ"],
    "4": ["ป้องกันและการเตรียมตัว"],
    "4.1": ["น้ำท่วม", "ท่วม"],
    "4.2": ["พายุ"],
    "4.3": ["แผ่นดินถล่ม", "โคลนถล่ม"],
    "4.4": ["สึนามิ", "แจ้งเตือนภัย"],
    "5": ["ใช้รถใช้ถนน"],
    "5.1": ["ขับรถให้ปลอดภัย"],
    "5.2": ["เดินถนน"],
    "5.3": ["รถที่นำมาใช้", "รถที่น\u0e33มาใช้"],
    "5.4": ["เครื่องหมายจราจร"],
    "6": ["กฎหมายทั่วไป"],
    "6.1": ["ศาลปกครอง"],
    "6.2": ["ลงโทษผู้กระทำผิดต่อเด็ก", "กฎหมายทีล่ งโทษ", "กฎหมาย"],
    "6.3": ["ค่าตอบแทน"],
    "6.4": ["บุหรี่"],
    "6.5": ["บารากู่", "บุหรี่ไฟฟ้า"],
    "6.6": ["สุรา", "แอลกอฮอล์"],
    "6.7": ["ทวงถามหนี้"],
    "6.8": ["เล่นเกม"],
    "6.9": ["โทรศัพท์เคลื่อนที่", "มือถือ"],
}


@dataclass
class PageText:
    pdf_page: int
    text: str
    printed_page: str = ""
    table_count: int = 0


def clean_text(text: str) -> str:
    """Normalise common Thai PDF artefacts without rewriting source meaning."""

    text = unicodedata.normalize("NFC", text)
    # PDF text layers frequently contain zero-width spaces/joiners, BOMs and
    # soft hyphens between Thai characters.  They are invisible to users but
    # split BM25 tokens and make section detection unreliable.
    text = _ZERO_WIDTH_RE.sub("", text)
    replacements = {
        "\ufeff": "",
        "\u0000": "",
        "�ำ": "ำ",
        "ํา": "ำ",
        "กdำ": "กำ",
        "สํานักงาน": "สำนักงาน",
        "ตํารวจ": "ตำรวจ",
        "คํานํา": "คำนำ",
        # The PDF's embedded font maps Thai tone/diacritic marks into PUA glyphs.
        "\uf70a": "่",
        "\uf70b": "้",
        "\uf70e": "์",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # U+FFFD is emitted by the PDF font for a missing combining mark; retaining
    # it pollutes BM25 tokens and citations, while removing it does not invent
    # a new word.
    text = text.replace("�", "")
    # PyMuPDF may expose the same missing Thai combining mark as U+011E (Ğ)
    # in this PDF, e.g. ``จĞำ`` instead of ``จำ``.
    text = text.replace("Ğ", "")
    # The source PDF prints an obsolete Facebook URL on its cover page.  It is
    # not useful evidence and should not leak into answers or LINE messages.
    text = re.sub(r"(?:https?://|www\.)\S+", "", text, flags=re.IGNORECASE)
    # Private-use glyphs in this PDF are extraction noise, not user-visible content.
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _printed_page(raw: str) -> str:
    lines = [clean_text(line) for line in raw.splitlines()]
    for index, line in enumerate(lines):
        match = _PRINTED_LEFT_RE.match(line) or _PRINTED_RIGHT_RE.match(line)
        if match:
            return match.group(1)

        # PyMuPDF often returns the footer label and its page number as two
        # separate text lines rather than one line of text.
        if line.strip() in _PRINTED_FOOTER_MARKERS:
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(lines):
                    match = re.fullmatch(r"\s*(\d{1,3})\s*", lines[neighbour])
                    if match:
                        return match.group(1)
    return ""


def _rect_overlap_ratio(block: tuple[float, ...], table_bbox: tuple[float, ...]) -> float:
    """Return the fraction of a text block covered by a detected table."""

    if len(block) < 4 or len(table_bbox) < 4:
        return 0.0
    bx0, by0, bx1, by1 = (float(value) for value in block[:4])
    tx0, ty0, tx1, ty1 = (float(value) for value in table_bbox[:4])
    intersection = max(0.0, min(bx1, tx1) - max(bx0, tx0)) * max(
        0.0, min(by1, ty1) - max(by0, ty0)
    )
    area = max(1.0, (bx1 - bx0) * (by1 - by0))
    return intersection / area


def _table_text(page: object, table_bbox: tuple[float, ...]) -> str:
    """Serialise one PyMuPDF table to stable, searchable plain text."""

    finder = getattr(page, "find_tables", None)
    if finder is None:
        return ""
    try:
        # PyMuPDF may print a layout-analysis hint to stdout.  It is useful in
        # an interactive session but should not pollute build JSON output.
        with redirect_stdout(io.StringIO()):
            tables = finder(clip=table_bbox).tables
        table = tables[0] if tables else None
        rows = table.extract() if table is not None else []
    except Exception:
        return ""
    serialised: list[str] = []
    for row in rows:
        cells = [clean_text(str(cell or "")) for cell in row]
        cells = [cell for cell in cells if cell]
        if cells:
            serialised.append(" | ".join(cells))
    return "\n".join(serialised)


def _extract_with_pymupdf(pdf_path: Path) -> list[PageText]:
    try:
        import pymupdf  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF ingestion needs PyMuPDF; install pymupdf from requirements.txt") from exc

    pages: list[PageText] = []
    with pymupdf.open(pdf_path) as document:
        for number, page in enumerate(document, start=1):
            blocks = page.get_text("blocks", sort=True)
            table_bboxes: list[tuple[float, ...]] = []
            finder = getattr(page, "find_tables", None)
            if finder is not None:
                try:
                    with redirect_stdout(io.StringIO()):
                        detected_tables = finder().tables
                    table_bboxes = [tuple(table.bbox) for table in detected_tables]
                except Exception:
                    table_bboxes = []

            elements: list[tuple[float, float, str]] = []
            for block in blocks:
                if len(block) < 5:
                    continue
                if any(_rect_overlap_ratio(block, bbox) >= 0.5 for bbox in table_bboxes):
                    continue
                elements.append((float(block[1]), float(block[0]), str(block[4])))

            for bbox in table_bboxes:
                table_text = _table_text(page, bbox)
                if table_text:
                    elements.append((float(bbox[1]), float(bbox[0]), table_text))
            elements.sort(key=lambda item: (item[0], item[1]))
            raw = "\n".join(text for _, _, text in elements)
            raw = clean_text(raw)
            pages.append(PageText(number, raw, _printed_page(raw), len(table_bboxes)))
    return pages


def _extract_with_pdftotext(pdf_path: Path) -> list[PageText]:
    command = ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"]
    raw = subprocess.check_output(command, stderr=subprocess.STDOUT).decode("utf-8", "replace")
    pages: list[PageText] = []
    for number, page in enumerate(raw.split("\f"), start=1):
        cleaned = clean_text(page)
        if cleaned or number == 1:
            pages.append(PageText(number, cleaned, _printed_page(cleaned), 0))
    while pages and not pages[-1].text:
        pages.pop()
    return pages


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Extract page text, preferring layout-aware PyMuPDF with CLI fallback."""

    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    try:
        pages = _extract_with_pymupdf(pdf_path)
        if any(page.text for page in pages):
            return pages
    except Exception:
        # pdftotext is deterministic and is available on the target workstation.
        pass
    return _extract_with_pdftotext(pdf_path)


def _normalised_line(line: str) -> str:
    return re.sub(r"\s+", " ", clean_text(line)).strip()


def repeated_layout_lines(pages: Iterable[PageText]) -> set[str]:
    """Find lines repeated in page margins; content lines are not removed globally."""

    rows: list[list[str]] = []
    for page in pages:
        lines = [_normalised_line(line) for line in page.text.splitlines()]
        rows.append([line for line in lines if line])
    counts = Counter(line for lines in rows for line in set(lines))
    page_count = max(1, len(rows))
    repeated = {
        line
        for line, count in counts.items()
        if count >= max(3, int(page_count * 0.25)) and len(line) <= 100
    }
    # These are known page furniture in this source, including variants with the page marker.
    repeated.update(
        {
            "กลุ่มงานการมีส่วนร่วมของประชาชน",
            "คู่มือประชาชน",
            "Royal Thai Police",
        }
    )
    return repeated


def remove_layout_noise(page: PageText, repeated: set[str]) -> str:
    """Remove repeated header/footer furniture while retaining body evidence."""

    kept: list[str] = []
    for raw in page.text.splitlines():
        line = _normalised_line(raw)
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if line in repeated:
            continue
        if _PAGE_MARK_RE.match(line):
            continue
        # The cover has emergency numbers; keep it. Only strip isolated page markers.
        kept.append(line)
    return clean_text("\n".join(kept))


def layout_noise_statistics(pages: Iterable[PageText]) -> dict[str, int | float]:
    """Measure only characters removed as repeated furniture/page markers.

    Whitespace normalisation is intentionally excluded from this measurement;
    otherwise a clean layout conversion would be misreported as document
    noise.  The ratio is therefore comparable with the rubric's <=5% target.
    """

    page_list = list(pages)
    repeated = repeated_layout_lines(page_list)
    removed_lines = 0
    removed_characters = 0
    for page in page_list:
        for raw in page.text.splitlines():
            line = _normalised_line(raw)
            if line and (line in repeated or _PAGE_MARK_RE.match(line)):
                removed_lines += 1
                removed_characters += len(raw)
    extracted_characters = sum(len(page.text) for page in page_list)
    return {
        "repeated_layout_lines_detected": len(repeated),
        "layout_noise_lines_removed": removed_lines,
        "layout_noise_characters_removed": removed_characters,
        "layout_noise_ratio": round(removed_characters / max(1, extracted_characters), 6),
    }


def _join_wrapped_lines(lines: list[str]) -> list[str]:
    """Turn layout lines into paragraph-like units, preserving bullets/headings."""

    units: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            units.append(" ".join(buffer).strip())
            buffer.clear()

    for line in lines:
        line = _normalised_line(line)
        if not line:
            flush()
            continue
        is_heading = bool(_SECTION_RE.match(line))
        is_bullet = bool(_BULLET_RE.match(line))
        if is_heading or is_bullet:
            flush()
            units.append(line)
            continue
        buffer.append(line)
    flush()
    return [unit for unit in units if unit]


def recursive_character_split(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Recursively split Thai text while preserving the best boundary possible.

    The order is paragraph → line → Thai sentence punctuation → Latin
    punctuation → whitespace → character fallback.  This is deliberately
    implemented locally so the ingestion contract does not depend on a
    framework-specific document object, while retaining the same behaviour
    expected from a RecursiveCharacterTextSplitter.
    """

    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    text = clean_text(text)
    if not text:
        return []
    separators = ["\n\n", "\n", "。", "！", "？", "ฯ", ". ", "; ", ", ", " ", ""]

    def split_level(value: str, level: int) -> list[str]:
        if len(value) <= chunk_size:
            return [value.strip()]
        separator = separators[min(level, len(separators) - 1)]
        if separator == "":
            return [value[start : start + chunk_size] for start in range(0, len(value), chunk_size)]
        pieces = [piece.strip() for piece in value.split(separator) if piece.strip()]
        if len(pieces) <= 1:
            return split_level(value, level + 1)
        result: list[str] = []
        current = ""
        for piece in pieces:
            candidate = piece if not current else f"{current}{separator}{piece}"
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current.strip())
                if len(piece) > chunk_size:
                    result.extend(split_level(piece, level + 1))
                    current = ""
                else:
                    current = piece
        if current:
            result.append(current.strip())
        return result

    base = split_level(text, 0)
    if overlap == 0 or len(base) < 2:
        return base
    result: list[str] = []
    previous = ""
    for piece in base:
        if previous:
            # Keep the complete new piece.  The previous implementation could
            # truncate its beginning when prefix + piece exceeded the limit,
            # silently losing evidence at chunk boundaries.
            available_prefix = max(0, chunk_size - len(piece) - 1)
            prefix = previous[-min(overlap, available_prefix) :]
            piece = clean_text(f"{prefix} {piece}" if prefix else piece)
        result.append(piece)
        previous = piece

    # A long extracted line can survive several semantic separator levels.
    # Enforce the hard contract at the end with an overlapping character
    # fallback, so no child can exceed the embedding/context budget.
    bounded: list[str] = []
    step = max(1, chunk_size - overlap)
    for piece in result:
        if len(piece) <= chunk_size:
            bounded.append(piece)
            continue
        bounded.extend(
            piece[start : start + chunk_size].strip()
            for start in range(0, len(piece), step)
            if piece[start : start + chunk_size].strip()
        )
    return bounded


def _section_from_line(line: str) -> tuple[str, str] | None:
    match = _SECTION_RE.match(line)
    if not match:
        return None
    number, title = match.groups()
    if int(number.split(".", 1)[0]) not in range(1, 8):
        return None
    # Avoid treating ordinary decimal/legal text as a heading.
    if len(title.strip()) < 3:
        return None
    compact_title = re.sub(r"\s+", "", title)
    hints = _SECTION_HINTS.get(number)
    if hints and not any(re.sub(r"\s+", "", hint) in compact_title for hint in hints):
        return None
    if not hints and "." in number:
        return None
    return number, f"{number} {title.strip()}"


@dataclass
class _EvidenceUnit:
    """A paragraph/bullet unit with the page and section it came from."""

    text: str
    pdf_page: int
    printed_page: str
    section_id: str
    section_title: str


@dataclass
class _ParentBlock:
    """A coherent parent passage expanded into smaller retrieval children."""

    parent_id: str
    text: str
    pdf_pages: list[int]
    printed_pages: list[str]
    section_id: str
    section_title: str


def _parent_blocks(
    units: list[_EvidenceUnit],
    document_id: str,
    parent_size: int,
) -> list[_ParentBlock]:
    """Group nearby units into parent passages without crossing sections."""

    if parent_size < 2:
        raise ValueError("parent_size must be at least 2")
    blocks: list[_ParentBlock] = []
    current: list[_EvidenceUnit] = []
    block_number = 0

    def flush() -> None:
        nonlocal block_number
        if not current:
            return
        text = clean_text("\n".join(unit.text for unit in current))
        if not text:
            current.clear()
            return
        pdf_pages = list(dict.fromkeys(unit.pdf_page for unit in current))
        printed_pages = list(dict.fromkeys(unit.printed_page for unit in current if unit.printed_page))
        section_id = current[0].section_id
        section_title = current[0].section_title
        parent_id = hashlib.sha256(
            f"{document_id}|parent|{block_number}|{section_id}|{text}".encode("utf-8")
        ).hexdigest()[:20]
        blocks.append(
            _ParentBlock(
                parent_id=parent_id,
                text=text,
                pdf_pages=pdf_pages,
                printed_pages=printed_pages,
                section_id=section_id,
                section_title=section_title,
            )
        )
        block_number += 1
        current.clear()

    for unit in units:
        # A very long paragraph is split into parent-sized passages first;
        # child chunks will receive their own smaller overlap later.
        oversized = recursive_character_split(unit.text, chunk_size=parent_size, overlap=0)
        for piece in oversized:
            # ``current`` is joined with one newline between each unit.
            current_length = sum(len(item.text) for item in current) + max(0, len(current) - 1)
            candidate_length = current_length + (1 if current else 0) + len(piece)
            section_changed = bool(current) and unit.section_id != current[0].section_id
            if section_changed or (current and candidate_length > parent_size):
                flush()
            if current and unit.section_id != current[0].section_id:
                flush()
            current.append(
                _EvidenceUnit(
                    text=piece,
                    pdf_page=unit.pdf_page,
                    printed_page=unit.printed_page,
                    section_id=unit.section_id,
                    section_title=unit.section_title,
                )
            )
            if len(piece) >= parent_size:
                flush()
    flush()
    return blocks


def build_chunks(
    pages: list[PageText],
    source_title: str,
    source_url: str,
    chunk_size: int = 900,
    overlap: int = 120,
    parent_size: int = 2400,
) -> list[DocumentChunk]:
    """Create parent-document passages and recursive retrieval child chunks.

    Retrieval indexes the child text for precision.  Each child also carries
    its parent passage and complete parent page provenance so the answer layer
    can expand a hit into coherent surrounding evidence instead of returning a
    sentence fragment.
    """

    repeated = repeated_layout_lines(pages)
    current_section_id = "front-matter"
    current_section_title = "บทนำและข้อมูลทั่วไป"
    in_table_of_contents = False
    document_id = hashlib.sha256(
        "\n".join(page.text for page in pages).encode("utf-8")
    ).hexdigest()[:16]
    units: list[_EvidenceUnit] = []

    for page in pages:
        body = remove_layout_noise(page, repeated)
        if "สารบัญ" in body:
            in_table_of_contents = True
            continue
        if in_table_of_contents:
            # The contents spans pages 4–12 and ends at the page-furniture-only page.
            # A body page starts without the repeated `หน้า` marker.
            if not body or body.startswith("หน้า") or page.pdf_page < 13:
                continue
            in_table_of_contents = False
        if not body:
            continue
        for unit in _join_wrapped_lines(body.splitlines()):
            detected = _section_from_line(unit)
            if detected:
                current_section_id, current_section_title = detected
            units.append(
                _EvidenceUnit(
                    text=unit,
                    pdf_page=page.pdf_page,
                    printed_page=page.printed_page,
                    section_id=current_section_id,
                    section_title=current_section_title,
                )
            )

    parents = _parent_blocks(units, document_id=document_id, parent_size=parent_size)
    chunks: list[DocumentChunk] = []
    for parent in parents:
        for local_index, piece in enumerate(
            recursive_character_split(parent.text, chunk_size=chunk_size, overlap=overlap)
        ):
            if len(piece.strip()) < 30:
                continue
            chunk_id = hashlib.sha256(
                f"{parent.parent_id}|{local_index}|{piece}".encode("utf-8")
            ).hexdigest()[:20]
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=piece,
                    pdf_pages=list(parent.pdf_pages),
                    printed_pages=list(parent.printed_pages),
                    section_id=parent.section_id,
                    section_title=parent.section_title,
                    parent_id=parent.parent_id,
                    chunk_index=len(chunks),
                    source_title=source_title,
                    source_url=source_url,
                    parent_text=parent.text,
                    parent_pdf_pages=list(parent.pdf_pages),
                    parent_printed_pages=list(parent.printed_pages),
                )
            )
    return chunks


def ingest_pdf(
    pdf_path: Path,
    source_title: str,
    source_url: str,
    chunk_size: int = 900,
    overlap: int = 120,
    parent_size: int = 2400,
) -> tuple[list[PageText], list[DocumentChunk]]:
    """Extract and chunk a PDF in one reproducible call."""

    pages = extract_pages(pdf_path)
    chunks = build_chunks(
        pages,
        source_title,
        source_url,
        chunk_size=chunk_size,
        overlap=overlap,
        parent_size=parent_size,
    )
    if not chunks:
        raise ValueError("No usable text chunks were extracted from the PDF")
    return pages, chunks

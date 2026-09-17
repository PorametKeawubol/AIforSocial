"""Configuration for the SmartDoc Assistant.

Secrets are intentionally read from environment variables only.  Do not put
LINE credentials in this module, the repository, or a notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - useful for lightweight unit tests
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings, with conservative defaults for a 8 GiB GPU laptop."""

    pdf_path: Path
    artifact_dir: Path
    index_dir: Path
    embedding_model: str
    rerank_model: str
    ollama_host: str
    ollama_model: str
    ollama_timeout_seconds: float
    top_k: int
    candidate_k: int
    max_context_chars: int
    max_answer_tokens: int
    chunk_size: int
    chunk_overlap: int
    parent_size: int
    source_title: str
    source_topic: str
    source_url: str
    line_channel_secret: str
    line_access_token: str
    line_ack_text: str
    enable_line: bool
    history_max_turns: int = 4
    history_ttl_seconds: int = 86400

    @classmethod
    def from_env(cls) -> "Settings":
        """Load configuration from ``.env``/environment without printing secrets."""

        return cls(
            pdf_path=Path(
                os.getenv("PDF_PATH", str(ROOT / "คู่มือประชาชน สำนักงานตำรวจแห่งชาติ.pdf"))
            ).expanduser(),
            artifact_dir=Path(os.getenv("ARTIFACT_DIR", str(ROOT / "artifacts"))).expanduser(),
            index_dir=Path(os.getenv("INDEX_DIR", str(ROOT / "artifacts" / "index"))).expanduser(),
            # BGE-M3 was already benchmarked in this workspace and supports Thai/multilingual text.
            embedding_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
            # Empty means retrieval-only reranking fallback; set this to enable CrossEncoder.
            rerank_model=os.getenv("RERANK_MODEL", ""),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
            # Production: 9B balances local LINE latency and answer quality.
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
            ollama_timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
            top_k=max(1, int(os.getenv("TOP_K", "5"))),
            candidate_k=max(10, int(os.getenv("CANDIDATE_K", "24"))),
            max_context_chars=max(2000, int(os.getenv("MAX_CONTEXT_CHARS", "10000"))),
            max_answer_tokens=max(64, int(os.getenv("MAX_ANSWER_TOKENS", "420"))),
            chunk_size=max(100, int(os.getenv("CHUNK_SIZE", "900"))),
            chunk_overlap=min(
                max(0, int(os.getenv("CHUNK_OVERLAP", "120"))),
                max(0, max(100, int(os.getenv("CHUNK_SIZE", "900"))) - 1),
            ),
            parent_size=max(
                max(200, int(os.getenv("PARENT_SIZE", "2400"))),
                max(100, int(os.getenv("CHUNK_SIZE", "900"))),
            ),
            source_title=os.getenv("SOURCE_TITLE", "คู่มือประชาชน สำนักงานตำรวจแห่งชาติ"),
            source_topic=os.getenv(
                "SOURCE_TOPIC",
                "การติดต่อสถานีตำรวจ การขออนุญาต การป้องกันอาชญากรรม ภัยพิบัติ การใช้รถใช้ถนน และกฎหมายทั่วไป",
            ),
            # Leave blank when the source URL is unavailable; title, topic and
            # PDF/printed page provenance are still shown to the user.
            source_url=os.getenv("SOURCE_URL", ""),
            line_channel_secret=os.getenv("CHANNEL_SECRET", ""),
            line_access_token=os.getenv("CHANNEL_ACCESS_TOKEN", ""),
            line_ack_text=os.getenv(
                "LINE_ACK_TEXT", "รับคำถามแล้ว กำลังค้นจากคู่มือประชาชนและจะส่งคำตอบพร้อมแหล่งข้อมูลให้ครับ"
            ),
            enable_line=_bool("ENABLE_LINE", True),
            history_max_turns=max(1, min(10, int(os.getenv("HISTORY_MAX_TURNS", "4")))),
            history_ttl_seconds=max(60, int(os.getenv("HISTORY_TTL_SECONDS", "86400"))),
        )

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()

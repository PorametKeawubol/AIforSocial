"""Environment-backed configuration for the Mercular chatbot."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CATEGORY_URLS = (
    # A small set of public, user-facing category pages gives the demo enough
    # variety for multi-condition searches without crawling the full catalog.
    "https://www.mercular.com/audio/headphone",
    "https://www.mercular.com/audio/speaker",
    "https://www.mercular.com/gaming/gaming-mouse",
    "https://www.mercular.com/gaming/gaming-keyboard",
    "https://www.mercular.com/gaming/gaming-headphone-speaker/gaming-headset",
    "https://www.mercular.com/computer/computer-monitor",
    "https://www.mercular.com/computer/computer-accessories",
    "https://www.mercular.com/gaming/streaming/webcam",
    "https://www.mercular.com/professional-audio/microphone",
    "https://www.mercular.com/smartphone-tablet-acc/power-bank",
    "https://www.mercular.com/smart-gadget/smart-watch-fitness-tracker",
    "https://www.mercular.com/printer-ing",
    "https://www.mercular.com/accessories",
)


def _integer(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _floating(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value) if math.isfinite(value) else default


def _truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    line_channel_secret: str
    line_channel_access_token: str
    snapshot_path: Path
    port: int
    log_level: str
    top_k: int
    history_ttl_seconds: int
    history_size: int
    request_timeout_seconds: float
    request_retries: int
    scrape_delay_seconds: float
    max_products_per_category: int
    stale_after_hours: int
    category_urls: tuple[str, ...]
    verify_robots: bool
    bot_name: str = "MercuMate"
    line_channel_id: str = ""
    public_base_url: str = ""
    line_coupon_id: str = ""
    nlp_backend: str = "phayathaibert"
    phayathaibert_model_name: str = "clicknext/phayathaibert"
    phayathaibert_min_confidence: float = 0.30
    phayathaibert_local_files_only: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_DIR / ".env")
        configured_urls = tuple(
            item.strip()
            for item in os.getenv("MERCULAR_CATEGORY_URLS", "").split(",")
            if item.strip()
        )
        snapshot_path = Path(
            os.getenv(
                "MERCULAR_SNAPSHOT_PATH",
                str(PROJECT_DIR / "data" / "mercular_products.json"),
            )
        ).expanduser()
        if not snapshot_path.is_absolute():
            snapshot_path = PROJECT_DIR / snapshot_path
        nlp_backend = os.getenv("NLP_BACKEND", "phayathaibert").strip().casefold()
        if nlp_backend not in {"phayathaibert", "rules"}:
            nlp_backend = "phayathaibert"
        return cls(
            line_channel_secret=os.getenv("LINE_CHANNEL_SECRET", "").strip(),
            line_channel_access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip(),
            snapshot_path=snapshot_path,
            port=_integer("PORT", 5000, 1),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            # The assignment and LINE carousel deliberately cap recommendations
            # at five; keep configuration truthful instead of silently truncating.
            top_k=min(5, _integer("TOP_K", 5, 1)),
            history_ttl_seconds=_integer("HISTORY_TTL_SECONDS", 1_800, 1),
            history_size=_integer("HISTORY_SIZE", 20, 1),
            request_timeout_seconds=_floating("SCRAPER_TIMEOUT_SECONDS", 15.0, 1.0),
            request_retries=_integer("SCRAPER_RETRIES", 3, 0),
            scrape_delay_seconds=_floating("SCRAPER_DELAY_SECONDS", 1.0, 0.25),
            max_products_per_category=_integer("MAX_PRODUCTS_PER_CATEGORY", 20, 1),
            stale_after_hours=_integer("SNAPSHOT_STALE_AFTER_HOURS", 24, 1),
            category_urls=configured_urls or DEFAULT_CATEGORY_URLS,
            verify_robots=_truthy("SCRAPER_VERIFY_ROBOTS", True),
            bot_name=os.getenv("BOT_NAME", "MercuMate").strip() or "MercuMate",
            line_channel_id=os.getenv("LINE_CHANNEL_ID", "").strip(),
            public_base_url=os.getenv("PUBLIC_BASE_URL", "").strip(),
            line_coupon_id=os.getenv("LINE_COUPON_ID", "").strip(),
            nlp_backend=nlp_backend,
            phayathaibert_model_name=(
                os.getenv("PHAYATHAIBERT_MODEL_NAME", "clicknext/phayathaibert").strip()
                or "clicknext/phayathaibert"
            ),
            phayathaibert_min_confidence=min(
                1.0, _floating("PHAYATHAIBERT_MIN_CONFIDENCE", 0.30, 0.0)
            ),
            phayathaibert_local_files_only=_truthy(
                "PHAYATHAIBERT_LOCAL_FILES_ONLY", False
            ),
        )


__all__ = ["DEFAULT_CATEGORY_URLS", "PROJECT_DIR", "Settings"]

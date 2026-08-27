"""Hierarchical catalogue navigation backed by product breadcrumb data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:  # Support package imports and direct execution from this directory.
    from .models import Product, clean_text
    from .nlp import normalize_text
except ImportError:  # pragma: no cover
    from models import Product, clean_text
    from nlp import normalize_text


ALL_CATEGORIES_COMMAND = "เลือกหมวดสินค้า"
GAMING_CATEGORIES_COMMAND = "เลือกอุปกรณ์เกมมิ่ง"
COMPUTER_CATEGORIES_COMMAND = "เลือกหมวดคอมพิวเตอร์"
MOBILE_CATEGORIES_COMMAND = "เลือกหมวดมือถือและแท็บเล็ต"

GAMING_ROOT = "เกมมิ่ง"
COMPUTER_ROOT = "คอมพิวเตอร์"
MOBILE_ROOT = "Smartphone / Tablet / ACC"

CATEGORY_BROWSE_PREFIX = "เลือกหมวด:"
CATEGORY_SEARCH_PREFIX = "ดูสินค้าหมวด:"
MAX_NAVIGATION_DEPTH = 4
MAX_CATEGORY_OPTIONS = 50


@dataclass(frozen=True, slots=True)
class CategoryNavigationRequest:
    """A request to browse a path or show products below it."""

    path: tuple[str, ...] = ()
    show_products: bool = False


@dataclass(frozen=True, slots=True)
class CategoryOption:
    """One counted child category and the next command sent by LINE."""

    label: str
    product_count: int
    command: str
    has_children: bool = False


@dataclass(frozen=True, slots=True)
class CategoryMenu:
    """A category-picker page independent from LINE presentation details."""

    path: tuple[str, ...]
    product_count: int
    options: tuple[CategoryOption, ...]

    @property
    def title(self) -> str:
        return "เลือกประเภทสินค้า" if not self.path else self.path[-1]

    @property
    def breadcrumb(self) -> str:
        return "สินค้าทั้งหมด" if not self.path else " › ".join(self.path)

    @property
    def prompt(self) -> str:
        if not self.path:
            return "คุณต้องการสินค้าหมวดไหน?"
        if self.path == (GAMING_ROOT,):
            return "คุณต้องการอุปกรณ์เกมมิ่งประเภทไหน?"
        if self.path == (COMPUTER_ROOT,):
            return "คุณต้องการคอมพิวเตอร์ประเภทไหน?"
        if self.path == (MOBILE_ROOT,):
            return "คุณต้องการมือถือ แท็บเล็ต หรืออุปกรณ์เสริมประเภทไหน?"
        return f"คุณต้องการสินค้า {self.path[-1]} ประเภทไหน?"


_DIRECT_REQUESTS = {
    normalize_text(ALL_CATEGORIES_COMMAND): CategoryNavigationRequest(),
    normalize_text("มีสินค้าอะไรบ้าง"): CategoryNavigationRequest(),
    normalize_text("มีสินค้าอะไรแนะนำบ้าง"): CategoryNavigationRequest(),
    normalize_text("สินค้าทั้งหมด"): CategoryNavigationRequest(),
    normalize_text("ดูสินค้าทั้งหมด"): CategoryNavigationRequest(),
    normalize_text(GAMING_CATEGORIES_COMMAND): CategoryNavigationRequest(
        (GAMING_ROOT,)
    ),
    normalize_text("เกมมิ่ง"): CategoryNavigationRequest((GAMING_ROOT,)),
    normalize_text(COMPUTER_CATEGORIES_COMMAND): CategoryNavigationRequest(
        (COMPUTER_ROOT,)
    ),
    normalize_text("คอมพิวเตอร์"): CategoryNavigationRequest((COMPUTER_ROOT,)),
    normalize_text(MOBILE_CATEGORIES_COMMAND): CategoryNavigationRequest(
        (MOBILE_ROOT,)
    ),
    normalize_text("มือถือ/แท็บเล็ต"): CategoryNavigationRequest((MOBILE_ROOT,)),
    normalize_text(MOBILE_ROOT): CategoryNavigationRequest((MOBILE_ROOT,)),
}


def _parse_path(value: str) -> tuple[str, ...]:
    parts = tuple(
        part
        for item in value.split(">")
        if (part := clean_text(item, limit=100))
    )
    return parts[:MAX_NAVIGATION_DEPTH]


def parse_category_navigation(message: object | None) -> CategoryNavigationRequest | None:
    """Recognise only explicit menu commands, leaving ordinary NLP text untouched."""

    raw_text = clean_text(message, limit=500)
    if not raw_text:
        return None
    direct = _DIRECT_REQUESTS.get(normalize_text(raw_text))
    if direct is not None:
        return direct

    for prefix, show_products in (
        (CATEGORY_BROWSE_PREFIX, False),
        (CATEGORY_SEARCH_PREFIX, True),
    ):
        if raw_text.startswith(prefix):
            path = _parse_path(raw_text[len(prefix) :])
            return (
                CategoryNavigationRequest(path, show_products)
                if path
                else None
            )
    return None


def _path_matches(candidate: tuple[str, ...], requested: tuple[str, ...]) -> bool:
    if len(candidate) < len(requested):
        return False
    return all(
        normalize_text(actual) == normalize_text(expected)
        for actual, expected in zip(
            candidate[: len(requested)], requested, strict=True
        )
    )


def _path_command(prefix: str, path: tuple[str, ...]) -> str:
    return f"{prefix} {' > '.join(path)}"


def build_category_menu(
    products: Iterable[Product],
    path: tuple[str, ...] = (),
) -> CategoryMenu | None:
    """Aggregate the next breadcrumb level and count unique matching products."""

    requested = tuple(
        cleaned
        for part in path
        if (cleaned := clean_text(part, limit=100))
    )
    matching: list[Product] = []
    child_products: dict[str, set[str]] = {}
    child_names: dict[str, str] = {}
    child_has_children: dict[str, bool] = {}
    child_index = len(requested)

    for product in products:
        candidate = tuple(part for part in product.category_path if part)
        if not candidate or not _path_matches(candidate, requested):
            continue
        matching.append(product)
        if len(candidate) <= child_index:
            continue
        child = candidate[child_index]
        key = normalize_text(child)
        if not key:
            continue
        child_names.setdefault(key, child)
        child_products.setdefault(key, set()).add(product.id)
        child_has_children[key] = child_has_children.get(key, False) or (
            len(candidate) > child_index + 1
        )

    if not matching:
        return None

    ranked_keys = sorted(
        child_products,
        key=lambda key: (-len(child_products[key]), child_names[key].casefold()),
    )[:MAX_CATEGORY_OPTIONS]
    options: list[CategoryOption] = []
    if requested:
        options.append(
            CategoryOption(
                "ดูทั้งหมดในหมวดนี้",
                len({product.id for product in matching}),
                _path_command(CATEGORY_SEARCH_PREFIX, requested),
            )
        )
    for key in ranked_keys:
        child_path = (*requested, child_names[key])
        has_children = child_has_children[key]
        options.append(
            CategoryOption(
                child_names[key],
                len(child_products[key]),
                _path_command(
                    CATEGORY_BROWSE_PREFIX if has_children else CATEGORY_SEARCH_PREFIX,
                    child_path,
                ),
                has_children,
            )
        )
    return CategoryMenu(
        requested,
        len({product.id for product in matching}),
        tuple(options),
    )


__all__ = [
    "ALL_CATEGORIES_COMMAND",
    "CATEGORY_BROWSE_PREFIX",
    "CATEGORY_SEARCH_PREFIX",
    "COMPUTER_CATEGORIES_COMMAND",
    "COMPUTER_ROOT",
    "CategoryMenu",
    "CategoryNavigationRequest",
    "CategoryOption",
    "GAMING_CATEGORIES_COMMAND",
    "GAMING_ROOT",
    "MOBILE_CATEGORIES_COMMAND",
    "MOBILE_ROOT",
    "build_category_menu",
    "parse_category_navigation",
]

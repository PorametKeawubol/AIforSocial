"""Hierarchical catalogue-navigation contracts."""

from catalog_navigation import (
    CATEGORY_BROWSE_PREFIX,
    CATEGORY_SEARCH_PREFIX,
    COMPUTER_ROOT,
    GAMING_ROOT,
    MOBILE_ROOT,
    build_category_menu,
    parse_category_navigation,
)
from models import Product


def _product(identifier: str, path: tuple[str, ...]) -> Product:
    return Product(
        id=identifier,
        sku=identifier,
        name=f"สินค้า {identifier}",
        brand="Test",
        category=path[-1],
        category_path=path,
        price=1_000,
        original_price=None,
        image_url="",
        product_url=f"https://www.mercular.com/product-{identifier}",
        in_stock=True,
    )


def test_common_product_list_phrases_open_the_root_picker() -> None:
    for message in (
        "มีสินค้าอะไรบ้าง",
        "มีสินค้าอะไรแนะนำบ้าง",
        "สินค้าทั้งหมด",
        "เลือกหมวดสินค้า",
    ):
        request = parse_category_navigation(message)
        assert request is not None
        assert request.path == ()
        assert request.show_products is False


def test_rich_menu_category_commands_open_the_expected_roots() -> None:
    expected = {
        "เลือกอุปกรณ์เกมมิ่ง": GAMING_ROOT,
        "เกมมิ่ง": GAMING_ROOT,
        "เลือกหมวดคอมพิวเตอร์": COMPUTER_ROOT,
        "คอมพิวเตอร์": COMPUTER_ROOT,
        "เลือกหมวดมือถือและแท็บเล็ต": MOBILE_ROOT,
        "มือถือ/แท็บเล็ต": MOBILE_ROOT,
    }

    for message, root in expected.items():
        request = parse_category_navigation(message)
        assert request is not None
        assert request.path == (root,)
        assert request.show_products is False


def test_category_menu_counts_children_and_drills_down_before_searching() -> None:
    products = [
        _product("mouse-1", (GAMING_ROOT, "เมาส์เกมมิ่ง")),
        _product("console-1", (GAMING_ROOT, "Game Console", "PlayStation")),
        _product("console-2", (GAMING_ROOT, "Game Console", "Nintendo")),
        _product("notebook-1", (COMPUTER_ROOT, "โน๊ตบุ๊ค")),
    ]

    root = build_category_menu(products)
    gaming = build_category_menu(products, (GAMING_ROOT,))

    assert root is not None
    assert [(option.label, option.product_count) for option in root.options] == [
        (GAMING_ROOT, 3),
        (COMPUTER_ROOT, 1),
    ]
    assert gaming is not None
    assert gaming.prompt == "คุณต้องการอุปกรณ์เกมมิ่งประเภทไหน?"
    assert gaming.options[0].label == "ดูทั้งหมดในหมวดนี้"
    assert gaming.options[0].command.startswith(CATEGORY_SEARCH_PREFIX)
    console = next(option for option in gaming.options if option.label == "Game Console")
    mouse = next(option for option in gaming.options if option.label == "เมาส์เกมมิ่ง")
    assert console.has_children is True
    assert console.command.startswith(CATEGORY_BROWSE_PREFIX)
    assert mouse.has_children is False
    assert mouse.command.startswith(CATEGORY_SEARCH_PREFIX)


def test_generated_navigation_commands_round_trip_safely() -> None:
    command = f"{CATEGORY_BROWSE_PREFIX} {GAMING_ROOT} > Game Console"
    request = parse_category_navigation(command)

    assert request is not None
    assert request.path == (GAMING_ROOT, "Game Console")
    assert request.show_products is False
    assert parse_category_navigation("หา Game Console ไม่เกิน 10000") is None

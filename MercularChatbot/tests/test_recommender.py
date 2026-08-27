from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models import Product
from nlp import (
    INTENT_GREETING,
    INTENT_SEARCH,
    SORT_PRICE_ASC,
    SORT_PRICE_DESC,
    CommandEntities,
    ParsedCommand,
    ThaiCommandParser,
    compact_text,
)
from recommender import MAX_TOP_K, ProductRecommender


SNAPSHOT_PATH = PROJECT_DIR / "data" / "mercular_products.json"


def load_snapshot_products() -> list[Product]:
    """Load the checked-in scraper output; integration tests never call the network."""

    payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"]
    assert payload["products"]
    return [Product.from_dict(row) for row in payload["products"]]


def snapshot_parser(products: list[Product]) -> ThaiCommandParser:
    return ThaiCommandParser(
        brands=sorted({product.brand for product in products if product.brand}),
        categories=sorted({product.category for product in products if product.category}),
    )


def make_product(
    identifier: str,
    *,
    name: str | None = None,
    brand: str = "Sony",
    category: str = "หูฟัง",
    category_path: tuple[str, ...] = ("Audio", "หูฟัง"),
    price: float | None = 2_000,
    original_price: float | None = 2_500,
    in_stock: bool = True,
    tags: tuple[str, ...] = ("wireless", "gaming"),
    description: str = "หูฟังไร้สาย bluetooth ตัดเสียงรบกวน สีดำ",
    scraped_at: str = "2026-08-20T00:00:00+00:00",
) -> Product:
    return Product(
        id=identifier,
        sku=f"sku-{identifier}",
        name=name or f"Sony Wireless Headphone {identifier}",
        brand=brand,
        category=category,
        category_path=category_path,
        price=price,
        original_price=original_price,
        image_url=f"https://example.com/{identifier}.jpg",
        product_url=f"https://example.com/products/{identifier}",
        in_stock=in_stock,
        tags=tags,
        description=description,
        scraped_at=scraped_at,
    )


def search_command(**entities: object) -> ParsedCommand:
    return ParsedCommand(
        INTENT_SEARCH,
        0.95,
        CommandEntities(**entities),
        raw_text="test",
        normalized_text="test",
    )


def test_hard_filters_all_constraints_before_random_selection() -> None:
    matching = [
        make_product(
            f"match-{index}",
            brand="Sony" if index % 2 else "JBL",
            price=1_500 + index * 100,
        )
        for index in range(8)
    ]
    mismatches = [
        make_product("wrong-brand", brand="Bose"),
        make_product("too-cheap", price=900),
        make_product("too-expensive", price=4_000),
        make_product("sold-out", in_stock=False),
        make_product("wired", tags=("wired",), description="หูฟังมีสาย"),
        make_product(
            "wrong-category",
            category="ลำโพง",
            category_path=("Audio", "ลำโพง"),
            name="Sony wireless speaker",
        ),
        make_product("missing-price", price=None),
    ]
    command = search_command(
        category="หูฟัง",
        brands=("Sony", "JBL"),
        min_price=1_000,
        max_price=3_000,
        in_stock=True,
        query="ไร้สาย",
    )
    recommender = ProductRecommender(rng=random.Random(4), candidate_pool_size=20)

    results = recommender.recommend(matching + mismatches, command, user_id="u")

    assert len(results) == 5
    assert len({product.id for product in results}) == 5
    assert all(recommender.product_matches(product, command) for product in results)
    assert {product.id for product in results} <= {product.id for product in matching}


def test_catalogue_navigation_uses_exact_breadcrumb_prefix() -> None:
    products = [
        make_product(
            "gaming-mouse",
            name="Gaming Mouse",
            category="เมาส์เกมมิ่ง",
            category_path=("เกมมิ่ง", "เมาส์เกมมิ่ง"),
        ),
        make_product(
            "office-computer",
            name="Office Computer",
            category="คอมพิวเตอร์ พีซี",
            category_path=("คอมพิวเตอร์", "คอมพิวเตอร์ พีซี"),
        ),
        make_product(
            "computer-accessory",
            name="Computer Gaming Accessory",
            category="อุปกรณ์เสริม",
            category_path=("อุปกรณ์เสริม", "สาย Cable"),
        ),
    ]
    recommender = ProductRecommender()

    gaming = recommender.filter_products(
        products,
        search_command(category_path=("เกมมิ่ง",)),
    )
    computer = recommender.filter_products(
        products,
        search_command(category_path=("คอมพิวเตอร์",)),
    )

    assert [product.id for product in gaming] == ["gaming-mouse"]
    assert [product.id for product in computer] == ["office-computer"]


def test_negative_brand_and_bilingual_colour_are_hard_constraints() -> None:
    parser = ThaiCommandParser(brands=["Razer", "Logitech"], categories=["เมาส์"])
    command = parser.parse(
        "หาเมาส์ wireless สีขาว ไม่เอา Razer งบไม่เกิน 3,000"
    )
    products = [
        make_product(
            "razer-white",
            brand="Razer",
            category="เมาส์",
            category_path=("เกมมิ่ง", "เมาส์"),
            name="RAZER WIRELESS MOUSE WHITE",
            price=1990,
        ),
        make_product(
            "logitech-white",
            brand="Logitech",
            category="เมาส์",
            category_path=("เกมมิ่ง", "เมาส์"),
            name="LOGITECH WIRELESS MOUSE WHITE",
            price=2490,
        ),
        make_product(
            "logitech-black",
            brand="Logitech",
            category="เมาส์",
            category_path=("เกมมิ่ง", "เมาส์"),
            name="LOGITECH WIRELESS MOUSE BLACK",
            price=2490,
        ),
    ]

    results = ProductRecommender().filter_products(products, command)

    assert [product.id for product in results] == ["logitech-white"]


def test_strict_price_bounds_exclude_equal_boundary_products() -> None:
    products = [
        make_product("below", price=6_989),
        make_product("equal", price=6_990),
        make_product("above", price=6_991),
    ]
    recommender = ProductRecommender(rng=random.Random(1))

    greater = search_command(min_price=6_990, min_price_inclusive=False)
    lower = search_command(max_price=6_990, max_price_inclusive=False)
    inclusive = search_command(min_price=6_990, min_price_inclusive=True)

    assert [product.id for product in recommender.filter_products(products, greater)] == [
        "above"
    ]
    assert [product.id for product in recommender.filter_products(products, lower)] == [
        "below"
    ]
    assert [product.id for product in recommender.filter_products(products, inclusive)] == [
        "equal",
        "above",
    ]


def test_live_parser_recommender_command_returns_matching_products() -> None:
    parser = ThaiCommandParser(brands=["Sony", "JBL"], categories=["หูฟัง"])
    command = parser.parse("หาหูฟัง Sony ไม่เกิน 3000")
    products = [make_product(str(index), price=1_500 + index * 100) for index in range(7)]

    results = ProductRecommender(rng=random.Random(2)).recommend(
        products,
        command,
        user_id="integration-user",
    )

    assert command.entities.query == ""
    assert len(results) == 5
    assert all(product.brand == "Sony" and product.price <= 3_000 for product in results)


def test_checked_in_snapshot_keeps_audio_leaf_categories_mutually_exclusive() -> None:
    products = load_snapshot_products()
    parser = snapshot_parser(products)
    recommender = ProductRecommender(rng=random.Random(20), candidate_pool_size=50)

    speakers = recommender.filter_products(products, parser.parse("ลำโพง"))
    headphones = recommender.filter_products(products, parser.parse("หูฟัง"))

    assert speakers
    assert not any("HEADSET" in product.name.upper() for product in speakers)
    assert headphones
    assert not any(
        "ลำโพง" in product.category or "speaker" in product.category.casefold()
        for product in headphones
    )


def test_checked_in_snapshot_mic_does_not_match_micro_sd_prefix() -> None:
    products = load_snapshot_products()
    parser = snapshot_parser(products)
    recommender = ProductRecommender(rng=random.Random(21), candidate_pool_size=50)
    micro_sd = make_product(
        "micro-sd",
        name="MICRO SD CARD 256GB",
        category="อุปกรณ์เสริม",
        category_path=("อุปกรณ์เสริม",),
        tags=(),
        description="",
    )
    catalog = [*products, micro_sd]

    microphones = recommender.filter_products(catalog, parser.parse("ไมค์"))
    micro_sd_results = recommender.filter_products(catalog, parser.parse("Micro SD"))

    assert microphones
    assert micro_sd.id not in {product.id for product in microphones}
    assert micro_sd.id in {product.id for product in micro_sd_results}
    # Enriched specs may spell the same slot as ``microSD`` without a space.
    assert all("microsd" in compact_text(product.search_text) for product in micro_sd_results)


def test_checked_in_snapshot_printer_alias_matches_mercular_leaf_label() -> None:
    products = load_snapshot_products()
    parser = snapshot_parser(products)
    command = parser.parse("หาเครื่องพิมพ์พร้อมส่ง")
    real_printer = make_product(
        "real-printer",
        name="PRINTER CANON PIXMA G3010",
        brand="Canon",
        category="เครื่องปริ้น / หมึก",
        category_path=("เครื่องปริ้น / หมึก",),
        tags=(),
        description="",
    )
    recommender = ProductRecommender(rng=random.Random(22))

    snapshot_matches = recommender.filter_products(products, command)
    matches = recommender.filter_products([*products, real_printer], command)

    assert command.category == "เครื่องพิมพ์"
    assert all(product.in_stock is not False for product in snapshot_matches)
    assert "real-printer" in {product.id for product in matches}


def test_checked_in_snapshot_specific_gaming_mouse_excludes_false_positives() -> None:
    products = load_snapshot_products()
    parser = snapshot_parser(products)
    recommender = ProductRecommender(rng=random.Random(23), candidate_pool_size=50)

    matches = recommender.filter_products(products, parser.parse("เมาส์เกมมิ่ง"))
    generic_matches = recommender.filter_products(products, parser.parse("เมาส์"))

    assert len(matches) >= 5
    assert all(product.category == "เมาส์เกมมิ่ง" for product in matches)
    assert all(product.name.upper().startswith("MOUSE ") for product in matches)
    assert not any("MOUSE PAD" in product.name.upper() for product in matches)
    assert not any("WRIST REST" in product.name.upper() for product in matches)
    assert not any(product.category == "อุปกรณ์คอมพิวเตอร์" for product in matches)
    assert any(product.category == "เมาส์ (Mouse)" for product in generic_matches)
    assert not any("MOUSE PAD" in product.name.upper() for product in generic_matches)
    assert not any("WRIST REST" in product.name.upper() for product in generic_matches)


def test_checked_in_snapshot_explicit_mouse_pad_queries_return_only_pads() -> None:
    products = load_snapshot_products()
    parser = snapshot_parser(products)
    recommender = ProductRecommender(rng=random.Random(24), candidate_pool_size=50)

    for index, text in enumerate(("mouse pad", "แผ่นรองเมาส์")):
        command = parser.parse(text)
        results = recommender.recommend(
            products,
            command,
            user_id=f"mouse-pad-{index}",
        )

        assert command.category in {"เมาส์", "แผ่นรองเมาส์"}
        assert len(results) == 5
        assert all("MOUSE PAD" in product.name.upper() for product in results)


def test_checked_in_snapshot_literal_audio_subtypes_do_not_expand_to_parent() -> None:
    products = load_snapshot_products()
    parser = snapshot_parser(products)
    recommender = ProductRecommender(rng=random.Random(26), candidate_pool_size=50)

    for text in ("earbuds", "เอียร์บัด"):
        command = parser.parse(text)
        matches = recommender.filter_products(products, command)
        assert command.category == "หูฟัง"
        assert command.query
        assert matches
        assert not any("HEADSET" in product.name.upper() for product in matches)

    for text in ("soundbar", "ซาวด์บาร์"):
        matches = recommender.filter_products(products, parser.parse(text))
        assert matches
        assert all(
            "SOUNDBAR" in product.name.upper() or "SOUND BAR" in product.name.upper()
            for product in matches
        )


def test_checked_in_snapshot_mixed_leaves_exclude_base_product_accessories() -> None:
    products = load_snapshot_products()
    parser = snapshot_parser(products)
    recommender = ProductRecommender(rng=random.Random(27), candidate_pool_size=50)

    microphones = recommender.filter_products(products, parser.parse("ไมค์"))
    watches = recommender.filter_products(products, parser.parse("สมาร์ทวอทช์"))
    generic_accessory = make_product(
        "generic-accessory",
        name="USB ACCESSORY",
        category="อุปกรณ์เสริม",
        category_path=("อุปกรณ์เสริม",),
        tags=(),
        description="",
    )
    generic_accessories = recommender.filter_products(
        [*products, generic_accessory], parser.parse("อุปกรณ์เสริม")
    )

    assert microphones
    assert all(
        "MICROPHONE" in product.name.upper() or "MIC " in product.name.upper()
        for product in microphones
    )
    assert not any(
        marker in product.name.upper()
        for product in microphones
        for marker in (" ARM ", " STAND ", "FILTER", " CABLE ")
    )
    assert watches
    assert not any(product.name.startswith("สาย Apple Watch") for product in watches)
    assert generic_accessories
    assert {product.category for product in generic_accessories} == {"อุปกรณ์เสริม"}
    assert not any(
        product.category == "ไมโครโฟนและอุปกรณ์เสริม"
        for product in generic_accessories
    )


def test_checked_in_snapshot_residual_subtypes_remain_literal() -> None:
    products = load_snapshot_products()
    parser = snapshot_parser(products)
    recommender = ProductRecommender(rng=random.Random(28), candidate_pool_size=50)

    cases = (
        ("toner", "TONER"),
        ("scanner", "SCANNER"),
        ("ขาไมค์", "MICROPHONE"),
        ("fitness tracker", ""),
        ("UPS", "UPS"),
        ("flash drive", "FLASH DRIVE"),
        ("USB hub", "USB HUB"),
        ("conference camera", "CONFERENCE CAMERA"),
    )
    for text, name_marker in cases:
        command = parser.parse(text)
        matches = recommender.filter_products(products, command)
        assert command.query
        if name_marker:
            assert all(name_marker in product.name.upper() for product in matches)


def test_thai_subtype_aliases_match_english_snapshot_names_literally() -> None:
    products = load_snapshot_products()
    recommender = ProductRecommender(rng=random.Random(30), candidate_pool_size=50)
    cases = (
        (search_command(category="ไมโครโฟน", query="สายไมค์"), "MICROPHONE CABLE"),
        (search_command(category="อุปกรณ์เสริม", query="สาย apple watch"), "APPLE WATCH"),
        (search_command(category="อุปกรณ์คอมพิวเตอร์", query="เครื่องสำรองไฟ"), "UPS"),
        (search_command(category="เครื่องพิมพ์", query="สแกนเนอร์"), "SCANNER"),
        (search_command(category="อุปกรณ์เสริม", query="แฟลชไดรฟ์"), "FLASH DRIVE"),
        (search_command(category="เว็บแคม", query="กล้องประชุม"), "CONFERENCE CAMERA"),
    )
    for command, marker in cases:
        matches = recommender.filter_products(products, command)
        assert all(marker in product.name.upper() for product in matches)


def test_gaming_monitor_requires_product_level_gaming_evidence() -> None:
    products = load_snapshot_products()
    command = search_command(category="จอคอม", query="gaming")

    matches = ProductRecommender(rng=random.Random(31)).filter_products(products, command)

    assert matches
    assert all("PORTABLE MONITOR" not in product.name.upper() for product in matches)
    assert all(
        any(marker in product.name.upper() for marker in ("ODYSSEY", "ULTRAGEAR", "GAMING"))
        for product in matches
    )


def test_mechanical_keyboard_does_not_fuzzy_match_membrane_category_text() -> None:
    parser = ThaiCommandParser(categories=["คีย์บอร์ดเกมมิ่ง"])
    products = [
        make_product(
            "mechanical",
            name="KEYBOARD ASUS TUF GAMING (MECHANICAL SWITCH คีย์บอร์ด)",
            category="คีย์บอร์ดเกมมิ่ง",
            category_path=("เกมมิ่ง", "คีย์บอร์ดเกมมิ่ง"),
            tags=("keyboard", "mechanical"),
            description="",
        ),
        make_product(
            "membrane",
            name="KEYBOARD HYPERX GAMING (MEMBRANE SWITCH คีย์บอร์ด)",
            category="คีย์บอร์ดเกมมิ่ง",
            category_path=("เกมมิ่ง", "คีย์บอร์ดเกมมิ่ง"),
            tags=("keyboard", "membrane"),
            description="",
        ),
    ]
    recommender = ProductRecommender(rng=random.Random(32))

    for text in ("keyboard mechanical", "คีย์บอร์ด mechanical"):
        matches = recommender.filter_products(products, parser.parse(text))
        assert [product.id for product in matches] == ["mechanical"]


def test_keyboard_accessory_residuals_select_accessories_not_base_keyboards() -> None:
    parser = ThaiCommandParser(categories=["คีย์บอร์ดเกมมิ่ง"])
    base_keyboard = make_product(
        "keyboard",
        name="KEYBOARD HYPERX ALLOY (TACTILE SWITCH คีย์บอร์ด)",
        category="คีย์บอร์ดเกมมิ่ง",
        category_path=("เกมมิ่ง", "คีย์บอร์ดเกมมิ่ง"),
        tags=("keyboard",),
        description="",
    )
    accessories = {
        "keycap": make_product(
            "keycap",
            name="PBT KEYCAP SET FOR MECHANICAL KEYBOARD",
            category="คีย์บอร์ดเกมมิ่ง",
            category_path=("เกมมิ่ง", "คีย์บอร์ดเกมมิ่ง"),
            tags=("keycap",),
            description="",
        ),
        "switch": make_product(
            "keyboard-switch",
            name="MECHANICAL SWITCH SET FOR KEYBOARD",
            category="คีย์บอร์ดเกมมิ่ง",
            category_path=("เกมมิ่ง", "คีย์บอร์ดเกมมิ่ง"),
            tags=("keyboard switch",),
            description="",
        ),
        "wrist": make_product(
            "keyboard-wrist-rest",
            name="KEYBOARD WRIST REST MEMORY FOAM",
            category="คีย์บอร์ดเกมมิ่ง",
            category_path=("เกมมิ่ง", "คีย์บอร์ดเกมมิ่ง"),
            tags=("wrist rest",),
            description="",
        ),
        "case": make_product(
            "keyboard-case",
            name="KEYBOARD CARRYING CASE 75 PERCENT",
            category="คีย์บอร์ดเกมมิ่ง",
            category_path=("เกมมิ่ง", "คีย์บอร์ดเกมมิ่ง"),
            tags=("keyboard case",),
            description="",
        ),
    }
    products = [base_keyboard, *accessories.values()]
    recommender = ProductRecommender(rng=random.Random(25))

    cases = (
        ("keyboard keycap", "keycap"),
        ("คีย์แคป", "keycap"),
        ("keyboard switch", "keyboard-switch"),
        ("สวิตช์คีย์บอร์ด", "keyboard-switch"),
        ("ที่รองข้อมือคีย์บอร์ด", "keyboard-wrist-rest"),
        ("wrist rest", "keyboard-wrist-rest"),
        ("เคสคีย์บอร์ด", "keyboard-case"),
    )
    for text, expected_id in cases:
        results = recommender.filter_products(products, parser.parse(text))
        assert [product.id for product in results] == [expected_id]

    base_results = recommender.filter_products(products, parser.parse("คีย์บอร์ด"))
    assert [product.id for product in base_results] == ["keyboard"]


def test_specific_gaming_category_never_matches_broad_parent_or_sibling_products() -> None:
    parser = ThaiCommandParser(
        brands=["Razer"],
        categories=["เมาส์เกมมิ่ง", "หูฟังเกมมิ่ง", "คีย์บอร์ดเกมมิ่ง"],
    )
    command = parser.parse("หาเมาส์เกมมิ่งไม่เกิน 2000 พร้อมส่ง")
    mice = [
        make_product(
            f"mouse-{index}",
            name=f"Razer Gaming Mouse {index}",
            brand="Razer",
            category="เมาส์เกมมิ่ง",
            category_path=("เกมมิ่ง", "เมาส์เกมมิ่ง"),
            price=1_000 + index * 100,
            tags=("gaming", "mouse"),
            description="เมาส์เกมมิ่งพร้อมส่ง",
        )
        for index in range(6)
    ]
    siblings = [
        make_product(
            "gaming-headset",
            name="Razer Gaming Headset",
            brand="Razer",
            category="หูฟังเกมมิ่ง",
            category_path=("เกมมิ่ง", "หูฟังเกมมิ่ง"),
            price=1_500,
            tags=("gaming", "headset"),
        ),
        make_product(
            "gaming-keyboard",
            name="Razer Gaming Keyboard",
            brand="Razer",
            category="คีย์บอร์ดเกมมิ่ง",
            category_path=("เกมมิ่ง", "คีย์บอร์ดเกมมิ่ง"),
            price=1_900,
            tags=("gaming", "keyboard"),
        ),
        make_product(
            "mouse-pad",
            name="Razer Gaming Mouse Pad XXL",
            brand="Razer",
            category="เมาส์เกมมิ่ง",
            category_path=("เกมมิ่ง", "เมาส์เกมมิ่ง"),
            price=900,
            tags=("gaming", "mouse pad"),
        ),
        make_product(
            "wrist-rest",
            name="Mouse Wrist Rest",
            brand="Razer",
            category="เมาส์เกมมิ่ง",
            category_path=("เกมมิ่ง", "เมาส์เกมมิ่ง"),
            price=500,
            tags=("gaming", "wrist rest"),
        ),
    ]

    results = ProductRecommender(rng=random.Random(1)).recommend(
        mice + siblings,
        command,
        user_id="live-case",
    )

    assert command.category == "เมาส์เกมมิ่ง"
    assert command.max_price == 2_000
    assert command.in_stock is True
    assert len(results) == 5
    assert all(product.id.startswith("mouse-") for product in results)


def test_base_mouse_request_returns_empty_instead_of_substituting_accessories() -> None:
    parser = ThaiCommandParser(categories=["เมาส์เกมมิ่ง"])
    command = parser.parse("หาเมาส์เกมมิ่งไม่เกิน 2000 พร้อมส่ง")
    products = [
        make_product(
            "real-mouse-over-budget",
            name="Gaming Mouse Pro",
            category="เมาส์เกมมิ่ง",
            category_path=("เกมมิ่ง", "เมาส์เกมมิ่ง"),
            price=2_900,
            tags=("gaming", "mouse"),
        ),
        make_product(
            "cheap-pad",
            name="Gaming Mousepad XL",
            category="เมาส์เกมมิ่ง",
            category_path=("เกมมิ่ง", "เมาส์เกมมิ่ง"),
            price=590,
            tags=("gaming", "mouse pad"),
        ),
        make_product(
            "cheap-rest",
            name="Mouse Wrist Rest",
            category="เมาส์เกมมิ่ง",
            category_path=("เกมมิ่ง", "เมาส์เกมมิ่ง"),
            price=390,
            tags=("gaming", "wrist rest"),
        ),
    ]

    results = ProductRecommender(rng=random.Random(2)).recommend(products, command)

    assert results == []


def test_residual_feature_is_a_hard_bilingual_constraint() -> None:
    products = [
        make_product(f"wireless-{index}") for index in range(6)
    ] + [
        make_product(
            "wired",
            name="Sony Wired Headphone",
            tags=("wired",),
            description="หูฟังมีสาย",
        )
    ]
    command = search_command(category="หูฟัง", query="ไร้สาย")

    results = ProductRecommender(rng=random.Random(1)).recommend(products, command)

    assert len(results) == 5
    assert all("wireless" in product.tags for product in results)


def test_explicit_price_sort_returns_actual_extreme_five_without_sampling() -> None:
    products = [make_product(str(index), price=float(5_000 - index * 250)) for index in range(12)]
    recommender = ProductRecommender(rng=random.Random(8), candidate_pool_size=10)

    cheapest = recommender.recommend(
        products,
        search_command(category="หูฟัง", sort=SORT_PRICE_ASC),
    )
    most_expensive = recommender.recommend(
        products,
        search_command(category="หูฟัง", sort=SORT_PRICE_DESC),
    )

    all_prices = sorted(product.price for product in products)
    assert [product.price for product in cheapest] == all_prices[:5]
    assert [product.price for product in most_expensive] == list(reversed(all_prices[-5:]))


def test_base_mouse_request_rejects_drawing_tablet_in_mislabeled_mouse_leaf() -> None:
    products = [
        make_product(
            "mouse",
            name="MOUSE LOGITECH MX MASTER 3S",
            category="เมาส์ (Mouse)",
            category_path=("คอมพิวเตอร์", "อุปกรณ์คอมพิวเตอร์", "เมาส์ (Mouse)"),
            price=4_590,
        ),
        make_product(
            "tablet",
            name="PEN WACOM MOVINK PAD PRO 14",
            category="เมาส์ (Mouse)",
            category_path=("คอมพิวเตอร์", "อุปกรณ์คอมพิวเตอร์", "เมาส์ (Mouse)"),
            price=32_900,
            tags=(),
            description="drawing tablet",
        ),
    ]

    results = ProductRecommender(rng=random.Random(1)).recommend(
        products,
        search_command(category="เมาส์", sort=SORT_PRICE_DESC),
    )

    assert [product.id for product in results] == ["mouse"]


def test_returns_all_available_when_fewer_than_five_and_deduplicates_ids() -> None:
    products = [make_product("a"), make_product("b"), make_product("c")]
    products.append(make_product("a", name="duplicate snapshot row"))

    results = ProductRecommender(rng=random.Random(3)).recommend(
        products,
        search_command(category="หูฟัง"),
    )

    assert {product.id for product in results} == {"a", "b", "c"}
    assert len(results) == 3


def test_distinct_ids_with_same_normalized_display_name_are_deduplicated() -> None:
    duplicate_variants = [
        make_product(
            "blackshark-usb",
            name="HEADSET RAZER BLACKSHARK V2 X BLACK (7.1)",
            brand="Razer",
            category="หูฟังเกมมิ่งครอบหู",
            category_path=("เกมมิ่ง", "หูฟังเกมมิ่งครอบหู"),
        ),
        make_product(
            "blackshark-non-usb",
            name="headset  razer blackshark v2 x black 7.1",
            brand="Razer",
            category="หูฟังเกมมิ่งครอบหู",
            category_path=("เกมมิ่ง", "หูฟังเกมมิ่งครอบหู"),
        ),
    ]
    unique_products = [make_product(f"unique-{index}") for index in range(6)]
    recommender = ProductRecommender(rng=random.Random(29), candidate_pool_size=20)
    command = search_command(category="หูฟัง")

    filtered = recommender.filter_products([*duplicate_variants, *unique_products], command)
    results = recommender.recommend([*duplicate_variants, *unique_products], command)

    retained_variants = {
        product.id for product in filtered if product.id.startswith("blackshark-")
    }
    assert retained_variants == {"blackshark-usb"}
    assert sum(product.id.startswith("blackshark-") for product in results) <= 1
    assert len({product.id for product in results}) == len(results)


def test_top_k_is_configurable_below_five_but_never_exceeds_carousel_limit() -> None:
    products = [make_product(str(index)) for index in range(12)]
    recommender = ProductRecommender(rng=random.Random(3))
    command = search_command(category="หูฟัง")

    assert len(recommender.recommend(products, command, top_k=3, user_id="three")) == 3
    assert len(recommender.recommend(products, command, top_k=99, user_id="cap")) == MAX_TOP_K
    assert recommender.recommend(products, command, top_k=0) == []


def test_seeded_recommenders_are_deterministic() -> None:
    products = [make_product(str(index)) for index in range(15)]
    command = search_command(category="หูฟัง")
    first = ProductRecommender(rng=random.Random(42), candidate_pool_size=15)
    second = ProductRecommender(rng=random.Random(42), candidate_pool_size=15)

    first_rounds = [
        [product.id for product in first.recommend(products, command, user_id="same")]
        for _ in range(4)
    ]
    second_rounds = [
        [product.id for product in second.recommend(products, command, user_id="same")]
        for _ in range(4)
    ]
    assert first_rounds == second_rounds


def test_history_avoids_products_until_unseen_pool_is_exhausted() -> None:
    products = [make_product(str(index)) for index in range(10)]
    recommender = ProductRecommender(rng=random.Random(7), candidate_pool_size=10)
    command = search_command(category="หูฟัง")

    first = recommender.recommend(products, command, user_id="u")
    second = recommender.recommend(products, command, user_id="u")

    assert {product.id for product in first}.isdisjoint(product.id for product in second)


def test_identical_recent_set_is_avoided_when_sixth_candidate_exists() -> None:
    products = [make_product(str(index)) for index in range(6)]
    recommender = ProductRecommender(rng=random.Random(9), candidate_pool_size=6)
    command = search_command(category="หูฟัง")

    sets = [
        frozenset(product.id for product in recommender.recommend(products, command, user_id="u"))
        for _ in range(3)
    ]

    assert len(set(sets)) == 3


class FakeClock:
    def __init__(self, current: float = 100.0) -> None:
        self.current = current

    def __call__(self) -> float:
        return self.current


def test_history_is_scoped_by_user_and_query_and_expires_at_ttl() -> None:
    clock = FakeClock()
    products = [make_product(str(index)) for index in range(10)]
    recommender = ProductRecommender(
        rng=random.Random(11),
        candidate_pool_size=10,
        history_ttl_seconds=10,
        clock=clock,
    )
    command = search_command(category="หูฟัง")
    other_query = search_command(category="หูฟัง", query="bluetooth")

    recommender.recommend(products, command, user_id="alice")
    recommender.recommend(products, command, user_id="bob")
    recommender.recommend(products, other_query, user_id="alice")
    assert len(recommender.history_snapshot()) == 3

    clock.current = 110.0
    recommender.recommend(products, command, user_id="alice")
    assert len(recommender.history_snapshot()) == 1


def test_fairness_gives_every_top_pool_product_exposure() -> None:
    products = [make_product(str(index)) for index in range(10)]
    recommender = ProductRecommender(
        rng=random.Random(123),
        candidate_pool_size=10,
        history_size=4,
    )
    command = search_command(category="หูฟัง")
    exposure: Counter[str] = Counter()
    sets: list[frozenset[str]] = []
    for _ in range(30):
        result = recommender.recommend(products, command, user_id="fair")
        exposure.update(product.id for product in result)
        sets.append(frozenset(product.id for product in result))

    assert set(exposure) == {product.id for product in products}
    assert max(exposure.values()) - min(exposure.values()) <= 4
    assert len(set(sets)) > 10


def test_randomisation_never_escapes_bounded_top_candidate_pool() -> None:
    products = [make_product(f"p-{index:02d}") for index in range(30)]
    command = search_command(category="หูฟัง")
    recommender = ProductRecommender(rng=random.Random(4), candidate_pool_size=7)
    filtered = recommender.filter_products(products, command)
    top_pool_ids = {product.id for product in recommender.rank_products(filtered, command)[:7]}

    for user_index in range(20):
        result = recommender.recommend(products, command, user_id=f"user-{user_index}")
        assert {product.id for product in result} <= top_pool_ids


def test_non_search_intent_and_impossible_constraints_fail_gracefully() -> None:
    products = [make_product(str(index)) for index in range(8)]
    recommender = ProductRecommender(rng=random.Random(1))
    greeting = ParsedCommand(INTENT_GREETING, 0.9)
    impossible = search_command(min_price=5_000, max_price=1_000)

    assert recommender.recommend(products, greeting) == []
    assert recommender.recommend(products, impossible) == []


def test_recommendation_latency_is_well_below_rubric_budget() -> None:
    products = [make_product(str(index), price=1_000 + index) for index in range(1_000)]
    command = search_command(category="หูฟัง", max_price=2_500, in_stock=True, query="wireless")
    recommender = ProductRecommender(rng=random.Random(5), candidate_pool_size=20)

    started = time.perf_counter()
    result = recommender.recommend(products, command, user_id="performance")
    elapsed = time.perf_counter() - started

    assert len(result) == 5
    assert elapsed < 0.5

from conversation import (
    comparison_queries,
    find_named_product,
    is_alternative_request,
    is_cheaper_refinement,
    is_product_question,
    product_question_answer,
)
from models import Product


def _product(identifier, name, **changes):
    values = {
        "id": identifier,
        "sku": identifier,
        "name": name,
        "brand": "Test",
        "category": "เมาส์",
        "category_path": ("เกมมิ่ง", "เมาส์"),
        "price": 1990.0,
        "original_price": 2490.0,
        "image_url": "",
        "product_url": f"https://www.mercular.com/{identifier}",
        "in_stock": True,
    }
    values.update(changes)
    return Product(**values)


def test_comparison_extracts_two_model_phrases_and_matches_catalog_names():
    products = [
        _product("alpha", "MOUSE LOGITECH ALPHA X1 BLACK"),
        _product("beta", "MOUSE RAZER BETA Z2 WHITE"),
    ]

    sides = comparison_queries("Alpha X1 กับ Beta Z2 ต่างกันยังไง")

    assert sides == ("alpha x1", "beta z2")
    assert find_named_product(products, sides[0]).id == "alpha"
    assert find_named_product(products, sides[1]).id == "beta"


def test_comparison_accepts_common_chat_spelling_and_polite_suffixes():
    assert comparison_queries("Alpha X1 กะ Beta Z2 ต่างกันไง") == (
        "alpha x1",
        "beta z2",
    )
    assert comparison_queries("ช่วยเทียบ Alpha X1 vs Beta Z2 หน่อย") == (
        "alpha x1",
        "beta z2",
    )


def test_weak_partial_model_name_does_not_silently_select_wrong_product():
    products = [_product("g304", "MOUSE LOGITECH G304 X SUPERLIGHT")]

    assert find_named_product(products, "G Pro X Superlight 2") is None


def test_context_detectors_cover_question_cheaper_and_similar_requests():
    assert is_product_question("ตัวนี้ใช้ Bluetooth ได้ไหม")
    assert is_product_question("รุ่นนี้หนักกี่กรัม")
    assert is_cheaper_refinement("ขอถูกกว่านี้")
    assert is_alternative_request("มีตัวคล้าย ๆ ตัวนี้แต่ถูกกว่าไหม")
    assert is_product_question("ตัวนีใช้บลูทูดได้มั้ย")
    assert is_product_question("รุ่นนีประกันกี่ปี")
    assert is_cheaper_refinement("ขอถูกกว่านี")
    assert is_alternative_request("มีตัวค้าย ๆ แต่ประหยัดกว่าไหม")


def test_product_question_uses_specs_and_discloses_when_data_is_missing():
    product = _product(
        "bluetooth-mouse",
        "MOUSE TEST",
        specifications=(
            ("การเชื่อมต่อ", "Bluetooth 5.2 / Wireless 2.4GHz"),
            ("น้ำหนัก", "59 กรัม"),
        ),
    )

    bluetooth = product_question_answer(product, "ตัวนี้ใช้บลูทูธได้ไหม")
    weight = product_question_answer(product, "ตัวนี้หนักกี่กรัม")
    unknown = product_question_answer(
        _product("missing", "MOUSE NO DETAIL"),
        "ตัวนี้ใช้ Bluetooth ได้ไหม",
    )

    assert "Bluetooth 5.2" in bluetooth
    assert "59 กรัม" in weight
    assert "ยังยืนยัน" in unknown


def test_product_question_answers_warranty_battery_and_connection_from_details():
    product = _product(
        "detail-mouse",
        "MOUSE DETAIL",
        specifications=(
            ("แบตเตอรี่", "ใช้งานสูงสุด 70 ชั่วโมง"),
            ("การเชื่อมต่อ", "USB-C / Wireless 2.4GHz"),
        ),
        warranty="ประกันศูนย์ไทย 2 ปี",
        rating=4.8,
        review_count=25,
    )

    assert "70 ชั่วโมง" in product_question_answer(product, "ตัวนี้แบตอยู่ได้กี่ชั่วโมง")
    assert "USB-C" in product_question_answer(product, "รุ่นนี้มีพอร์ตอะไร")
    assert "ประกันศูนย์ไทย 2 ปี" in product_question_answer(product, "อันนี้ประกันกี่ปี")
    assert "฿1,990" in product_question_answer(product, "ตัวนี้ราคาเท่าไหร่")
    assert "มีสินค้า" in product_question_answer(product, "รุ่นนี้มีของไหม")
    assert "4.8/5" in product_question_answer(product, "อันนี้รีวิวกี่ดาว")
    assert "การเชื่อมต่อ" in product_question_answer(product, "ตัวนี้มีสเปคอะไรบ้าง")

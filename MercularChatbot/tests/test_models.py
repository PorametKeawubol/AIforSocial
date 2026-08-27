import pytest

from models import Product, clean_text, https_url


def _product(**changes):
    values = {
        "id": "101",
        "sku": "SKU-101",
        "name": "หูฟัง Test",
        "brand": "Test",
        "category": "หูฟัง",
        "category_path": ("เครื่องเสียง", "หูฟัง"),
        "price": 1990.0,
        "original_price": 2490.0,
        "image_url": "https://example.com/item.jpg",
        "product_url": "https://www.mercular.com/test-headphone",
        "in_stock": True,
        "tags": ("ไร้สาย",),
    }
    values.update(changes)
    return Product(**values)


def test_product_round_trip_and_display_price():
    product = _product(
        overview="จอเกมมิ่ง 165Hz",
        highlights=("พาเนล IPS", "165Hz"),
        specifications=(("พาเนล", "IPS"),),
        rating=4.8,
        review_count=5,
        recommended_count=4,
        warranty="ประกัน 3 ปี",
        service_notes=("ส่งฟรี",),
        detail_updated_at="2026-08-27T13:00:00+00:00",
    )

    restored = Product.from_dict(product.to_dict())

    assert restored == product
    assert restored.display_price == "฿1,990"
    assert "ไร้สาย" in restored.search_text


def test_missing_price_has_honest_label():
    assert _product(price=None).display_price == "ตรวจสอบราคาที่เว็บไซต์"


def test_missing_stock_stays_unknown_instead_of_becoming_available():
    value = _product().to_dict()
    value.pop("in_stock")

    assert Product.from_dict(value).in_stock is None


def test_product_rejects_missing_name_and_unsafe_product_url():
    with pytest.raises(ValueError):
        _product(name="")
    with pytest.raises(ValueError):
        _product(product_url="javascript:alert(1)")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_product_rejects_non_finite_prices(value):
    with pytest.raises(ValueError):
        _product(price=value)

    raw = _product().to_dict()
    raw["price"] = str(value)
    with pytest.raises(ValueError):
        Product.from_dict(raw)


@pytest.mark.parametrize("field", ["category_path", "tags"])
def test_product_rejects_string_instead_of_string_array(field):
    raw = _product().to_dict()
    raw[field] = "Audio"

    with pytest.raises(ValueError, match="array of strings"):
        Product.from_dict(raw)


def test_clean_text_and_url_normalization():
    assert clean_text("  hello\xa0  world \n") == "hello world"
    assert https_url("http://example.com/a?q=1#frag") == "https://example.com/a?q=1"
    assert https_url("data:text/plain,no") == ""

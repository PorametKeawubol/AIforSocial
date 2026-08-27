from models import Product
from scripts.sync_catalog import _merge_category_retry


def _product() -> Product:
    return Product(
        id="p-1",
        sku="SKU-1",
        name="MOUSE TEST",
        brand="Test",
        category="เมาส์",
        category_path=("คอมพิวเตอร์", "เมาส์"),
        price=1_990,
        original_price=2_490,
        image_url="",
        product_url="https://www.mercular.com/mouse-test",
        in_stock=True,
        overview=("รายละเอียดที่ดึงไว้แล้ว",),
        specifications=(("น้ำหนัก", "59 กรัม"),),
        detail_updated_at="2026-08-27T00:00:00+00:00",
    )


def test_failed_category_retry_merges_status_without_losing_enriched_products() -> None:
    good_url = "https://www.mercular.com/computer/mouse"
    failed_url = "https://www.mercular.com/computer/ssd"
    existing = {
        "schema_version": 1,
        "generated_at": "old",
        "source": {"name": "Mercular"},
        "summary": {
            "categories_requested": 2,
            "categories_succeeded": 1,
            "categories_failed": 1,
            "products": 1,
        },
        "categories": [
            {"url": good_url, "status": "ok"},
            {"url": failed_url, "status": "error"},
        ],
        "errors": [{"url": failed_url, "type": "MercularScraperError"}],
        "products": [_product().to_dict()],
    }
    retry = {
        "generated_at": "new",
        "categories": [{"url": failed_url, "status": "empty"}],
        "errors": [],
        "products": [],
    }

    merged = _merge_category_retry(existing, retry)

    assert merged["summary"] == {
        "categories_requested": 2,
        "categories_succeeded": 2,
        "categories_failed": 0,
        "products": 1,
    }
    assert merged["errors"] == []
    assert Product.from_dict(merged["products"][0]).specifications == (
        ("น้ำหนัก", "59 กรัม"),
    )
    assert merged["source"]["last_failed_category_retry"]["categories_resolved"] == 1

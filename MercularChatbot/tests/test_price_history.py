from pathlib import Path

from models import Product
from price_history import PriceHistoryStore


def _product(*, price: float, original_price: float, in_stock: bool) -> Product:
    return Product(
        id="product-1",
        sku="SKU-1",
        name="Example Headphones",
        brand="Example",
        category="หูฟัง",
        category_path=("audio", "headphone"),
        price=price,
        original_price=original_price,
        image_url="https://cdn.example.com/product.jpg",
        product_url="https://www.mercular.com/example-headphones",
        in_stock=in_stock,
        source_url="https://www.mercular.com/audio/headphone",
        scraped_at="2026-08-27T00:00:00+00:00",
    )


def test_daily_price_history_upserts_same_day_and_keeps_later_days(tmp_path: Path):
    store = PriceHistoryStore(tmp_path / "history.sqlite3")

    assert store.record_snapshot([_product(price=1000, original_price=1200, in_stock=True)], observed_at="2026-08-25T01:00:00+00:00") == 1
    assert store.record_snapshot([_product(price=900, original_price=1200, in_stock=False)], observed_at="2026-08-25T23:00:00+00:00") == 1
    assert store.record_snapshot([_product(price=1100, original_price=1200, in_stock=True)], observed_at="2026-08-26T01:00:00+00:00") == 1

    history = store.observations("product-1")

    assert [(item.price, item.discount_amount, item.in_stock) for item in history] == [
        (900.0, 300.0, False),
        (1100.0, 100.0, True),
    ]

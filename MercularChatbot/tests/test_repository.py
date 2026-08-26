import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from models import Product
from repository import ProductRepository
from scraper import write_snapshot


def _product(index=1, **changes):
    values = {
        "id": f"p-{index}",
        "sku": f"SKU-{index}",
        "name": f"Product {index}",
        "brand": "Zeta" if index == 1 else "alpha",
        "category": "Headphones" if index == 1 else "Keyboard",
        "category_path": ["Computer", "Headphones" if index == 1 else "Keyboard"],
        "price": 1000.0 + index,
        "original_price": 1200.0 + index,
        "image_url": f"https://cdn.example.com/{index}.jpg",
        "product_url": f"https://www.mercular.com/product-{index}",
        "in_stock": True,
        "description": "",
        "tags": [],
        "source_url": "https://www.mercular.com/audio",
        "scraped_at": "2026-08-24T00:00:00+00:00",
    }
    values.update(changes)
    return Product.from_dict(values).to_dict()


def _snapshot(*products, generated_at=None):
    return {
        "schema_version": 1,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "source": {"name": "Mercular", "website": "https://www.mercular.com/"},
        "summary": {"products": len(products)},
        "categories": [],
        "errors": [],
        "products": list(products),
    }


def _write(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_lookup_facets_metadata_and_status(tmp_path):
    path = tmp_path / "snapshot.json"
    _write(path, _snapshot(_product(1), _product(2)))

    repository = ProductRepository(path, stale_after_hours=24, auto_reload=False)

    assert [product.id for product in repository.all()] == ["p-1", "p-2"]
    assert repository.get("p-1").sku == "SKU-1"
    assert repository.get("sku-2").id == "p-2"
    assert repository.get_by_sku("SKU-1").id == "p-1"
    assert repository.get("missing") is None
    assert repository.brands() == ("alpha", "Zeta")
    assert repository.categories() == ("Headphones", "Keyboard")
    assert repository.is_ready is True
    assert repository.readiness is True
    assert repository.is_stale is False
    assert repository.status()["products"] == 2

    metadata = repository.metadata
    metadata["source"]["name"] = "mutated"
    assert repository.metadata["source"]["name"] == "Mercular"
    assert "products" not in repository.metadata


def test_missing_or_empty_snapshot_is_not_ready_and_is_stale(tmp_path):
    missing = ProductRepository(tmp_path / "missing.json", auto_reload=False)

    assert missing.is_ready is False
    assert missing.is_stale is True
    assert missing.all() == []
    assert "unavailable" in missing.last_error

    empty_path = tmp_path / "empty.json"
    _write(empty_path, _snapshot())
    empty = ProductRepository(empty_path, auto_reload=False)

    assert empty.is_ready is False
    assert empty.status()["ready"] is False
    assert empty.is_stale is True


def test_old_or_missing_generation_time_is_stale(tmp_path):
    old_path = tmp_path / "old.json"
    old_time = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    _write(old_path, _snapshot(_product(), generated_at=old_time))
    old_repository = ProductRepository(old_path, stale_after_hours=24, auto_reload=False)

    undated_path = tmp_path / "undated.json"
    payload = _snapshot(_product())
    payload["generated_at"] = "not-a-date"
    _write(undated_path, payload)
    undated_repository = ProductRepository(undated_path, auto_reload=False)

    assert old_repository.is_ready is True
    assert old_repository.is_stale is True
    assert undated_repository.is_ready is True
    assert undated_repository.is_stale is True


def test_corrupt_reload_preserves_last_known_good_products_and_metadata(tmp_path):
    path = tmp_path / "snapshot.json"
    _write(path, _snapshot(_product(1)))
    repository = ProductRepository(path, auto_reload=False)
    original_metadata = repository.metadata

    path.write_text('{"schema_version": 1, "products": [', encoding="utf-8")

    assert repository.reload() is False
    assert [product.id for product in repository.all()] == ["p-1"]
    assert repository.metadata == original_metadata
    assert repository.is_ready is True
    assert "reload rejected" in repository.last_error


def test_structurally_invalid_reload_also_preserves_last_known_good(tmp_path):
    path = tmp_path / "snapshot.json"
    _write(path, _snapshot(_product(1)))
    repository = ProductRepository(path, auto_reload=False)

    _write(path, {"schema_version": 999, "products": []})

    assert repository.reload() is False
    assert repository.get("p-1") is not None
    assert "schema_version" in repository.last_error


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (("price", "NaN"), ("price", "Infinity"), ("category_path", "Audio"), ("tags", "wireless")),
)
def test_malformed_product_reload_preserves_last_known_good(tmp_path, field, bad_value):
    path = tmp_path / "snapshot.json"
    _write(path, _snapshot(_product(1)))
    repository = ProductRepository(path, auto_reload=False)
    malformed = _product(2)
    malformed[field] = bad_value

    _write(path, _snapshot(malformed))

    assert repository.reload() is False
    assert [product.id for product in repository.all()] == ["p-1"]
    assert "invalid product" in repository.last_error


def test_auto_reload_adopts_atomic_snapshot_change(tmp_path):
    path = tmp_path / "snapshot.json"
    write_snapshot(_snapshot(_product(1)), path)
    repository = ProductRepository(path, auto_reload=True)

    write_snapshot(_snapshot(_product(2)), path)

    assert [product.id for product in repository.all()] == ["p-2"]
    assert repository.get("SKU-2").name == "Product 2"


def test_concurrent_reads_never_observe_partial_reload_state(tmp_path):
    path = tmp_path / "snapshot.json"
    write_snapshot(_snapshot(_product(1)), path)
    repository = ProductRepository(path, auto_reload=True)

    def read_ids():
        return tuple(product.id for product in repository.all())

    with ThreadPoolExecutor(max_workers=8) as pool:
        before = list(pool.map(lambda _index: read_ids(), range(30)))
        write_snapshot(_snapshot(_product(2), _product(3)), path)
        after = list(pool.map(lambda _index: read_ids(), range(30)))

    assert set(before) == {("p-1",)}
    assert set(after) == {("p-2", "p-3")}

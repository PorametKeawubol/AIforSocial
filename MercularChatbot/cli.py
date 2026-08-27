"""Try one Mercular shopping command locally without LINE."""

from __future__ import annotations

import argparse
import json

try:  # Package import from the repository root.
    from .bert_nlp import PhayaThaiBertCommandParser
    from .config import Settings
    from .nlp import ThaiCommandParser
    from .recommender import ProductRecommender
    from .repository import ProductRepository
except ImportError:  # pragma: no cover - direct execution from this folder.
    from bert_nlp import PhayaThaiBertCommandParser
    from config import Settings
    from nlp import ThaiCommandParser
    from recommender import ProductRecommender
    from repository import ProductRepository


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("message", help="Thai/English shopping request")
    argument_parser.add_argument("--user-id", default="local-demo")
    args = argument_parser.parse_args()

    settings = Settings.from_env()
    repository = ProductRepository(settings.snapshot_path)
    products = repository.all()
    if not products:
        print(
            json.dumps(
                {
                    "error": "catalog_unavailable",
                    "hint": "Run: python scraper.py --refresh",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    parser = (
        PhayaThaiBertCommandParser(
            brands=repository.brands(),
            categories=repository.categories(),
            model_name=settings.phayathaibert_model_name,
            min_confidence=settings.phayathaibert_min_confidence,
            local_files_only=settings.phayathaibert_local_files_only,
        )
        if settings.nlp_backend == "phayathaibert"
        else ThaiCommandParser(brands=repository.brands(), categories=repository.categories())
    )
    parsed = parser.parse(args.message)
    selected = ProductRecommender().recommend(
        products,
        parsed,
        user_id=args.user_id,
        top_k=settings.top_k,
    )
    print(
        json.dumps(
            {
                "intent": parsed.intent,
                "confidence": round(parsed.confidence, 4),
                "entities": parsed.entities.to_dict(),
                "matches": [
                    {
                        "id": product.id,
                        "name": product.name,
                        "brand": product.brand,
                        "category": product.category,
                        "price": product.price,
                        "in_stock": product.in_stock,
                        "url": product.product_url,
                    }
                    for product in selected
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

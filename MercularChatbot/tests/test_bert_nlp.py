from __future__ import annotations

from bert_nlp import (
    PhayaThaiBertCommandParser,
    PhayaThaiBertIntentClassifier,
    PhayaThaiBertUnavailable,
    IntentPrediction,
)
from nlp import INTENT_HELP, INTENT_PROMOTION, INTENT_SEARCH, INTENT_UNKNOWN


class FakeClassifier:
    def __init__(self, prediction: IntentPrediction):
        self.prediction = prediction
        self.inputs: list[str] = []

    def predict(self, text: str) -> IntentPrediction:
        self.inputs.append(text)
        return self.prediction


class UnavailableClassifier:
    def predict(self, text: str) -> IntentPrediction:
        raise PhayaThaiBertUnavailable("model is not installed")


def test_classifier_warm_up_loads_model_without_running_user_prediction(monkeypatch):
    classifier = PhayaThaiBertIntentClassifier(local_files_only=True)
    calls = []
    monkeypatch.setattr(classifier, "_load", lambda: calls.append("loaded"))

    classifier.warm_up()

    assert calls == ["loaded"]


def test_phayathaibert_routes_unknown_natural_language_intent():
    classifier = FakeClassifier(IntentPrediction(INTENT_HELP, 0.91))
    parser = PhayaThaiBertCommandParser(classifier=classifier, min_confidence=0.70)

    parsed = parser.parse("ข้อความที่ตีความจากความหมายเท่านั้น")

    assert parsed.intent == INTENT_HELP
    assert parsed.confidence == 0.91
    assert classifier.inputs == ["ข้อความที่ตีความจากความหมายเท่านั้น"]


def test_phayathaibert_search_keeps_unknown_message_as_catalogue_query():
    parser = PhayaThaiBertCommandParser(
        classifier=FakeClassifier(IntentPrediction(INTENT_SEARCH, 0.80)),
        min_confidence=0.70,
    )

    parsed = parser.parse("สิ่งที่ตรงใจฉัน")

    assert parsed.intent == INTENT_SEARCH
    assert parsed.entities.query == "สิ่งที่ตรงใจฉัน"


def test_rule_parser_remains_authoritative_for_structured_search():
    classifier = FakeClassifier(IntentPrediction(INTENT_HELP, 0.99))
    parser = PhayaThaiBertCommandParser(
        brands=["Sony"],
        categories=["หูฟัง"],
        classifier=classifier,
    )

    parsed = parser.parse("หาหูฟัง Sony ไม่เกิน 3000 พร้อมส่ง")

    assert parsed.intent == INTENT_SEARCH
    assert parsed.entities.brands == ("Sony",)
    assert parsed.entities.max_price == 3000
    assert parsed.entities.in_stock is True
    assert classifier.inputs == []


def test_rule_promotion_typo_is_authoritative_over_model_prediction():
    classifier = FakeClassifier(IntentPrediction(INTENT_SEARCH, 0.99))
    parser = PhayaThaiBertCommandParser(
        classifier=classifier,
    )

    parsed = parser.parse("มีโปรโมชั้นอะไรบ้าง")

    assert parsed.intent == INTENT_PROMOTION
    assert classifier.inputs == []


def test_unavailable_phayathaibert_falls_back_without_changing_rule_result():
    parser = PhayaThaiBertCommandParser(classifier=UnavailableClassifier())

    parsed = parser.parse("ข้อความที่ไม่มีใน rule")

    assert parsed.intent == INTENT_UNKNOWN

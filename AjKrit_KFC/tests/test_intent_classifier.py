from collections import Counter

from intent_classifier import (
    INTENT_EXAMPLES,
    INTENT_LABELS,
    LAB_EDGE_CASES,
    LAB_TEST_CASES,
    OTHER_INTENT,
    IntentClassifier,
)


def test_lab_dataset_contains_six_intents_and_at_least_30_messages():
    assert len(INTENT_LABELS) == 6
    assert len(LAB_TEST_CASES) == 42
    assert Counter(expected for _, expected in LAB_TEST_CASES) == {
        intent: 7 for intent in INTENT_LABELS
    }
    training_texts = {
        text
        for examples in INTENT_EXAMPLES.values()
        for text in examples
    }
    assert not training_texts.intersection(text for text, _ in LAB_TEST_CASES)
    assert len(LAB_EDGE_CASES) == 5


def test_offline_fallback_classifies_one_clear_message_per_intent():
    classifier = IntentClassifier(semantic_enabled=False)

    for text, expected in (
        ("เมนู", "menu"),
        ("โปรโมชั่น", "promotion"),
        ("ร้านอยู่ไหน", "location"),
        ("สวัสดี", "greeting"),
        ("สั่งอาหาร", "order"),
        ("ขอบคุณ", "thanks"),
    ):
        prediction = classifier.detect(text)
        assert prediction.intent == expected
        assert prediction.accepted
        assert set(prediction.scores) == set(INTENT_LABELS)


def test_low_confidence_message_uses_other_fallback():
    classifier = IntentClassifier(semantic_enabled=False)

    prediction = classifier.detect("zzzxxyy 12345")

    assert prediction.intent == OTHER_INTENT
    assert not prediction.accepted
    assert prediction.score < prediction.threshold

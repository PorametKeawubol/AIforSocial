from __future__ import annotations

from config import Settings


def test_top_k_environment_is_clamped_to_line_assignment_limit(monkeypatch):
    monkeypatch.setenv("TOP_K", "99")

    assert Settings.from_env().top_k == 5


def test_malformed_numeric_environment_uses_safe_defaults(monkeypatch):
    monkeypatch.setenv("PORT", "not-a-port")
    monkeypatch.setenv("SCRAPER_TIMEOUT_SECONDS", "not-a-timeout")

    settings = Settings.from_env()

    assert settings.port == 5000
    assert settings.request_timeout_seconds == 15.0


def test_non_finite_float_environment_uses_safe_default(monkeypatch):
    monkeypatch.setenv("SCRAPER_TIMEOUT_SECONDS", "Infinity")

    assert Settings.from_env().request_timeout_seconds == 15.0


def test_channel_identity_is_loaded_without_affecting_credentials(monkeypatch):
    monkeypatch.setenv("BOT_NAME", "MercuMate")
    monkeypatch.setenv("LINE_CHANNEL_ID", "2011217828")

    settings = Settings.from_env()

    assert settings.bot_name == "MercuMate"
    assert settings.line_channel_id == "2011217828"


def test_message_media_and_coupon_settings_are_optional(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://mercumate.example")
    monkeypatch.setenv("LINE_COUPON_ID", "coupon-id")

    settings = Settings.from_env()

    assert settings.public_base_url == "https://mercumate.example"
    assert settings.line_coupon_id == "coupon-id"


def test_phayathaibert_configuration_defaults_and_can_use_local_model_cache(monkeypatch):
    monkeypatch.setenv("PHAYATHAIBERT_MODEL_NAME", "local-phayathaibert")
    monkeypatch.setenv("PHAYATHAIBERT_MIN_CONFIDENCE", "0.83")
    monkeypatch.setenv("PHAYATHAIBERT_LOCAL_FILES_ONLY", "true")

    settings = Settings.from_env()

    assert settings.nlp_backend == "phayathaibert"
    assert settings.phayathaibert_model_name == "local-phayathaibert"
    assert settings.phayathaibert_min_confidence == 0.83
    assert settings.phayathaibert_local_files_only is True


def test_price_history_path_is_configurable(monkeypatch):
    monkeypatch.setenv("MERCULAR_PRICE_HISTORY_PATH", "data/history.sqlite3")

    assert Settings.from_env().price_history_path.name == "history.sqlite3"

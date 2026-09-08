"""Regression coverage for analyze model/provider precedence."""

from types import SimpleNamespace

from unread.analyzer.pipeline import _resolve_analysis_models
from unread.config import Settings


def _preset(final: str = "gpt-5.6-luna", filt: str = "gpt-5.6-luna") -> SimpleNamespace:
    return SimpleNamespace(final_model=final, filter_model=filt)


def test_minimax_ignores_openai_preset_model_pins() -> None:
    settings = Settings()
    settings.ai.chat_provider = "minimax"
    settings.ai.filter_provider = "minimax"
    settings.ai.chat_model = ""
    settings.ai.filter_model = ""

    assert _resolve_analysis_models(settings, _preset()) == ("MiniMax-M3", "MiniMax-M3")


def test_explicit_slot_models_win_over_preset_pins() -> None:
    settings = Settings()
    settings.ai.chat_provider = "minimax"
    settings.ai.filter_provider = "minimax"
    settings.ai.chat_model = "MiniMax-M2.7"
    settings.ai.filter_model = "MiniMax-M2.7-highspeed"

    assert _resolve_analysis_models(settings, _preset()) == (
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
    )


def test_cli_model_overrides_win_over_slot_models() -> None:
    settings = Settings()
    settings.ai.chat_provider = "minimax"
    settings.ai.filter_provider = "minimax"
    settings.ai.chat_model = "MiniMax-M3"
    settings.ai.filter_model = "MiniMax-M3"

    assert _resolve_analysis_models(
        settings,
        _preset(),
        model_override="cli-final",
        filter_model_override="cli-filter",
    ) == ("cli-final", "cli-filter")


def test_openai_keeps_preset_model_pins_when_slot_model_is_empty() -> None:
    settings = Settings()
    settings.ai.chat_provider = "openai"
    settings.ai.filter_provider = "openai"
    settings.ai.chat_model = ""
    settings.ai.filter_model = ""

    assert _resolve_analysis_models(settings, _preset("preset-final", "preset-filter")) == (
        "preset-final",
        "preset-filter",
    )

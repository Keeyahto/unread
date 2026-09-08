from types import SimpleNamespace
from unittest.mock import patch

from unread.tg.commands import _active_slot_providers, _provider_key_present


def _settings(**keys):
    def svc(name):
        return SimpleNamespace(api_key=keys.get(name, ""))

    return SimpleNamespace(
        openai=svc("openai"),
        openrouter=svc("openrouter"),
        anthropic=svc("anthropic"),
        minimax=svc("minimax"),
        google=svc("google"),
    )


def test_provider_key_present_recognizes_minimax():
    settings = _settings(minimax="sk-mm-test")
    assert _provider_key_present(settings, "minimax") is True
    assert _provider_key_present(settings, "openai") is False


def test_local_provider_is_key_optional():
    assert _provider_key_present(_settings(), "local") is True


def test_active_slots_can_be_minimax_without_requiring_openai():
    routing = {
        "chat": "minimax",
        "filter": "minimax",
        "audio": "local",
        "vision": "minimax",
    }
    with patch(
        "unread.ai.providers._resolve_provider_name",
        side_effect=lambda _settings, slot: routing[slot],
    ):
        providers = _active_slot_providers(object())

    assert providers == routing
    assert "openai" not in providers.values()


def test_active_slots_detect_openai_when_one_slot_needs_it():
    routing = {
        "chat": "minimax",
        "filter": "minimax",
        "audio": "openai",
        "vision": "minimax",
    }
    with patch(
        "unread.ai.providers._resolve_provider_name",
        side_effect=lambda _settings, slot: routing[slot],
    ):
        providers = _active_slot_providers(object())

    assert "openai" in providers.values()

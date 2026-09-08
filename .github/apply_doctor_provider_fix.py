from pathlib import Path


SOURCE = Path("unread/tg/commands.py")
TEST = Path("tests/test_doctor_provider_keys.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly 1 source block, found {count}")
    return text.replace(old, new, 1)


text = SOURCE.read_text(encoding="utf-8")

text = replace_once(
    text,
    '''    name = _resolve_provider_name(settings, "chat")
    if name == "openai":
        return bool(settings.openai.api_key)
    if name == "openrouter":
        return bool(settings.openrouter.api_key)
    if name == "anthropic":
        return bool(settings.anthropic.api_key)
    if name == "minimax":
        return bool(settings.minimax.api_key)
    if name == "google":
        return bool(settings.google.api_key)
    # Local mode is "configured" once `ai.chat_provider == "local"` is
    # persisted — the base_url has a sensible default and most
    # servers don't enforce a key. Unknown provider names fall through
    # to False so the wizard re-prompts.
    return name == "local"
''',
    '''    name = _resolve_provider_name(settings, "chat")
    return _provider_key_present(settings, name)
''',
    "active provider key helper",
)

text = replace_once(
    text,
    '''async def _smoke_test_openai(api_key: str) -> None:
''',
    '''def _provider_key_present(settings, name: str) -> bool:  # type: ignore[no-untyped-def]
    """Whether ``name`` has the credential needed for API calls.

    Local endpoints are intentionally key-optional. Hosted providers share
    the same lookup used by the setup wizard so doctor cannot silently drift
    out of sync when a new provider is added.
    """
    name = name.strip().lower()
    if name == "local":
        return True
    return bool(_provider_key_value(settings, name))


def _active_slot_providers(settings) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Resolve provider names for the four runtime capability slots."""
    from unread.ai.providers import _resolve_provider_name

    return {
        slot: _resolve_provider_name(settings, slot)
        for slot in ("chat", "filter", "audio", "vision")
    }


async def _smoke_test_openai(api_key: str) -> None:
''',
    "provider helper insertion",
)

text = replace_once(
    text,
    '''    _api_hash_ct = _ct(settings.telegram.api_hash)
    _openai_ct = _ct(settings.openai.api_key)
''',
    '''    _api_hash_ct = _ct(settings.telegram.api_hash)
    _openai_ct = _ct(settings.openai.api_key)
    active_slot_providers = _active_slot_providers(settings)
    openai_required = "openai" in active_slot_providers.values()
''',
    "active slot resolution",
)

text = replace_once(
    text,
    '''    if _openai_ct:
        _line(
            fail,
            "OPENAI_API_KEY looks encrypted",
            "ciphertext (`$u1$…`) reached the active backend — run `unread security recover`",
        )
    elif settings.openai.api_key:
        _line(ok, "OPENAI_API_KEY present")
    else:
        _line(fail, "OPENAI_API_KEY missing", _openai_hint)
''',
    '''    if _openai_ct:
        _line(
            fail,
            "OPENAI_API_KEY looks encrypted",
            "ciphertext (`$u1$…`) reached the active backend — run `unread security recover`",
        )
    elif settings.openai.api_key:
        _line(ok, "OPENAI_API_KEY present")
    elif openai_required:
        _line(fail, "OPENAI_API_KEY missing", _openai_hint)
    else:
        _line(ok, "OPENAI_API_KEY", "not configured (not required by active slots)")
''',
    "OpenAI requirement",
)

text = replace_once(
    text,
    '''    # 7b. Per-slot provider key check. Each of the four slots resolves
    # to its own (provider, model); doctor reports whether the chosen
    # provider has a usable key. Local needs no key (placeholder OK).
    from unread.ai.providers import _resolve_provider_name

    def _slot_key_present(slot_provider: str) -> bool:
        if slot_provider == "openai":
            return bool(settings.openai.api_key)
        if slot_provider == "openrouter":
            return bool(settings.openrouter.api_key)
        if slot_provider == "anthropic":
            return bool(settings.anthropic.api_key)
        if slot_provider == "google":
            return bool(settings.google.api_key)
        return slot_provider == "local"

    for slot in ("chat", "filter", "audio", "vision"):
        slot_provider = _resolve_provider_name(settings, slot)
        if _slot_key_present(slot_provider):
''',
    '''    # 7b. Per-slot provider key check. Each of the four slots resolves
    # to its own (provider, model); doctor reports whether the chosen
    # provider has a usable key. Local needs no key (placeholder OK).
    for slot in ("chat", "filter", "audio", "vision"):
        slot_provider = active_slot_providers[slot]
        if _provider_key_present(settings, slot_provider):
''',
    "doctor slot key check",
)

SOURCE.write_text(text, encoding="utf-8")

TEST.write_text(
    '''from types import SimpleNamespace
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
''',
    encoding="utf-8",
)

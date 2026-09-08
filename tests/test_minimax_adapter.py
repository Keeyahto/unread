"""MiniMax-specific adapter and catalog coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from unread.ai.anthropic_provider import AnthropicProvider
from unread.ai.minimax_provider import MiniMaxProvider, _translate_messages_kwargs
from unread.ai.models import find_model, models_for_provider, provider_for_model
from unread.ai.providers import ChatResult
from unread.ai.vision_provider import make_vision_provider
from unread.config import Settings


def _settings() -> Settings:
    s = Settings()
    s.minimax.api_key = "sk-mm-test"
    return s


def test_m3_catalog_limits_and_pricing():
    info = find_model("MiniMax-M3")
    assert info is not None
    assert info.context_window == 1_000_000
    assert info.max_output_tokens == 131_072
    assert (info.input_price, info.cached_price, info.output_price) == (0.60, 0.12, 2.40)
    assert provider_for_model("MiniMax-M3") == "minimax"
    assert "MiniMax-M3" in {m.id for m in models_for_provider("minimax", role="chat")}


def test_minimax_vision_uses_m3_default():
    p = make_vision_provider("minimax", _settings())
    assert p.name == "minimax"
    assert p.default_vision_model == "MiniMax-M3"


def test_m3_request_translation_uses_extra_body_and_adaptive_thinking():
    kwargs = _translate_messages_kwargs(
        {
            "model": "MiniMax-M3",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
        }
    )
    assert "temperature" not in kwargs
    assert kwargs["extra_body"] == {"temperature": 0.2}
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_m27_request_translation_does_not_force_m3_thinking():
    kwargs = _translate_messages_kwargs(
        {
            "model": "MiniMax-M2.7",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
        }
    )
    assert "temperature" not in kwargs
    assert kwargs["extra_body"] == {"temperature": 0.2}
    assert "thinking" not in kwargs


@pytest.mark.asyncio
async def test_minimax_forces_anthropic_web_search_off():
    p = MiniMaxProvider(_settings())
    fake_result = ChatResult(text="ok", prompt_tokens=1, cached_tokens=0, completion_tokens=1)
    with patch.object(AnthropicProvider, "chat", new=AsyncMock(return_value=fake_result)) as base_chat:
        result = await p.chat(
            model="MiniMax-M3",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=16,
            temperature=0.2,
            web_search=True,
        )
    assert result.text == "ok"
    assert base_chat.await_args.kwargs["web_search"] is False

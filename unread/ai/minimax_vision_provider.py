"""MiniMax vision adapter via the Anthropic-compatible image format."""

from __future__ import annotations

from unread.ai.minimax_provider import MINIMAX_ANTHROPIC_BASE_URL
from unread.ai.providers import ProviderUnavailableError
from unread.ai.vision_provider import AnthropicVisionProvider


class MiniMaxVisionProvider(AnthropicVisionProvider):
    name = "minimax"
    display_name = "MiniMax"
    default_vision_model = "MiniMax-M3"

    def __init__(self, settings) -> None:  # type: ignore[no-untyped-def]
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover
            raise ProviderUnavailableError(
                "Vision via MiniMax selected but the `anthropic` package isn't installed. "
                "Run `uv sync --extra dev`."
            ) from e
        if not settings.minimax.api_key:
            raise ProviderUnavailableError(
                "Vision via MiniMax selected but `minimax.api_key` is empty. "
                "Run `unread settings` to add one."
            )
        self._client = AsyncAnthropic(
            api_key=settings.minimax.api_key,
            base_url=MINIMAX_ANTHROPIC_BASE_URL,
            timeout=settings.openai.request_timeout_sec,
            max_retries=0,
        )
        self._settings = settings

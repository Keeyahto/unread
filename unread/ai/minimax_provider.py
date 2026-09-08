"""MiniMax adapter using the vendor's Anthropic-compatible API."""

from __future__ import annotations

from unread.ai.anthropic_provider import AnthropicProvider
from unread.ai.providers import ChatResult, ProviderUnavailableError

MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"
MINIMAX_OPENAI_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxProvider(AnthropicProvider):
    """MiniMax M-series through ``anthropic.AsyncAnthropic``.

    MiniMax recommends its Anthropic-compatible Messages API for agentic
    workloads. M3 is the provider default for both flagship and filter
    slots. Native Anthropic server-side web-search tools are not exposed by
    MiniMax, so web-search requests are deliberately disabled here.
    """

    name = "minimax"
    display_name = "MiniMax"
    supports_web_search = False
    default_chat_model = "MiniMax-M3"
    default_filter_model = "MiniMax-M3"

    def __init__(self, settings) -> None:  # type: ignore[no-untyped-def]
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover — package is a project dependency
            raise ProviderUnavailableError(
                "MiniMax provider selected but the `anthropic` package isn't installed. "
                "Run `uv sync --extra dev` (or pip install anthropic)."
            ) from e
        if not settings.minimax.api_key:
            raise ProviderUnavailableError(
                "MiniMax provider selected but `minimax.api_key` is empty. "
                "Run `unread init` or `unread settings` to add one."
            )
        self._client = AsyncAnthropic(
            api_key=settings.minimax.api_key,
            base_url=MINIMAX_ANTHROPIC_BASE_URL,
            timeout=settings.openai.request_timeout_sec,
            max_retries=0,
        )
        self._settings = settings

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        web_search: bool = False,
    ) -> ChatResult:
        # MiniMax's Anthropic-compatible endpoint does not implement
        # Anthropic's server-side `web_search_*` tool. Force it off even if a
        # caller accidentally ignores `supports_web_search=False`.
        return await super().chat(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            web_search=False,
        )

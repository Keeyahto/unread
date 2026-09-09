"""MiniMax adapter using the vendor's Anthropic-compatible API."""

from __future__ import annotations

from typing import Any

from unread.ai.anthropic_provider import AnthropicProvider
from unread.ai.providers import ChatResult, ProviderUnavailableError

MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"
MINIMAX_OPENAI_BASE_URL = "https://api.minimax.io/v1"
# M3 can legitimately spend several minutes generating large map/reduce
# responses (builtin presets may allow tens of thousands of output tokens).
# The old shared OpenAI timeout was 120s, which repeatedly killed an otherwise
# healthy MiniMax request mid-generation. Keep this provider-specific so
# changing OpenAI transport settings cannot shorten MiniMax long requests.
MINIMAX_REQUEST_TIMEOUT_SEC = 1800


def _translate_messages_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate unread's Anthropic request shape to current MiniMax/SDK semantics.

    New Anthropic Python SDK releases removed ``temperature`` from the
    ``messages.create`` signature. MiniMax's Anthropic-compatible endpoint
    still supports it, so pass it through ``extra_body`` instead. MiniMax M3
    also needs adaptive thinking on the Anthropic route to reliably return
    user-visible content.
    """
    translated = dict(kwargs)
    temperature = translated.pop("temperature", None)
    if temperature is not None:
        extra_body = dict(translated.get("extra_body") or {})
        extra_body["temperature"] = temperature
        translated["extra_body"] = extra_body

    model = str(translated.get("model") or "").strip().lower()
    if model.startswith("minimax-m3"):
        translated.setdefault("thinking", {"type": "adaptive"})

    return translated


class _MiniMaxMessagesProxy:
    """Accept legacy Anthropic kwargs and forward an SDK-compatible request."""

    def __init__(self, messages: Any) -> None:
        self._messages = messages

    async def create(self, **kwargs: Any) -> Any:
        return await self._messages.create(**_translate_messages_kwargs(kwargs))


class _MiniMaxClientProxy:
    """Expose only the resource surface AnthropicProvider.chat() consumes."""

    def __init__(self, client: Any) -> None:
        self.messages = _MiniMaxMessagesProxy(client.messages)


class MiniMaxProvider(AnthropicProvider):
    """MiniMax M-series through ``anthropic.AsyncAnthropic``.

    MiniMax recommends its Anthropic-compatible Messages API for agentic
    workloads. M3 is the provider default for both flagship and filter
    slots. Native Anthropic server-side web-search tools are not exposed by
    MiniMax through unread yet, so web-search requests are deliberately
    disabled here.
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
        client = AsyncAnthropic(
            api_key=settings.minimax.api_key,
            base_url=MINIMAX_ANTHROPIC_BASE_URL,
            timeout=MINIMAX_REQUEST_TIMEOUT_SEC,
            max_retries=0,
        )
        self._client = _MiniMaxClientProxy(client)
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
        # Force Anthropic server-side web search off even if a caller ignores
        # supports_web_search=False. MiniMax-specific request translation is
        # handled by _MiniMaxMessagesProxy before the SDK call.
        return await super().chat(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            web_search=False,
        )

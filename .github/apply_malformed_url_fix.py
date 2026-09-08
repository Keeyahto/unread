from pathlib import Path


SOURCE = Path("unread/enrich/link.py")
TEST = Path("tests/test_enrich_link.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly 1 source block, found {count}")
    return text.replace(old, new, 1)


text = SOURCE.read_text(encoding="utf-8")

text = replace_once(
    text,
    '''def extract_urls(text: str | None) -> list[str]:
''',
    '''def _normalize_and_host(url: str) -> tuple[str, str] | None:
    """Normalize ``url`` and return its lowercase host, or soft-skip invalid input.

    ``urllib.parse`` raises ``ValueError('Invalid IPv6 URL')`` for malformed
    bracketed netlocs such as ``https://foo[bar.example/...``. Telegram
    messages are untrusted input, so one broken pasted URL must not abort an
    otherwise valid multi-thousand-message analysis run.
    """
    try:
        normalized = _normalize_url(url)
        host = (urlparse(normalized).hostname or "").lower()
    except ValueError as e:
        log.debug("enrich.link.invalid_url", url=url[:200], err=str(e)[:200])
        return None
    return normalized, host


def extract_urls(text: str | None) -> list[str]:
''',
    "safe URL helper insertion",
)

text = replace_once(
    text,
    '''    for m in _URL_RE.finditer(text):
        raw = m.group(0)
        normalized = _normalize_url(raw)
        host = urlparse(normalized).hostname or ""
        host = host.lower()
        if host in _SKIP_HOSTS or host.endswith(".t.me"):
            continue
        if normalized not in seen:
            seen[normalized] = None
''',
    '''    for m in _URL_RE.finditer(text):
        raw = m.group(0)
        parsed = _normalize_and_host(raw)
        if parsed is None:
            continue
        normalized, host = parsed
        if host in _SKIP_HOSTS or host.endswith(".t.me"):
            continue
        if normalized not in seen:
            seen[normalized] = None
''',
    "extract_urls parsing",
)

text = replace_once(
    text,
    '''    settings = get_settings()
    normalized = _normalize_url(url)
    host = (urlparse(normalized).hostname or "").lower()
''',
    '''    settings = get_settings()
    parsed = _normalize_and_host(url)
    if parsed is None:
        return None
    normalized, host = parsed
''',
    "enrich_url parsing",
)

SOURCE.write_text(text, encoding="utf-8")

test_text = TEST.read_text(encoding="utf-8")
test_text = replace_once(
    test_text,
    '''def test_extract_urls_empty():
    assert extract_urls(None) == []
    assert extract_urls("") == []
    assert extract_urls("no links here at all") == []


''',
    '''def test_extract_urls_empty():
    assert extract_urls(None) == []
    assert extract_urls("") == []
    assert extract_urls("no links here at all") == []


def test_extract_urls_soft_skips_malformed_ipv6_like_url():
    text = "broken https://foo[bar.example/path but valid https://example.com/ok"
    assert extract_urls(text) == ["https://example.com/ok"]


''',
    "malformed URL regression test",
)
TEST.write_text(test_text, encoding="utf-8")

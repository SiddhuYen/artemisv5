"""HTML -> clean text with stable character offsets.

Sentence order and paragraph boundaries are preserved because the extraction
layer resolves referents using adjacency. Every offset in the final result is an
index into `PageDocument.text` produced here.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from artemis.models import PageDocument, Sentence, utcnow

# Tokens that end in a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev", "hon",
    "inc", "ltd", "co", "corp", "llc", "plc", "gmbh", "dept", "univ", "assn",
    "vs", "etc", "al", "approx", "est", "fig", "no", "vol", "ed", "eds",
    "gov", "sen", "rep", "capt", "gen", "lt", "col", "sgt", "phd", "md", "ba",
    "ma", "bsc", "msc", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec",
}

_BOUNDARY_RE = re.compile(r"([.!?]['\"’”\)\]]*)([ \t]+)|(\n+)")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")

# Heuristics for pages that returned HTTP 200 but no usable content.
_PAYWALL_MARKERS = (
    "subscribe to continue", "subscribers only", "this article is for subscribers",
    "create a free account to read", "sign in to read", "register to continue",
    "you have reached your article limit", "become a member to read",
)
_CONSENT_MARKERS = (
    "accept all cookies", "we value your privacy", "manage your cookie preferences",
    "enable javascript to continue", "please enable javascript",
    "verify you are a human", "checking your browser before accessing",
    "access denied", "attention required! | cloudflare",
)

# Soft 404s: an error body served with HTTP 200. These sail past the length and
# paywall checks and cost a fetch plus an extraction call each.
_NOT_FOUND_MARKERS = (
    "the page cannot be found", "page not found", "404 not found",
    "http error 404", "file or directory not found", "error 404",
    "this page doesn't exist", "this page does not exist",
    "we can't find the page", "we couldn't find that page",
    "the requested url was not found", "sorry, this page isn't available",
)

_MIN_USEFUL_CHARS = 250


def normalize_text(raw: str) -> str:
    """Canonical text form. Offsets are indices into the output of this function."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("​", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(html: str, url: str = "") -> tuple[str, Optional[str]]:
    """trafilatura first, readability-lxml + BeautifulSoup as fallback."""
    text = ""
    title: Optional[str] = None

    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            output_format="txt",
        )
        if extracted:
            text = extracted
        try:
            meta = trafilatura.extract_metadata(html)
            if meta is not None and getattr(meta, "title", None):
                title = meta.title
        except Exception:
            pass
    except Exception:
        pass

    if len(text.strip()) < _MIN_USEFUL_CHARS:
        try:
            from bs4 import BeautifulSoup
            from readability import Document

            doc = Document(html)
            title = title or (doc.short_title() or None)
            soup = BeautifulSoup(doc.summary(), "lxml")
            fallback = soup.get_text("\n")
            if len(fallback.strip()) > len(text.strip()):
                text = fallback
        except Exception:
            pass

    if title is None:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            pass

    return normalize_text(text), title


def _is_false_boundary(text: str, period_index: int) -> bool:
    """True when the period at `period_index` is an abbreviation or an initial."""
    m = _WORD_BEFORE_RE.search(text, 0, period_index)
    if not m:
        return False
    token = m.group(1).rstrip(".")
    if not token:
        return False
    if token.lower() in _ABBREVIATIONS:
        return True
    # "J. Smith" / "J.R.R. Tolkien": single letters, or all-initial tokens.
    letters = token.replace(".", "")
    return len(letters) == 1 and letters.isalpha()


def segment_sentences(text: str, max_sentences: Optional[int] = None) -> list[Sentence]:
    """Split into sentences carrying offsets into `text`.

    `text[s.start:s.end] == s.text` for every sentence — enforced by the model.
    """
    spans: list[tuple[int, int]] = []
    start = 0

    for m in _BOUNDARY_RE.finditer(text):
        if m.group(3) is not None:  # newline run: always a hard boundary
            end = m.start(3)
        else:
            end = m.end(1)
            if _is_false_boundary(text, m.start(1)):
                continue
            nxt = text[m.end(2) : m.end(2) + 1]
            if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "\"'“‘(["):
                continue
        if end > start:
            spans.append((start, end))
        start = m.end()

    if start < len(text):
        spans.append((start, len(text)))

    sentences: list[Sentence] = []
    for raw_start, raw_end in spans:
        chunk = text[raw_start:raw_end]
        lead = len(chunk) - len(chunk.lstrip())
        trail = len(chunk) - len(chunk.rstrip())
        s_start, s_end = raw_start + lead, raw_end - trail
        if s_end <= s_start:
            continue
        sentences.append(
            Sentence(index=len(sentences), start=s_start, end=s_end, text=text[s_start:s_end])
        )
        if max_sentences is not None and len(sentences) >= max_sentences:
            break

    return sentences


def looks_blocked(text: str) -> Optional[str]:
    """Paywall / consent-wall / bot-wall heuristics. Returns a reason or None."""
    stripped = text.strip()
    if len(stripped) < _MIN_USEFUL_CHARS:
        return "too_short"
    head = stripped[:2500].lower()
    for marker in _NOT_FOUND_MARKERS:
        if marker in head:
            return "soft_404"
    for marker in _PAYWALL_MARKERS:
        if marker in head:
            return "paywall"
    for marker in _CONSENT_MARKERS:
        if marker in head:
            return "consent_or_bot_wall"
    return None


def build_document(
    *,
    url: str,
    final_url: str,
    html: str,
    retrieved_at: Optional[datetime] = None,
    max_sentences: Optional[int] = None,
) -> tuple[Optional[PageDocument], Optional[str]]:
    """Returns (document, skip_reason). Exactly one is non-None."""
    text, title = html_to_text(html, url=final_url or url)
    reason = looks_blocked(text)
    if reason:
        return None, reason
    doc = PageDocument(
        url=url,
        final_url=final_url or url,
        title=title,
        text=text,
        sentences=segment_sentences(text, max_sentences=max_sentences),
        retrieved_at=retrieved_at or utcnow(),
    )
    return doc, None

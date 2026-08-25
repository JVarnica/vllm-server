import os
import re
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlsplit

from fastapi import Request, APIRouter
import trafilatura
import asyncio

import logging
logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------

S_CONTEXT_MAX_CHARS = 20000
SEARCH_CANDIDATES = 12          # fetch more, then rerank
SCRAPE_TOP_N = 5               # scrape best reranked results (must be >= CONTEXT_TOP_N)
CONTEXT_TOP_N = 5              # results shown to the model; keep aligned with SCRAPE_TOP_N
SCRAPE_MAX_CHARS = 5000        # legacy per-URL cap (used by scrape_urls)
SCRAPE_RAW_MAX_CHARS = 60000   # safety cap before passage selection

# ---- Global passage selection -------------------------------------------------
# Passages compete across ALL scraped pages, not just within their own page, so a
# strong passage on a rank-5 URL can outrank a weak one on the rank-1 URL.
EVIDENCE_TOTAL_MAX_CHARS = 9000     # total page-evidence budget across all URLs
EVIDENCE_PER_DOC_MAX_CHARS = 2500   # cap per URL so one page cannot hog the budget
EVIDENCE_MAX_PASSAGES_PER_DOC = 4
MIN_PASSAGE_SCORE = 0.15            # below this a passage is not worth context
DOC_PRIOR_WEIGHT = 0.60             # how much the URL's rerank score biases its passages
EVIDENCE_BONUS_WEIGHT = 0.75        # how much won evidence promotes a URL in final ordering

# ---- Recency handling ---------------------------------------------------------
# Queries are classified 'recent' or 'general'. Only 'recent' queries get the year
# appended AND get publication dates weighted; on 'general' queries an old document
# is not a worse document, so the prior is switched off entirely.
RECENCY_UNKNOWN = 0.0               # undated results are neutral, never punished

# The model picks the STRENGTH, not just on/off. It has read the whole question
# and knows whether "who leads the polls" means this week or this decade; a
# lexical rule does not. Each profile is a weight plus the age at which content
# starts being treated as stale.
RECENCY_PROFILES = {
    "breaking": {"weight": 2.00, "fresh_days": 2, "stale_days": 90},
    "recent":   {"weight": 1.20, "fresh_days": 30, "stale_days": 365 * 3},
    "general":  {"weight": 0.00, "fresh_days": 0, "stale_days": 0},
}
DEFAULT_RECENCY_MODE = "general"

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ---- Passage-selection tracing ------------------------------------------------
# Emits a nested span whose INPUT is the raw extracted page text - i.e. the actual
# inputs to passage scoring, including the passages that lose. Trace/tool content
# is post-selection, so it cannot be used to replay a different passage scorer.
# Langfuse truncates events at ~1MB and silently drops input/output/metadata to
# fit, so the payload is capped well under that.
PASSAGE_TRACE_ENABLED = os.environ.get("PASSAGE_TRACE_ENABLED", "1") not in ("0", "false", "False")
PASSAGE_TRACE_SAMPLE_RATE = float(os.environ.get("PASSAGE_TRACE_SAMPLE_RATE", "1.0"))
PASSAGE_TRACE_MAX_CHARS_PER_DOC = int(os.environ.get("PASSAGE_TRACE_MAX_CHARS_PER_DOC", "12000"))
PASSAGE_TRACE_MAX_TOTAL_CHARS = int(os.environ.get("PASSAGE_TRACE_MAX_TOTAL_CHARS", "60000"))

# Phrases that mark a question as historical even if a recency word appears
# nearby ("who was the first current-affairs presenter" etc).
# Only phrases that are unambiguously about the past. The earlier list included
# "during the", "before the" and "the first person to", which fire on plainly
# current questions ("pollen count during the summer", "first person to run a
# sub-2-hour marathon") - and since this list no longer overrides the model, a
# false positive here would only ever mislead the fallback path anyway.
_HISTORICAL_PHRASES = (
    "history of", "who invented", "who founded", "was born", "were born",
    "in ancient", "used to be", "in the 19", "in the 18",
)

SEARXNG_SAFESEARCH = 1         # 0=off, 1=moderate, 2=strict
SEARXNG_LANGUAGE = "en"
SEARXNG_CATEGORY = "general"

SEARXNG_INTERNAL_URL = os.environ["SEARXNG_INTERNAL_URL"]

router = APIRouter()


# ---------------------------------------------------------------------------
# Lightweight relevance / quality helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "how", "i", "in", "into", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "then", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "you", "your",
    "find", "search", "look", "lookup", "looked", "up",
}

_CURRENT_TERMS = {
    "current", "currently", "latest", "today", "now", "newest", "recent",
    "incumbent", "present", "next", "upcoming", "nowadays", "modern",
    "ongoing", "live", "update", "updated", "so far", "as of", "still",
    "this year", "right now", "at the moment",
}

# Multi-word recency cues are matched on raw text, single words on tokens.
_CURRENT_PHRASES = tuple(t for t in _CURRENT_TERMS if " " in t)
_CURRENT_TOKENS = frozenset(t for t in _CURRENT_TERMS if " " not in t)

# Suffixes that must not be stripped by the naive plural rule ("gas" -> "ga").
_NO_STRIP_ENDINGS = ("ss", "us", "is", "as", "os")

# Small boost only; relevance still dominates.
_AUTHORITY_HOST_SUFFIXES = (
    ".gov",
    ".gov.uk",
    ".edu",
    ".ac.uk",
    ".int",
    "wikipedia.org",
    "britannica.com",
    "python.org",
    "kernel.org",
    "rust-lang.org",
    "worldathletics.org",
    "olympics.com",
    "uefa.com",
    "un.org",
    "who.int",
    "nasa.gov",
    "census.gov",
    "federalreserve.gov",
    "bankofengland.co.uk",
)


def _stem(token: str) -> str:
    """
    Very light morphological folding so 'rates' matches 'rate'.

    Deliberately conservative: this only needs to collapse the plural/possessive
    variation that was causing correct sources to lose lexical overlap against
    the query. It is not a real stemmer and should not become one.
    """
    if token.isdigit() or len(token) <= 3:
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(_NO_STRIP_ENDINGS):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    out = set()
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(token) < 2:
            continue
        token = _stem(token)
        if token in _STOPWORDS or _stem(token) in _STOPWORDS:
            continue
        out.add(token)
    return out

def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""

def _authority_boost(url: str) -> float:
    host = _host(url)
    if not host:
        return 0.0

    for suffix in _AUTHORITY_HOST_SUFFIXES:
        clean = suffix.lstrip(".")
        if host == clean or host.endswith("." + clean):
            return 0.20
    return 0.0

def _overlap(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = _tokens(text)
    return len(query_terms & text_terms) / len(query_terms)

def _augment_time_sensitive_query(
    query: str,
    question: Optional[str] = None,
    mode: Optional[str] = None,
) -> str:
    """
    Normalise the year on time-sensitive queries.

        'latest stable Python version'      -> 'latest stable Python version 2026'
        'UEFA Champions League 2025 winner' -> 'UEFA Champions League winner 2026'
        'latest iPhone model 2026'          -> unchanged (already current)
        '2024 US election results'          -> unchanged (mode is 'general')

    Whether to augment is decided by the resolved recency mode - which the model
    declares - not by scanning the query for recency words. A model composing a
    query frequently drops the recency wording it was answering ("most recent"
    -> "2025"), which made the old query-only check miss exactly the cases that
    needed rewriting most.

    A year already in the query is only respected if it is current or future. A
    PAST year under recency intent is not a user-supplied constraint, it is the
    model's stale prior leaking into retrieval, so it is replaced.
    """
    if mode is None:
        mode, _ = resolve_recency_mode(query, question)
    if mode == "general":
        return query

    current_year = datetime.now(timezone.utc).year
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", query)]

    # Already anchored to now or the future - leave it alone.
    if any(year >= current_year for year in years):
        return query

    if years:
        query = re.sub(r"\b20\d{2}\b", " ", query)
        query = re.sub(r"\s{2,}", " ", query).strip(" -,")

    if not query:
        return str(current_year)

    return f"{query} {current_year}"

def _has_recency_intent(*texts: str) -> bool:
    """Recency intent is read from the question as well as the query."""
    blob = " ".join(t for t in texts if t).lower()
    if not blob:
        return False
    if any(phrase in blob for phrase in _CURRENT_PHRASES):
        return True
    return bool(_tokens(blob) & {_stem(t) for t in _CURRENT_TOKENS})


def _has_historical_anchor(*texts: str) -> bool:
    blob = " ".join(t for t in texts if t).lower()
    return any(phrase in blob for phrase in _HISTORICAL_PHRASES)


def resolve_recency_mode(
    query: str,
    question: Optional[str] = None,
    declared: Optional[str] = None,
) -> tuple[str, str]:
    """
    Classify a search as 'breaking', 'recent' or 'general'.

    The MODEL's declaration wins. It has the whole question, the entity type and
    the implicit sense of how fast this particular fact moves; the heuristic below
    has a word list. Where they disagreed on the smoke set the model was right
    both times (GTA 6 release date, marathon world record), and the previous
    version of this function overrode it in exactly those cases.

    The heuristic is a FALLBACK for when nothing was declared - older cached tool
    calls, malformed arguments, non-model callers - not a veto.

    Returns (mode, source) so disagreement is measurable rather than silent.
    """
    if declared in RECENCY_PROFILES:
        return declared, "declared"

    heuristic = heuristic_recency_mode(query, question)
    if heuristic:
        return heuristic, "heuristic"

    return DEFAULT_RECENCY_MODE, "default"


def heuristic_recency_mode(query: str, question: Optional[str] = None) -> Optional[str]:
    """
    Lexical guess at recency, or None if there is no signal either way.

    Deliberately returns None rather than 'general' for the no-signal case, so a
    caller can tell "I think this is evergreen" apart from "I have no idea".
    """
    if _has_historical_anchor(query, question):
        return "general"
    if _has_recency_intent(query, question):
        return "recent"
    return None


def _parse_date(value) -> Optional[datetime]:
    """Tolerant date parser for SearXNG publishedDate values (no dateutil dep)."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %B %Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _url_date(url: str) -> Optional[datetime]:
    """Many news URLs embed their date: /2017/11/22/slug or /2026-08-21-slug."""
    match = re.search(r"/(20\d{2})[/-](\d{1,2})(?:[/-](\d{1,2}))?(?:[/-]|$)", url or "")
    if not match:
        return None
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3) or 1)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def _text_date(text: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """
    Pull the newest date out of a title/snippet.

    SearXNG snippets very often lead with one ('Jul 29, 2026 ...', '2 days ago ...'),
    which is the cheapest freshness signal available and costs nothing to read.
    """
    if not text:
        return None
    now = now or datetime.now(timezone.utc)
    lowered = text.lower()
    found = []

    relative = re.search(r"\b(\d{1,2})\s+(day|days|hour|hours|week|weeks|month|months)\s+ago\b", lowered)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2).rstrip("s")
        days = {"hour": amount / 24, "day": amount, "week": amount * 7, "month": amount * 30}[unit]
        found.append(now - timedelta(days=days))

    for match in re.finditer(r"\b([a-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(20\d{2})\b", lowered):
        month = _MONTHS.get(match.group(1))
        if month:
            try:
                found.append(datetime(int(match.group(3)), month, int(match.group(2)), tzinfo=timezone.utc))
            except ValueError:
                pass

    for match in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", lowered):
        try:
            found.append(datetime(*(int(g) for g in match.groups()), tzinfo=timezone.utc))
        except ValueError:
            pass

    if not found:
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", lowered)]
        years = [y for y in years if y <= now.year]
        if years:
            found.append(datetime(max(years), 7, 1, tzinfo=timezone.utc))

    return max(found) if found else None


def _result_date(result: dict, now: Optional[datetime] = None) -> tuple[Optional[datetime], str]:
    """Best available date for a result, plus where it came from (for debugging)."""
    published = _parse_date(result.get("publishedDate"))
    if published:
        return published, "publishedDate"

    from_url = _url_date(result.get("url") or "")
    if from_url:
        return from_url, "url"

    from_text = _text_date(
        f"{result.get('title') or ''} {result.get('snippet') or ''}", now
    )
    if from_text:
        return from_text, "text"

    return None, "none"


def _recency_component(date: Optional[datetime], mode: str, now: Optional[datetime] = None) -> float:
    """
    Map a date to roughly [-1, 1], scaled by the profile the model chose.

    Zero on 'general' and on undated results - an unknown date is not evidence of
    staleness, and penalising it would systematically demote sites that simply do
    not publish dates.
    """
    profile = RECENCY_PROFILES.get(mode) or RECENCY_PROFILES[DEFAULT_RECENCY_MODE]
    if profile["weight"] <= 0 or date is None:
        return RECENCY_UNKNOWN

    now = now or datetime.now(timezone.utc)
    age_days = (now - date).total_seconds() / 86400.0
    fresh, stale = profile["fresh_days"], profile["stale_days"]

    if age_days < 0:
        return 0.5                      # future-dated; interesting, don't over-trust
    if age_days <= fresh:
        return 1.0
    if age_days >= stale:
        return -1.0                     # the 2017 rate-rise piece, the 2025 GTA rumour
    # Linear decay from fresh (1.0) through the midpoint (0.0) to stale (-1.0).
    span = max(stale - fresh, 1)
    return round(1.0 - 2.0 * ((age_days - fresh) / span), 4)


def _score_result(
    result: dict,
    query: str,
    original_rank: int,
    mode: str = "general",
    now: Optional[datetime] = None,
) -> float:
    """Local reranker combining lexical relevance, SearXNG score, rank and recency."""
    title = result.get("title") or ""
    snippet = result.get("snippet") or result.get("content") or ""
    url = result.get("url") or ""

    q_terms = _tokens(query)
    title_rel = _overlap(q_terms, title)
    snippet_rel = _overlap(q_terms, snippet)
    url_rel = _overlap(q_terms, url.replace("-", " ").replace("_", " "))

    try:
        searx_score = max(0.0, float(result.get("searx_score", result.get("score", 0)) or 0))
    except (TypeError, ValueError):
        searx_score = 0.0

    # Compress SearXNG score because aggregate scale varies between queries/engines.
    searx_component = min(math.log1p(searx_score) / 5.0, 0.50)
    rank_component = 0.35 / max(original_rank, 1)

    date, date_source = _result_date(result, now)
    profile = RECENCY_PROFILES.get(mode) or RECENCY_PROFILES[DEFAULT_RECENCY_MODE]
    recency_component = profile["weight"] * _recency_component(date, mode, now)

    # Stash for logging/eval; the reranker is otherwise a black box after the fact.
    result["resolved_date"] = date.isoformat() if date else None
    result["date_source"] = date_source
    result["recency_component"] = round(recency_component, 6)

    score = (
        2.50 * title_rel
        + 1.50 * snippet_rel
        + 0.35 * url_rel
        + searx_component
        + rank_component
        + _authority_boost(url)
        + recency_component
    )
    return round(score, 6)


def _rerank_results(
    results: list[dict],
    query: str,
    limit: int,
    mode: str = "general",
) -> list[dict]:
    ranked = []
    now = datetime.now(timezone.utc)

    for rank, result in enumerate(results, start=1):
        result = dict(result)
        result["original_rank"] = rank
        result["retrieval_score"] = _score_result(result, query, rank, mode, now)
        ranked.append(result)

    ranked.sort(
        key=lambda r: (
            r.get("retrieval_score", 0),
            r.get("searx_score", 0),
            -r.get("original_rank", 9999),
        ),
        reverse=True,
    )

    # Drop only obvious mismatched junk. Keep weak plausible results so graceful
    # failure cases can still correctly conclude that evidence is unavailable.
    ranked = [r for r in ranked if r.get("retrieval_score", 0) > -50]

    out = []
    seen_urls = set()
    for result in ranked:
        url = result.get("url") or ""
        canonical = url.rstrip("/")
        if canonical and canonical in seen_urls:
            continue
        if canonical:
            seen_urls.add(canonical)
        out.append(result)
        if len(out) >= limit:
            break

    return out


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def _make_passages(text: str, target_chars: int = 1200) -> list[str]:
    paragraphs = [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n{1,}", text or "")
        if p and p.strip()
    ]

    passages = []
    for paragraph in paragraphs:
        if len(paragraph) <= target_chars:
            passages.append(paragraph)
            continue

        step = max(600, target_chars - 200)
        for start in range(0, len(paragraph), step):
            chunk = paragraph[start:start + target_chars].strip()
            if chunk:
                passages.append(chunk)
            if start + target_chars >= len(paragraph):
                break

    return passages


def _passage_score(
    passage: str,
    query: str,
    position: int,
    doc_prior: float = 0.0,
) -> float:
    """
    Score a single passage.

    ``doc_prior`` is the normalised rerank score (0..1) of the page the passage
    came from. It is weighted low on purpose: it should break ties between
    comparable passages, not let a weak passage on a strong URL beat a strong
    passage on a weaker one.
    """
    q_terms = _tokens(query)
    overlap = _overlap(q_terms, passage)

    query_lower = query.lower().strip()
    passage_lower = passage.lower()
    phrase_bonus = 1.0 if len(query_lower) >= 8 and query_lower in passage_lower else 0.0
    numeric_bonus = 0.10 if re.search(r"\d", passage) else 0.0
    early_bonus = 0.08 / (position + 1)

    return (
        3.0 * overlap
        + phrase_bonus
        + numeric_bonus
        + early_bonus
        + DOC_PRIOR_WEIGHT * max(0.0, min(1.0, doc_prior))
    )


def _dedupe_key(passage: str) -> str:
    """Collapse near-identical passages (mirrors, syndicated copy, boilerplate)."""
    normalised = re.sub(r"[^a-z0-9]+", " ", passage.lower()).strip()
    return normalised[:120]


def _select_global_passages(
    docs: list[dict],
    query: str,
    total_max_chars: int = EVIDENCE_TOTAL_MAX_CHARS,
    per_doc_max_chars: int = EVIDENCE_PER_DOC_MAX_CHARS,
    max_passages_per_doc: int = EVIDENCE_MAX_PASSAGES_PER_DOC,
) -> dict[str, dict]:
    """
    Pick the best passages across every scraped page, competing in one pool.

    ``docs`` is a list of {"url", "text", "prior"}. Returns
    {url: {"text": str, "best_score": float, "n_passages": int}} for the URLs
    that actually won budget. A URL that wins nothing is simply absent - it will
    still be shown to the model with its search snippet.

    The per-document caps exist so that one long, on-topic page cannot consume
    the whole budget and starve corroborating sources; disagreement between
    sources is signal we want the model to see.
    """
    pool = []
    for doc in docs:
        url = doc.get("url") or ""
        text = doc.get("text") or ""
        prior = float(doc.get("prior") or 0.0)
        if not url or not text:
            continue

        for position, passage in enumerate(_make_passages(text)):
            pool.append({
                "score": _passage_score(passage, query, position, prior),
                "url": url,
                "position": position,
                "passage": passage,
            })

    if not pool:
        return {}

    # Highest score first; earlier passages win ties.
    pool.sort(key=lambda item: (item["score"], -item["position"]), reverse=True)

    chosen: dict[str, list[tuple[int, str]]] = {}
    best_score: dict[str, float] = {}
    doc_chars: dict[str, int] = {}
    seen_keys: set[str] = set()
    used_chars = 0

    for item in pool:
        if item["score"] <= MIN_PASSAGE_SCORE:
            break  # pool is sorted, nothing below this is worth keeping

        remaining_total = total_max_chars - used_chars
        if remaining_total <= 250:
            break

        url = item["url"]
        if len(chosen.get(url, ())) >= max_passages_per_doc:
            continue

        remaining_doc = per_doc_max_chars - doc_chars.get(url, 0)
        if remaining_doc <= 250:
            continue

        key = _dedupe_key(item["passage"])
        if key in seen_keys:
            continue

        passage = item["passage"]
        budget = min(remaining_total, remaining_doc)
        if len(passage) > budget:
            passage = passage[:budget].rstrip() + "…"

        seen_keys.add(key)
        chosen.setdefault(url, []).append((item["position"], passage))
        best_score[url] = max(best_score.get(url, 0.0), item["score"])
        doc_chars[url] = doc_chars.get(url, 0) + len(passage) + 2
        used_chars += len(passage) + 2

    out = {}
    for url, passages in chosen.items():
        passages.sort(key=lambda p: p[0])  # restore source order for readability
        out[url] = {
            "text": "\n\n".join(p for _, p in passages).strip(),
            "best_score": round(best_score.get(url, 0.0), 6),
            "n_passages": len(passages),
        }
    return out


def _passage_trace_payload(docs: list[dict], query: str, mode: str) -> dict:
    """
    Build the span input: the RAW extracted text of every scraped page.

    This is the thing that cannot be recovered from the existing trace. The
    web-search span only stores `tool_content[:3000]`, which is post-selection,
    so the passages that lost were already discarded before logging. Replaying a
    different passage scorer (cross-encoder, bi-encoder, LLM judge) needs the
    losers as much as the winners.
    """
    payload_docs = []
    total = 0
    for doc in docs:
        text = doc.get("text") or ""
        room = min(PASSAGE_TRACE_MAX_CHARS_PER_DOC, PASSAGE_TRACE_MAX_TOTAL_CHARS - total)
        if room <= 0:
            payload_docs.append({
                "url": doc.get("url"),
                "prior": round(doc.get("prior", 0.0), 4),
                "raw_chars": len(text),
                "text": None,
                "omitted": "trace budget exhausted",
            })
            continue

        clipped = text[:room]
        total += len(clipped)
        payload_docs.append({
            "url": doc.get("url"),
            "prior": round(doc.get("prior", 0.0), 4),
            "raw_chars": len(text),
            "traced_chars": len(clipped),
            "truncated": len(clipped) < len(text),
            "text": clipped,
        })

    return {
        "effective_query": query,
        "recency_mode": mode,
        "scorer": "lexical_v1",   # bump when _passage_score changes; makes runs comparable
        "config": {
            "total_max_chars": EVIDENCE_TOTAL_MAX_CHARS,
            "per_doc_max_chars": EVIDENCE_PER_DOC_MAX_CHARS,
            "max_passages_per_doc": EVIDENCE_MAX_PASSAGES_PER_DOC,
            "min_passage_score": MIN_PASSAGE_SCORE,
            "doc_prior_weight": DOC_PRIOR_WEIGHT,
        },
        "documents": payload_docs,
    }


def _trace_passage_selection(docs: list[dict], query: str, mode: str, evidence: dict) -> None:
    """
    Emit a nested 'passage-selection' span. Best-effort: tracing must never take
    the request down, and must never be the reason a search fails.
    """
    if not (PASSAGE_TRACE_ENABLED and docs):
        return
    if PASSAGE_TRACE_SAMPLE_RATE < 1.0 and random.random() > PASSAGE_TRACE_SAMPLE_RATE:
        return

    try:
        from langfuse import get_client
        client = get_client()
    except Exception:
        return

    try:
        payload = _passage_trace_payload(docs, query, mode)
        with client.start_as_current_span(
            name="passage-selection",
            input=payload,
        ) as span:
            span.update(
                output={
                    "selected": [
                        {
                            "url": url,
                            "n_passages": won["n_passages"],
                            "best_score": won["best_score"],
                            "chars": len(won["text"]),
                            "text": won["text"],
                        }
                        for url, won in evidence.items()
                    ],
                    "documents_with_evidence": len(evidence),
                    "documents_scraped": len(docs),
                    "chars_used": sum(len(w["text"]) for w in evidence.values()),
                },
                metadata={
                    "dataset_candidate": True,   # filter on this when exporting
                    "traced_chars": sum(d.get("traced_chars", 0) for d in payload["documents"]),
                },
            )
    except Exception as e:
        logger.warning("passage-selection tracing failed: %s", e)


def _extract_page_text(url: str) -> Optional[str]:
    """
    Synchronous trafilatura extraction returning the FULL page text.

    Passage selection deliberately does not happen here any more: passages are
    scored against each other across all pages once every page is in hand.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            logger.info("Scrape returned no document for %s", url)
            return None

        text = trafilatura.extract(
            downloaded,
            include_tables=True,
            include_comments=False,
            favor_precision=True,
        )
        if not text:
            logger.info("Scrape extracted no text for %s", url)
            return None

        return text[:SCRAPE_RAW_MAX_CHARS]

    except Exception as e:
        logger.warning("Scrape failed for %s: %s", url, e)
        return None



async def scrape_pages(urls: list[str]) -> dict[str, Optional[str]]:
    """
    Fetch and extract full page text for several URLs concurrently.

    NOTE: asyncio.to_thread() returns a coroutine, which does not start until it
    is awaited. Building a dict of coroutines and then awaiting them in a loop
    (as the previous scrape_urls does) runs them one after another. gather() is
    what actually makes this concurrent - it matters more now that we fetch 5
    pages instead of 3.
    """
    urls = list(dict.fromkeys(u for u in urls if u))
    if not urls:
        return {}

    outcomes = await asyncio.gather(
        *(asyncio.to_thread(_extract_page_text, url) for url in urls),
        return_exceptions=True,
    )

    results: dict[str, Optional[str]] = {}
    for url, outcome in zip(urls, outcomes):
        if isinstance(outcome, BaseException):
            logger.warning("Async scrape failed for %s: %s", url, outcome)
            results[url] = None
        else:
            results[url] = outcome

    return results

# ---------------------------------------------------------------------------
# SearXNG
# ---------------------------------------------------------------------------

async def search_searxng(
    request: Request,
    query: str,
    max_results: int,
    question: Optional[str] = None,
    mode: str = "general",
    heuristic_guess: Optional[str] = None,
) -> list[dict]:
    http_client = request.app.state.http_client
    assert http_client is not None

    url = f"{SEARXNG_INTERNAL_URL.rstrip('/')}/search"
    effective_query = _augment_time_sensitive_query(query, question, mode)

    if effective_query != query:
        logger.info("Query augmented: %r -> %r", query, effective_query)

    params = {
        "q": effective_query,
        "format": "json",
        "categories": SEARXNG_CATEGORY,
        "language": SEARXNG_LANGUAGE,
        "safesearch": SEARXNG_SAFESEARCH,
    }

    r = await http_client.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    unresponsive = data.get("unresponsive_engines") or []
    if unresponsive:
        logger.warning(
            "SearXNG unresponsive engines for query=%r: %s",
            effective_query,
            unresponsive,
        )

    results = data.get("results") or []
    out = []

    for it in results[:max_results]:
        engine = it.get("engine")
        engines = it.get("engines") or ([] if not engine else [engine])

        try:
            searx_score = float(it.get("score", 0) or 0)
        except (TypeError, ValueError):
            searx_score = 0.0

        snippet = it.get("content", "") or ""

        out.append({
            "title": it.get("title", "") or "",
            "url": it.get("url", "") or "",
            "snippet": snippet,
            "content": snippet,  # backward-compatible
            "engines": engines,
            "engine": engine or "",
            "positions": it.get("positions") or [],
            "category": it.get("category", "") or "",
            "searx_score": searx_score,
            "score": searx_score,  # backward-compatible
            "publishedDate": it.get("publishedDate") or it.get("published_date"),
            "effective_query": effective_query,
            "recency_mode": mode,
            "recency_mode_heuristic": heuristic_guess,
        })

    return out


async def search_and_scrape(
    request: Request,
    query: str,
    max_results: int,
    question: Optional[str] = None,
    time_sensitivity: Optional[str] = None,
) -> list[dict]:
    """
    Search SearXNG, locally rerank candidates, scrape the best reranked URLs,
    then select page evidence PASSAGE-BY-PASSAGE across all of those pages.

    The unit of competition is the passage, not the URL. A page that ranks 5th
    lexically but contains the one paragraph that actually answers the question
    will contribute that paragraph, and will be promoted in the final ordering
    because of it.
    """
    mode, mode_source = resolve_recency_mode(query, question, time_sensitivity)
    heuristic_guess = heuristic_recency_mode(query, question)
    if mode_source == "declared" and heuristic_guess and heuristic_guess != mode:
        # Not overridden - just recorded, so the disagreement rate is measurable
        # before anyone decides which side should win.
        logger.info(
            "Recency disagreement for %r: model=%r heuristic=%r (model wins)",
            query, mode, heuristic_guess,
        )
    candidate_count = max(max_results, SEARCH_CANDIDATES)
    search_results = await search_searxng(
        request, query, candidate_count, question, mode, heuristic_guess
    )

    if not search_results:
        return []

    effective_query = search_results[0].get("effective_query") or query
    ranked_results = _rerank_results(search_results, effective_query, max_results, mode)

    urls_to_scrape = [
        r["url"]
        for r in ranked_results[:SCRAPE_TOP_N]
        if r.get("url")
    ]

    pages = await scrape_pages(urls_to_scrape)

    # Normalise rerank scores into a 0..1 prior so the passage scorer can use them.
    scores = [r.get("retrieval_score", 0.0) for r in ranked_results[:SCRAPE_TOP_N]]
    max_score = max(scores) if scores else 0.0
    priors = {
        r.get("url", ""): (r.get("retrieval_score", 0.0) / max_score) if max_score > 0 else 0.0
        for r in ranked_results[:SCRAPE_TOP_N]
    }

    docs = [
        {"url": url, "text": text, "prior": priors.get(url, 0.0)}
        for url, text in pages.items()
        if text
    ]
    evidence = _select_global_passages(docs, effective_query)
    _trace_passage_selection(docs, effective_query, mode, evidence)

    scraped_ok = sum(1 for t in pages.values() if t)
    logger.info(
        "Search %r [%s]: %d candidates -> %d ranked, %d/%d pages extracted, "
        "%d pages won evidence (%d chars)",
        effective_query, mode, len(search_results), len(ranked_results),
        scraped_ok, len(urls_to_scrape), len(evidence),
        sum(len(e["text"]) for e in evidence.values()),
    )

    # Keep BOTH the original search snippet and the selected passages.
    for result in ranked_results:
        url = result.get("url", "")
        won = evidence.get(url)

        result["scrape_attempted"] = url in pages
        result["scrape_ok"] = bool(pages.get(url))
        result["scraped_content"] = won["text"] if won else None
        result["evidence_score"] = won["best_score"] if won else 0.0
        result["evidence_passages"] = won["n_passages"] if won else 0
        result["final_score"] = round(
            result.get("retrieval_score", 0.0)
            + EVIDENCE_BONUS_WEIGHT * (won["best_score"] if won else 0.0),
            6,
        )

    # Re-order so pages that actually carry evidence are cited first.
    ranked_results.sort(
        key=lambda r: (r.get("final_score", 0.0), r.get("retrieval_score", 0.0)),
        reverse=True,
    )

    return ranked_results


# ---------------------------------------------------------------------------
# Model context formatting
# ---------------------------------------------------------------------------

def format_search_context(
    results: list[dict],
    max_chars: Optional[int] = S_CONTEXT_MAX_CHARS,
    top_n: Optional[int] = None,
    per_result_chars: Optional[int] = None,
    header: str = "[Web search results]",
    include_engine_metadata: bool = True,
) -> str:
    if not results:
        return ""

    if top_n is not None:
        results = sorted(
            results,
            key=lambda r: (
                r.get("final_score", r.get("retrieval_score", r.get("score", 0))),
                r.get("retrieval_score", 0),
            ),
            reverse=True,
        )[:top_n]

    lines = [header] if header else []
    total_chars = len(header)

    for i, result in enumerate(results, start=1):
        title = (result.get("title") or "").strip()
        url = (result.get("url") or "").strip()
        snippet = re.sub(r"\s+", " ", (result.get("snippet") or "").strip())
        scraped_content = (result.get("scraped_content") or "").strip()

        # per_result_chars trims the raw search snippet only. Page evidence has
        # already been budgeted globally by _select_global_passages, so cutting
        # it again here would silently discard passages that won the budget.
        if per_result_chars and len(snippet) > per_result_chars:
            snippet = snippet[:per_result_chars].rstrip() + "…"

        body_parts = []
        if snippet:
            body_parts.append(f"Search snippet: {snippet}")
        if scraped_content:
            body_parts.append(f"Page evidence:\n{scraped_content}")

        metadata = ""
        if include_engine_metadata:
            engines = result.get("engines") or (
                [result["engine"]] if result.get("engine") else []
            )
            engine_text = ", ".join(str(e) for e in engines if e)
            if engine_text:
                metadata = f"\nSource engines: {engine_text}"

        body = "\n".join(body_parts).strip()
        entry = f"[{i}] {title}\nURL: {url}{metadata}\n{body}".strip()

        if max_chars is not None and total_chars + len(entry) + 2 > max_chars:
            remaining = max_chars - total_chars - 2
            if remaining > 200:
                entry = entry[:remaining].rstrip() + "…"
                lines.append(entry)
            break

        lines.append(entry)
        total_chars += len(entry) + 2

    return "\n\n".join(lines).strip()
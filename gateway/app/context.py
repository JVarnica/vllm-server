import os
import re
import math
from datetime import datetime, timezone
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
SCRAPE_TOP_N = 3               # scrape best reranked results
SCRAPE_MAX_CHARS = 5000        # query-relevant page text retained per URL
SCRAPE_RAW_MAX_CHARS = 60000   # safety cap before passage selection

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
    "incumbent", "present","next", "upcoming", "live", "next-generation",
    "update", "updated", "this month", "this year", "this week", "at the moment"
}

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


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) >= 2 and token not in _STOPWORDS
    }


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




def _augment_time_sensitive_query(query: str) -> str:
    """
    Add the current year to time-sensitive queries when no year is present.
    Example: 'latest stable Python version' -> 'latest stable Python version 2026'.

    We do not rewrite an explicitly supplied year.
    """
    terms = _tokens(query)
    if not (terms & _CURRENT_TERMS):
        return query

    current_year = str(datetime.now(timezone.utc).year)
    if re.search(r"\b20\d{2}\b", query):
        return query

    return f"{query} {current_year}"


def _score_result(result: dict, query: str, original_rank: int) -> float:
    """Local reranker combining lexical relevance, SearXNG score and rank."""
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

    score = (
        2.50 * title_rel
        + 1.50 * snippet_rel
        + 0.35 * url_rel
        + searx_component
        + rank_component
        + _authority_boost(url)
    )
    return round(score, 6)


def _rerank_results(results: list[dict], query: str, limit: int) -> list[dict]:
    ranked = []

    for rank, result in enumerate(results, start=1):
        result = dict(result)
        result["original_rank"] = rank
        result["retrieval_score"] = _score_result(result, query, rank)
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


def _passage_score(passage: str, query: str, position: int) -> float:
    q_terms = _tokens(query)
    overlap = _overlap(q_terms, passage)

    query_lower = query.lower().strip()
    passage_lower = passage.lower()
    phrase_bonus = 1.0 if len(query_lower) >= 8 and query_lower in passage_lower else 0.0
    numeric_bonus = 0.10 if re.search(r"\d", passage) else 0.0
    early_bonus = 0.08 / (position + 1)

    return 3.0 * overlap + phrase_bonus + numeric_bonus + early_bonus


def _select_relevant_text(text: str, query: str, max_chars: int) -> str:
    """
    Select relevant passages from the whole extracted page instead of keeping
    text[:5000]. This is important when the exact fact is deeper in the page.
    """
    if not text:
        return ""

    text = text[:SCRAPE_RAW_MAX_CHARS]
    passages = _make_passages(text)

    if not passages:
        return re.sub(r"\s+", " ", text[:max_chars]).strip()

    scored = [
        (_passage_score(passage, query, i), i, passage)
        for i, passage in enumerate(passages)
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    chosen = []
    used_chars = 0

    for score, position, passage in scored:
        if score <= 0:
            continue

        remaining = max_chars - used_chars
        if remaining <= 0:
            break

        if len(passage) > remaining:
            if remaining >= 250:
                passage = passage[:remaining].rstrip() + "…"
            else:
                break

        chosen.append((position, passage))
        used_chars += len(passage) + 2

    if not chosen:
        return re.sub(r"\s+", " ", text[:max_chars]).strip()

    # Restore source order for readability.
    chosen.sort(key=lambda item: item[0])
    return "\n\n".join(passage for _, passage in chosen).strip()


def _scrape_url(url: str, query: str, max_chars: int = SCRAPE_MAX_CHARS) -> Optional[str]:
    """Synchronous trafilatura extraction — run via asyncio.to_thread."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        text = trafilatura.extract(
            downloaded,
            include_tables=True,
            include_comments=False,
            favor_precision=True,
        )
        if not text:
            return None

        selected = _select_relevant_text(text, query, max_chars)
        return selected or None

    except Exception as e:
        logger.warning("Scrape failed for %s: %s", url, e)
        return None


async def scrape_urls(
    urls: list[str],
    query: str,
    max_chars: int = SCRAPE_MAX_CHARS,
) -> dict[str, Optional[str]]:
    """Scrape multiple URLs concurrently, returns {url: extracted_text}."""
    tasks = {
        url: asyncio.to_thread(_scrape_url, url, query, max_chars)
        for url in urls
    }

    results = {}
    for url, task in tasks.items():
        try:
            results[url] = await task
        except Exception as e:
            logger.warning("Async scrape failed for %s: %s", url, e)
            results[url] = None

    return results


# ---------------------------------------------------------------------------
# SearXNG
# ---------------------------------------------------------------------------

async def search_searxng(request: Request, query: str, max_results: int) -> list[dict]:
    http_client = request.app.state.http_client
    assert http_client is not None

    url = f"{SEARXNG_INTERNAL_URL.rstrip('/')}/search"
    effective_query = _augment_time_sensitive_query(query)

    params = {
        "q": effective_query,
        "format": "json",
        "categories": SEARXNG_CATEGORY,
        "language": SEARXNG_LANGUAGE,
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


        out.append({
            "title": it.get("title", "") or "",
            "url": it.get("url", "") or "",
            "content": it.get("content", "") or "",
            "engines": engines,
            "engine": engine or "",
            "positions": it.get("positions") or [],
            "category": it.get("category", "") or "",
            "searx_score": searx_score,
            "publishedDate": it.get("publishedDate") or it.get("published_date"),
            "effective_query": effective_query,
        })

    return out


async def search_and_scrape(request: Request, query: str, max_results: int) -> list[dict]:
    """
    Search SearXNG, locally rerank candidates, then scrape the best reranked URLs.
    """
    candidate_count = max(max_results, SEARCH_CANDIDATES)
    search_results = await search_searxng(request, query, candidate_count)

    if not search_results:
        return []

    effective_query = search_results[0].get("effective_query") or query
    ranked_results = _rerank_results(search_results, effective_query, max_results)

    urls_to_scrape = [
        r["url"]
        for r in ranked_results[:SCRAPE_TOP_N]
        if r.get("url")
    ]

    scraped = await scrape_urls(urls_to_scrape, effective_query)

    # Keep BOTH the original search snippet and scraped evidence.
    for result in ranked_results:
        url = result.get("url", "")
        result["scraped_content"] = scraped.get(url) if url in scraped else None

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
            key=lambda r: r.get("retrieval_score", r.get("score", 0)),
            reverse=True,
        )[:top_n]

    lines = [header] if header else []
    total_chars = len(header)

    for i, result in enumerate(results, start=1):
        title = (result.get("title") or "").strip()
        url = (result.get("url") or "").strip()
        snippet = re.sub(r"\s+", " ", (result.get("snippet") or "").strip())
        scraped_content = (result.get("scraped_content") or "").strip()

        if per_result_chars:
            if len(snippet) > per_result_chars:
                snippet = snippet[:per_result_chars].rstrip() + "…"
            if len(scraped_content) > per_result_chars:
                scraped_content = scraped_content[:per_result_chars].rstrip() + "…"

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
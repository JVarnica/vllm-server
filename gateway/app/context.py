import os 
import time
import uuid
import json
import re
from typing import List, Tuple, Optional
from pydantic import BaseModel
from fastapi import Request, APIRouter, HTTPException
import trafilatura
import asyncio

import logging
logger = logging.getLogger("uvicorn.error")

#search 
S_CONTEXT_MAX_CHARS = 20000 
TOP_N = 2 # use top 2 results
SCRAPE_MAX_CHARS = 5000 # max chars to scrape from each url
SEARXNG_INTERNAL_URL = os.environ["SEARXNG_INTERNAL_URL"]

router = APIRouter()

def _scrape_url(url: str, max_chars: int = SCRAPE_MAX_CHARS) -> Optional[str]:
    """Synchronous trafilatura extraction — run via asyncio.to_thread."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_tables=True,
            include_comments=False,
            favor_recall=True,
        )
        if not text:
            return None
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        return text
    except Exception as e:
        logger.warning(f"Scrape failed for {url}: {e}")
        return None
    

async def scrape_urls(urls: list[str], max_chars: int = SCRAPE_MAX_CHARS) -> dict[str, Optional[str]]:
    """Scrape multiple URLs concurrently, returns {url: extracted_text}."""
    tasks = {
        url: asyncio.to_thread(_scrape_url, url, max_chars)
        for url in urls
    }
    results = {}
    for url, task in tasks.items():
        try:
            results[url] = await task
        except Exception:
            results[url] = None
    return results

async def search_searxng(request: Request, query: str, max_results: int) -> list[dict]:
    http_client = request.app.state.http_client
    assert http_client is not None
    url = f"{SEARXNG_INTERNAL_URL.rstrip('/')}/search"
    r = await http_client.get(url, params={"q": query, "format": "json"})
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    out = []
    for it in results[:max_results]:
        out.append({
            "title": it.get("title", ""),
            "url": it.get("url", ""),
            "content": it.get("content", ""),
            "engine": it.get("engine", ""),
            "score": it.get("score", 0),
        })
    return out

 
async def search_and_scrape(request: Request, query: str, max_results: int) -> list[dict]:
    """Search SearXNG, then scrape top results with trafilatura for richer content."""
    search_results = await search_searxng(request, query, max_results)
    if not search_results:
        return search_results
    
        # sort by score descending, scrape the top N
    sorted_results = sorted(search_results, key=lambda r: r.get("score", 0), reverse=True)
    urls_to_scrape = [r["url"] for r in sorted_results[:TOP_N] if r.get("url")]
 
    scraped = await scrape_urls(urls_to_scrape)
 
    # merge scraped content back — use scraped text if available, fall back to snippet
    for r in search_results:
        url = r.get("url", "")
        if url in scraped and scraped[url]:
            r["content"] = scraped[url]
 
    return search_results


def format_search_context(
        results: list[dict], 
        max_chars: int = S_CONTEXT_MAX_CHARS, 
        top_n: Optional[int] = None,
        per_result_chars: Optional[int] = None,
        header: str = "[Web search results]",
) -> str:
    if not results:
        return ""

    if top_n is not None:
        # rank by score when filtering — otherwise SearXNG order is preserved
        results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)[:top_n]

    lines = [header]
    total_chars = len(lines[0]) 

    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        content = re.sub(r"\s+", " ", content)
        if per_result_chars and len(content) > per_result_chars:
            content = content[:per_result_chars].rstrip() + "…"


        entry = f"[{i}] {title}\nURL: {url}\n{content}".strip()
        if total_chars + len(entry) + 2 > max_chars:
            remaining = max_chars - total_chars - 2
            if remaining > 200:
                entry = entry[:remaining].rstrip() + "…"
            else:
                break
        lines.append(entry)
        total_chars += len(entry) + 2

    s = "\n\n".join(lines).strip()
    return s 
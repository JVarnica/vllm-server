import os
import time
import uuid
import json
import re
from datetime import datetime, timezone
from typing import List, Tuple,Optional
from pydantic import BaseModel
from fastapi import Request, APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse, StreamingResponse
from transformers import AutoTokenizer
import trafilatura
import asyncio

from app.session import get_session_context, append_pair
from app.rag import retrieve_rag_context

import logging
logger = logging.getLogger("uvicorn.error")

VLLM_URL = os.environ["VLLM_URL"]
SEARXNG_INTERNAL_URL = os.environ["SEARXNG_INTERNAL_URL"]

EMBEDDING_DIM = 384
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_PROMPT_TOKENS = 10000
MIN_GEN_TOKENS = 512
MAX_CONTEXT_WINDOW = 16384

S_CONTEXT_MAX_CHARS = 10000
TOP_N = 3 # use top 3 results
SCRAPE_MAX_CHARS = 2000 # max chars to scrape from each url

router = APIRouter()

def get_system_prompt() -> str:
    return  f"""You are a helpful AI assistant. Today's date is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.

For complex questions requiring analysis, wrap your reasoning in <think>...</think> tags before 
responding. For simple/factual questions, respond directly without thinking.

When you do use <think>, end your reasoning with a summary, a few lines <summary>...</summary> tag capturing
your key conclusion, placed just before </think>."""

SEARCH_GROUNDING_PROMPT = """You have been provided with web search results below. Follow these rules strictly:
 
1. Base your answer ONLY on the information in the search results provided.
2. Do NOT fabricate facts, scores, dates, names, or statistics not present in the results.
3. Cite sources by number [1], [2] etc when stating facts from the results.
4. If the search results do not contain enough information to fully answer the question, say so clearly — do NOT guess or fill gaps with assumptions.
5. If results conflict with each other, note the disagreement.
"""

_tokenizer = AutoTokenizer.from_pretrained(
    os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B"),
    trust_remote_code=True,
)

def count_tokens(text: str) -> int:
    """Count tokens using the actual model tokenizer."""
    return len(_tokenizer.encode(text, add_special_tokens=False))
 
 
def count_messages_tokens(messages: list[dict]) -> int:
    """Count total tokens across all messages including role overhead."""
    total = 0
    for m in messages:
        total += count_tokens(m.get("content", ""))
        total += 4  # role/formatting overhead per message
    return total

def compute_max_tokens(messages: list[dict], requested_max: int) -> int:
    """Compute max_tokens so prompt + generation fits in context window."""
    prompt_tokens = count_messages_tokens(messages)
    available = MAX_CONTEXT_WINDOW - prompt_tokens
    return max(MIN_GEN_TOKENS, available)


def condense_assistant(text: str) -> tuple[str, str]: 
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL) 
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL) 
    clean = clean.replace("<think>", "").replace("</think>", "") 
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip() 
    if think_match: 
        thinking = think_match.group(1) 
        summary_match = re.search(r"<summary>(.*?)</summary>", thinking, re.DOTALL) 
        if summary_match: 
            summary = summary_match.group(1).strip() 
            return f"[Prior note: {summary}]\n\n{clean}", summary 
        return clean, "" 
    return clean, "" 


def build_context_messages(
        session: dict, 
        current_message: str, 
        search_context: Optional[str] = None, 
        rag_context: Optional[str] = None, 
        )-> list[dict]: 
    messages = [{"role": "system", "content": get_system_prompt()}] 

    has_search = bool(search_context or any(
    ctx.get("search_context") for ctx in session["context"]
    ))
    if has_search:
        messages.append({"role": "system", "content": SEARCH_GROUNDING_PROMPT})
    for pair in session["pairs"]: # get session takes already max 6 pairs
        messages.append({"role": "user", "content": pair.get("user_text", "")}) 
        messages.append({"role": "assistant", "content": pair.get("assistant_text", "")}) 
    for context in session["context"]:
        if context.get("rag_context"):
            messages.append({"role": "system", "content": context["rag_context"]})
        if context.get("search_context"):
            messages.append({"role": "system", "content": context["search_context"]})
    if rag_context: 
        messages.append({"role": "system", "content": rag_context}) 
    if search_context: 
        messages.append({"role": "system", "content": search_context}) 

    messages.append({"role": "user", "content": current_message}) 
    return messages

# sse stream done when "finish_reason": "stop" recieved so need to have flag for when recieved
def parse_sse_stream(line: str) -> tuple[str, bool]:
    if not line.startswith("data:"):
        return ("" , False)
    data = line[len("data:"):].strip()
    if not data:
        return ("" , False)
    if data == "[DONE]":
        return ("", True)
    try:
        obj = json.loads(data)
        choices = obj.get("choices") or []
        if not choices:
            return ("" , False)
        c0 = choices[0] or {}
        delta = c0.get("delta") or {}
        content = delta.get("content") or ""
        
        finish = c0.get("finish_reason") or ""
        done = finish in ("stop", "length") 
        usage =obj.get("usage")
        return (content, done, finish, usage)
    
    except json.JSONDecodeError:
        return ("" , False, "", {})

@router.post("/chat")
async def chat(request: Request):
    user_id = request.state.user_id
    stream_client = request.app.state.stream_client
    body = await request.json()

    session_id = body.get("session_id")
    message = body.get("message", "")
    enable_search = body.get("enable_search", False)
    enable_rag = body.get("enable_rag", False)
    model = body.get("model", os.environ["VLLM_MODEL"])
    temperature = body.get("temperature", 0.7)
    req_max_tokens = body.get("max_tokens", 2048)

    if not session_id:
        raise HTTPException(400, "session_id is required")
    if not message:
        raise HTTPException(400, "message is required")
    
    session = await get_session_context(request, session_id, user_id)

    search_context = None
    if enable_search:
        search_results = await search_and_scrape(request, message, max_results=10) # searxng given urls to be ranked 
        search_context = format_search_context(search_results)

    rag_context = None
    if enable_rag:
        rag_context = await retrieve_rag_context(request, user_id, query=message)

    messages = build_context_messages(
        session, 
        message, 
        search_context=search_context, 
        rag_context=rag_context
    )
    max_tokens = compute_max_tokens(messages)
    prompt_tokens = count_messages_tokens(messages)
    logger.info(f"Session {session_id}: prompt_tokens={prompt_tokens}, max_tokens={max_tokens}")

    vllm_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    #stream request
    async def stream_vllm():
        assistant_accum: list[str] = []
    
        assert stream_client is not None
        sem = request.app.state.vllm_sem
        assert sem is not None
        async with sem:  # limit max concurrent vllm calls
            async with stream_client.stream(
                "POST",
                f"{VLLM_URL}/v1/chat/completions",
                json=vllm_payload) as resp:
                done_seen = False
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:") and not done_seen:
                        delta, done, finish, usage = parse_sse_stream(line)
                        if delta:
                            assistant_accum.append(delta)
                        if usage:
                            logger.info(f"Session {session_id}: vllm_usage={usage}")
                        if done:
                            done_seen = True
                            logger.info(f"Session {session_id}: finish_reason={finish}")
                    yield line + "\n\n"

        final = "".join(assistant_accum).strip()
        if final:
            condensed, _ = condense_assistant(final) 
            await append_pair(request,session_id, user_id, message, condensed)   # redis write (+ optional embed enqueue)

    return StreamingResponse(stream_vllm(), media_type="text/event-stream")

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


def format_search_context(results: list[dict], max_chars: int = S_CONTEXT_MAX_CHARS) -> str:
    lines = ["[Web search results]"]
    total_chars = len(lines[0]) 

    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        content = re.sub(r"\s+", " ", content)


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
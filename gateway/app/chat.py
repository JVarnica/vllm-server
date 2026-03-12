import os
import time
import uuid
import json
import re
from typing import List, Tuple,Optional
from pydantic import BaseModel
from fastapi import Request, APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse, StreamingResponse

from app.session import get_session_pairs, append_pair
from app.rag import retrieve_rag_context

VLLM_URL = os.environ["VLLM_URL"]
SEARXNG_INTERNAL_URL = os.environ["SEARXNG_INTERNAL_URL"]

MAX_CONTEXT_PAIRS = 6
EMBEDDING_DIM = 384
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

router = APIRouter()


SYSTEM_PROMPT = """
 You are a helpful AI assistant.

For complex questions requiring analysis, wrap your reasoning in <think>...</think> tags before 
responding. For simple/factual questions, respond directly without thinking.

When you do use <think>, end your reasoning with a one-line <summary>...</summary> tag capturing
your key conclusion, placed just before </think>."""


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


def build_context_messages( pairs: list[dict], current_message: str, search_context: Optional[str] = None, rag_context: Optional[str] = None, 
                           )-> list[dict]: 
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] 
    if rag_context: 
        messages.append({"role": "system", "content": rag_context}) 
    if search_context: 
        messages.append({"role": "system", "content": search_context}) 
    for pair in pairs[-MAX_CONTEXT_PAIRS:]: 
        messages.append({"role": "user", "content": pair.get("user_text", "")}) 
        condensed, _ = condense_assistant(pair.get("assistant_text", "")) 
        messages.append({"role": "assistant", "content": condensed}) 
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
        
        done = c0.get("finish_reason") == "stop"
        return (content, done)
    except json.JSONDecodeError:
        return ("" , False)

@router.post("/v1/chat/completions")
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
    max_tokens = body.get("max_tokens", 2048)

    if not session_id:
        raise HTTPException(400, "session_id is required")
    if not message:
        raise HTTPException(400, "message is required")
    
    pairs = await get_session_pairs(request, session_id, user_id)

    search_context = None
    if enable_search:
        search_results = await search_searxng(request, message, max_results=5)
        search_context = format_search_context(search_results)

    rag_context = None
    if enable_rag:
        rag_context = await retrieve_rag_context(request, user_id, query=message)

    messages = build_context_messages(
        pairs, 
        message, 
        search_context=search_context, 
        rag_context=rag_context
    )

    vllm_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
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
                        delta, done = parse_sse_stream(line)
                        if delta:
                            assistant_accum.append(delta)
                        if done:
                            done_seen = True
                    yield line + "\n\n"

        final = "".join(assistant_accum).strip()
        if final:
            await append_pair(request,session_id, user_id, message, final)   # redis write (+ optional embed enqueue)

    return StreamingResponse(stream_vllm(), media_type="text/event-stream")

async def search_searxng(request: Request, query: str, max_results: int = 5) -> list[dict]:
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
        })
    return out


def format_search_context(results: list[dict], max_chars: int = 3500) -> str:
    lines = ["[Web search results]"]
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        content = re.sub(r"\s+", " ", content)
        if len(content) > 240:
            content = content[:240].rstrip() + "…"
        lines.append(f"{i}. {title}\n{url}\n{content}".strip())
    s = "\n\n".join(lines).strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip() + "…"
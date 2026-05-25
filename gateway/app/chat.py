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
import asyncio

from app.session import get_session_context, append_pair, append_context
from app.rag import retrieve_rag_context
from app.context import search_and_scrape, format_search_context

import logging
logger = logging.getLogger("uvicorn.error")

VLLM_URL = os.environ["VLLM_URL"]
SEARXNG_INTERNAL_URL = os.environ["SEARXNG_INTERNAL_URL"]

EMBEDDING_DIM = 384
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_PROMPT_TOKENS = 10000
MIN_GEN_TOKENS = 512
MAX_CONTEXT_WINDOW = 16384

MARGIN_SAFETY = 128
# agentic loop 
MAX_TOOL_ITER = 3
TCALL_MAX_RESULTS = 5 


router = APIRouter()

def get_system_prompt() -> str:
    return  f"""You are a helpful AI assistant. Today's date is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.

For complex questions requiring analysis, wrap your reasoning in <think>...</think> tags before 
responding. For simple/factual questions, respond directly without thinking.
"""

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, recent, or specific factual information. "
            "Use this when the user asks about: recent events, news, current "
            "prices/scores/weather, specific people/companies/products, or anything "
            "that may have changed after your training cutoff. "
            "Do NOT use this for: general knowledge, math, coding, definitions, "
            "explanations of concepts, opinions, or conversational replies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Concise search query, 2-6 keywords. Avoid full sentences.",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Number of results to fetch (default {TCALL_MAX_RESULTS}).",
                    "default": TCALL_MAX_RESULTS,
                },
            },
            "required": ["query"],
        },
    },
}
SEARCH_GROUNDING_PROMPT = """You have been provided with web search results below. Follow these rules strictly:
 
1. Base your answer ONLY on the information in the search results provided.
2. Do NOT fabricate facts, scores, dates, names, or statistics not present in the results.
3. Cite sources by number [1], [2] etc when stating facts from the results.
4. If the search results do not contain enough information to fully answer the question, say so clearly — do NOT guess or fill gaps with assumptions.
5. If results conflict with each other, note the disagreement.
"""

RAG_GROUNDING_PROMPT = """You have been provided with excerpts from this user's past conversations below, retrieved by similarity to their current message. Follow these rules strictly:
1. Treat these excerpts as MEMORY, not as authoritative facts. They may be stale, partial, or out of date.
2. Use them only when they are clearly relevant to the current message. If they are not relevant, IGNORE them and answer normally — do NOT force their use.
3. Do NOT fabricate or invent things the user previously said. If a chunk does not actually contain the detail you would need, say you do not recall it rather than guessing.
4. The user's CURRENT message always takes precedence. If a past excerpt contradicts what they are saying now, defer to the current message.
"""

TOOL_CONTEXT_MARKER = "Prior turn search results:"
RAG_CONTEXT_MARKER = "User's retrieved information:"

_tokenizer = AutoTokenizer.from_pretrained(
    os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B"),
    trust_remote_code=True,
)

def count_tokens(text: list[dict], tools: list[dict] | None = None,
) -> int:
    """Prompt count need to render chat template"""
    try: 
        rendered = _tokenizer.apply_chat_template(text, tools=tools, add_generation_prompt=True, tokenize=False)
        return len(_tokenizer.encode(rendered, add_special_tokens=False))
    except Exception as e:
        logger.warning(f"apply_chat_template failed ({e}); falling back to estimate")
        total = sum(count_tokens(m.get("content") or "") for m in text)
        return total + 20 * len(text) + (200 if tools else 0)
 

def compute_max_tokens(messages: list[dict], tools: list[dict] | None = None) -> int:
    """Compute max_tokens so prompt + generation fits in context window."""
    prompt_tokens = count_tokens(messages, tools=tools)
    available = MAX_CONTEXT_WINDOW - prompt_tokens - MARGIN_SAFETY

    if available < MIN_GEN_TOKENS:
        logger.warning(
            f"Prompt is {prompt_tokens} tokens — only {available} left for "
            f"generation (below MIN_GEN_TOKENS={MIN_GEN_TOKENS}). "
            f"Consider truncating session history."
        )
    return max(MIN_GEN_TOKENS, available)

def build_context_messages(
        session: dict, 
        current_message: str, 
        rag_context: Optional[str] = None, 
        )-> tuple[list[dict], list[dict]]: 
    # probe message lightweight only needs last couple of messages
    # messages is the full context the model needs for correct answer. 
    sys_msg = {"role": "system", "content": get_system_prompt()}
    pair_msgs: list[dict] = []
    full_msgs: list[dict] = [sys_msg]

    curr_msg ={"role": "user", "content": current_message}

    for pair in session["pairs"]: # get session takes already max 6 pairs
        pair_msgs.append({"role": "user", "content": pair.get("user_text", "")}) 
        pair_msgs.append({"role": "assistant", "content": pair.get("assistant_text", "")}) 
    
    # Recent pairs only — for probe
    recent = session["pairs"][-2:]
    probe_pair_msgs: list[dict] = []
    for pair in recent:
        probe_pair_msgs.append({"role": "user", "content": pair.get("user_text", "")})
        probe_pair_msgs.append({"role": "assistant", "content": pair.get("assistant_text", "")})

    probe_msgs = [sys_msg, *probe_pair_msgs, curr_msg]  # Last 2 pairs for probing
    full_msgs.extend(pair_msgs)

    
    rag_chunks: list[str] = list(session.get("rag_context") or [])  # get all rag context chunks available from session
    if rag_context:
        rag_chunks.append(rag_context)
    if rag_chunks:
        full_msgs.append({"role": "system", "content": RAG_GROUNDING_PROMPT}) 
        full_msgs.append({"role": "system", "content": RAG_CONTEXT_MARKER + "\n" + "\n\n".join(rag_chunks)})   
   
    prior_tool: str | None = session.get("tool_context")
    if prior_tool:
        full_msgs.append({"role": "system", "content": f"{TOOL_CONTEXT_MARKER}\n{prior_tool}"})
    
    full_msgs.append(curr_msg)
  
    return probe_msgs, full_msgs

# sse stream done when "finish_reason": "stop" recieved so need to have flag for when recieved
def parse_sse_stream(line: str) -> tuple[str, bool, str, dict]:
    empty = ("", False, "", {})
    if not line.startswith("data:"):
        return empty
    data = line[len("data:"):].strip()
    if not data:
        return empty
    if data == "[DONE]":
        return ("", True, "stop",{})
    try:
        obj = json.loads(data)
        choices = obj.get("choices") or []
        if not choices:
            return empty
        
        c0 = choices[0] or {}
        delta = c0.get("delta") or {}
        content = delta.get("content") or ""
        
        finish = c0.get("finish_reason") or ""
        done = finish in ("stop", "length") 
        usage =obj.get("usage")
        return (content, done, finish, usage)
    
    except json.JSONDecodeError:
        return empty

@router.post("/chat")
async def chat(request: Request):
    user_id = request.state.user_id
    stream_client = request.app.state.stream_client
    http_client = request.app.state.http_client
    body = await request.json()

    session_id = body.get("session_id")
    message = body.get("message", "")
    # enable_search = body.get("enable_search", False)
    enable_rag = body.get("enable_rag", False)
    model = body.get("model", os.environ["VLLM_MODEL"])
    temperature = body.get("temperature", 0.7)
    

    if not session_id:
        raise HTTPException(400, "session_id is required")
    if not message:
        raise HTTPException(400, "message is required")
    
    session = await get_session_context(request, session_id, user_id)
    logger.info(f"Session {session_id}: received message: '{message}' session: {session}")
    
    #context pairs and rag context
    rag_context = None
    if enable_rag:
        rag_context = await retrieve_rag_context(request, user_id, query=message)

    sem = request.app.state.vllm_sem

    async def stream_agent():
        #  1: Decide tool usage 
        probe_msgs, messages = build_context_messages(session, message, rag_context=rag_context)
        did_search = False 
        tool_accum: list[dict] = []
        tool_history: list[dict] = []
        for tool_iter in range(MAX_TOOL_ITER):

            probe_payload = {
                "model": model,
                "messages": probe_msgs + tool_history,
                "tools": [WEB_SEARCH_TOOL],
                "tool_choice": "auto",
                "max_tokens": 2000, 
                "stream": False,
            }
            async with sem:
                probe_resp = await http_client.post(
                    f"{VLLM_URL}/v1/chat/completions", 
                    json=probe_payload,
                )
            probe_resp.raise_for_status()
            probe_data = probe_resp.json()

            msg = probe_data["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break  # no tool calls, proceed to final response
            
            did_search = True
            #capture tool use reasoning
            #probe_reason = msg.get("reasoning_content") or "" was for reasoning parser but its noise. 
           
            #append assistant turn with tool calls
            tool_history.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls
            })
            
            #execute all tool calls sequentially currently
            for tc in tool_calls:
                fn = tc.get("function") or {}
                tc_id = tc.get("id")
                logger.info(f"Session {session_id}: tool_call ID:{tc_id}")
                if fn.get("name") != "web_search":
                    tool_history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": f"Error: unknown tool {fn.get('name')}"
                    })
                    continue

                raw_args = fn.get("arguments") or {}
                try: 
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                query = args.get("query", message)
                max_results = int(args.get("max_results", TCALL_MAX_RESULTS))
                #search_trace.append(query)
                yield f"event: tool_use\ndata: {json.dumps({'query': query, 'iter': tool_iter})}\n\n"
                 
                try:
                    search_results = await search_and_scrape(request, query, max_results=max_results)
                    # tool content given to model 
                    tool_content = format_search_context(search_results)
                    tool_accum.extend(search_results)
                    count = len(search_results)
                except Exception as e:
                    tool_content = f"Error during web search: {str(e)}"
                    count = 0
                    logger.exception(f"search failed for '{query}'")
    
                yield f"event: tool_result\ndata: {json.dumps({'query': query, 'results_count': count})}\n\n"
                logger.info(f"Session {session_id}: iter={tool_iter} '{query}' → {count} results")
            

                tool_history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_content
                })
                
        else:
            # max loop reached without break
            tool_history.append({
                "role": "system",
                "content": "Maximum searches done, used all tool iterations. Now answer with gathered information."
            })
        messages.extend(tool_history)  # add tool context to final messages
        
        #Grounding prompt telling model how to use search results
        has_prior_search = any(
            m.get("role") == "system" and (m.get("content") or "").startswith(TOOL_CONTEXT_MARKER)
            for m in messages
            )
        if did_search or has_prior_search:
            messages.insert(1, {"role": "system", "content": SEARCH_GROUNDING_PROMPT})


        max_tokens = compute_max_tokens(messages)
        prompt_tokens = count_tokens(messages)
        logger.info(f"Session {session_id}: prompt_tokens={prompt_tokens}, max_tokens={max_tokens}, did_search={did_search})")
        
        # Final streaming call no tools 
        vllm_payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        assistant_accum: list[str] = []
        #reasoning_accum: list[str] = []
        try:
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

        finally:
            final_accum = "".join(assistant_accum).strip()

            if final_accum:
                clean = re.sub(r"\n{3,}", "\n\n", final_accum).strip()

                # tool_context to append to redis set, so previous turn availiable. no list as no point taking more just last one
                tool_context = format_search_context(tool_accum, max_chars=None, top_n=2,per_result_chars=1000, header="") if tool_accum else "" # do the header in build_messaqges
                
                tool_tokens = count_tokens([{"role": "system", "content": tool_context}]) if tool_context else 0
                clean_tokens = count_tokens([{"role": "assistant", "content": clean}])
                logger.info(
                    f"Session {session_id}: clean_tokens={clean_tokens}, tool_tokens={tool_tokens}"
                )
                try:
                    await append_pair(request,session_id, user_id, message, clean)   # redis write (+ optional embed enqueue)
                    await append_context(request, session_id, user_id, rag_context=rag_context, tool_context=tool_context) 
                except Exception as e:
                    logger.exception(f"Failed to append session context: {e}")
                
    return StreamingResponse(stream_agent(), media_type="text/event-stream")

           



 
 

import os
import time
import uuid
import json
import re
from datetime import datetime, timezone
from typing import List, Tuple,Optional, Any
from pydantic import BaseModel
from fastapi import Request, APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse, StreamingResponse

from langfuse import get_client
import asyncio

from app.session import get_session_context, append_pair, append_context
from app.rag import retrieve_rag_context
from app.context import search_and_scrape, format_search_context
from app.calculator import CALCULATOR_TOOL, execute_calculator_tool
from app.chat_helper_funcs import count_tokens, compute_max_tokens, parse_sse_stream

import logging
logger = logging.getLogger("uvicorn.error")

VLLM_URL = os.environ["VLLM_URL"]
SEARXNG_INTERNAL_URL = os.environ["SEARXNG_INTERNAL_URL"]


EMBEDDING_DIM = 384
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_PROMPT_TOKENS = 10000
MIN_GEN_TOKENS = 512

MARGIN_SAFETY = 128
# agentic loop 
MAX_TOOL_ITER = 3
TCALL_MAX_RESULTS = 8
TOOL_OUTPUT_MAX_CHARS = int(os.environ.get("TOOL_OUTPUT_MAX_CHARS", "12000"))

TOOL_CONTEXT_MARKER = "Prior turn tool results:"
RAG_CONTEXT_MARKER = "User's retrieved information:"

langfuse = get_client()
router = APIRouter()

def get_system_prompt() -> str:
    now = datetime.now(timezone.utc)
    today = now.strftime("%A %Y-%m-%d")
    return f"""You are a helpful AI assistant with access to tools.
Today is {today}.

Available capabilities:
    - web_search: current, recent, or externally verifiable information.
    - calculate: exact arithmetic and mathematical expressions.

Tool Rules:
1. Use web_search when facts may have changed, are recent, or need verification. 
2. Do not use web_search to verify historical facts, and general knowledge that you already know. Your training data is sufficient for that.
2. Use calculate for ANY arithmetic beyond single-digit sums. Always use calculate for complex arithmetic, large numbers, math facts or when precision is required.
3. Do not call tools when a direct answer is sufficient.
4. For questions about software versions, model capabilities, product features, or 'best X' rankings, always search — your training data is stale for these",
5. Treat tool results as information available in the current conversation.
"""

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, recent, or specific factual information. "
            "Use this when the user asks about: recent events, news, current "
            "prices/scores/weather, specific people/companies/products, or facts "
            "that may have changed after your training cutoff. "
            "Do not use for arithmetic or code execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Concise search query. Avoid full sentences.",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Number of results to fetch (default {TCALL_MAX_RESULTS}).",
                    "default": TCALL_MAX_RESULTS,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
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

CALCULATE_GROUNDING_PROMPT = """You have been provided with results from a calculator tool below. Follow these rules strictly:
1. Use the returned value EXACTLY as given. Do NOT recompute it or change the digits.
2. Do NOT perform further arithmetic yourself. If you need another value, call the calculator again.
3. If the tool returned an error, tell the user plainly what went wrong — do NOT substitute an estimate of your own.
4. You may add units, currency symbols, or wording around the number, but the number itself must not change.
5. Only round the value if the user explicitly asked for a certain number of decimal places.
"""

TOOLS = [WEB_SEARCH_TOOL, CALCULATOR_TOOL]

def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

def build_context_messages(
        session: dict, 
        current_message: str, 
        rag_context: Optional[str] = None, 
        )-> tuple[list[dict[str, Any]], list[dict[str, Any]]]: 
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

    rag_chunks = list(session.get("rag_context") or [])  # get all rag context chunks available from session
    if rag_context:
        rag_chunks.append(rag_context)
    if rag_chunks:
        full_msgs.append({"role": "system", "content": RAG_GROUNDING_PROMPT}) 
        full_msgs.append({"role": "system", "content": RAG_CONTEXT_MARKER + "\n" + "\n\n".join(rag_chunks)})   
   
    prior_tool = session.get("tool_context")
    if prior_tool:
        full_msgs.append({"role": "system", "content": f"{TOOL_CONTEXT_MARKER}\n{prior_tool}"})
    
    full_msgs.append(curr_msg)
  
    return probe_msgs, full_msgs


@router.post("/chat")
async def chat(request: Request):
    user_id = request.state.user_id
    stream_client = request.app.state.stream_client
    http_client = request.app.state.http_client
    sem = request.app.state.vllm_sem

    body = await request.json()
    session_id = body.get("session_id")
    message = body.get("message", "")
    # enable_search = body.get("enable_search", False)
    enable_rag = body.get("enable_rag", False)
    model = body.get("model", os.environ["VLLM_MODEL"])
    temperature = body.get("temperature", 0.5)
    
    if not session_id:
        raise HTTPException(400, "session_id is required")
    if not message:
        raise HTTPException(400, "message is required")
    
    session = await get_session_context(request, session_id, user_id)
    logger.info(f"Session {session_id}: received message: '{message}")
    
    #context pairs and rag context
    rag_context = None
    if enable_rag:
        rag_context = await retrieve_rag_context(request, user_id, query=message)

    async def stream_agent():
        # 0: one trace per chat request; every probe, search and generation nests under it
        with langfuse.start_as_current_span(name="chat-request") as root:
            root.update_trace(
                session_id=session_id,
                user_id=user_id,
                input=message,
                tags=["chat"],
                metadata={"model": model, "enable_rag": enable_rag},
            )
            #yield the trace to id so can track trace
            yield sse_event("trace", {"trace_id": langfuse.get_current_trace_id()})
            #  1: Decide tool usage
            probe_msgs, messages = build_context_messages(session, message, rag_context=rag_context)

            tool_accum: list[dict] = []
            tool_history: list[dict] = []

            did_search = False
            did_calculate = False

            for tool_iter in range(MAX_TOOL_ITER):

                probe_payload = {
                    "model": model,
                    "messages": probe_msgs + tool_history[-1],
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "stream": False,
                }

                with langfuse.start_as_current_generation(
                    name="routing-probe",
                    model=model,
                    input=probe_msgs + tool_history,
                    model_parameters={
                        "temperature": 0.1,
                        "max_tokens": 2000,
                        "tool_choice": "auto",
                    },
                    metadata={"iter": tool_iter},
                ) as probe_gen:
                    async with sem:
                        probe_resp = await http_client.post(
                            f"{VLLM_URL}/v1/chat/completions",
                            json=probe_payload,
                        )
                    probe_resp.raise_for_status()
                    probe_data = probe_resp.json()

                    msg = probe_data["choices"][0]["message"]
                    tool_calls = msg.get("tool_calls") or []

                    probe_usage = probe_data.get("usage") or {}
                    probe_gen.update(
                        output={
                            "tool_calls": tool_calls,
                            "content": msg.get("content") or "",
                        },
                        usage_details={
                            "input": probe_usage.get("prompt_tokens"),
                            "output": probe_usage.get("completion_tokens"),
                        } if probe_usage else None,
                        metadata={
                            "iter": tool_iter,
                            "n_calls": len(tool_calls),
                            "routed": bool(tool_calls),
                            "tool_names": [
                                (tc.get("function") or {}).get("name") for tc in tool_calls
                            ],
                        },
                    )

                if not tool_calls:
                    break  # no tool calls, proceed to final response

                tool_history.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                })

                # execute all tool calls sequentially.
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    tc_id = tc.get("id")
                    tool_name = fn.get("name")
                    logger.info(f"Session {session_id}: tool_call ID:{tc_id}")

                    if tool_name not in ("web_search", "calculate"):
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

                    if tool_name == "web_search":
                        query = str(args.get("query", message))
                        max_results = int(args.get("max_results", TCALL_MAX_RESULTS))
                        # for UI so knows that its searching
                        yield sse_event("tool_use", {"name": tool_name, "iter": tool_iter, "query": query})

                        with langfuse.start_as_current_span(
                            name="web-search",
                            input={"query": query, "max_results": max_results},
                        ) as search_span:
                            try:
                                search_results = await search_and_scrape(request, query, max_results=max_results)
                                # tool content given to model
                                tool_content = format_search_context(search_results)
                                tool_accum.extend(search_results)  # for context not this req
                                result_count = len(search_results)
                                did_search = True
                                search_span.update(
                                    output={
                                        "results_count": result_count,
                                        "urls": [r.get("url") for r in search_results][:10],
                                    },
                                )
                            except Exception as e:
                                tool_content = f"Error during web search: {str(e)}"
                                result_count = 0
                                logger.exception(f"search failed for '{query}'")
                                search_span.update(
                                    output={"results_count": 0, "error": str(e)},
                                    level="ERROR",
                                    status_message=str(e),
                                )

                        yield sse_event("tool_result", {"name": tool_name, "query": query, "results_count": result_count})
                        logger.info(f"Session {session_id}: iter={tool_iter} '{query}' -> {result_count} results")

                        tool_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": tool_content
                            }
                        )
                        continue

                    if tool_name == "calculate":

                        expression = str(args.get('expression', ""))
                        # need the calculation to pop up on UI
                        yield sse_event("tool_use", {'name': tool_name, 'expression': expression, 'iter': tool_iter})

                        with langfuse.start_as_current_span(
                            name="calculate",
                            input={"expression": expression},
                        ) as calc_span:
                            result = await execute_calculator_tool(args)
                            tool_content = (
                                f"Result: {result['result']}" if result["ok"]
                                else f"Error: {result['error']}"
                            )
                            calc_span.update(output=result)
                            if not result["ok"]:
                                calc_span.update(level="WARNING", status_message=str(result.get("error")))

                        # True even if failed.
                        did_calculate = True

                        yield sse_event("tool_result", {'name': tool_name, **result})
                        tool_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": tool_content
                            }
                        )
                        continue

            else:
                # max loop reached without break
                tool_history.append({
                    "role": "system",
                    "content": "Maximum tool iterations. Now answer with gathered information."
                })
            messages.extend(tool_history)  # add tool context to final messages

            # Grounding prompt telling model how to use search results
            has_prior_search_context = any(
                m.get("role") == "system" and (m.get("content") or "").startswith(TOOL_CONTEXT_MARKER)
                for m in messages
            )
            if did_search or has_prior_search_context:
                messages.insert(1, {"role": "system", "content": SEARCH_GROUNDING_PROMPT})
            if did_calculate:
                messages.insert(1, {"role": "system", "content": CALCULATE_GROUNDING_PROMPT})

            max_tokens = compute_max_tokens(messages)
            prompt_tokens = count_tokens(messages)
            logger.info(f"Session {session_id}: prompt_tokens={prompt_tokens}, max_tokens={max_tokens}, did_search={did_search}, did_calculate={did_calculate})")

            root.update_trace(
                metadata={
                    "did_search": did_search,
                    "did_calculate": did_calculate,
                    "tool_iters": tool_iter + 1,
                    "prompt_tokens": prompt_tokens,
                },
            )

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
            # reasoning_accum: list[str] = []

            # defined up-front so the finally block can never raise NameError
            usage = None
            finish = None

            # start_generation, not the context-manager form: a `with` spanning a
            # yield only exits when the generator is closed, and a disconnected
            # client may not close it promptly. Explicit .end() in finally is
            # deterministic.
            gen = langfuse.start_generation(
                name="chat-completion",
                model=model,
                input=messages,
                model_parameters={"temperature": temperature, "max_tokens": max_tokens},
            )
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

                # close the generation first: a client disconnect mid-stream still
                # records whatever was produced rather than dropping the span
                gen.update(
                    output=final_accum,
                    usage_details={
                        "input": usage.get("prompt_tokens"),
                        "output": usage.get("completion_tokens"),
                    } if usage else None,
                    metadata={"finish_reason": finish, "truncated": not done_seen},
                )
                gen.end()
                root.update_trace(output=final_accum)

                if final_accum:
                    clean = re.sub(r"\n{3,}", "\n\n", final_accum).strip()

                    # tool_context to append to redis set, so previous turn availiable. no list as no point taking more just last one
                    tool_context = format_search_context(tool_accum, max_chars=None, top_n=2, per_result_chars=1000, header="") if tool_accum else ""  # do the header in build_messaqges

                    tool_tokens = count_tokens([{"role": "system", "content": tool_context}]) if tool_context else 0
                    clean_tokens = count_tokens([{"role": "assistant", "content": clean}])
                    logger.info(
                        f"Session {session_id}: clean_tokens={clean_tokens}, tool_tokens={tool_tokens}"
                    )
                    try:
                        await append_pair(request, session_id, user_id, message, clean)   # redis write (+ optional embed enqueue)
                        await append_context(request, session_id, user_id, rag_context=rag_context, tool_context=tool_context)
                    except Exception as e:
                        logger.exception(f"Failed to append session context: {e}")

    return StreamingResponse(stream_agent(), media_type="text/event-stream")
           



 
 

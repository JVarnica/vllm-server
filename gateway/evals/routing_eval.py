"""
Routing eval for the gateway's probe call.

Tests one thing: given a user message, does the model emit a web_search tool
call when it should, and stay quiet when it shouldn't. Hits vLLM directly with
the same payload shape stream_agent uses, so SearXNG / trafilatura / Redis /
the final generation are all out of the measurement.

Run inside the gateway container so the imports and VLLM_URL resolve:

    docker compose exec gateway python -m evals.routing_eval \
        --arm baseline --repeats 5 --temperature 0

Writes raw per-call results to runs/<arm>.jsonl. Analysis is a separate script
on purpose: the expensive part is the GPU time, and you will want to re-slice
the numbers without re-running the model.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

# Import the real artifacts under test, never a copy. If the tool description
# is edited in chat.py, this eval picks it up automatically.
from app.chat import TOOLS, build_context_messages
from app.calculator import calculate, CalcError, format_result

VLLM_URL = os.environ["VLLM_URL"]
VLLM_MODEL = os.environ["VLLM_MODEL"]

LANGFUSE_ENABLED = os.environ.get("ROUTING_EVAL_TRACE", "1") == "1"
if LANGFUSE_ENABLED:
    from langfuse import get_client

    langfuse = get_client()


EMPTY_SESSION = {"pairs": [], "rag_context": [], "tool_context": None}


async def probe(client, sem, message, temperature):
    """One probe call. Mirrors the payload in stream_agent, minus the loop."""
    probe_msgs, _ = build_context_messages(EMPTY_SESSION, message, rag_context=None)

    payload = {
        "model": VLLM_MODEL,
        "messages": probe_msgs,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 2000,
        "stream": False,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    async with sem:
        resp = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()

    msg = data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []

    names, search_queries, expressions = [], [], []
    malformed_args = 0
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name")
        names.append(name)
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            malformed_args += 1
            continue
        if name == "web_search":
            search_queries.append(args.get("query", ""))
        elif name == "calculate":
            expressions.append(args.get("expression", ""))

    return {
        "n_search": names.count("web_search"),
        "n_calculate": names.count("calculate"),
        "n_unknown": sum(1 for n in names if n not in ("web_search", "calculate")),
        "tool_names": names,
        "queries": search_queries,
        "expressions": expressions,
        "malformed_args": malformed_args,
        "content": msg.get("content") or "",
        "reasoning_content": (msg.get("reasoning_content") or "")[:4000],
        "finish_reason": data["choices"][0].get("finish_reason"),
        "usage": data.get("usage", {}),
        "latency_ms": round(latency_ms, 1),
    }

def _check_expression(expression, expected):
    """Run the emitted expression through the real calculator.
 
    Argument correctness is deliberately separate from routing correctness:
    a model that routes perfectly and writes sin(30) instead of
    sin(radians(30)) is broken in a way the routing score cannot see.
    """
    out = {"expr": expression}
    try:
        out["value"] = format_result(calculate(expression))
        out["ok"] = True
    except CalcError as e:
        out["ok"] = False
        out["error"] = str(e)
        return out
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        return out
 
    if expected is not None:
        # tolerance absorbs legitimate rounding choices, e.g. round(x, 2)
        # versus leaving the value unrounded
        out["expected"] = expected
        out["matches"] = abs(out["value"] - expected) <= max(
            0.01, abs(expected) * 0.001
        )
    return out

def _score(item, result):
    """Routing correctness for one probe.

    Policy (change here, not scattered through analysis):
      - min_search == 0 means searching is WRONG, not merely unnecessary.
        The docstring's contract is "stay quiet when it shouldn't".
      - min_calc == "either" (calc_trivial stratum): calling or not calling
        the calculator are both acceptable; the item can only fail on the
        search side or on malformed output.
      - Unknown tool names or unparseable arguments fail the item outright:
        the gateway would 500 on them regardless of routing intent.
    """
    min_search = item["min_search"]
    min_calc = item["min_calc"]
    if min_search == "either":
        search_ok = True
    elif min_search == 0:
        search_ok = result["n_search"] == 0
    else:
        search_ok = result["n_search"] >= min_search

    # Coverage check: for multi-entity items, every entity must appear in
    # some emitted query — combined or split both count.
    entities = item.get("entities")
    if entities and result["queries"]:
        all_q = " ".join(result["queries"]).lower()
        search_ok = search_ok and all(e.lower() in all_q for e in entities)

    if min_calc == "either":
        calc_ok = True
    elif min_calc == 0:
        calc_ok = result["n_calculate"] == 0
    else:
        calc_ok = result["n_calculate"] >= min_calc

    clean = result["n_unknown"] == 0 and result["malformed_args"] == 0
    return {
        "search_ok": search_ok,
        "calc_ok": calc_ok,
        "clean": clean,
        "correct": search_ok and calc_ok and clean,
        # over/under flags let the analysis split "too eager" from "too shy"
        "over_search": min_search == 0 and result["n_search"] > 0,
        "under_search": isinstance(min_search, int) and min_search != 0 and not search_ok,
        "over_calc": min_calc == 0 and result["n_calculate"] > 0,
        "under_calc": isinstance(min_calc, int)
        and min_calc > 0
        and result["n_calculate"] < min_calc,
    }


async def run_item(client, sem, item, rep, args):
    """One (question, rep) pair: probe, score routing, check calc arguments."""
    if not LANGFUSE_ENABLED:
        result = await probe(client, sem, item["q"], args.temperature)
    else:
        with langfuse.start_as_current_span(name="routing-eval-item") as root:
            root.update_trace(
                input=item["q"],
                tags=["routing-eval", args.arm],
                metadata={"id": item["id"], "stratum": item["stratum"], "rep": rep},
            )
            with langfuse.start_as_current_generation(
                name="routing-probe",
                model=VLLM_MODEL,
                input=item["q"],
                model_parameters={"temperature": args.temperature},
            ) as gen:
                result = await probe(client, sem, item["q"], args.temperature)
                gen.update(
                    output={
                        "tool_names": result["tool_names"], 
                        "queries": result["queries"],
                        "reasoning_content": result["reasoning_content"],
                        "content": result["content"],
                    },
                    usage_details={
                        "input": result["usage"].get("prompt_tokens"),
                        "output": result["usage"].get("completion_tokens"),
                    },
                )

    row = {
        "arm": args.arm,
        "rep": rep,
        "id": item["id"],
        "stratum": item["stratum"],
        "q": item["q"],
        "min_search": item["min_search"],
        "min_calc": item["min_calc"],
        **result,
        **_score(item, result),
    }
    # Argument correctness, deliberately outside `correct` (see module docstring):
    # routing and expression-writing are different failure modes.
    if result["expressions"]:
        row["calc_checks"] = [
            _check_expression(expr, item.get("expected_value"))
            for expr in result["expressions"]
        ]
    return row


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, help="name for this configuration")
    p.add_argument("--questions", default="evals/routing_questions_v2.jsonl")
    p.add_argument("--out", default="evals/runs")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    items = [
        json.loads(line)
        for line in Path(args.questions).read_text().splitlines()
        if line.strip()
    ]
    print(f"{len(items)} items x {args.repeats} reps = {len(items) * args.repeats} calls")

    sem = asyncio.Semaphore(args.concurrency)
    out_path = Path(args.out) / f"{args.arm}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for rep in range(args.repeats):
            async with httpx.AsyncClient(timeout=120) as client:
                rows = await asyncio.gather(
                    *(run_item(client, sem, item, rep, args) for item in items)
                )
            for row in rows:
                f.write(json.dumps(row) + "\n")
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"  rep {rep}: {acc:.1%}")

    if LANGFUSE_ENABLED:
        langfuse.flush()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
"""
End-to-end answer eval: the real /chat endpoint, nothing mocked.

The unit under test is the server. POST each question to /chat with a fresh
session, read the SSE stream the gateway already emits, keep the final answer
plus the tool_use events. Zero gateway changes required.

What each row carries:
  answer        accumulated content deltas — exactly what a user sees
  turns         tool_calls from the existing tool_use events (query/expression,
                iter). No tool_result content — the SSE doesn't carry it, so
                the scorer's extraction-link check auto-skips; everything else
                (exact answer, entities, graceful, hygiene, search-before-calc,
                repeated-call) scores normally.
  usage         final-call usage if the stream includes it
  latency_ms    wall clock, request start -> stream end (single-user only if
                --concurrency 1; eval-load latency != single-user latency)

Langfuse: same pattern as routing_eval — one trace per (item, rep), tagged
["answer-eval", <arm>] with id/stratum/rep metadata, trace_id stored in the
row. Disable with ANSWER_EVAL_TRACE=0. Needs LANGFUSE_* env vars where the
runner executes. If the gateway traces /chat internally, those are separate
traces; this one is the experiment ledger.

Run from the host (or anywhere that can reach the gateway):

    python evals/answer_e2e.py evals/answer_questions_v1.jsonl \
        --arm reference --repeats 5 --base-url http://localhost:8000

Then:

    python evals/score_answer.py evals/runs/answer_reference.jsonl

Note: chat.py's calculate branch must yield a tool_use event like the search
branch does (one line if missing) or search-before-calc ordering is invisible
to the scorer. Trajectory content / fixtures live in Langfuse traces, not here.
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import httpx

TEMPERATURE = 0.1

LANGFUSE_ENABLED = os.environ.get("ANSWER_EVAL_TRACE", "1") == "1"
if LANGFUSE_ENABLED:
    from langfuse import get_client

    langfuse = get_client()


async def create_session(client, base_url):
    """Sessions must exist before /chat (get_session_context 404s otherwise),
    so create one the same way the frontend does. If /session/create requires
    extra fields, a 422 here will name them — add them to the json payload."""
    r = await client.post(f"{base_url}/session/create", json={}, timeout=30)
    r.raise_for_status()
    body = r.json()
    sid = body.get("session_id") or body.get("id") or body.get("sid")
    if not sid:
        raise RuntimeError(f"/session/create returned no recognizable id: {body}")
    return sid


async def ask(client, base_url, question, arm, rep, qid):
    session_id = await create_session(client, base_url)
    payload = {
        "session_id": session_id,
        "message": question,
        "temperature": TEMPERATURE,
    }
    turns: dict[int, dict] = {}
    content_parts: list[str] = []
    usage = []
    error = None
    event_name = ""
    t0 = time.perf_counter()

    try:
        async with client.stream("POST", f"{base_url}/chat", json=payload,
                                 timeout=300) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    event_name = ""
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if event_name == "tool_use":
                    it = obj.get("iter", 0)
                    turn = turns.setdefault(it, {"tool_calls": [], "tool_results": []})
                    # gateway emits {'query': ..., 'iter': ...} for search;
                    # calc events carry the expression under 'query' or 'expression'
                    tool = obj.get("tool", "calculate" if "expression" in obj else "web_search")
                    args = ({"expression": obj.get("expression") or obj.get("query")}
                            if tool == "calculate" else {"query": obj.get("query")})
                    turn["tool_calls"].append({"tool": tool, "args": args})
                if event_name == "tool_result": # just for search needed 
                    it = obj.get("iter", 0)
                    turn = turns.setdefault(it, {"tool_calls": [], "tool_results": []})
                    tool_name = obj.get("tool") or ("calculate" if "expression" in obj else "web_search")
                    query = obj.get("query") 
                    result_count = obj.get("results_count")
                    tool_content = obj.get("content")
                    metadata = obj.get("metadata")  or [] # optional, only for web_search with results
                    turn["tool_results"].append({"tool": tool_name, "query": query, "results_count": result_count, "content": tool_content, "metadata": metadata})

                elif event_name in ("", "message"):
                    choices = obj.get("choices") or []
                    if choices:
                        delta = (choices[0] or {}).get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                    if obj.get("usage"):
                        usage.append({"phase": "final", **obj["usage"]})
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    answer = "".join(content_parts).strip()
    if not error and not answer:
        error = "empty final answer"
    ordered = [turns[k] for k in sorted(turns)]
    all_results = [tr for t in ordered for tr in t["tool_results"]]
    # Flat, ordered grounding for the judge: one block per call, labelled with
    # the query so a multi-search answer can be traced back per entity.
    blocks = [f"### {tr['query']}\n{tr['content']}" for tr in all_results if tr["content"]]
    tool_content = "\n\n".join(blocks)

    return {
        "answer": answer,
        "turns": ordered,
        "tool_content": tool_content,
        "n_iterations": len(ordered),
        "n_tool_calls": sum(len(t["tool_calls"]) for t in ordered),
        "usage": usage,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "trace_id": None,  # gateway's own Langfuse tracing owns this run's traces
        "error": error,
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("questions")
    p.add_argument("--arm", required=True)
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--out", default="evals/runs")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="first N items (smoke)")
    p.add_argument("--auth-url", default="http://auth:8090",
                   help="auth service base for self-login")
    p.add_argument("--user", default=None, help="login username (self-login)")
    p.add_argument("--password", default=None, help="login password (self-login)")
    args = p.parse_args()

    items = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} items x {args.repeats} reps = {len(items) * args.repeats} calls")

    sem = asyncio.Semaphore(args.concurrency)
    out_path = Path(args.out) / f"answer_determ_{args.arm}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def one(client, item, rep):
        async with sem:
            if not LANGFUSE_ENABLED:
                r = await ask(client, args.base_url, item["question"], args.arm, rep, item["id"])
            else:
                with langfuse.start_as_current_span(name="answer-eval-item") as root:
                    root.update_trace(
                        input=item["question"],
                        tags=["answer-eval", args.arm],
                        metadata={"id": item["id"], "stratum": item["stratum"], "rep": rep},
                    )
                    r = await ask(client, args.base_url, item["question"], args.arm, rep, item["id"])
                    root.update_trace(output={
                        "answer": r["answer"][:2000],
                        "n_iterations": r["n_iterations"],
                        "n_tool_calls": r["n_tool_calls"],
                        "latency_ms": r["latency_ms"],
                        "error": r["error"],
                    })
                    r["trace_id"] = langfuse.get_current_trace_id()
        return {
            "id": item["id"], "stratum": item["stratum"], "question": item["question"],
            "rep": rep, "run": args.arm, "model": None,
            # Preserve both the new evaluation schema and the legacy
            # `expected` object during migration. The deterministic scorer and
            # semantic judge need these fields in the immutable run record;
            # they must not depend on re-reading a question file that may later
            # change.
            "expected_behavior": item.get("expected_behavior"),
            "tool_policy": item.get("tool_policy") or {},
            "oracle": item.get("oracle") or {},
            "judge": item.get("judge") or {},
            **r,
        }

    async def get_headers():
        """Fresh credentials per rep. Prefers self-login (EVAL_USER/EVAL_PASS)
        so tokens can't expire mid-run; falls back to a static
        EVAL_AUTH_HEADER; empty if neither is set."""
        user = args.user or os.environ.get("EVAL_USER")
        pw = args.password or os.environ.get("EVAL_PASS")
        if user and pw:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{args.auth_url}/login",
                                 json={"username": user, "password": pw}, timeout=30)
                r.raise_for_status()
                return {"Authorization": f"Bearer {r.json()['access_token']}"}
        raw = os.environ.get("EVAL_AUTH_HEADER", "")
        if raw and ":" in raw:
            k, v = raw.split(":", 1)
            return {k.strip(): v.strip()}
        raise SystemExit(
            "No auth configured: pass --user/--password (or EVAL_USER/EVAL_PASS, "
            "or EVAL_AUTH_HEADER). The gateway requires a Bearer token."
        )

    with out_path.open("w") as f:
        for rep in range(args.repeats):
            headers = await get_headers()  # re-login every rep: outlives any TTL
            async with httpx.AsyncClient(headers=headers) as client:
                rows = await asyncio.gather(*(one(client, it, rep) for it in items))
            for row in rows:
                f.write(json.dumps(row) + "\n")
            n_err = sum(1 for r in rows if r["error"])
            print(f"  rep {rep}: done, errors={n_err}")

    if LANGFUSE_ENABLED:
        langfuse.flush()
    print(f"wrote {out_path}")
    print(f"score: python evals/score_answer.py {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

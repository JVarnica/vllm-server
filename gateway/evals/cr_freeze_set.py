"""
Build a frozen search set: questions -> /chat -> whatever the search pipeline returned.

Phase 1 of two. This runs the real endpoint once per question and records the
tool calls and tool results. Phase 2 (answer_from_freeze.py) replays those
results into the model with no network, so answer-quality changes can be
attributed to the model or the prompt rather than to the web moving underneath.

Deliberately not in here, versus answer_e2e_v1.py:
  - no --arm / --repeats: a freeze is one pass. Variance belongs in the answer
    phase, replayed against identical context.
  - no eval-ledger Langfuse tracing: the gateway already traces /chat, and the
    raw page text lives in its passage-selection span (TRACES_DATASET=1).
  - no judge/oracle scoring: this writes inputs, it does not grade.

The live answer IS captured, but only as an incidental baseline - /chat cannot
return search results without also answering. Phase 2 should ignore it.

    TRACES_DATASET=1 python evals/freeze_set.py evals/answer_questions_v1.jsonl \
        --out evals/freeze/freeze_v2.jsonl --base-url http://localhost:8000
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

TEMPERATURE = 0.1


async def login(auth_url, user, password):
    """Fresh token up front. A freeze is a single pass, so one login covers it."""
    if user and password:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{auth_url}/login",
                             json={"username": user, "password": password}, timeout=30)
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


async def create_session(client, base_url):
    r = await client.post(f"{base_url}/session/create", json={}, timeout=30)
    r.raise_for_status()
    body = r.json()
    sid = body.get("session_id") or body.get("id") or body.get("sid")
    if not sid:
        raise RuntimeError(f"/session/create returned no recognizable id: {body}")
    return sid


async def capture(client, base_url, question, tool_policy=None):
    """One question -> one freeze row's worth of search state."""
    session_id = await create_session(client, base_url)
    payload = {"session_id": session_id, "message": question, "temperature": TEMPERATURE}

    turns: dict[int, dict] = {}
    content_parts: list[str] = []
    trace_id = None
    error = None
    event_name = ""
    t0 = time.perf_counter()

    try:
        async with client.stream("POST", f"{base_url}/chat", json=payload, timeout=300) as resp:
            resp.raise_for_status()
            # Only present if the gateway sets it; see note at the bottom of this file.
            trace_id = resp.headers.get("x-trace-id")

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
                    turn = turns.setdefault(obj.get("iter", 0),
                                            {"tool_calls": [], "tool_results": []})
                    tool = obj.get("tool") or ("calculate" if "expression" in obj else "web_search")
                    args = ({"expression": obj.get("expression") or obj.get("query")}
                            if tool == "calculate"
                            else {"query": obj.get("query"),
                                  "time_sensitivity": obj.get("time_sensitivity")})
                    turn["tool_calls"].append({"tool": tool, "args": args})

                elif event_name == "tool_result":
                    turn = turns.setdefault(obj.get("iter", 0),
                                            {"tool_calls": [], "tool_results": []})
                    turn["tool_results"].append({
                        "tool": obj.get("tool") or ("calculate" if "expression" in obj else "web_search"),
                        "query": obj.get("query"),
                        # The query the model asked for is not the query that ran.
                        "effective_query": obj.get("effective_query"),
                        "recency_mode": obj.get("recency_mode"),
                        "results_count": obj.get("results_count"),
                        "content": obj.get("content"),
                        "metadata": obj.get("metadata") or [],
                    })

                elif event_name in ("", "message"):
                    choices = obj.get("choices") or []
                    if choices:
                        delta = (choices[0] or {}).get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    ordered = [turns[k] for k in sorted(turns)]
    results = [tr for t in ordered for tr in t["tool_results"]]

    searches = [tr for tr in results if tr["tool"] == "web_search"]
    policy = (tool_policy or {}).get("search")

    if not error and policy == "required" and not searches:
        # A freeze row with no search results replays as empty context in phase 2
        # and reads as a model failure. Flag it here instead.
        error = "search required but no web_search result captured"
    elif not error and searches and not any(tr.get("content") for tr in searches):
        error = "web_search ran but returned no content"

    return {
        "session_id": session_id,
        "trace_id": trace_id,
        "turns": ordered,
        # Flat, ordered grounding for phase 2: one block per call, labelled with
        # the query so a multi-search answer can be traced back per entity.
        "tool_content": "\n\n".join(
            f"### {tr['query']}\n{tr['content']}" for tr in results if tr.get("content")
        ),
        "n_iterations": len(ordered),
        "n_tool_calls": sum(len(t["tool_calls"]) for t in ordered),
        "n_searches": len(searches),
        # Flattened for the post-freeze QA pass; the detail stays in turns.
        "queries": [
            {"query": tr["query"], "effective_query": tr["effective_query"],
             "recency_mode": tr["recency_mode"], "results_count": tr["results_count"]}
            for tr in searches
        ],
        "live_answer": "".join(content_parts).strip(),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "error": error,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("questions")
    p.add_argument("--out", default="evals/freeze/freeze_set.jsonl")
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="first N items (smoke)")
    p.add_argument("--stratum", default=None, help="freeze only this stratum")
    p.add_argument("--retries", type=int, default=1,
                   help="re-run rows that errored; a failed freeze row is dead weight")
    p.add_argument("--auth-url", default="http://auth:8090")
    p.add_argument("--user", default=os.environ.get("EVAL_USER"))
    p.add_argument("--password", default=os.environ.get("EVAL_PASS"))
    args = p.parse_args()

    items = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    if args.stratum:
        items = [i for i in items if i.get("stratum") == args.stratum]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no items matched")

    if os.environ.get("TRACES_DATASET") != "1":
        print("WARNING: TRACES_DATASET != 1 - no passage-selection spans will be "
              "emitted, so raw page text will not be recoverable for this freeze.")

    print(f"freezing {len(items)} questions -> {args.out}")

    headers = await login(args.auth_url, args.user, args.password)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(client, item):
        async with sem:
            captured = await capture(client, args.base_url, item["question"],
                                     item.get("tool_policy"))
        # Carry the whole question item through, so the freeze row is
        # self-contained and phase 2 never has to re-read a file that may
        # have changed since the freeze.
        return {**item, **captured}

    async with httpx.AsyncClient(headers=headers) as client:
        rows = await asyncio.gather(*(one(client, it) for it in items))

        for attempt in range(args.retries):
            failed = [r for r in rows if r["error"]]
            if not failed:
                break
            print(f"  retry {attempt + 1}: {len(failed)} rows")
            by_id = {r["id"]: r for r in rows}
            redone = await asyncio.gather(*(one(client, by_id[r["id"]]) for r in failed))
            for r in redone:
                if not r["error"]:
                    by_id[r["id"]] = r
            rows = [by_id[r["id"]] for r in rows]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report(rows, out_path)


def report(rows, out_path):
    """
    Post-freeze QA. A freeze you have not inspected is a freeze you cannot trust,
    and every problem below is cheaper to fix now than after phase 2 has run
    against it.
    """
    import collections

    n_err = sum(1 for r in rows if r["error"])
    print(f"\nwrote {out_path}  ({len(rows)} rows, {n_err} errors)")

    by_stratum = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        st = by_stratum[r.get("stratum", "?")]
        st[0] += 1
        st[1] += 0 if r["error"] else 1
        st[2] += r.get("n_searches", 0)

    print(f"\n  {'stratum':<24}{'ok':>6}{'n':>5}{'searches':>10}")
    print("  " + "-" * 45)
    for name, (n, ok, searches) in sorted(by_stratum.items()):
        print(f"  {name:<24}{ok:>6}{n:>5}{searches:>10}")

    queries = [q for r in rows for q in r.get("queries", [])]
    modes = collections.Counter(q["recency_mode"] or "none" for q in queries)
    no_eff = sum(1 for q in queries if not q["effective_query"])
    empty = sum(1 for q in queries if not q["results_count"])
    n_traced = sum(1 for r in rows if r.get("trace_id"))

    print(f"\n  {len(queries)} searches across {len(rows)} questions")
    print(f"  recency_mode: {dict(modes)}")
    print(f"  trace_id captured: {n_traced}/{len(rows)}")

    if no_eff:
        print(f"  WARNING: {no_eff} searches missing effective_query - the gateway "
              f"is not emitting it, so this freeze is not reproducible.")
    if empty:
        print(f"  WARNING: {empty} searches returned zero results.")
    if modes and not modes.keys() - {"general", "none"}:
        # Every search classified 'general' means the model never declared
        # time_sensitivity and the heuristic found nothing either - the whole
        # recency path is inert and the freeze will not exercise it.
        print("  WARNING: no search was classified 'recent' or 'breaking'. "
              "Check that the model is emitting time_sensitivity.")

    for r in rows:
        if r["error"]:
            print(f"    {r['id']} [{r.get('stratum')}] {r['error']}")

    unfilled = sum(1 for r in rows if (r.get("oracle") or {}).get("fill_after_freeze"))
    if unfilled:
        print(f"\n  {unfilled} rows still need oracles filled from the frozen "
              f"context before scoring.")


if __name__ == "__main__":
    asyncio.run(main())

# ---------------------------------------------------------------------------
# Joining a freeze row to its raw page text
#
# The row records what the model SAW (post-selection). The raw extracted text
# lives in the passage-selection span. To join them you need the gateway's
# trace id on the row, which means chat.py setting it on the response:
#
#     response.headers["X-Trace-Id"] = langfuse.get_current_trace_id()
#
# Without it, trace_id is None and the join has to fall back to matching on
# query + timestamp, which is fragile when the same question is frozen twice.
# ---------------------------------------------------------------------------
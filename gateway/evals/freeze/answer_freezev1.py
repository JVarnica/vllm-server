"""
Phase 2 of the freeze pipeline: replay frozen search results into the model and
record the answer. No network, no gateway, no SearXNG.

Phase 1 (cr_freeze_set.py) captured what search returned for each question.
This replays that identical evidence through the production prompt assembly and
calls vLLM directly, so a change in answer quality is attributable to the model,
the prompt, or the formatter - never to the web moving underneath.

What this deliberately does NOT do:
  - No routing probe for web_search. Routing is measured separately
    (routing_eval, 99.0% over 620 calls); re-deciding it here would add variance
    to the one thing this eval isolates. The search tool call is synthesized.
  - No Langfuse tracing, for the same reason phase 1 skipped it: this writes
    answers, and the judge phase grades them. Keep the ledger in one place.
  - No scoring. Join this output to the freeze on `id` in the judge.

The calculator IS executed for real. It is deterministic and offline, so there
is nothing to freeze, and tool_choice is forced so a routing miss can never cost
a row in a formatter comparison.

    docker compose exec gateway python -m evals.answer_from_freeze \
        evals/freeze/freeze_setv1.jsonl \
        --run formatter_sweep_v1 --formatter prod,narrow,wide,metadata \
        --repeats 10 --temperature 0.1 --concurrency 16

Check the budget before spending 400+ calls on it:

    docker compose exec gateway python -m evals.answer_from_freeze \
        evals/freeze/freeze_setv1.jsonl --formatter all --dry-run

Diff one replayed prompt against a live /chat trace to prove the harness is
faithful (do this once, after any edit to chat.py's message assembly):

    docker compose exec gateway python -m evals.answer_from_freeze \
        evals/freeze/freeze_setv1.jsonl --dump-prompt cc-01
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.calculator import CALCULATOR_TOOL, execute_calculator_tool
from app.chat import (
    CALCULATE_GROUNDING_PROMPT,
    SEARCH_GROUNDING_PROMPT,
    TOOL_CONTEXT_MARKER,
    build_context_messages,
    get_system_prompt,
)
from app.chat_helper_funcs import MIN_GEN_TOKENS, compute_max_tokens, count_tokens
from app.context3 import CONTEXT_TOP_N, format_search_context

VLLM_URL = os.environ["VLLM_URL"]
DEFAULT_MODEL = os.environ["VLLM_MODEL"]

PROBE_TEMPERATURE = 0.1
PROBE_MAX_TOKENS = 2000


def pin_system_prompt(prompt: str, frozen_at: str | None) -> str:
    """
    get_system_prompt() stamps today's date into the prompt, so the same freeze
    replayed a month apart is not the same experiment - the model's sense of
    what counts as "current" shifts against fixed evidence.

    This rewrites the date line to the freeze date. It is a patch over the real
    fix, which is `def get_system_prompt(now: datetime | None = None)` in
    chat.py, called by both /chat and this script.
    """
    if not frozen_at:
        return prompt
    when = datetime.strptime(frozen_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return re.sub(r"^Today is .*$",
                  f"Today is {when.strftime('%A %Y-%m-%d')}.",
                  prompt, count=1, flags=re.MULTILINE)


def build_answer_messages(
    session: dict,
    message: str,
    tool_history: list[dict],
    *,
    rag_context: str | None = None,
    did_search: bool = False,
    did_calculate: bool = False,
) -> list[dict]:
    """
    Mirrors the tail of chat.py's stream_agent (the block after the tool loop).

    This is the one piece of prompt logic duplicated between the endpoint and
    the harness, and therefore the one piece that can silently drift. Lift it
    into chat.py and import it from both - then this function disappears and
    --dump-prompt stops being load-bearing.
    """
    _, messages = build_context_messages(session, message, rag_context=rag_context)
    messages.extend(tool_history)

    has_prior_search_context = any(
        m.get("role") == "system" and (m.get("content") or "").startswith(TOOL_CONTEXT_MARKER)
        for m in messages
    )
    # Insert order matters: search goes in at 1, then calculate displaces it to
    # 2. Same as production.
    if did_search or has_prior_search_context:
        messages.insert(1, {"role": "system", "content": SEARCH_GROUNDING_PROMPT})
    if did_calculate:
        messages.insert(1, {"role": "system", "content": CALCULATE_GROUNDING_PROMPT})
    return messages

def synth_search_history(row: dict) -> tuple[list[dict], str]:
    """
    Rebuild the assistant/tool messages that real tool calls produce.

    Production shape, per iteration: ONE assistant message carrying every
    tool_call for that iteration, then ONE role="tool" message per call, keyed
    by call id. A multi-entity question issues two searches and gets two tool
    messages - flattening them into one changes what the template renders.
    """
    entries = row.get("tool_content") or []
    if not entries:
        raise ValueError(f"row {row['id']} has no tool_content to replay")

    calls, tool_msgs, blocks = [], [], []
    for i, entry in enumerate(entries):
        content = entry.get("content")
        if not content:
            raise ValueError(f"row {row['id']}: entry {i} has no content")
        query = entry.get("query")
        call_id = f"call_frozen_{row['id']}_{i}"
        calls.append({"id": call_id, "type": "function",
                    "function": {"name": "web_search",
                                "arguments": json.dumps({"query": query})}})
        tool_msgs.append({"role": "tool", "tool_call_id": call_id, "content": content})
        blocks.append(content)

    history = [{"role": "assistant", "content": "", "tool_calls": calls}, *tool_msgs]
    return history, "\n\n".join(blocks)


           


async def run_calculator_turn(client, model, probe_msgs, tool_history):
    """
    One forced calculator turn, reproducing iteration 2 of the production loop.

    Two details that are easy to get wrong:
      - production probes with probe_msgs + tool_history (the lightweight list),
        not the full message set. Using the full set here would change the
        probe's input distribution.
      - tool_choice is "required", not "auto". Routing is already measured; if
        it misfires here a formatter arm silently loses a row and the arms stop
        being comparable.
    """
    payload = {
        "model": model,
        "messages": probe_msgs + tool_history,
        "tools": [CALCULATOR_TOOL],
        "tool_choice": "required",
        "temperature": PROBE_TEMPERATURE,
        "max_tokens": PROBE_MAX_TOKENS,
        "stream": False,
    }
    resp = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=300)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return [], []

    added = [{"role": "assistant", "content": msg.get("content") or "",
              "tool_calls": tool_calls}]
    calls = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        raw = fn.get("arguments") or {}
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        result = await execute_calculator_tool(args)
        added.append({
            "role": "tool",
            "tool_call_id": tc.get("id"),
            "content": (f"Result: {result['result']}" if result["ok"]
                        else f"Error: {result['error']}"),
        })
        calls.append({"expression": args.get("expression"), **result})
    return added, calls


def assemble(row: dict, pin_date: bool):
    """Everything up to the vLLM call. Pure, so --dry-run can use it too."""
    session = {"pairs": [], "rag_context": [], "tool_context": None}
    tool_history, context = synth_search_history(row)

    probe_msgs, _ = build_context_messages(session, row["question"])
    messages = build_answer_messages(session, row["question"], tool_history,
                                     did_search=True, did_calculate=False)

    if pin_date:
        pinned = pin_system_prompt(get_system_prompt(), row.get("frozen_at"))
        for m in messages + probe_msgs:
            if m.get("role") == "system" and (m.get("content") or "").startswith("You are a helpful"):
                m["content"] = pinned
    return probe_msgs, tool_history, messages, context


async def answer_one(client, row, rep, args):
    model = args.model
    t0 = time.perf_counter()
    calc_calls: list[dict] = []
    error = None

    try:
        probe_msgs, tool_history, messages, context = assemble(row, args.pin_date)

        needs_calc = (row.get("tool_policy") or {}).get("calculator") == "required"
        if needs_calc and not args.no_calculator:
            added, calc_calls = await run_calculator_turn(client, model, probe_msgs, tool_history)
            if added:
                tool_history = tool_history + added
                messages = build_answer_messages(
                    {"pairs": [], "rag_context": [], "tool_context": None},
                    row["question"], tool_history,
                    did_search=True, did_calculate=True,
                )
                if args.pin_date:
                    pinned = pin_system_prompt(get_system_prompt(), row.get("frozen_at"))
                    for m in messages:
                        if m.get("role") == "system" and (m.get("content") or "").startswith("You are a helpful"):
                            m["content"] = pinned

        max_tokens = compute_max_tokens(messages)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if args.seed is not None:
            payload["seed"] = args.seed + rep

        resp = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=600)
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
        msg = choice["message"]
        usage = body.get("usage") or {}

        answer = (msg.get("content") or "").strip()
        answer = re.sub(r"\n{3,}", "\n\n", answer)

        return {
            "id": row["id"],
            "run": args.run,
            "rep": rep,
            "stratum": row.get("stratum"),
            "question": row.get("question"),
            "tool_policy": row.get("tool_policy"),
            "oracle": row.get("oracle"),
            "frozen_at": row.get("frozen_at"),
            "answer": answer,
            # Kept separate so the judge grades the answer, not the scratchpad.
            "reasoning": (msg.get("reasoning_content") or "").strip() or None,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "max_tokens_allowed": max_tokens,
            "context_chars": len(context),
            "n_results_shown": len(row.get("tool_content") or []),
            "calculator_calls": calc_calls or None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": None,
            "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        return {
            "id": row["id"], "run": args.run, "rep": rep,
            "stratum": row.get("stratum"), "answer": "", "reasoning": None,
            "finish_reason": None, "prompt_tokens": None, "completion_tokens": None,
            "max_tokens_allowed": None, "context_chars": None,
            "n_results_shown": None, "calculator_calls": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": error,
            "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def dry_run(rows, pin_date):
    """Token budget check. compute_max_tokens squeezes generation against an
    8192 window, and a formatter that pushes the prompt up truncates the answer
    rather than failing - which reads to the judge as an incomplete answer."""
    ctx, toks, gens = [], [], []
 
    for row in rows:
        _, _, messages, context = assemble(row, pin_date)
        ctx.append(len(context))
        toks.append(count_tokens(messages))
        gens.append(compute_max_tokens(messages))

    tight = sum(1 for g in gens if g <= MIN_GEN_TOKENS)
    print(f"{int(statistics.median(ctx)):>9}{max(ctx):>9}"
              f"{int(statistics.median(toks)):>9}{max(toks):>9}{min(gens):>9}{tight:>7}")
    print("\n'tight' = rows where the prompt left only the MIN_GEN_TOKENS floor. "
          "Any non-zero value means that arm is being truncated, not outperformed.")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("freeze", help="freeze jsonl from cr_freeze_set.py")
    p.add_argument("--out", default=None, help="default: evals/runs/<run>.jsonl")
    p.add_argument("--run", default="answer_v1", help="run label written to every row")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=None,
                   help="base seed; rep index is added, so reps stay distinct "
                        "but the sweep is reproducible")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stratum", default=None)
    p.add_argument("--id", default=None, help="single row, for debugging")
    p.add_argument("--no-calculator", action="store_true",
                   help="skip the forced calculator turn; calc rows answer from "
                        "search evidence alone")
    p.add_argument("--no-pin-date", dest="pin_date", action="store_false",
                   help="use today's date in the system prompt instead of the "
                        "freeze date (not reproducible)")
    p.add_argument("--dry-run", action="store_true",
                   help="assemble prompts and report token budget, no vLLM calls")
    p.add_argument("--dump-prompt", default=None, metavar="ID",
                   help="print the assembled messages for one row and exit")
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.freeze).read_text().splitlines() if l.strip()]
    if args.stratum:
        rows = [r for r in rows if r.get("stratum") == args.stratum]
    if args.id:
        rows = [r for r in rows if r["id"] == args.id]
    if args.limit:
        rows = rows[: args.limit]

    skipped = [r["id"] for r in rows if r.get("error")]
    rows = [r for r in rows if not r.get("error")]
    if skipped:
        print(f"skipping {len(skipped)} freeze rows that errored: {skipped}")
    if not rows:
        raise SystemExit("no usable freeze rows matched")

    if args.dump_prompt:
        row = next((r for r in rows if r["id"] == args.dump_prompt), None)
        if not row:
            raise SystemExit(f"{args.dump_prompt} not in freeze")
        _, _, messages, _ = assemble(row, args.pin_date)
        for m in messages:
            print(f"--- {m['role']} " + ("(tool_calls)" if m.get("tool_calls") else ""))
            print(m.get("content") or json.dumps(m.get("tool_calls"), indent=2))
        print(f"\nprompt_tokens={count_tokens(messages)} "
              f"max_tokens={compute_max_tokens(messages)}")
        return

    if args.dry_run:
        dry_run(rows, args.pin_date)
        return

    jobs = [(r, rep) for r in rows for rep in range(args.repeats)]
    print(f"{len(rows)} rows x {args.repeats} reps "
          f"= {len(jobs)} calls -> {args.model}")

    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(client, row, rep):
        nonlocal done
        async with sem:
            out = await answer_one(client, row, rep, args)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(jobs)}")
        return out

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(one(client, r, rep) for r, rep in jobs))

    out_path = Path(args.out or f"evals/runs/{args.run}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    report(results, out_path, args)


def report(results, out_path, args):
    """
    No grading here - the judge does that. This is the pre-judge sanity pass:
    every problem below invalidates a formatter comparison, and all of them are
    cheaper to catch now than after 400 rows have been scored.
    """
    import collections

    print(f"\nwrote {out_path}  ({len(results)} rows)")

    ok = [r for r in results if not r["error"]]
    errs = len(results) - len(ok)
    trunc = sum(1 for r in ok if r["finish_reason"] == "length")
    empty = sum(1 for r in ok if not r["answer"])
    toks = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]] or [0]
    alen = [len(r["answer"]) for r in ok] or [0]
    lat = [r["latency_ms"] for r in ok] or [0]
    print(f" \n  {'n':>5}{errs:>5}{trunc:>7}{empty:>7}"
            f"{int(statistics.median(toks)):>9}{int(statistics.median(alen)):>9}"
            f"{int(statistics.median(lat)):>9}")

    # A row whose context barely changes between arms cannot discriminate
    # between them. In freeze_setv1 these are the searches that came back with
    # one result - they contribute noise and no signal to a formatter sweep.
    thin = sorted({r["id"] for r in results
                   if r["context_chars"] is not None and r["context_chars"] < 1000})
    if thin:
        print(f"\n  {len(thin)} rows produced under 1000 chars of context: {thin}")
        print("  These render near-identically under every variant. Re-freeze "
              "them or drop them before reading a formatter delta.")

    trunc_ids = sorted({r["id"] for r in results if r["finish_reason"] == "length"})
    if trunc_ids:
        print(f"\n  WARNING: {len(trunc_ids)} ids hit the token ceiling: {trunc_ids}")
        print("  A truncated answer scores as incomplete. Fix the budget, not the prompt.")

    bad_calc = [r for r in results
                for c in (r["calculator_calls"] or [])
                if not c.get("ok")]
    if bad_calc:
        print(f"\n  WARNING: {len(bad_calc)} calculator calls failed:")
        for r in bad_calc[:5]:
            for c in r["calculator_calls"]:
                if not c.get("ok"):
                    print(f"    {r['id']} rep{r['rep']}: {c.get('expression')!r} "
                          f"-> {c.get('error')}")

    errs = [r for r in results if r["error"]]
    if errs:
        print(f"\n  {len(errs)} errored rows:")
        for r in errs[:10]:
            print(f"    {r['id']} rep{r['rep']}: {r['error']}")

    if args.repeats > 1:
        # Answer-length spread is a crude proxy, but a formatter whose answers
        # vary wildly across reps at temperature 0.1 is unstable, and that
        # instability will swamp any judge delta between arms.
     
        spreads = []
        per_id = collections.defaultdict(list)
        for r in ok:
            if not r["error"]:
                per_id[r["id"]].append(len(r["answer"]))
        for lens in per_id.values():
            if len(lens) > 1:
                spreads.append(statistics.pstdev(lens))
        mean_len = statistics.mean([l for lens in per_id.values() for l in lens] or [0])
        mean_spread = statistics.mean(spreads) if spreads else 0
        print(
            f"\n  {'mean answer len':>16}"
            f"{'mean sd across reps':>22}"
        )
        print("  " + "-" * 38)

        print(
            f"  {int(mean_len):>16}"
            f"{int(mean_spread):>22}"
        )


if __name__ == "__main__":
    asyncio.run(main()) 
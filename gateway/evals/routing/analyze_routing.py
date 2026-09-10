"""
Analysis for routing_eval runs. Pure post-processing: no model calls.

    python -m evals.analyze_routing evals/runs/baseline.jsonl
    python -m evals.analyze_routing evals/runs/baseline.jsonl evals/runs/no_toolcall_line.jsonl

With one file: per-stratum accuracy, over/under-trigger split, calculator
argument accuracy, latency percentiles.

With two files: all of the above per arm, plus exact McNemar on paired
(id, rep) outcomes — the test your plan calls for. Pairing on rep as well as
id is only meaningful at temperature 0; at temperature > 0 reps are not
paired samples, and the script warns rather than pretending they are.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import quantiles


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        sys.exit(f"{path}: empty run file")
    return rows


def pct(part, whole):
    return f"{part / whole:.1%}" if whole else "  n/a"


def latency_summary(rows):
    xs = sorted(r["latency_ms"] for r in rows)
    if len(xs) < 2:
        return f"n={len(xs)}"
    q = quantiles(xs, n=100)
    return f"p50 {q[49]:.0f}ms  p90 {q[89]:.0f}ms  p99 {q[98]:.0f}ms"


def report_arm(rows):
    arm = rows[0].get("arm", "?")
    n = len(rows)
    print(f"\n=== {arm}  ({n} calls, {len({r['id'] for r in rows})} items, "
          f"{len({r['rep'] for r in rows})} reps) ===")
    print(f"overall accuracy: {pct(sum(r['correct'] for r in rows), n)}"
          f"    latency: {latency_summary(rows)}")

    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[r["stratum"]].append(r)

    print(f"\n{'stratum':<22}{'acc':>7}{'over_s':>8}{'under_s':>9}"
          f"{'over_c':>8}{'under_c':>9}{'n':>6}")
    for stratum in sorted(by_stratum):
        rs = by_stratum[stratum]
        print(f"{stratum:<22}"
              f"{pct(sum(r['correct'] for r in rs), len(rs)):>7}"
              f"{sum(r['over_search'] for r in rs):>8}"
              f"{sum(r['under_search'] for r in rs):>9}"
              f"{sum(r['over_calc'] for r in rs):>8}"
              f"{sum(r['under_calc'] for r in rs):>9}"
              f"{len(rs):>6}")

    # Items that ever failed, worst first — the re-slicing you actually do.
    fails = defaultdict(list)
    for r in rows:
        if not r["correct"]:
            fails[r["id"]].append(r)
    if fails:
        print("\nfailing items (fails/reps):")
        reps = len({r["rep"] for r in rows})
        for item_id, frs in sorted(fails.items(), key=lambda kv: -len(kv[1])):
            print(f"  {item_id:<12} {len(frs)}/{reps}")
            snippet = (frs[0].get("reasoning_content") or "").replace("\n", " ")[:200]
            if snippet:
                print(f"      └ {snippet}")

    # Calculator argument accuracy — separate axis from routing, on purpose.
    checks = [c for r in rows for c in r.get("calc_checks", [])]
    if checks:
        ok = sum(c["ok"] for c in checks)
        matched = [c for c in checks if "matches" in c]
        hit = sum(c["matches"] for c in matched)
        print(f"\ncalculator arguments: {ok}/{len(checks)} evaluable"
              + (f", {hit}/{len(matched)} match expected_value" if matched else ""))
        bad = [c for c in checks if not c["ok"] or c.get("matches") is False]
        for c in bad[:10]:
            detail = c.get("error") or f"got {c.get('value')} want {c.get('expected')}"
            print(f"  BAD  {c['expr']!r}: {detail}")


def mcnemar(rows_a, rows_b):
    key = lambda r: (r["id"], r["rep"])
    a = {key(r): r["correct"] for r in rows_a}
    b = {key(r): r["correct"] for r in rows_b}
    common = a.keys() & b.keys()
    if not common:
        sys.exit("no overlapping (id, rep) pairs between arms — different "
                 "question sets or repeat counts?")
    only_a, only_b = len(a) - len(common), len(b) - len(common)
    if only_a or only_b:
        print(f"\nwarning: unpaired calls dropped (arm A: {only_a}, arm B: {only_b})")

    b01 = sum(1 for k in common if a[k] and not b[k])  # A right, B wrong
    b10 = sum(1 for k in common if b[k] and not a[k])  # B right, A wrong
    n = b01 + b10

    name_a, name_b = rows_a[0].get("arm", "A"), rows_b[0].get("arm", "B")
    print(f"\n=== McNemar: {name_a} vs {name_b} ({len(common)} paired calls) ===")
    print(f"{name_a} right / {name_b} wrong: {b01}")
    print(f"{name_b} right / {name_a} wrong: {b10}")
    if n == 0:
        print("no discordant pairs — arms identical on this set, p = 1.0")
        return
    # exact two-sided binomial test on the discordant pairs
    tail = sum(math.comb(n, i) for i in range(0, min(b01, b10) + 1)) / 2**n
    p = min(1.0, 2 * tail)
    print(f"exact two-sided p = {p:.4f}"
          + ("   (temperature > 0 in either run makes rep-pairing dubious)"
             if any(r.get("temperature") for r in (rows_a[0], rows_b[0])) else ""))


def main():
    paths = sys.argv[1:]
    if not paths or len(paths) > 2:
        sys.exit(__doc__)
    runs = [load(p) for p in paths]
    for rows in runs:
        report_arm(rows)
    if len(runs) == 2:
        mcnemar(*runs)


if __name__ == "__main__":
    main()
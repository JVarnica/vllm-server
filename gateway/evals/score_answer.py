"""Deterministic scorer for the gateway answer eval.

Runs BEFORE any LLM judge. Stdlib only, offline, rescoreable.
Reads run jsonl rows (schema from the handoff doc), writes scored jsonl,
prints a per-stratum table, and emits a needs_judge list for the rows
the programmatic checks cannot decide.

Checks implemented (design decision 3):
  1. exact-answer containment   (expected_value / expected_values_alt, tolerance)
  2. substring containment      (expected_substrings, case-insensitive)
  3. entity coverage in ANSWER  (entities — multi_entity carryover)
  4. trajectory shape           (chained: search-before-calc, extraction link,
                                 terminates in budget, no repeated identical calls)
  5. graceful failure           (junk-fixture items: admit failure, no fabricated numbers)
  6. output hygiene             (no <think>/<tool_call>/raw tool JSON in final text)
Plus per-item aggregation: pass rate over reps + answer stability.

Answer-key convention (in the questions file, copied into each row's `expected`):
  expected_value        float|int   primary numeric answer (post format_result)
  expected_values_alt   [num]       alternate representations (e.g. 15 for 0.15)
  tolerance             float       optional override; default max(0.01, 0.1%)
  expected_substrings   [str]       ALL must appear (case-insensitive)
  entities              [str]       ALL must appear in answer
  trajectory            {...}       presence => chained checks run
      requires_search_before_calc  bool
  graceful              bool        presence/true => graceful-failure checks run
  needs_judge           bool        open/freshness item: hygiene-only here,
                                    groundedness left to the judge

Selftest: python score_answer.py --selftest
Score:    python score_answer.py run.jsonl [-o scored.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------- number utils

MAX_DECIMAL_PLACES = 5  # keep in sync with app/calculator.py

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# numbers that appear constantly in prose/expressions and prove nothing
_TRIVIAL_NUMS = {0.0, 1.0, 2.0, 100.0, 1000.0}


def format_result(value):
    """Mirror of calculator.format_result so containment matches tool output."""
    if not isinstance(value, float):
        return value
    if math.isnan(value) or math.isinf(value):
        return value
    rounded = round(value, MAX_DECIMAL_PLACES)
    if rounded == 0 and value != 0:
        rounded = float(f"{value:.{MAX_DECIMAL_PLACES}g}")
    if rounded.is_integer():
        return int(rounded)
    return rounded


def extract_numbers(text: str) -> list[float]:
    """All numeric tokens in text, commas stripped. '1,234.5' -> 1234.5."""
    out = []
    for tok in _NUM_RE.findall(text or ""):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def extract_numbers_excluding_identifiers(text: str) -> list[float]:
    """Like extract_numbers, but drops tokens glued to a letter or wrapped in
    [] brackets — i.e. citation markers ([1][2]) and identifiers (3ITVP, G1)
    that aren't claims about anything. Post-hoc span check against the
    ORIGINAL text, not a lookahead inside the pattern: a lookahead invites
    the engine to backtrack to a shorter match that clears it (e.g. "340g"
    would backtrack to "34" and still pass), which corrupts real numbers
    instead of dropping them. Checking the actual neighboring character
    after a normal greedy match avoids that entirely.
    Used only by check_graceful's fabrication heuristic — extract_numbers()
    itself stays permissive because this model glues numbers to units
    constantly ("595g", "340g") and that must keep matching everywhere else.
    """
    text = text or ""
    out = []
    for m in _NUM_RE.finditer(text):
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        if before.isalpha() or before == "[" or after.isalpha() or after == "]":
            continue
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            continue
    return out


def default_tol(expected: float) -> float:
    return max(0.01, abs(expected) * 0.001)


def num_matches(candidate: float, expected: float, tol: float) -> bool:
    return abs(candidate - expected) <= tol


# ---------------------------------------------------------------- row helpers

def iter_tool_calls(row):
    """Yield (iteration_index, tool_name, args_dict) across all turns."""
    for i, turn in enumerate(row.get("turns") or []):
        for tc in turn.get("tool_calls") or []:
            name = tc.get("tool") or tc.get("name") or ""
            args = tc.get("args") or tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            yield i, name, args


def all_search_content(row) -> str:
    """Concatenated content of every web_search tool_result (fixture text)."""
    parts = []
    for turn in row.get("turns") or []:
        for tr in turn.get("tool_results") or []:
            prov = tr.get("provenance") or tr
            if (prov.get("tool") or "") == "web_search":
                parts.append(str(prov.get("content") or tr.get("content") or ""))
    return "\n".join(parts)


# ---------------------------------------------------------------- checks
# Each check returns (passed: bool|None, detail: str). None = not applicable.

def check_exact_answer(row, exp):
    if "expected_value" not in exp:
        return None, ""
    targets = [exp["expected_value"]] + list(exp.get("expected_values_alt") or [])
    targets = [float(format_result(float(t))) for t in targets]
    tol = exp.get("tolerance")
    answer_nums = extract_numbers(row.get("answer") or "")
    for t in targets:
        t_tol = tol if tol is not None else default_tol(t)
        for n in answer_nums:
            if num_matches(n, t, t_tol):
                row["_matched_value"] = n  # the token that IS the answer
                return True, f"matched {n} ~ {t}"
    return False, f"none of {targets} in answer numbers {answer_nums[:8]}"


def _missing_groups(groups, ans_lower):
    """Each element is a str or a list of alias strings; any alias satisfies
    the group. Returns the first alias of every unsatisfied group."""
    missing = []
    for g in groups:
        aliases = [g] if isinstance(g, str) else list(g)
        if not any(a.lower() in ans_lower for a in aliases):
            missing.append(aliases[0])
    return missing


def check_substrings(row, exp):
    subs = exp.get("expected_substrings")
    if not subs:
        return None, ""
    missing = _missing_groups(subs, (row.get("answer") or "").lower())
    return (not missing), (f"missing {missing}" if missing else "all present")


def check_entities(row, exp):
    ents = exp.get("entities")
    if not ents:
        return None, ""
    missing = _missing_groups(ents, (row.get("answer") or "").lower())
    return (not missing), (f"missing {missing}" if missing else "all covered")


def check_trajectory(row, exp):
    if "trajectory" not in exp:
        return None, ""
    traj = exp.get("trajectory") or {}
    problems = []
    calls = list(iter_tool_calls(row))

    # terminates within budget: error field is the authority for loop blowout
    if row.get("error"):
        problems.append(f"error: {row['error']}")

    # no repeated identical calls
    seen = set()
    for _, name, args in calls:
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            problems.append(f"repeated call {name} {key[1][:80]}")
            break
        seen.add(key)

    first_search = next((i for i, n, _ in calls if n == "web_search"), None)
    first_calc = next((i for i, n, _ in calls if n == "calculate"), None)

    if traj.get("requires_search_before_calc"):
        if first_calc is None:
            problems.append("never called calculate")
        elif first_search is None:
            problems.append("never searched")
        elif first_calc < first_search:
            problems.append("calculated before searching")

    # extraction link: >=1 non-trivial number in a calc expression must
    # appear in prior search content (proves the value came from retrieval).
    # Only checkable when tool_result content was captured — endpoint runs
    # without content capture skip it rather than false-failing.
    if first_calc is not None and first_search is not None:
        content_nums = set(extract_numbers(all_search_content(row)))
        linked = False
        expr_nums = []
        for _, name, args in calls:
            if name != "calculate":
                continue
            nums = [n for n in extract_numbers(str(args.get("expression", "")))
                    if n not in _TRIVIAL_NUMS]
            expr_nums.extend(nums)
            if any(any(num_matches(cn, en, default_tol(en)) for cn in content_nums)
                   for en in nums):
                linked = True
        if expr_nums and content_nums and not linked:
            problems.append(f"no expression number {expr_nums[:5]} found in search results")

    return (not problems), "; ".join(problems) or "trajectory ok"


_ADMIT_RE = re.compile(
    r"(couldn'?t|could not|unable to|wasn'?t able|not able to|"
    r"didn'?t (?:find|return)|failed to (?:find|retrieve)|"
    r"don'?t have (?:enough|reliable|sufficient)|"
    r"cannot (?:confirm|verify|find|determine)|can'?t (?:confirm|verify|find|determine)|"
    r"cannot be (?:determined|found|verified|confirmed)|"
    r"could not be (?:determined|found|verified|confirmed)|"
    r"no(?:\s+\w+){0,4}\s+(?:information|data|results)\b|"          # bounded-gap: "no officially verified GDP data"
    r"(?:does|do) not (?:contain|include|mention|provide)(?:\s+\w+){0,3}\s+"
    r"(?:information|data|details)|"                                # "do not contain any information about"
    r"(?:does|do) not exist|"
    r"not a recognized|"
    r"is not available|"
    r"no valid answer (?:can|could) be (?:derived|given|provided))",
    re.IGNORECASE,
)


_RETRIEVAL_META_RE = re.compile(
    r"\b(?:top|first)\s+\d+\s+(?:search\s+)?(?:results?|sources?|links?)\b",
    re.IGNORECASE,
)


def check_graceful(row, exp):
    if not exp.get("graceful"):
        return None, ""
    ans = row.get("answer") or ""
    problems = []
    if not _ADMIT_RE.search(ans):
        problems.append("no admission of failure")
    # fabrication proxy: any number in the answer not present in the question
    q_nums = set(extract_numbers(row.get("question") or ""))
    ans_for_fabrication = _RETRIEVAL_META_RE.sub("", ans)  # "top 5 results" isn't a claim
    fabricated = [n for n in extract_numbers_excluding_identifiers(ans_for_fabrication)
                  if n not in _TRIVIAL_NUMS
                  and not any(num_matches(n, q, default_tol(q)) for q in q_nums)]
    if fabricated:
        problems.append(f"fabricated numbers {fabricated[:5]}")
    return (not problems), "; ".join(problems) or "admitted failure cleanly"


_HYGIENE_PATTERNS = [
    ("<think>", "think tag leaked"),
    ("</think>", "think close tag leaked"),
    ("<tool_call>", "tool_call tag leaked"),
    ('"arguments":', "raw tool-call JSON leaked"),
    ('{"name":', "raw tool-call JSON leaked"),
    ("tool_result", "tool_result marker leaked"),
]


def check_hygiene(row, exp):
    ans = row.get("answer") or ""
    hits = [msg for pat, msg in _HYGIENE_PATTERNS if pat in ans]
    return (not hits), "; ".join(sorted(set(hits))) or "clean"


CHECKS = [
    ("exact_answer", check_exact_answer),
    ("substrings", check_substrings),
    ("entities", check_entities),
    ("trajectory", check_trajectory),
    ("graceful", check_graceful),
    ("hygiene", check_hygiene),
]


def score_row(row) -> dict:
    exp = row.get("expected") or {}
    results, failures = {}, []
    applicable = 0
    for name, fn in CHECKS:
        passed, detail = fn(row, exp)
        results[name] = {"passed": passed, "detail": detail}
        if passed is not None:
            applicable += 1
            if not passed:
                failures.append(f"{name}: {detail}")
    # hard failure: transport/loop error always fails the row
    if row.get("error") and "trajectory" not in [f.split(":")[0] for f in failures]:
        failures.append(f"error: {row['error']}")

    needs_judge = bool(exp.get("needs_judge"))
    # judge-bound rows: programmatic verdict covers hygiene(+entities) only;
    # 'passed' means "nothing deterministic failed", not "correct".
    passed = not failures
    return {
        "checks": results,
        "prog_passed": passed,
        "prog_failures": failures,
        "n_checks_applicable": applicable,
        "needs_judge": needs_judge,
        "final_verdict": None if needs_judge and passed else passed,
    }


# ---------------------------------------------------------------- aggregation

def aggregate(rows):
    if not rows:
        print("no rows to score — run file is empty")
        return
    by_stratum = defaultdict(lambda: {"pass": 0, "total": 0})
    by_item = defaultdict(list)
    for r in rows:
        s = r.get("stratum", "?")
        by_stratum[s]["total"] += 1
        if r["_score"]["prog_passed"]:
            by_stratum[s]["pass"] += 1
        by_item[r.get("id", "?")].append(r)

    print(f"\n{'stratum':<24}{'acc':>8}{'n':>6}")
    print("-" * 38)
    tp = tt = 0
    for s in sorted(by_stratum):
        d = by_stratum[s]
        tp += d["pass"]; tt += d["total"]
        print(f"{s:<24}{d['pass'] / d['total']:>8.1%}{d['total']:>6}")
    print("-" * 38)
    print(f"{'TOTAL (programmatic)':<24}{tp / tt:>8.1%}{tt:>6}")

    # per-item pass rate + stability
    print(f"\n{'-- flaky / failing items --'}")
    any_flag = False
    for item_id, item_rows in sorted(by_item.items()):
        n = len(item_rows)
        p = sum(1 for r in item_rows if r["_score"]["prog_passed"])
        exp = item_rows[0].get("expected") or {}
        stability = ""
        if "expected_value" in exp and n > 1:
            target = float(format_result(float(exp["expected_value"])))
            tol = exp.get("tolerance")
            tol = tol if tol is not None else default_tol(target)
            primaries = []
            for r in item_rows:
                if "_matched_value" in r:
                    primaries.append(r["_matched_value"])
                else:  # failing rep: nearest number to expected, if any
                    nums = extract_numbers(r.get("answer") or "")
                    primaries.append(min(nums, key=lambda x: abs(x - target))
                                     if nums else None)
            present = [p for p in primaries if p is not None]
            # cluster values within tolerance of each other (transitive chain
            # over sorted values) — "stable" means every rep landed in the
            # same accepted-value neighborhood, not that they printed
            # byte-identical strings
            clusters = 0
            if present:
                clusters = 1
                for a, b in zip(sorted(present), sorted(present)[1:]):
                    if abs(b - a) > tol:
                        clusters += 1
            if clusters > 1:
                stability = f"  UNSTABLE answers={primaries}"
        if p < n or stability:
            any_flag = True
            fail_detail = next((r["_score"]["prog_failures"]
                                for r in item_rows if r["_score"]["prog_failures"]), [])
            print(f"  {item_id} [{item_rows[0].get('stratum','?')}] "
                  f"{p}/{n}{stability}")
            for f in fail_detail[:3]:
                print(f"      {f}")
    if not any_flag:
        print("  none")

    judge_rows = [r for r in rows if r["_score"]["needs_judge"]]
    print(f"\nneeds_judge: {len(judge_rows)} rows "
          f"({len({r.get('id') for r in judge_rows})} items) — groundedness judge only")


# ---------------------------------------------------------------- selftest
# Lesson from routing eval: debug the eval before believing it. Every check
# gets a forced PASS and a forced FAIL.

def _mkrow(answer, expected, turns=None, question="q?", error=None):
    return {"id": "t", "stratum": "test", "question": question, "rep": 0,
            "answer": answer, "expected": expected, "turns": turns or [],
            "error": error}


def selftest():
    cases = [
        # exact answer
        ("exact pass", _mkrow("The total is 1,234.57 USD.", {"expected_value": 1234.567}), True),
        ("exact fail", _mkrow("The total is 999.", {"expected_value": 1234.567}), False),
        ("exact alt-form pass", _mkrow("That's 15% overall.",
                                       {"expected_value": 0.15, "expected_values_alt": [15]}), True),
        # substrings / entities
        ("substr fail", _mkrow("Paris is the capital.", {"expected_substrings": ["Lyon"]}), False),
        ("entities pass", _mkrow("Arsenal beat Chelsea 2-1.", {"entities": ["Arsenal", "Chelsea"]}), True),
        ("entities alias pass", _mkrow("The Galaxy S26 outsold it.",
                                       {"entities": [["Samsung S26", "Galaxy S26"]]}), True),
        ("entities alias fail", _mkrow("The Pixel 11 outsold it.",
                                       {"entities": [["Samsung S26", "Galaxy S26"]]}), False),
        ("exact pass with leading year", _mkrow("In 2026, the result is £595.",
                                                {"expected_value": 595}), True),        # trajectory
        ("chain pass", _mkrow(
            "GDP grew by 512.5.",
            {"expected_value": 512.5, "trajectory": {"requires_search_before_calc": True}},
            turns=[
                {"tool_calls": [{"tool": "web_search", "args": {"query": "gdp"}}],
                 "tool_results": [{"provenance": {"tool": "web_search",
                                                  "content": "GDP was 2050 billion in 2024"}}]},
                {"tool_calls": [{"tool": "calculate", "args": {"expression": "2050 * 0.25"}}],
                 "tool_results": [{"provenance": {"tool": "calculate", "content": "512.5"}}]},
            ]), True),
        ("chain fail: calc first", _mkrow(
            "x", {"trajectory": {"requires_search_before_calc": True}},
            turns=[
                {"tool_calls": [{"tool": "calculate", "args": {"expression": "3*4"}}],
                 "tool_results": []},
                {"tool_calls": [{"tool": "web_search", "args": {"query": "gdp"}}],
                 "tool_results": [{"provenance": {"tool": "web_search", "content": "12"}}]},
            ]), False),
        ("chain fail: number not from fixture", _mkrow(
            "x", {"trajectory": {"requires_search_before_calc": True}},
            turns=[
                {"tool_calls": [{"tool": "web_search", "args": {"query": "gdp"}}],
                 "tool_results": [{"provenance": {"tool": "web_search",
                                                  "content": "revenue reached 12000 last year"}}]},
                {"tool_calls": [{"tool": "calculate", "args": {"expression": "777 * 3"}}],
                 "tool_results": []},
            ]), False),
        ("chain fail: repeated identical call", _mkrow(
            "x", {"trajectory": {}},
            turns=[
                {"tool_calls": [{"tool": "web_search", "args": {"query": "same"}}], "tool_results": []},
                {"tool_calls": [{"tool": "web_search", "args": {"query": "same"}}], "tool_results": []},
            ]), False),
        # graceful failure
        ("graceful pass", _mkrow("I couldn't find reliable information on this.",
                                 {"graceful": True}), True),
        ("graceful fail: hallucinated", _mkrow("The population is 4,210,000.",
                                               {"graceful": True}), False),
        ("graceful ok: repeats question number", _mkrow(
            "I couldn't verify the 2024 figure.", {"graceful": True},
            question="What was the 2024 figure?"), True),
        # real phrasing from the reference run (gf-01/02/04) that the old
        # regex missed — regression cases, not synthetic
        ("graceful pass: does-not-exist phrasing", _mkrow(
            "The Republic of Vantoria is not a recognized sovereign state in "
            "the real world and does not exist in official geopolitical or "
            "economic records.", {"graceful": True}), True),
        ("graceful pass: no-valid-answer phrasing", _mkrow(
            "The provided web search results do not contain any information "
            "about the prize. No valid answer can be derived from the given "
            "data.", {"graceful": True}), True),
        ("graceful pass: cannot-be-determined phrasing", _mkrow(
            "The current stock price of Quorvex Industries cannot be "
            "determined from the provided search results.",
            {"graceful": True}), True),
        ("graceful ok: citation numbers not fabrication", _mkrow(
            "No information found. Results reference an unrelated repo "
            "3ITVP and the G1 phone [1][2][3][4].", {"graceful": True}), True),
        ("graceful ok: retrieval-count not fabrication", _mkrow(
            "None of the top 5 search results explicitly mention this "
            "company. Data is not available.", {"graceful": True}), True),
        ("exact pass: unit glued to number (no space)", _mkrow(
            "85g/person x 7 people = 595g. You will need 595g of flour.",
            {"expected_value": 595}), True),
        # hygiene
        ("hygiene fail: think leak", _mkrow("<think>hmm</think>The answer is 4.",
                                            {"expected_value": 4}), False),
        # error always fails
        ("error fails row", _mkrow("The answer is 4.", {"expected_value": 4},
                                   error="tool_iter exceeded"), False),
    ]
    bad = 0
    for name, row, want in cases:
        got = score_row(row)["prog_passed"]
        ok = got == want
        bad += (not ok)
        print(f"  [{'ok' if ok else 'BROKEN'}] {name}: expected {want}, got {got}"
              + ("" if ok else f"  -> {score_row(row)['prog_failures']}"))
    print(f"\nselftest: {len(cases) - bad}/{len(cases)} passing")

    # stability clustering (lives in aggregate(), not score_row())
    def cluster_count(values, target, tol):
        present = [v for v in values if v is not None]
        if not present:
            return 0
        c = 1
        for a, b in zip(sorted(present), sorted(present)[1:]):
            if abs(b - a) > tol:
                c += 1
        return c

    stab_cases = [
        ("same value, different precision -> 1 cluster",
         [1552.87, 1552.87, 1552.87109375], 1552.871, 1.553, 1),
        ("wide-tolerance rounding -> 1 cluster",
         [299792.0, 299792.458, 300000.0], 299792.0, 500, 1),
        ("genuine flip -> 2+ clusters",
         [13170.62, 2012.0, 13170.62, 2119.69, 13170.62], 13170.7, 60, 2),
    ]
    stab_bad = 0
    for name, vals, target, tol, want in stab_cases:
        got = cluster_count(vals, target, tol)
        ok = (got == want) if want == 1 else (got >= want)
        stab_bad += (not ok)
        print(f"  [{'ok' if ok else 'BROKEN'}] {name}: got {got} cluster(s)")
    print(f"stability clustering: {len(stab_cases) - stab_bad}/{len(stab_cases)} passing")

    return bad == 0 and stab_bad == 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_file", nargs="?", help="run jsonl from the GPU phase")
    ap.add_argument("-o", "--out", help="scored jsonl (default: <run>_scored.jsonl)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.run_file:
        ap.error("run_file required (or --selftest)")

    rows = []
    with open(args.run_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    for r in rows:
        r["_score"] = score_row(r)

    out = args.out or args.run_file.rsplit(".jsonl", 1)[0] + "_scored.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    aggregate(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
"""Deterministic pre-judge scorer for end-to-end answer evaluations.

This scorer does NOT decide semantic answer quality. It:
  * rejects runs with undeniable technical/output failures (hard gates),
  * checks explicit tool policy,
  * records cheap answer diagnostics,
  * emits a compact judge_input for every run that survives the gates.

Expected question schema (copied into each run row):
{
  "expected_behavior": "answer|refuse|clarify|social_response",
  "tool_policy": {
    "search": "required|optional|forbidden|not_required",
    "calculator": "required|optional|forbidden|not_required",
    "ordered_calls": ["web_search", "calculate"],
    "calculation_must_use_retrieved_value": true
  },
  "oracle": {
    "reference_answer": "..." | null,
    "reference_mode": "static|retrieved_evidence",
    "required_facts": ["..."],
    "required_entities": [["alias one", "alias two"]],
    "forbidden_claims": ["..."],
    "numeric_answers": [{"value": 53.65, "tolerance": 0.02, "unit": "GBP/person"}],
    "numeric_answer_alternatives": [{"value": 0.5}, {"value": 50, "unit": "%"}],
    "calculation_dependency": {"source_value": 53.65},
    "freshness_required": true,
    "refusal_reason": "...",
    "required_properties": ["..."]
  },
  "judge": {"enabled": true, "dimensions": ["correctness", "groundedness"]}
}


Usage:
    python score_answer_v2.py run.jsonl [-o scored.jsonl]
    python score_answer_v2.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict

MAX_DECIMAL_PLACES = 5
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_TRIVIAL_NUMS = {0.0, 1.0, 2.0, 100.0, 1000.0}
_VALID_POLICIES = {"required", "optional", "forbidden", "not_required"}

_HYGIENE_PATTERNS = (
    ("<think>", "think tag leaked"),
    ("</think>", "think close tag leaked"),
    ("<tool_call>", "tool-call tag leaked"),
    ('"arguments":', "raw tool-call JSON leaked"),
    ('{"name":', "raw tool-call JSON leaked"),
    ("tool_result", "tool-result marker leaked"),
)

_REFUSAL_RE = re.compile(
    r"\b(couldn'?t|could not|unable to|cannot|can'?t|didn'?t find|"
    r"failed to (?:find|retrieve|verify)|not enough (?:information|evidence)|"
    r"no reliable (?:information|evidence|data)|not available|"
    r"cannot be (?:verified|confirmed|determined|found))\b",
    re.IGNORECASE,
)


def format_result(value):
    if not isinstance(value, float) or math.isnan(value) or math.isinf(value):
        return value
    rounded = round(value, MAX_DECIMAL_PLACES)
    if rounded == 0 and value != 0:
        rounded = float(f"{value:.{MAX_DECIMAL_PLACES}g}")
    return int(rounded) if rounded.is_integer() else rounded


def extract_numbers(text: str) -> list[float]:
    out = []
    for token in _NUM_RE.findall(text or ""):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return out


def default_tolerance(value: float) -> float:
    return max(0.01, abs(value) * 0.001)


def number_matches(candidate: float, target: float, tolerance: float) -> bool:
    return abs(candidate - target) <= tolerance


def iter_tool_calls(row):
    for turn_index, turn in enumerate(row.get("turns") or []):
        for call in turn.get("tool_calls") or []:
            name = call.get("tool") or call.get("name") or ""
            args = call.get("args") or call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            yield turn_index, name, args


def search_evidence(row) -> str:
    parts = []
    for turn in row.get("turns") or []:
        for result in turn.get("tool_results") or []:
            provenance = result.get("provenance") or result
            if provenance.get("tool") == "web_search":
                parts.append(str(provenance.get("content") or result.get("content") or ""))
    return "\n".join(p for p in parts if p)


def normalize_expected(row) -> dict:
    """Translate legacy rows to the new shape without changing source files."""
    expected = dict(row.get("expected") or {})
    behavior = row.get("expected_behavior") or expected.get("expected_behavior")
    if not behavior:
        behavior = "refuse" if expected.get("graceful") else (
            "social_response" if row.get("stratum") == "conversational" else "answer"
        )

    tool_policy = dict(row.get("tool_policy") or expected.get("tool_policy") or {})
    legacy_min_search = row.get("min_search")
    if "search" not in tool_policy and legacy_min_search is not None:
        tool_policy["search"] = (
            "required" if legacy_min_search == 1 else
            "forbidden" if legacy_min_search == 0 else "optional"
        )
    trajectory = expected.get("trajectory") or {}
    if trajectory.get("requires_search_before_calc"):
        tool_policy.setdefault("search", "required")
        tool_policy.setdefault("calculator", "required")
        tool_policy.setdefault("ordered_calls", ["web_search", "calculate"])
        tool_policy.setdefault("calculation_must_use_retrieved_value", True)

    oracle = dict(row.get("oracle") or expected.get("oracle") or {})
    if "expected_value" in expected and not (
        "numeric_answers" in oracle or "numeric_answer_alternatives" in oracle
    ):
        values = [expected["expected_value"], *(expected.get("expected_values_alt") or [])]
        specs = [
            {"value": value, **({"tolerance": expected["tolerance"]}
                                if "tolerance" in expected else {})}
            for value in values
        ]
        if len(specs) == 1:
            oracle["numeric_answers"] = specs
        else:
            # Legacy expected_values_alt means any listed value is acceptable.
            oracle["numeric_answer_alternatives"] = specs
    oracle.setdefault("required_entities", expected.get("entities") or [])
    oracle.setdefault("required_substrings", expected.get("expected_substrings") or [])

    judge = dict(row.get("judge") or expected.get("judge") or {})
    judge.setdefault("enabled", True)
    return {
        "expected_behavior": behavior,
        "tool_policy": tool_policy,
        "oracle": oracle,
        "judge": judge,
    }


def result(passed, detail="", severity="diagnostic", applicable=True):
    return {
        "passed": passed if applicable else None,
        "detail": detail,
        "severity": severity,
    }


def check_runtime(row):
    if row.get("error"):
        return result(False, str(row["error"]), "hard_gate")
    if not (row.get("answer") or "").strip():
        return result(False, "empty final answer", "hard_gate")
    return result(True, "request completed with a non-empty answer", "hard_gate")


def check_hygiene(row):
    answer = row.get("answer") or ""
    hits = sorted({message for pattern, message in _HYGIENE_PATTERNS if pattern in answer})
    return result(not hits, "; ".join(hits) or "clean", "hard_gate")


def check_tool_policy(row, expected):
    policy = expected["tool_policy"]
    calls = list(iter_tool_calls(row))
    names = [name for _, name, _ in calls]
    failures = []

    mapping = {"search": "web_search", "calculator": "calculate"}
    for field, tool_name in mapping.items():
        setting = policy.get(field)
        if setting is None:
            continue
        if setting not in _VALID_POLICIES:
            failures.append(f"invalid {field} policy {setting!r}")
        elif setting == "required" and tool_name not in names:
            failures.append(f"required tool {tool_name} not called")
        elif setting == "forbidden" and tool_name in names:
            failures.append(f"forbidden tool {tool_name} called")

    ordered = policy.get("ordered_calls") or []
    cursor = -1
    for required_name in ordered:
        try:
            cursor = names.index(required_name, cursor + 1)
        except ValueError:
            failures.append(f"ordered tool {required_name} missing or out of order")
            break

    return result(not failures, "; ".join(failures) or "tool policy satisfied", "hard_gate")


def check_duplicate_calls(row):
    seen = set()
    duplicates = []
    for _, name, args in iter_tool_calls(row):
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            duplicates.append(f"{name} {key[1][:100]}")
        seen.add(key)
    return result(not duplicates, "; ".join(duplicates[:3]) or "none", "diagnostic")


def _aliases(group):
    return [group] if isinstance(group, str) else list(group)


def _contains_entity(answer_lower: str, alias) -> bool:
    """Case-insensitive entity/phrase match without matching inside larger words."""
    alias_lower = str(alias).strip().lower()
    if not alias_lower:
        return False
    pattern = rf"(?<!\w){re.escape(alias_lower)}(?!\w)"
    return re.search(pattern, answer_lower) is not None


def _missing_entity_groups(groups, answer_lower):
    missing = []
    for group in groups or []:
        aliases = _aliases(group)
        if not any(_contains_entity(answer_lower, alias) for alias in aliases):
            missing.append(aliases)
    return missing


def _missing_substring_groups(groups, answer_lower):
    missing = []
    for group in groups or []:
        aliases = _aliases(group)
        if not any(str(alias).lower() in answer_lower for alias in aliases):
            missing.append(aliases)
    return missing


def check_entities(row, expected):
    groups = expected["oracle"].get("required_entities") or []
    if not groups:
        return result(None, severity="diagnostic", applicable=False)
    missing = _missing_entity_groups(groups, (row.get("answer") or "").lower())
    return result(not missing, f"missing {missing}" if missing else "all mentioned")


def check_substrings(row, expected):
    groups = expected["oracle"].get("required_substrings") or []
    if not groups:
        return result(None, severity="diagnostic", applicable=False)
    missing = _missing_substring_groups(groups, (row.get("answer") or "").lower())
    return result(not missing, f"missing {missing}" if missing else "all present")


def _match_numeric_spec(candidates, spec):
    target = float(spec["value"])
    tolerance = float(spec.get("tolerance", default_tolerance(target)))
    hit = next((n for n in candidates if number_matches(n, target, tolerance)), None)
    if hit is None:
        return None
    return {"target": target, "matched": hit, "unit": spec.get("unit")}


def check_numeric(row, expected):
    """Require every numeric_answers item; allow any numeric_answer_alternatives item."""
    oracle = expected["oracle"]
    required = oracle.get("numeric_answers") or []
    alternatives = oracle.get("numeric_answer_alternatives") or []
    if not required and not alternatives:
        return result(None, severity="diagnostic", applicable=False)

    candidates = extract_numbers(row.get("answer") or "")
    matched_required = []
    missing_required = []

    for spec in required:
        match = _match_numeric_spec(candidates, spec)
        if match is None:
            missing_required.append(spec)
        else:
            matched_required.append(match)

    matched_alternative = None
    if alternatives:
        for spec in alternatives:
            matched_alternative = _match_numeric_spec(candidates, spec)
            if matched_alternative is not None:
                break

    required_ok = not missing_required
    alternatives_ok = not alternatives or matched_alternative is not None
    passed = required_ok and alternatives_ok

    detail = {
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_alternative": matched_alternative,
    }
    if alternatives and matched_alternative is None:
        detail["unmatched_alternatives"] = alternatives
    return result(passed, json.dumps(detail, ensure_ascii=False))


def check_behavior_signal(row, expected):
    behavior = expected["expected_behavior"]
    refusal_signal = bool(_REFUSAL_RE.search(row.get("answer") or ""))
    if behavior == "refuse":
        passed = refusal_signal
        detail = "refusal language found" if passed else "no obvious refusal language"
    elif behavior in {"answer", "social_response"}:
        passed = not refusal_signal
        detail = "no refusal signal" if passed else "possible refusal in answerable item"
    else:
        return result(None, severity="diagnostic", applicable=False)
    return result(passed, detail)


def _expected_retrieved_values(expected):
    """Return explicit/strongly implied values that should flow from search into calculation."""
    oracle = expected["oracle"]
    dependency = oracle.get("calculation_dependency") or {}

    raw_values = []
    if dependency.get("source_value") is not None:
        raw_values.append(dependency["source_value"])
    raw_values.extend(dependency.get("source_values") or [])

    if raw_values:
        return [float(value) for value in raw_values], "oracle"

    # Conservative fallback: infer only when required_facts contain exactly one
    # unique non-trivial number. This handles rows such as "retrieved key count is 88".
    inferred = []
    for fact in oracle.get("required_facts") or []:
        inferred.extend(n for n in extract_numbers(str(fact)) if n not in _TRIVIAL_NUMS)
    unique = []
    for value in inferred:
        if not any(number_matches(value, seen, default_tolerance(seen)) for seen in unique):
            unique.append(value)
    if len(unique) == 1:
        return unique, "required_facts"
    return [], None


def check_retrieval_link(row, expected):
    policy = expected["tool_policy"]
    if not policy.get("calculation_must_use_retrieved_value"):
        return result(None, severity="diagnostic", applicable=False)

    evidence = search_evidence(row)
    if not evidence:
        return result(None, "search-result content not captured", "diagnostic", applicable=False)

    evidence_numbers = extract_numbers(evidence)
    expression_numbers = []
    for _, name, args in iter_tool_calls(row):
        if name == "calculate":
            expression_numbers.extend(extract_numbers(str(args.get("expression", ""))))

    expected_sources, source_kind = _expected_retrieved_values(expected)
    if expected_sources:
        details = []
        all_linked = True
        for source in expected_sources:
            tolerance = default_tolerance(source)
            in_evidence = any(number_matches(n, source, tolerance) for n in evidence_numbers)
            in_expression = any(number_matches(n, source, tolerance) for n in expression_numbers)
            details.append({
                "source_value": source,
                "present_in_evidence": in_evidence,
                "present_in_calculation": in_expression,
            })
            all_linked = all_linked and in_evidence and in_expression
        return result(
            all_linked,
            f"source={source_kind}; " + json.dumps(details, ensure_ascii=False),
        )

    # If the dataset does not identify the source value, retain a weaker overlap
    # diagnostic rather than pretending that arbitrary numeric overlap proves lineage.
    expression_nontrivial = [n for n in expression_numbers if n not in _TRIVIAL_NUMS]
    overlap = [
        expression for expression in expression_nontrivial
        if any(number_matches(expression, source, default_tolerance(source))
               for source in evidence_numbers)
    ]
    return result(
        bool(overlap),
        ("heuristic numeric overlap only: " + str(overlap[:5]))
        if overlap else
        f"no numeric overlap between calculation {expression_nontrivial[:5]} and evidence",
    )


def build_judge_input(row, expected, checks):
    """Build an independent judge payload; deterministic diagnostics are deliberately excluded."""
    oracle = expected["oracle"]
    evidence = search_evidence(row)
    return {
        "id": row.get("id"),
        "stratum": row.get("stratum"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "expected_behavior": expected["expected_behavior"],
        "oracle": oracle,
        "judge_dimensions": expected["judge"].get("dimensions") or [],
        "retrieved_evidence": evidence or None,
    }


def score_row(row):
    expected = normalize_expected(row)
    checks = {
        "runtime": check_runtime(row),
        "hygiene": check_hygiene(row),
        "tool_policy": check_tool_policy(row, expected),
        "duplicate_calls": check_duplicate_calls(row),
        "entities": check_entities(row, expected),
        "substrings": check_substrings(row, expected),
        "numeric": check_numeric(row, expected),
        "behavior_signal": check_behavior_signal(row, expected),
        "retrieval_link": check_retrieval_link(row, expected),
    }
    hard_failures = [
        f"{name}: {value['detail']}" for name, value in checks.items()
        if value["severity"] == "hard_gate" and value["passed"] is False
    ]
    diagnostic_failures = [
        f"{name}: {value['detail']}" for name, value in checks.items()
        if value["severity"] == "diagnostic" and value["passed"] is False
    ]
    hard_gate_passed = not hard_failures
    judge_enabled = bool(expected["judge"].get("enabled", True))
    judge_required = hard_gate_passed and judge_enabled
    return {
        "checks": checks,
        "hard_gate_passed": hard_gate_passed,
        "hard_failures": hard_failures,
        "diagnostic_failures": diagnostic_failures,
        "judge_required": judge_required,
        "final_verdict": False if not hard_gate_passed else None,
        "judge_input": build_judge_input(row, expected, checks) if judge_required else None,
    }


def aggregate(rows):
    total = len(rows)
    hard_pass = sum(r["_score"]["hard_gate_passed"] for r in rows)
    judge_queue = sum(r["_score"]["judge_required"] for r in rows)
    print(f"runs: {total}")
    print(f"hard-gate pass: {hard_pass}/{total} ({hard_pass / total:.1%})" if total else "hard-gate pass: n/a")
    print(f"queued for judge: {judge_queue}")

    failure_counts = Counter(
        failure.split(":", 1)[0]
        for row in rows for failure in row["_score"]["hard_failures"]
    )
    diagnostic_counts = Counter(
        failure.split(":", 1)[0]
        for row in rows for failure in row["_score"]["diagnostic_failures"]
    )
    if failure_counts:
        print("hard failures:", dict(failure_counts))
    if diagnostic_counts:
        print("diagnostic misses:", dict(diagnostic_counts))

    by_stratum = defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = by_stratum[row.get("stratum", "?")]
        bucket[1] += 1
        bucket[0] += int(row["_score"]["hard_gate_passed"])
    if by_stratum:
        print(f"\n{'stratum':<24}{'gate pass':>12}{'n':>6}")
        print("-" * 42)
        for stratum, (passed, count) in sorted(by_stratum.items()):
            print(f"{stratum:<24}{passed / count:>12.1%}{count:>6}")


def selftest():
    base = {
        "id": "t", "stratum": "calc_basic", "question": "What is 2+2?",
        "answer": "The answer is 4.", "turns": [], "error": None,
        "expected_behavior": "answer",
        "tool_policy": {"search": "forbidden", "calculator": "optional"},
        "oracle": {"numeric_answers": [{"value": 4, "tolerance": 0}]},
        "judge": {"enabled": True, "dimensions": ["correctness"]},
    }
    cases = [
        ("normal answer reaches judge", base, True, True),
        ("transport error hard-fails", {**base, "error": "timeout"}, False, False),
        ("hygiene leak hard-fails", {**base, "answer": "<think>x</think> 4"}, False, False),
        ("forbidden search hard-fails", {**base, "turns": [{"tool_calls": [
            {"tool": "web_search", "args": {"query": "2+2"}}]}]}, False, False),
        ("wrong number still reaches judge", {**base, "answer": "The answer is 5."}, True, True),
    ]
    failures = 0
    for name, row, gate_want, judge_want in cases:
        score = score_row(row)
        ok = score["hard_gate_passed"] == gate_want and score["judge_required"] == judge_want
        failures += int(not ok)
        print(f"[{'ok' if ok else 'BROKEN'}] {name}")

    regression_cases = []

    multi_required = {
        **base,
        "answer": "Only 10 is present.",
        "oracle": {"numeric_answers": [
            {"value": 10, "tolerance": 0},
            {"value": 20, "tolerance": 0},
        ]},
    }
    regression_cases.append((
        "all numeric_answers are required",
        score_row(multi_required)["checks"]["numeric"]["passed"] is False,
    ))

    exact_decimal = {
        **base,
        "answer": "0.1234567",
        "oracle": {"numeric_answers": [{"value": 0.1234567, "tolerance": 0}]},
    }
    regression_cases.append((
        "numeric target is not rounded before comparison",
        score_row(exact_decimal)["checks"]["numeric"]["passed"] is True,
    ))

    entity_boundary = {
        **base,
        "answer": "Russia is large.",
        "oracle": {"required_entities": [["US"]]},
    }
    regression_cases.append((
        "entity aliases do not match inside larger words",
        score_row(entity_boundary)["checks"]["entities"]["passed"] is False,
    ))

    retrieval_link = {
        **base,
        "answer": "88 squared is 7744.",
        "tool_policy": {
            "search": "required",
            "calculator": "required",
            "ordered_calls": ["web_search", "calculate"],
            "calculation_must_use_retrieved_value": True,
        },
        "oracle": {
            "required_facts": ["The retrieved key count is 88"],
            "numeric_answers": [{"value": 7744, "tolerance": 0}],
        },
        "turns": [{
            "tool_calls": [
                {"tool": "web_search", "args": {"query": "piano keys"}},
                {"tool": "calculate", "args": {"expression": "88 ** 2"}},
            ],
            "tool_results": [{
                "provenance": {"tool": "web_search", "content": "A standard piano has 88 keys."}
            }],
        }],
    }
    regression_cases.append((
        "retrieval dependency links the expected source value",
        score_row(retrieval_link)["checks"]["retrieval_link"]["passed"] is True,
    ))

    wrong_retrieval_link = {
        **retrieval_link,
        "turns": [{
            "tool_calls": [
                {"tool": "web_search", "args": {"query": "piano keys"}},
                {"tool": "calculate", "args": {"expression": "2026 - 3"}},
            ],
            "tool_results": [{
                "provenance": {"tool": "web_search", "content": "In 2026, a standard piano has 88 keys."}
            }],
        }],
    }
    regression_cases.append((
        "unrelated numeric overlap does not prove retrieval dependency",
        score_row(wrong_retrieval_link)["checks"]["retrieval_link"]["passed"] is False,
    ))

    for name, ok in regression_cases:
        failures += int(not ok)
        print(f"[{'ok' if ok else 'BROKEN'}] {name}")

    total = len(cases) + len(regression_cases)
    print(f"selftest: {total - failures}/{total} passing")
    return failures == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", nargs="?")
    parser.add_argument("-o", "--out")
    parser.add_argument("--judge-queue", help="optional JSONL containing judge_input objects")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.run_file:
        parser.error("run_file required (or --selftest)")

    with open(args.run_file, encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    for row in rows:
        row["_score"] = score_row(row)

    out = args.out or args.run_file.rsplit(".jsonl", 1)[0] + "_deterministic.jsonl"
    with open(out, "w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.judge_queue:
        with open(args.judge_queue, "w", encoding="utf-8") as queue:
            for row in rows:
                payload = row["_score"]["judge_input"]
                if payload is not None:
                    queue.write(json.dumps(payload, ensure_ascii=False) + "\n")

    aggregate(rows)
    print(f"\nwrote {out}")
    if args.judge_queue:
        print(f"wrote {args.judge_queue}")


if __name__ == "__main__":
    main()

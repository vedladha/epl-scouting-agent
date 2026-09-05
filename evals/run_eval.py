"""Runs the labeled cases through the agent and scores what comes back.

Two tiers. Offline replays each case's labeled weights straight through
find_players: deterministic, free, and it can gate CI. Live sends the ask to
the model and scores the translation it composes, which is the thing actually
worth measuring and the thing that costs money.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_tools import find_players, run_agent_turn  # noqa: E402
from cases import EvalCase, load_cases  # noqa: E402
from score import (FilterScore, ShortlistScore, WeightScore, find_leaks,  # noqa: E402
                   score_filters, score_refusal, score_shortlist, score_weights)

AGENT_MODEL_ENV = "EVAL_AGENT_MODEL"


def resolve_agent_model(model: str = None) -> str | None:
    return model or os.environ.get(AGENT_MODEL_ENV) or None


@dataclass
class CaseResult:
    case: EvalCase
    ranked: bool = False
    interpreted: dict = field(default_factory=dict)
    returned_ids: tuple[str, ...] = ()
    answer: str = ""
    weights: WeightScore = None
    filters: FilterScore = None
    shortlist: ShortlistScore = None
    refusal: object = None
    leaks: tuple = ()
    error: str = ""

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.leaks:
            return False
        if self.case.is_refusal:
            return bool(self.refusal and self.refusal.passed)
        if not self.ranked:
            return False
        if self.weights and (self.weights.sign_flips or self.weights.missing):
            return False
        if self.filters and not self.filters.is_exact:
            return False
        if self.shortlist and self.shortlist.misses:
            return False
        return True


def extract_calls(messages: list[dict], tool_name: str = "find_players") -> list[dict]:
    calls = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            continue
        for block in content or []:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                calls.append(dict(block.input))
    return calls


def final_text(messages: list[dict]) -> str:
    if not messages:
        return ""
    content = messages[-1].get("content")
    if isinstance(content, str):
        return content
    return "\n".join(b.text for b in content or []
                     if getattr(b, "type", None) == "text")


def _filters_of(call: dict) -> dict:
    keys = ("position", "min_age", "max_age", "club", "exclude_club", "min_minutes")
    return {k: call[k] for k in keys if call.get(k) is not None}


def _score_ranking(case: EvalCase, call: dict, returned: list[str]) -> dict:
    scores = {"shortlist": score_shortlist(list(case.shortlist), returned)}
    if case.mode == "rank":
        scores["weights"] = score_weights(case.weights, call.get("style_weights") or {})
        scores["filters"] = score_filters(case.filters, _filters_of(call))
    return scores


def replay_filters(case: EvalCase) -> dict:
    """A case may accept several readings of a filter; replaying needs one."""
    filters = dict(case.filters)
    value = filters.get("position")
    if isinstance(value, list):
        filters["position"] = value[0]
    return filters


def run_case_offline(case: EvalCase) -> CaseResult:
    result = CaseResult(case=case)
    if case.is_refusal:
        result.error = "refusal cases need the live tier"
        return result

    filters = replay_filters(case)
    payload = (find_players(similar_to_player_id=case.similar_to, **filters,
                            k=8) if case.mode == "similar"
               else find_players(style_weights=dict(case.weights), **filters, k=8))
    if "error" in payload:
        result.error = payload["error"]
        return result

    result.ranked = True
    result.interpreted = payload.get("interpreted_as", {})
    result.returned_ids = tuple(r["player_id"] for r in payload["results"])
    result.shortlist = score_shortlist(list(case.shortlist), list(result.returned_ids))
    return result


def run_case_live(case: EvalCase, client, model: str = None) -> CaseResult:
    result = CaseResult(case=case)
    messages = [{"role": "user", "content": case.ask}]
    try:
        kwargs = {"model": model} if model else {}
        messages = run_agent_turn(client, messages, **kwargs)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.answer = final_text(messages)
    result.leaks = tuple(find_leaks(result.answer))
    calls = extract_calls(messages)
    result.ranked = bool(calls)

    if case.is_refusal:
        result.refusal = score_refusal(result.answer, list(case.mentions), result.ranked)
        return result

    if not calls:
        result.error = "the agent ranked nothing"
        return result

    call = calls[0]
    result.interpreted = call
    if case.mode == "similar":
        got = call.get("similar_to_player_id")
        if got != case.similar_to:
            result.error = f"expected a similarity search on {case.similar_to}, got {got!r}"
            return result

    payload = find_players(**call)
    result.returned_ids = tuple(r["player_id"] for r in payload.get("results", []))
    scores = _score_ranking(case, call, list(result.returned_ids))
    result.weights = scores.get("weights")
    result.filters = scores.get("filters")
    result.shortlist = scores["shortlist"]
    return result


def summarize(results: list[CaseResult]) -> dict:
    ranked = [r for r in results if r.weights]
    labeled = [r for r in results if r.shortlist and r.shortlist.expected]
    refusals = [r for r in results if r.case.is_refusal and r.refusal]
    directions = [r.weights.direction_accuracy for r in ranked]
    return {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "errors": sum(bool(r.error) for r in results),
        "leaks": sum(len(r.leaks) for r in results),
        "weight_direction_accuracy": (sum(directions) / len(directions)
                                      if directions else None),
        "filters_exact": (sum(r.filters.is_exact for r in ranked) / len(ranked)
                          if ranked else None),
        "shortlist_recall": (sum(r.shortlist.recall for r in labeled) / len(labeled)
                             if labeled else None),
        "refusals_correct": (sum(r.refusal.passed for r in refusals) / len(refusals)
                             if refusals else None),
    }


def _percent(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def report(results: list[CaseResult], summary: dict, tier: str) -> str:
    lines = [f"Eval run ({tier} tier): {summary['passed']}/{summary['cases']} cases passed", ""]
    for result in results:
        notes = []
        if result.error:
            notes.append(result.error)
        if result.weights and result.weights.sign_flips:
            notes.append(f"wrong direction: {', '.join(result.weights.sign_flips)}")
        if result.weights and result.weights.missing:
            notes.append(f"not expressed: {', '.join(result.weights.missing)}")
        if result.weights and result.weights.extra:
            notes.append(f"unasked for: {', '.join(result.weights.extra)}")
        if result.filters and result.filters.mismatched:
            notes += [f"filter {k}: wanted {want!r}, got {got!r}"
                      for k, want, got in result.filters.mismatched]
        if result.shortlist and result.shortlist.misses:
            notes.append(f"not shortlisted: {', '.join(result.shortlist.misses)}")
        if result.refusal and result.refusal.unmentioned:
            notes.append(f"refusal never mentioned: {', '.join(result.refusal.unmentioned)}")
        for leak in result.leaks:
            notes.append(f"leak ({leak.kind}): {leak.quote}")

        mark = "ok" if result.passed else "FAIL"
        lines.append(f"  {mark:<6} {result.case.id:<24} {notes[0] if notes else ''}")
        for note in notes[1:]:
            lines.append(f"  {'':<6} {'':<24} {note}")

    lines += [
        "",
        f"  weight directions   {_percent(summary['weight_direction_accuracy'])}",
        f"  filters exact       {_percent(summary['filters_exact'])}",
        f"  shortlist recall    {_percent(summary['shortlist_recall'])}",
        f"  refusals correct    {_percent(summary['refusals_correct'])}",
        f"  leaks               {summary['leaks']}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="send each ask to the model; costs money")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--case", action="append", dest="only")
    args = parser.parse_args()

    cases = load_cases()
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
        if not cases:
            parser.error(f"no case matched {args.only}")

    if args.live:
        import anthropic
        model = resolve_agent_model(args.model)
        print(f"Running {len(cases)} cases against the model. This costs real money.\n")
        results = [run_case_live(c, anthropic.Anthropic(), model) for c in cases]
        tier = "live"
    else:
        results = [run_case_offline(c) for c in cases if not c.is_refusal]
        tier = "offline"

    summary = summarize(results)
    text = report(results, summary, tier)
    print(text)

    if args.out:
        args.out.write_text(json.dumps({
            "tier": tier,
            "summary": summary,
            "cases": [{"id": r.case.id, "passed": r.passed, "error": r.error,
                       "interpreted": r.interpreted,
                       "returned": list(r.returned_ids),
                       "leaks": [{"kind": l.kind, "quote": l.quote} for l in r.leaks]}
                      for r in results],
        }, indent=2, default=str))
        print(f"\nWrote {args.out}")

    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

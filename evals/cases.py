"""Loads and validates the labeled evaluation cases in cases.yaml."""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import FEATURE_COLUMNS  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "cases.yaml"

MODES = ("rank", "refuse", "similar")
FILTER_KEYS = ("position", "min_age", "max_age", "club", "exclude_club", "min_minutes")
POSITIONS = ("DF", "MF", "FW")


class CaseError(ValueError):
    pass


@dataclass(frozen=True)
class EvalCase:
    id: str
    ask: str
    mode: str
    weights: dict[str, float] = field(default_factory=dict)
    similar_to: str | None = None
    filters: dict[str, object] = field(default_factory=dict)
    shortlist: tuple[str, ...] = ()
    mentions: tuple[str, ...] = ()

    @property
    def is_refusal(self) -> bool:
        return self.mode == "refuse"


def _fail(case_id: str, message: str):
    raise CaseError(f"case {case_id!r}: {message}")


def _validate_weights(case_id: str, weights) -> dict[str, float]:
    if not isinstance(weights, dict) or not weights:
        _fail(case_id, "mode 'rank' needs a non-empty weights mapping")
    unknown = sorted(set(weights) - set(FEATURE_COLUMNS))
    if unknown:
        _fail(case_id, f"weights reference features that do not exist: {unknown}")
    for feature, weight in weights.items():
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            _fail(case_id, f"weight for {feature!r} is not a number: {weight!r}")
        if weight == 0:
            _fail(case_id, f"weight for {feature!r} is zero, which expresses nothing")
    return {feature: float(weight) for feature, weight in weights.items()}


def _validate_filters(case_id: str, filters) -> dict[str, object]:
    if filters is None:
        return {}
    if not isinstance(filters, dict):
        _fail(case_id, "filters must be a mapping")
    unknown = sorted(set(filters) - set(FILTER_KEYS))
    if unknown:
        _fail(case_id, f"unknown filter keys: {unknown}")
    position = filters.get("position")
    if position is not None:
        allowed = position if isinstance(position, list) else [position]
        if not allowed or any(p not in POSITIONS for p in allowed):
            _fail(case_id, f"position must be one of {list(POSITIONS)}, or a list "
                           f"of them when either reading is acceptable, got {position!r}")
    for key in ("min_age", "max_age", "min_minutes"):
        value = filters.get(key)
        if value is not None and not isinstance(value, int):
            _fail(case_id, f"{key} must be an integer, got {value!r}")
    return dict(filters)


def _validate_strings(case_id: str, key: str, values) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip()
                                           for v in values):
        _fail(case_id, f"{key} must be a list of non-empty strings")
    return tuple(values)


def _build_case(raw: dict) -> EvalCase:
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise CaseError(f"every case needs a string id: {raw!r}")

    ask = raw.get("ask")
    if not isinstance(ask, str) or not ask.strip():
        _fail(case_id, "ask must be a non-empty string")

    mode = raw.get("mode")
    if mode not in MODES:
        _fail(case_id, f"mode must be one of {list(MODES)}, got {mode!r}")

    unknown = sorted(set(raw) - {"id", "ask", "mode", "weights", "similar_to",
                                 "filters", "shortlist", "mentions"})
    if unknown:
        _fail(case_id, f"unknown keys: {unknown}")

    weights, similar_to = {}, None
    if mode == "rank":
        weights = _validate_weights(case_id, raw.get("weights"))
        if raw.get("similar_to"):
            _fail(case_id, "mode 'rank' cannot also set similar_to")
    elif mode == "similar":
        similar_to = raw.get("similar_to")
        if not isinstance(similar_to, str) or not similar_to.strip():
            _fail(case_id, "mode 'similar' needs a similar_to player_id")
        if raw.get("weights"):
            _fail(case_id, "mode 'similar' cannot also set weights")
    else:
        if raw.get("weights") or raw.get("similar_to") or raw.get("filters"):
            _fail(case_id, "a refusal case expects no weights, similar_to or filters")
        if not raw.get("mentions"):
            _fail(case_id, "a refusal case needs mentions naming what it must say")

    return EvalCase(
        id=case_id,
        ask=ask,
        mode=mode,
        weights=weights,
        similar_to=similar_to,
        filters=_validate_filters(case_id, raw.get("filters")),
        shortlist=_validate_strings(case_id, "shortlist", raw.get("shortlist")),
        mentions=_validate_strings(case_id, "mentions", raw.get("mentions")),
    )


def load_cases(path: Path = CASES_FILE) -> list[EvalCase]:
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise CaseError(f"{path} must contain a top-level 'cases' list")

    cases = [_build_case(raw) for raw in document["cases"]]
    if not cases:
        raise CaseError(f"{path} contains no cases")

    seen = set()
    for case in cases:
        if case.id in seen:
            raise CaseError(f"duplicate case id: {case.id!r}")
        seen.add(case.id)
    return cases


def referenced_player_ids(cases: list[EvalCase]) -> set[str]:
    ids = set()
    for case in cases:
        ids.update(case.shortlist)
        if case.similar_to:
            ids.add(case.similar_to)
    return ids


if __name__ == "__main__":
    loaded = load_cases()
    by_mode = {mode: sum(c.mode == mode for c in loaded) for mode in MODES}
    print(f"{len(loaded)} cases loaded from {CASES_FILE}")
    print("  " + ", ".join(f"{mode}: {count}" for mode, count in by_mode.items()))
    print(f"  {len(referenced_player_ids(loaded))} player ids referenced")

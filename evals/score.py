"""Scoring for the eval harness: translation, ranking, refusal and output leaks.

Every function here is pure. Nothing touches the database, the API or the clock,
so scores are reproducible from a recorded run.
"""
import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightScore:
    direction_accuracy: float
    cosine: float
    matched: tuple[str, ...]
    sign_flips: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not (self.sign_flips or self.missing or self.extra)


@dataclass(frozen=True)
class FilterScore:
    matched: tuple[str, ...]
    mismatched: tuple[tuple[str, object, object], ...]

    @property
    def is_exact(self) -> bool:
        return not self.mismatched


@dataclass(frozen=True)
class ShortlistScore:
    expected: int
    hits: tuple[str, ...]
    misses: tuple[str, ...]
    forbidden_hits: tuple[str, ...]

    @property
    def recall(self) -> float:
        return len(self.hits) / self.expected if self.expected else 1.0


@dataclass(frozen=True)
class RefusalScore:
    refused: bool
    mentioned: tuple[str, ...]
    unmentioned: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.refused and not self.unmentioned


@dataclass(frozen=True)
class Leak:
    kind: str
    quote: str


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def score_weights(expected: dict[str, float], actual: dict[str, float]) -> WeightScore:
    actual = {k: float(v) for k, v in (actual or {}).items() if float(v) != 0}
    expected = {k: float(v) for k, v in (expected or {}).items() if float(v) != 0}

    matched, sign_flips = [], []
    for feature, weight in expected.items():
        if feature not in actual:
            continue
        if _sign(actual[feature]) == _sign(weight):
            matched.append(feature)
        else:
            sign_flips.append(feature)

    missing = [f for f in expected if f not in actual]
    extra = [f for f in actual if f not in expected]
    accuracy = len(matched) / len(expected) if expected else 0.0

    return WeightScore(
        direction_accuracy=accuracy,
        cosine=cosine(expected, actual),
        matched=tuple(sorted(matched)),
        sign_flips=tuple(sorted(sign_flips)),
        missing=tuple(sorted(missing)),
        extra=tuple(sorted(extra)),
    )


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(float(a.get(k, 0.0)) * float(b.get(k, 0.0)) for k in keys)
    norm_a = math.sqrt(sum(float(a.get(k, 0.0)) ** 2 for k in keys))
    norm_b = math.sqrt(sum(float(b.get(k, 0.0)) ** 2 for k in keys))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def score_filters(expected: dict, actual: dict) -> FilterScore:
    actual = actual or {}
    matched, mismatched = [], []
    for key, value in (expected or {}).items():
        got = actual.get(key)
        if got == value:
            matched.append(key)
        else:
            mismatched.append((key, value, got))
    return FilterScore(tuple(sorted(matched)), tuple(sorted(mismatched)))


def score_shortlist(expected: list[str], returned: list[str],
                    forbidden: list[str] = None) -> ShortlistScore:
    returned_set = set(returned or [])
    expected = list(expected or [])
    hits = [p for p in expected if p in returned_set]
    misses = [p for p in expected if p not in returned_set]
    forbidden_hits = [p for p in (forbidden or []) if p in returned_set]
    return ShortlistScore(len(expected), tuple(hits), tuple(misses),
                          tuple(forbidden_hits))


def score_refusal(text: str, mentions: list[str], ranked: bool) -> RefusalScore:
    lowered = (text or "").lower()
    mentioned = [m for m in (mentions or []) if m.lower() in lowered]
    unmentioned = [m for m in (mentions or []) if m.lower() not in lowered]
    return RefusalScore(not ranked, tuple(mentioned), tuple(unmentioned))


LEAK_PATTERNS = (
    ("per_90_rate",
     re.compile(r"\d+(?:\.\d+)?(?:\s+[a-z-]+){0,4}\s*(?:per\s*90|/\s*90|\bp90)\b", re.I)),
    ("column_name", re.compile(r"\b[a-z]+(?:_[a-z0-9]+)*_p90\b", re.I)),
    ("statistical_jargon",
     re.compile(r"\b(?:z-?scores?|standard deviations?|fit[_ ]score|"
                r"sample[_ ]reliability|style[_ ]weights|shrinkage)\b", re.I)),
    ("footedness",
     re.compile(r"\b(?:left|right)[- ]?(?:footed|sided)\b|\bpreferred foot\b|"
                r"\b(?:naturally|predominantly)\s+(?:left|right)\b", re.I)),
    ("transfer_or_money",
     re.compile(r"[£$€]\s?\d|\b\d+(?:\.\d+)?\s?m(?:illion)?\b|\btransfer fee\b|"
                r"\bmarket value\b|\bprice tag\b|\bout of contract\b|"
                r"\b(?:likely|set|expected|reportedly)\s+to\s+(?:leave|join|sign)\b",
                re.I)),
    ("injury_inference",
     re.compile(r"\binjur(?:y|ed|ies)[- ]?(?:disrupted|hit|plagued)?\b", re.I)),
)

NEGATION_CUES = (
    "no ", "not ", "n't", "cannot", "can not", "never", "without",
    "outside what", "unable", "lacks", "has none", "there is none",
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def _is_negated(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(cue in lowered for cue in NEGATION_CUES)


def find_leaks(text: str, allow_negated: bool = True) -> list[Leak]:
    leaks = []
    for sentence in _sentences(text):
        for kind, pattern in LEAK_PATTERNS:
            if not pattern.search(sentence):
                continue
            if allow_negated and _is_negated(sentence):
                continue
            leaks.append(Leak(kind, sentence))
    return leaks

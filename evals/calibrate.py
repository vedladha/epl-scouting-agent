"""Measures the judge against answers whose verdict is already known.

An LLM judge that flags nothing scores a perfect record on a clean test set and
is worthless. These fixtures are answers this project has actually produced, or
close paraphrases of them, so the judge can be shown to catch what it should
and stay quiet when it should.
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import RULES, JudgeError, judge_answer, resolve_model  # noqa: E402

FIXTURES_FILE = Path(__file__).resolve().parent / "judge_fixtures.yaml"


@dataclass(frozen=True)
class Fixture:
    id: str
    ask: str
    answer: str
    expect: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not self.expect


@dataclass(frozen=True)
class FixtureResult:
    fixture: Fixture
    found: tuple[str, ...]
    error: str = ""

    @property
    def caught(self) -> bool:
        return set(self.fixture.expect) <= set(self.found)

    @property
    def extras(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.found) - set(self.fixture.expect)))

    @property
    def false_alarm(self) -> bool:
        return self.fixture.is_clean and bool(self.found)


@dataclass(frozen=True)
class Calibration:
    results: tuple[FixtureResult, ...]

    @property
    def violations(self) -> list[FixtureResult]:
        return [r for r in self.results if not r.fixture.is_clean]

    @property
    def clean(self) -> list[FixtureResult]:
        return [r for r in self.results if r.fixture.is_clean]

    @property
    def recall(self) -> float:
        rows = self.violations
        return sum(r.caught for r in rows) / len(rows) if rows else 1.0

    @property
    def with_extras(self) -> list[FixtureResult]:
        return [r for r in self.violations if r.caught and r.extras]

    @property
    def false_alarm_rate(self) -> float:
        rows = self.clean
        return sum(r.false_alarm for r in rows) / len(rows) if rows else 0.0

    @property
    def missed(self) -> list[FixtureResult]:
        return [r for r in self.violations if not r.caught]


def load_fixtures(path: Path = FIXTURES_FILE) -> list[Fixture]:
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("fixtures"), list):
        raise ValueError(f"{path} must contain a top-level 'fixtures' list")

    fixtures = []
    for raw in document["fixtures"]:
        expect = tuple(raw.get("expect") or ())
        unknown = sorted(set(expect) - set(RULES))
        if unknown:
            raise ValueError(f"fixture {raw.get('id')!r} expects unknown rules: {unknown}")
        fixtures.append(Fixture(raw["id"], raw["ask"], raw["answer"], expect))

    if not any(f.is_clean for f in fixtures):
        raise ValueError("fixtures need clean answers too, or false alarms go unmeasured")
    return fixtures


def calibrate(fixtures: list[Fixture], client, model: str = None) -> Calibration:
    results = []
    for fixture in fixtures:
        try:
            verdict = judge_answer(fixture.ask, fixture.answer, client=client, model=model)
            results.append(FixtureResult(fixture, verdict.rules_broken))
        except JudgeError as exc:
            results.append(FixtureResult(fixture, (), error=str(exc)))
    return Calibration(tuple(results))


def report(calibration: Calibration, model: str) -> str:
    lines = [f"Judge calibration: {model}", ""]
    for result in calibration.results:
        if result.error:
            mark, detail = "ERROR", result.error
        elif result.fixture.is_clean:
            mark = "ok" if not result.found else "FALSE ALARM"
            detail = ", ".join(result.found) or "no findings, as expected"
        elif result.caught and result.extras:
            mark = "ok +extra"
            detail = f"{', '.join(result.fixture.expect)}, plus {', '.join(result.extras)}"
        elif result.caught:
            mark, detail = "ok", ", ".join(result.found)
        else:
            mark = "MISSED"
            detail = (f"expected {', '.join(result.fixture.expect)}, got "
                      f"{', '.join(result.found) or 'nothing'}")
        lines.append(f"  {mark:<12} {result.fixture.id:<28} {detail}")

    lines += [
        "",
        f"  caught          {calibration.recall:.0%} of planted violations",
        f"  false alarms    {calibration.false_alarm_rate:.0%} of clean answers",
        f"  extra findings  {len(calibration.with_extras)} of "
        f"{len(calibration.violations)} planted answers",
    ]
    if calibration.with_extras:
        lines.append("")
        lines.append("  An extra finding is not counted as wrong. Over-flagging "
                     "is measured on the clean answers above.")
    if calibration.missed:
        lines.append("")
        lines.append("  A miss here means the judge is blind to that failure, "
                     "not that the answer was fine.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_FILE)
    args = parser.parse_args()

    import anthropic

    fixtures = load_fixtures(args.fixtures)
    model = resolve_model(args.model)
    print(f"Grading {len(fixtures)} fixtures with {model}. This costs real money.\n")
    print(report(calibrate(fixtures, anthropic.Anthropic(), model=args.model), model))


if __name__ == "__main__":
    main()

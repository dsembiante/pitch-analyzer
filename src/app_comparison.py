"""
Dataclasses and logic for session-to-session metric comparison.
"""

from dataclasses import dataclass

from app_pipeline import PipelineResult
from app_utils import METRIC_DISPLAY_ORDER, get_direction
from metrics._types import MetricResult

UNCHANGED_PCT_THRESHOLD = 2.0          # |pct_change| < this → unchanged
UNCHANGED_ABS_THRESHOLD_BY_UNIT: dict[str, float] = {
    "degrees":             1.0,
    "degrees_per_second":  1.0,
    "seconds":             0.05,
    "percent_body_height": 1.0,
    "":                    0.01,       # fallback for unknown units
}

# Neutral metrics whose |pct_change| exceeds this are classified "notable" (amber).
NEUTRAL_NOTABLE_PCT_THRESHOLD = 10.0
# Absolute-delta fallback used when pct_change is None (baseline near zero).
NEUTRAL_NOTABLE_ABS_THRESHOLD_BY_UNIT: dict[str, float] = {
    "degrees":             5.0,
    "degrees_per_second":  5.0,
    "seconds":             0.10,
    "percent_body_height": 5.0,
    "":                    0.05,
}


@dataclass
class MetricComparison:
    name: str
    display_name: str
    unit: str
    description: str
    direction: str          # "higher_better" | "lower_better" | "neutral"
    session_a: MetricResult
    session_b: MetricResult
    delta: float | None
    pct_change: float | None
    classification: str     # "improved" | "regressed" | "unchanged" | "notable" | "incomparable"


@dataclass
class ComparisonResult:
    session_a: PipelineResult
    session_b: PipelineResult
    comparisons: list[MetricComparison]
    summary: dict[str, int]


def compute_comparison(
    result_a: PipelineResult,
    result_b: PipelineResult,
) -> ComparisonResult:
    """Compare per-metric results from two pipeline runs, ordered by METRIC_DISPLAY_ORDER."""
    comparisons: list[MetricComparison] = []

    for name in METRIC_DISPLAY_ORDER:
        mr_a = result_a.metrics.get(name)
        mr_b = result_b.metrics.get(name)
        if mr_a is None or mr_b is None:
            continue

        # a. incomparable: either side has no value or an error
        if mr_a.value is None or mr_a.error or mr_b.value is None or mr_b.error:
            comparisons.append(MetricComparison(
                name=name,
                display_name=mr_a.display_name,
                unit=mr_a.unit,
                description=mr_a.description,
                direction="neutral",
                session_a=mr_a,
                session_b=mr_b,
                delta=None,
                pct_change=None,
                classification="incomparable",
            ))
            continue

        delta = mr_b.value - mr_a.value
        pct_change = (delta / mr_a.value) * 100 if abs(mr_a.value) > 1e-6 else None

        direction = get_direction(name)
        if direction == "neutral":
            if pct_change is not None:
                notable = abs(pct_change) >= NEUTRAL_NOTABLE_PCT_THRESHOLD
            else:
                threshold = NEUTRAL_NOTABLE_ABS_THRESHOLD_BY_UNIT.get(
                    mr_a.unit, NEUTRAL_NOTABLE_ABS_THRESHOLD_BY_UNIT[""]
                )
                notable = abs(delta) >= threshold
            classification = "notable" if notable else "unchanged"
        else:
            # c. meaningfulness check
            if pct_change is not None:
                meaningful = abs(pct_change) >= UNCHANGED_PCT_THRESHOLD
            else:
                threshold = UNCHANGED_ABS_THRESHOLD_BY_UNIT.get(
                    mr_a.unit, UNCHANGED_ABS_THRESHOLD_BY_UNIT[""]
                )
                meaningful = abs(delta) >= threshold

            # d/e. classify by direction if meaningful, else unchanged
            if not meaningful:
                classification = "unchanged"
            elif direction == "higher_better":
                classification = "improved" if delta > 0 else "regressed"
            else:  # lower_better
                classification = "improved" if delta < 0 else "regressed"

        comparisons.append(MetricComparison(
            name=name,
            display_name=mr_a.display_name,
            unit=mr_a.unit,
            description=mr_a.description,
            direction=direction,
            session_a=mr_a,
            session_b=mr_b,
            delta=delta,
            pct_change=pct_change,
            classification=classification,
        ))

    summary: dict[str, int] = {}
    for cmp in comparisons:
        summary[cmp.classification] = summary.get(cmp.classification, 0) + 1

    return ComparisonResult(
        session_a=result_a,
        session_b=result_b,
        comparisons=comparisons,
        summary=summary,
    )

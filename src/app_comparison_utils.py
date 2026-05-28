"""
Display helpers for the Compare Sessions page.
"""

import pandas as pd
from pandas.io.formats.style import Styler as _Styler

from app_comparison import ComparisonResult
from app_utils import format_metric_value, build_export_payload

_UNIT_LABELS: dict[str, str] = {
    "degrees":             "deg",
    "seconds":             "s",
    "percent_body_height": "%",
    "degrees_per_second":  "deg/s",
}


def format_delta(delta: float | None, unit: str) -> str:
    """Format a delta value with leading sign and short unit label.

    Precision matches format_metric_value: 2 dp for seconds, 1 dp for everything else.
    Signed-zero (-0.0) is normalized to positive zero.
    """
    if delta is None:
        return "—"
    if delta == 0.0:
        delta = 0.0  # collapse -0.0 → +0.0
    label = _UNIT_LABELS.get(unit, unit)
    if unit == "seconds":
        return f"{delta:+.2f} {label}"
    return f"{delta:+.1f} {label}"


def format_pct_change(pct: float | None) -> str:
    """Format a percent-change value with leading sign, or em-dash when unavailable.

    Signed-zero (-0.0) is normalized to positive zero.
    """
    if pct is None:
        return "—"
    if pct == 0.0:
        pct = 0.0  # collapse -0.0 → +0.0
    return f"{pct:+.1f}%"


def classification_to_color(classification: str) -> str:
    """Return a CSS rgba background color string for a classification, or '' for none."""
    return {
        "improved":     "rgba(16, 185, 129, 0.25)",
        "regressed":    "rgba(239, 68, 68, 0.25)",
        "unchanged":    "",
        "incomparable": "rgba(148, 163, 184, 0.20)",
    }.get(classification, "")


def direction_label(direction: str) -> str:
    """Return a short human phrase for an improvement direction."""
    return {
        "higher_better": "Higher is better",
        "lower_better":  "Lower is better",
        "neutral":       "No directional preference",
    }.get(direction, "No directional preference")


def direction_tooltip(metric_name: str, direction: str) -> str:
    """Return a one-line explanation of why a metric has its direction."""
    _TOOLTIPS: dict[str, str] = {
        "stride_length": (
            "A longer stride increases forward momentum and is generally associated "
            "with higher pitch velocity."
        ),
        "hip_shoulder_separation_max": (
            "More separation builds a larger stretch-shortening cycle, allowing more "
            "elastic energy to be transferred through the kinetic chain into the arm."
        ),
        "head_path_length": (
            "Less total head movement indicates better postural stability and a more "
            "consistent release point."
        ),
        "head_max_deviation": (
            "A smaller peak head excursion is associated with better posture control "
            "and a more repeatable delivery."
        ),
        "front_knee_extension_rate": (
            "A higher (more positive) rate reflects a stronger front-leg block, "
            "helping transfer lower-body energy up the kinetic chain."
        ),
    }
    if metric_name in _TOOLTIPS:
        return _TOOLTIPS[metric_name]
    return "This metric has no universally better direction; shown for reference."


def style_comparison_dataframe(
    rows: list[dict],
    classifications: list[str],
) -> _Styler:
    """Apply background colors to the Delta and % Change columns by classification."""
    df = pd.DataFrame(rows)

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for i, cls in enumerate(classifications):
            color = classification_to_color(cls)
            if color:
                for col in ("Delta", "% Change"):
                    if col in styles.columns:
                        styles.loc[i, col] = f"background-color: {color}"
        return styles

    return df.style.apply(_apply, axis=None)


def build_comparison_table_rows(comparison: ComparisonResult) -> list[dict]:
    """Return rows ready for st.dataframe, one per MetricComparison."""
    rows = []
    for cmp in comparison.comparisons:
        rows.append({
            "Metric":    cmp.display_name,
            "Session A": format_metric_value(cmp.session_a),
            "Session B": format_metric_value(cmp.session_b),
            "Delta":     format_delta(cmp.delta, cmp.unit),
            "% Change":  format_pct_change(cmp.pct_change),
        })
    return rows


def build_comparison_export_payload(
    comparison: ComparisonResult,
    metadata_a: dict,
    metadata_b: dict,
) -> dict:
    """Build a JSON-serializable export payload for a two-session comparison.

    metadata_a / metadata_b must contain the same keys that build_export_payload
    accepts as keyword args: athlete_name, pitch_type, session_date, handedness,
    video_filename.
    """
    return {
        "comparison_type": "two_session",
        "session_a": build_export_payload(comparison.session_a, **metadata_a),
        "session_b": build_export_payload(comparison.session_b, **metadata_b),
        "comparison": [
            {
                "metric":          cmp.name,
                "display_name":    cmp.display_name,
                "unit":            cmp.unit,
                "direction":       cmp.direction,
                "session_a_value": cmp.session_a.value,
                "session_b_value": cmp.session_b.value,
                "delta":           cmp.delta,
                "pct_change":      cmp.pct_change,
                "classification":  cmp.classification,
            }
            for cmp in comparison.comparisons
        ],
        "summary": comparison.summary,
    }

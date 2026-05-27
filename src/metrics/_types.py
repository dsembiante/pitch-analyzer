"""Shared result type for all metrics."""

from dataclasses import dataclass


@dataclass
class MetricResult:
    name: str
    display_name: str
    value: float | None
    unit: str
    frame: int | None = None
    phase: str | None = None
    notes: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "value": self.value,
            "unit": self.unit,
            "frame": self.frame,
            "phase": self.phase,
            "notes": self.notes,
            "error": self.error,
        }

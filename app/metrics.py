"""Minimal Prometheus-compatible metrics registry (stdlib only).

Tracks counters, gauges, and histograms in memory and renders the plain-text
exposition format that Prometheus scrapes. Kept dependency-free on purpose so
the project stays zero-paid-deps; small enough that the whole registry fits in
a single module.

Label sets are sorted so the same label set always produces the same key.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from typing import Iterable

_HISTOGRAM_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
    float("inf"),
)


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = defaultdict(Counter)
        self._gauges: dict[str, Counter] = defaultdict(Counter)
        # name -> {labels -> {le: count, sum, count_total}}
        self._histograms: dict[str, dict[tuple, dict]] = defaultdict(dict)
        self._help: dict[str, str] = {}

    def register(self, name: str, help_text: str) -> None:
        self._help[name] = help_text

    @staticmethod
    def _labels(labels: dict | None) -> tuple:
        return tuple(sorted((labels or {}).items()))

    def inc(self, name: str, labels: dict | None = None, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name][self._labels(labels)] += value

    def set_gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        with self._lock:
            self._gauges[name][self._labels(labels)] = value

    def observe(self, name: str, value: float, labels: dict | None = None) -> None:
        key = self._labels(labels)
        with self._lock:
            bucket = self._histograms[name].setdefault(key, {"le": {}, "sum": 0.0, "count": 0})
            bucket["sum"] += value
            bucket["count"] += 1
            for b in _HISTOGRAM_BUCKETS:
                bucket["le"].setdefault(b, 0.0)
                if value <= b:
                    bucket["le"][b] += 1.0

    # ------------------------------------------------------------------ #
    # Rendering (Prometheus text exposition format)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_labels(labels: Iterable[tuple]) -> str:
        parts = [f'{k}="{_escape(v)}"' for k, v in labels]
        return "{" + ",".join(parts) + "}" if parts else ""

    def render(self) -> str:
        lines: list[str] = []
        names = set(self._counters) | set(self._gauges) | set(self._histograms)
        with self._lock:
            for name in sorted(names):
                if name in self._histograms:
                    kind = "histogram"
                elif name in self._gauges:
                    kind = "gauge"
                else:
                    kind = "counter"
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} {kind}")
            for name in sorted(self._counters):
                for labels, value in sorted(self._counters[name].items()):
                    lines.append(f"{name}{self._format_labels(labels)} {_fmt(value)}")
            for name in sorted(self._gauges):
                for labels, value in sorted(self._gauges[name].items()):
                    lines.append(f"{name}{self._format_labels(labels)} {_fmt(value)}")
            for name in sorted(self._histograms):
                for labels, bucket in sorted(self._histograms[name].items()):
                    for b in sorted(bucket["le"]):
                        le = "+Inf" if b == float("inf") else _fmt(b)
                        le_labels = list(labels) + [("le", le)]
                        lines.append(
                            f"{name}_bucket{self._format_labels(le_labels)} {_fmt(bucket['le'][b])}"
                        )
                    lines.append(
                        f"{name}_sum{self._format_labels(labels)} {_fmt(bucket['sum'])}"
                    )
                    lines.append(
                        f"{name}_count{self._format_labels(labels)} {_fmt(bucket['count'])}"
                    )
        return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: float) -> str:
    if value != value:  # NaN
        return "NaN"
    if value == float("inf"):
        return "+Inf"
    if value == float("-inf"):
        return "-Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.9g}".rstrip(".")


metrics = MetricsRegistry()

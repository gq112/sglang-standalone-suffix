#!/usr/bin/env python3
"""Summarize a fixed-K=4 versus dynamic-K scaling sweep."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


LOG_METRICS = {
    "output_tok_s": r"Total E2E output throughput \(tok/s\):\s*([\d.]+)",
    "ttft_ms": r"Mean TTFT \(ms\):\s*([\d.]+)",
    "tpot_ms": r"Mean TPOT \(ms\):\s*([\d.]+)",
}
MEASUREMENT_RE = re.compile(r"measurement_bs(?P<bs>\d+)_n\d+\.log$")
TP0_METRIC_RE = re.compile(r'tp_rank="0"[^}]*\}\s+([\d.]+)$')
METRICS = (
    "sglang:dynamic_k8_request_total",
    "sglang:dynamic_k8_output_token_total",
    "sglang:dynamic_k8_draft_token_total",
)


def parse_log(path: Path) -> dict[str, float] | None:
    match = MEASUREMENT_RE.match(path.name)
    if match is None:
        return None
    content = path.read_text(encoding="utf-8", errors="replace")
    values = {"bs": float(match.group("bs"))}
    for name, pattern in LOG_METRICS.items():
        found = re.findall(pattern, content)
        if not found:
            return None
        values[name] = float(found[-1])
    return values


def read_metrics(path: Path) -> dict[str, float]:
    values = {metric: 0.0 for metric in METRICS}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        metric = line.split("{", 1)[0]
        if metric not in values:
            continue
        match = TP0_METRIC_RE.search(line)
        if match:
            values[metric] += float(match.group(1))
    return values


def main() -> None:
    root = Path(sys.argv[1])
    rows: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = case_dir.name
        measurements = []
        for log_path in case_dir.rglob("measurement_bs*_n*.log"):
            row = parse_log(log_path)
            if row is None:
                continue
            measurements.append((int(row["bs"]), log_path, row))

        # Metric snapshots are cumulative. Subtract the immediately preceding
        # measurement snapshot so every row describes that concurrency phase,
        # rather than bs10 + bs20 + ... since the K=8 probe.
        previous_by_experiment: dict[Path, dict[str, float]] = {}
        for bs, log_path, row in sorted(measurements):
            experiment_dir = log_path.parent
            before = previous_by_experiment.setdefault(
                experiment_dir,
                read_metrics(experiment_dir / "metrics_after_k8_probe_focus.prom"),
            )
            after = read_metrics(
                experiment_dir / f"metrics_after_measurement_bs{bs}_focus.prom"
            )
            draft = after[METRICS[2]] - before[METRICS[2]]
            row["long_rounds"] = after[METRICS[0]] - before[METRICS[0]]
            row["long_efficiency"] = (
                (after[METRICS[1]] - before[METRICS[1]]) / draft if draft else 0.0
            )
            rows[bs][label] = row
            previous_by_experiment[experiment_dir] = after

    labels = ["fixed_k4", "dynamic_k4_control"] + sorted(
        (path.name for path in root.glob("dynamic_k4_k*")),
        key=lambda value: int(value.rsplit("k", 1)[1]),
    )
    lines = [
        "# Dynamic-K Scaling Sweep",
        "",
        "All runs use the same workload. Varlen CUDA-graph patterns are disabled, so deltas isolate dynamic-K policy and target-verify scaling.",
        "",
        "| Concurrency | Config | Output tok/s | vs fixed K=4 | Mean TTFT (ms) | Mean TPOT (ms) | Long-K rounds | Long-K efficiency |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bs in sorted(rows):
        fixed = rows[bs].get("fixed_k4", {}).get("output_tok_s")
        for label in labels:
            row = rows[bs].get(label)
            if row is None:
                continue
            delta = (row["output_tok_s"] / fixed - 1) * 100 if fixed else 0.0
            lines.append(
                f"| {bs} | {label} | {row['output_tok_s']:.2f} | {delta:+.2f}% | "
                f"{row['ttft_ms']:.2f} | {row['tpot_ms']:.2f} | "
                f"{row['long_rounds']:.0f} | {row['long_efficiency']:.3f} |"
            )
    print("\n".join(lines))


if __name__ == "__main__":
    main()

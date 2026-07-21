#!/usr/bin/env python3
"""Summarize Arctic versus SRI global/dataset suffix forest results."""

from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


POLICIES = ("arctic", "sri_global", "sri_dataset")
LOG_RE = re.compile(r"measurement_bs(?P<concurrency>\d+)_n\d+\.log$")
OUTPUT_RE = re.compile(r"Total E2E output throughput \(tok/s\):\s*([\d.]+)")
SOURCE_RE = re.compile(r'source="(?P<source>[^"]+)"')
VALUE_RE = re.compile(r"\}\s+(?P<value>[\d.]+)$")


def output_throughput(path: Path) -> float | None:
    values = OUTPUT_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    return float(values[-1]) if values else None


def source_counts(path: Path) -> dict[str, float]:
    counts: dict[str, float] = defaultdict(float)
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("sglang:suffix_proposal_source_total{"):
            continue
        if 'tp_rank="0"' not in line:
            continue
        source = SOURCE_RE.search(line)
        value = VALUE_RE.search(line)
        if source and value:
            counts[source.group("source")] += float(value.group("value"))
    return counts


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} RESULTS_DIR")
    root = Path(sys.argv[1])
    throughputs: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    source_deltas: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for policy in POLICIES:
        for case in sorted(root.glob(f"r*_{policy}")):
            experiment = case / "dynamic_k4_k16"
            previous = source_counts(experiment / "metrics_after_k8_probe_focus.prom")
            for log in sorted(experiment.glob("measurement_bs*_n*.log")):
                match = LOG_RE.match(log.name)
                value = output_throughput(log)
                if not match or value is None:
                    continue
                concurrency = int(match.group("concurrency"))
                throughputs[policy][concurrency].append(value)
                after = source_counts(
                    experiment / f"metrics_after_measurement_bs{concurrency}_focus.prom"
                )
                source_deltas[policy][concurrency].append(
                    {source: after[source] - previous.get(source, 0.0) for source in after}
                )
                previous = after

    lines = [
        "# SRI Dataset-Tree Suffix Experiment",
        "",
        "All policies use the final K=4/16/8 policy. Values are medians across cyclic server runs.",
        "",
        "| Concurrency | Arctic cache tok/s | SRI global-only tok/s | SRI + dataset tree tok/s | Dataset tree vs SRI global |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    concurrencies = sorted({c for values in throughputs.values() for c in values})
    for concurrency in concurrencies:
        values = {policy: throughputs[policy].get(concurrency, []) for policy in POLICIES}
        if any(not rows for rows in values.values()):
            continue
        medians = {policy: statistics.median(rows) for policy, rows in values.items()}
        delta = (medians["sri_dataset"] / medians["sri_global"] - 1) * 100
        lines.append(
            f"| {concurrency} | {medians['arctic']:.2f} | "
            f"{medians['sri_global']:.2f} | {medians['sri_dataset']:.2f} | {delta:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Median positive-score proposal source rows",
            "",
            "These are per-measurement-phase deltas on TP0, not unique requests.",
            "",
            "| Policy | Concurrency | local | global | dataset | arctic |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in POLICIES:
        for concurrency, rows in sorted(source_deltas[policy].items()):
            sources = {
                source: statistics.median(row.get(source, 0.0) for row in rows)
                for source in ("local", "global", "dataset", "arctic")
            }
            lines.append(
                f"| {policy} | {concurrency} | {sources['local']:.0f} | "
                f"{sources['global']:.0f} | {sources['dataset']:.0f} | "
                f"{sources['arctic']:.0f} |"
            )
    print("\n".join(lines))


if __name__ == "__main__":
    main()

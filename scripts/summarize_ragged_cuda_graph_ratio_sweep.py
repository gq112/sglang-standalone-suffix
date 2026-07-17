#!/usr/bin/env python3
"""Summarize throughput and ragged CUDA-graph coverage across policy ratios."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


RANK_RE = re.compile(r'tp_rank="([^"]+)"')
METRICS = (
    "sglang:ragged_verify_cuda_graph_batch_total",
    "sglang:ragged_verify_eager_batch_total",
)


def read_metric_snapshot(path: Path) -> dict[str, float]:
    values = {metric: 0.0 for metric in METRICS}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            sample, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except ValueError:
            continue
        metric = sample.split("{", 1)[0]
        if metric not in values:
            continue
        rank = RANK_RE.search(sample)
        if rank is not None and rank.group(1) != "0":
            continue
        values[metric] += value
    return values


def read_throughput(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if row["config"] not in {"suffix_static_k4", "dynamic_k4_k8"}:
                continue
            rows.setdefault(int(row["concurrency"]), {})[row["config"]] = float(
                row["total_output_tok_s"]
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for ratio_dir in sorted(args.sweep_dir.glob("ratio_*")):
        result_path = ratio_dir / "throughput_comparison.tsv"
        dynamic_dir = ratio_dir / "dynamic_k4_k8"
        if not result_path.exists() or not dynamic_dir.is_dir():
            continue
        ratio = ratio_dir.name.removeprefix("ratio_").replace("_", ".")
        throughput = read_throughput(result_path)
        previous = read_metric_snapshot(
            dynamic_dir / "metrics_after_k8_probe_focus.prom"
        )
        for snapshot in sorted(
            dynamic_dir.glob("metrics_after_measurement_bs*_focus.prom")
        ):
            match = re.fullmatch(
                r"metrics_after_measurement_bs(\d+)_focus\.prom", snapshot.name
            )
            if match is None:
                continue
            concurrency = int(match.group(1))
            current = read_metric_snapshot(snapshot)
            graph = current[METRICS[0]] - previous[METRICS[0]]
            eager = current[METRICS[1]] - previous[METRICS[1]]
            previous = current
            static = throughput.get(concurrency, {}).get("suffix_static_k4")
            dynamic = throughput.get(concurrency, {}).get("dynamic_k4_k8")
            if static is None or dynamic is None:
                continue
            rows.append(
                (ratio, concurrency, static, dynamic, (dynamic / static - 1) * 100, graph, eager)
            )

    lines = [
        "# Ragged CUDA-Graph Ratio Sweep",
        "",
        "`ratio=1.0` is the eager Ragged control. Lower ratios increase graph coverage but also padding work.",
        "",
        "| Min K=8 ratio | Concurrency | Static K=4 tok/s | Ragged K=4/8 tok/s | Dynamic vs static | Graph batches | Eager batches | Graph hit rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ratio, concurrency, static, dynamic, delta, graph, eager in rows:
        hit_rate = 100 * graph / (graph + eager) if graph + eager else 0.0
        lines.append(
            f"| {ratio} | {concurrency} | {static:.2f} | {dynamic:.2f} | "
            f"{delta:+.2f}% | {graph:.0f} | {eager:.0f} | {hit_rate:.2f}% |"
        )

    output = args.sweep_dir / "ratio_sweep_summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize the four-layer end-to-end speculative-decoding comparison."""

from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


POLICIES = ("no_speculation", "standalone_k4", "suffix_static_k4", "dynamic_final")
EXPERIMENT_DIR = {
    "no_speculation": "no_speculation",
    "standalone_k4": "standalone_k4",
    "suffix_static_k4": "suffix_static_k4",
    "dynamic_final": "dynamic_k4_k16",
}
LOG_RE = re.compile(r"measurement_bs(?P<concurrency>\d+)_n\d+\.log$")
METRICS = {
    "tok_s": re.compile(r"Total E2E output throughput \(tok/s\):\s*([\d.]+)"),
    "ttft": re.compile(r"Mean TTFT \(ms\):\s*([\d.]+)"),
    "tpot": re.compile(r"Mean TPOT \(ms\):\s*([\d.]+)"),
    "ok": re.compile(r"Successful requests:\s*(\d+)/(\d+)"),
}


def read_log(path: Path) -> dict[str, float | str] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, float | str] = {}
    for name, pattern in METRICS.items():
        found = pattern.findall(text)
        if not found:
            return None
        result[name] = "/".join(found[-1]) if name == "ok" else float(found[-1])
    return result


def median(rows: list[dict[str, float | str]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def percent(numerator: float, denominator: float) -> str:
    return "-" if denominator == 0 else f"{(numerator / denominator - 1) * 100:+.2f}%"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} RESULTS_DIR")
    root = Path(sys.argv[1])
    data: dict[str, dict[int, list[dict[str, float | str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failures: list[str] = []

    for policy in POLICIES:
        for case in sorted(root.glob(f"r*_{policy}")):
            experiment = case / EXPERIMENT_DIR[policy]
            for log in sorted(experiment.glob("measurement_bs*_n*.log")):
                match = LOG_RE.match(log.name)
                values = read_log(log)
                if not match or values is None:
                    failures.append(str(log))
                    continue
                data[policy][int(match.group("concurrency"))].append(values)

    lines = [
        "# Full-Stack Final Policy Validation",
        "",
        "Four-layer comparison under identical workload. Values are medians over cyclic fresh-server runs.",
        "",
        "| Concurrency | No speculation | Standalone K=4 | Suffix static K=4 | Final dynamic K | Fixed speculation vs no-spec | Suffix fusion vs standalone | Dynamic K vs suffix static | Total dynamic vs no-spec |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_concurrency = sorted({c for values in data.values() for c in values})
    for concurrency in all_concurrency:
        rows = {policy: data[policy].get(concurrency, []) for policy in POLICIES}
        if any(not values for values in rows.values()):
            continue
        values = {policy: median(items, "tok_s") for policy, items in rows.items()}
        lines.append(
            f"| {concurrency} | {values['no_speculation']:.2f} | "
            f"{values['standalone_k4']:.2f} | {values['suffix_static_k4']:.2f} | "
            f"{values['dynamic_final']:.2f} | "
            f"{percent(values['standalone_k4'], values['no_speculation'])} | "
            f"{percent(values['suffix_static_k4'], values['standalone_k4'])} | "
            f"{percent(values['dynamic_final'], values['suffix_static_k4'])} | "
            f"{percent(values['dynamic_final'], values['no_speculation'])} |"
        )

    lines.extend(
        [
            "",
            "## Latency medians",
            "",
            "| Concurrency | Final dynamic TTFT (ms) | Suffix static TTFT (ms) | Final dynamic TPOT (ms) | Suffix static TPOT (ms) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for concurrency in all_concurrency:
        dynamic = data["dynamic_final"].get(concurrency, [])
        static = data["suffix_static_k4"].get(concurrency, [])
        if not dynamic or not static:
            continue
        lines.append(
            f"| {concurrency} | {median(dynamic, 'ttft'):.2f} | "
            f"{median(static, 'ttft'):.2f} | {median(dynamic, 'tpot'):.2f} | "
            f"{median(static, 'tpot'):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Run completeness",
            "",
            "| Policy | Concurrency | Completed/submitted values |",
            "| --- | ---: | --- |",
        ]
    )
    for policy in POLICIES:
        for concurrency, rows in sorted(data[policy].items()):
            lines.append(
                f"| {policy} | {concurrency} | {', '.join(str(row['ok']) for row in rows)} |"
            )
    if failures:
        lines.extend(["", "## Unparseable logs", "", *[f"- `{path}`" for path in failures]])
    print("\n".join(lines))


if __name__ == "__main__":
    main()

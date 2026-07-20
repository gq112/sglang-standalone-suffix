#!/usr/bin/env python3
"""Summarize alternating fixed-K=4 versus final dynamic-K A/B results."""

from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


OUTPUT_RE = re.compile(r"Total E2E output throughput \(tok/s\):\s*([\d.]+)")
TTFT_RE = re.compile(r"Mean TTFT \(ms\):\s*([\d.]+)")
TPOT_RE = re.compile(r"Mean TPOT \(ms\):\s*([\d.]+)")
LOG_RE = re.compile(r"measurement_bs(?P<concurrency>\d+)_n\d+\.log$")
TP0_RE = re.compile(r'tp_rank="0"[^}]*\}\s+([\d.]+)$')
TIER_RE = re.compile(r'draft_tokens="(?P<width>\d+)"')
TIER_METRIC = "sglang:dynamic_k_tier_request_total"


def metric_tiers(path: Path) -> dict[int, float]:
    result: dict[int, float] = defaultdict(float)
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(TIER_METRIC):
            continue
        tier = TIER_RE.search(line)
        value = TP0_RE.search(line)
        if tier and value:
            result[int(tier.group("width"))] += float(value.group(1))
    return result


def log_values(path: Path) -> dict[str, float] | None:
    values: dict[str, float] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for key, pattern in (("tok_s", OUTPUT_RE), ("ttft", TTFT_RE), ("tpot", TPOT_RE)):
        found = pattern.findall(text)
        if not found:
            return None
        values[key] = float(found[-1])
    return values


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} RESULTS_DIR")
    root = Path(sys.argv[1])
    if not root.is_dir():
        raise SystemExit(f"Results directory does not exist: {root}")

    runs: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    tier_rows: list[tuple[str, int, dict[int, float]]] = []
    for case in sorted(path for path in root.iterdir() if path.is_dir()):
        policy = "fixed" if case.name.endswith("_fixed_k4") else "dynamic" if case.name.endswith("_final_policy") else None
        if policy is None:
            continue
        previous_tiers: dict[Path, dict[int, float]] = {}
        measurements = []
        for log_path in case.rglob("measurement_bs*_n*.log"):
            match = LOG_RE.match(log_path.name)
            values = log_values(log_path)
            if match and values:
                measurements.append((int(match.group("concurrency")), log_path, values))
        for concurrency, log_path, values in sorted(measurements):
            runs[policy][concurrency].append(values)
            if policy != "dynamic":
                continue
            experiment = log_path.parent
            before = previous_tiers.setdefault(
                experiment,
                metric_tiers(experiment / "metrics_after_k8_probe_focus.prom"),
            )
            after = metric_tiers(
                experiment / f"metrics_after_measurement_bs{concurrency}_focus.prom"
            )
            phase = {width: after[width] - before.get(width, 0.0) for width in after}
            tier_rows.append((case.name, concurrency, phase))
            previous_tiers[experiment] = after

    lines = [
        "# Final Dynamic-K Alternating A/B",
        "",
        "Primary metric: total end-to-end output throughput. Values are medians across independently started alternating server runs.",
        "",
        "| Concurrency | Fixed K=4 median tok/s | Final policy median tok/s | Uplift | Fixed runs | Dynamic runs | Fixed median TTFT (ms) | Dynamic median TTFT (ms) | Fixed median TPOT (ms) | Dynamic median TPOT (ms) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for concurrency in sorted(set(runs["fixed"]) | set(runs["dynamic"])):
        fixed = runs["fixed"].get(concurrency, [])
        dynamic = runs["dynamic"].get(concurrency, [])
        if not fixed or not dynamic:
            continue
        fixed_tok = statistics.median(item["tok_s"] for item in fixed)
        dynamic_tok = statistics.median(item["tok_s"] for item in dynamic)
        fixed_ttft = statistics.median(item["ttft"] for item in fixed)
        dynamic_ttft = statistics.median(item["ttft"] for item in dynamic)
        fixed_tpot = statistics.median(item["tpot"] for item in fixed)
        dynamic_tpot = statistics.median(item["tpot"] for item in dynamic)
        uplift = (dynamic_tok / fixed_tok - 1.0) * 100
        lines.append(
            f"| {concurrency} | {fixed_tok:.2f} | {dynamic_tok:.2f} | {uplift:+.2f}% | "
            f"{len(fixed)} | {len(dynamic)} | {fixed_ttft:.2f} | {dynamic_ttft:.2f} | "
            f"{fixed_tpot:.2f} | {dynamic_tpot:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Dynamic tier coverage per run and measurement phase",
            "",
            "`K=8` must be nonzero at concurrency 24/30 to prove the high-batch fallback was exercised. Values are phase deltas for TP0 and count selected draft rows, not unique requests.",
            "",
            "| Run | Concurrency | K=8 selected rows | K=16 selected rows |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label, concurrency, tiers in tier_rows:
        lines.append(
            f"| {label} | {concurrency} | {tiers.get(8, 0.0):.0f} | {tiers.get(16, 0.0):.0f} |"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()

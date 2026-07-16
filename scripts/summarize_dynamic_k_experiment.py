#!/usr/bin/env python3
"""Summarize suffix/dynamic-K Prometheus counter deltas from one experiment."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


METRICS = (
    "sglang:suffix_proposal_total",
    "sglang:suffix_override_total",
    "sglang:dynamic_k8_request_total",
    "sglang:dynamic_k8_output_token_total",
    "sglang:dynamic_k8_draft_token_total",
    "sglang:dynamic_k_verify_batch_total",
    "sglang:dynamic_k_mixed_verify_batch_total",
    "sglang:dynamic_k_normal_verify_call_total",
    "sglang:dynamic_k_long_verify_call_total",
)
SNAPSHOTS = ("startup", "after_warmup", "after_k8_probe", "after_measurement")
LABEL_RE = re.compile(r'tp_rank="([^"]+)"')
MEASUREMENT_LOG_RE = re.compile(r"measurement_bs(?P<concurrency>\d+)_n\d+\.log$")

# `test_req.py` writes these exact labels at the end of every benchmark phase.
LOG_METRICS = {
    "successful_requests": r"Successful requests:\s*(\d+)/(\d+)",
    "duration_s": r"Benchmark duration \(s\):\s*([\d.]+)",
    "mean_decode_tok_s": r"Mean decoding throughput \(tok/s\):\s*([\d.]+)",
    "mean_output_tok_s": r"Mean output throughput \(tok/s\):\s*([\d.]+)",
    "total_output_tok_s": r"Total E2E output throughput \(tok/s\):\s*([\d.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s*([\d.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s*([\d.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s*([\d.]+)",
}
CONFIG_ORDER = (
    "no_speculation",
    "standalone_k4",
    "suffix_static_k4",
    "dynamic_k4_k4",
    "dynamic_k4_k8",
)


def read_snapshot(path: Path) -> dict[str, float]:
    values = {metric: 0.0 for metric in METRICS}
    if not path.exists():
        return values
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
        rank = LABEL_RE.search(sample)
        # Only TP0 owns the authoritative scheduler result. If a deployment
        # does not expose tp_rank, retain its unlabeled sample.
        if rank is not None and rank.group(1) != "0":
            continue
        values[metric] += value
    return values


def subtract(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {metric: after[metric] - before[metric] for metric in METRICS}


def parse_measurement_log(path: Path) -> dict[str, float | int] | None:
    match = MEASUREMENT_LOG_RE.search(path.name)
    if match is None:
        return None
    content = path.read_text(encoding="utf-8", errors="replace")
    row: dict[str, float | int] = {"concurrency": int(match.group("concurrency"))}
    for name, pattern in LOG_METRICS.items():
        values = re.findall(pattern, content)
        if not values:
            return None
        value = values[-1]
        if name == "successful_requests":
            completed, submitted = value
            row["successful_requests"] = int(completed)
            row["submitted_requests"] = int(submitted)
        else:
            row[name] = float(value)
    return row


def write_throughput_comparison(results_dir: Path) -> None:
    """Write raw and pivoted 10--24 benchmark comparisons from test_req logs."""
    rows: list[dict[str, float | int | str]] = []
    for config in CONFIG_ORDER:
        config_dir = results_dir / config
        if not config_dir.is_dir():
            continue
        for log_path in sorted(config_dir.glob("measurement_bs*_n*.log")):
            parsed = parse_measurement_log(log_path)
            if parsed is not None:
                rows.append({"config": config, **parsed})
    rows.sort(key=lambda row: (int(row["concurrency"]), CONFIG_ORDER.index(str(row["config"]))))
    if not rows:
        return

    raw_columns = (
        "concurrency",
        "config",
        "successful_requests",
        "submitted_requests",
        "duration_s",
        "total_output_tok_s",
        "mean_output_tok_s",
        "mean_decode_tok_s",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
    )
    raw_path = results_dir / "throughput_comparison.tsv"
    with raw_path.open("w", encoding="utf-8") as file:
        file.write("\t".join(raw_columns) + "\n")
        for row in rows:
            file.write("\t".join(str(row[column]) for column in raw_columns) + "\n")

    by_concurrency: dict[int, dict[str, dict[str, float | int | str]]] = {}
    for row in rows:
        by_concurrency.setdefault(int(row["concurrency"]), {})[str(row["config"])] = row

    markdown_lines = [
        "# Dynamic-K Throughput Comparison",
        "",
        "Primary metric: total end-to-end output throughput (token/s). "
        "Positive dynamic deltas are better.",
        "",
        "| Concurrency | No speculation | Standalone K=4 | Suffix static K=4 | Dynamic K=4/4 | Dynamic K=4/8 | K=4/4 vs static | K=4/8 vs K=4/4 | K=4/8 vs static |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for concurrency, config_rows in sorted(by_concurrency.items()):
        def throughput(config: str) -> float | None:
            row = config_rows.get(config)
            return None if row is None else float(row["total_output_tok_s"])

        no_spec = throughput("no_speculation")
        standalone = throughput("standalone_k4")
        suffix_static = throughput("suffix_static_k4")
        dynamic_k4_k4 = throughput("dynamic_k4_k4")
        dynamic = throughput("dynamic_k4_k8")
        delta_split = (
            (dynamic_k4_k4 / suffix_static - 1) * 100
            if dynamic_k4_k4 is not None and suffix_static not in (None, 0)
            else None
        )
        delta_k8 = (
            (dynamic / dynamic_k4_k4 - 1) * 100
            if dynamic is not None and dynamic_k4_k4 not in (None, 0)
            else None
        )
        delta_static = (
            (dynamic / suffix_static - 1) * 100
            if dynamic is not None and suffix_static not in (None, 0)
            else None
        )
        delta_standalone = (
            (dynamic / standalone - 1) * 100
            if dynamic is not None and standalone not in (None, 0)
            else None
        )
        format_value = lambda value: "-" if value is None else f"{value:.2f}"
        format_delta = lambda value: "-" if value is None else f"{value:+.2f}%"
        markdown_lines.append(
            "| "
            + " | ".join(
                (
                    str(concurrency),
                    format_value(no_spec),
                    format_value(standalone),
                    format_value(suffix_static),
                    format_value(dynamic_k4_k4),
                    format_value(dynamic),
                    format_delta(delta_split),
                    format_delta(delta_k8),
                    format_delta(delta_static),
                )
            )
            + " |"
        )

    markdown_lines.extend(
        [
            "",
            "## Dynamic versus suffix-static latency",
            "",
            "| Concurrency | Dynamic mean TTFT (ms) | Static mean TTFT (ms) | Dynamic mean TPOT (ms) | Static mean TPOT (ms) | Dynamic mean ITL (ms) | Static mean ITL (ms) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for concurrency, config_rows in sorted(by_concurrency.items()):
        dynamic = config_rows.get("dynamic_k4_k8")
        suffix_static = config_rows.get("suffix_static_k4")
        value = lambda row, key: "-" if row is None else f"{float(row[key]):.2f}"
        markdown_lines.append(
            f"| {concurrency} | {value(dynamic, 'mean_ttft_ms')} | "
            f"{value(suffix_static, 'mean_ttft_ms')} | "
            f"{value(dynamic, 'mean_tpot_ms')} | "
            f"{value(suffix_static, 'mean_tpot_ms')} | "
            f"{value(dynamic, 'mean_itl_ms')} | "
            f"{value(suffix_static, 'mean_itl_ms')} |"
        )

    markdown_path = results_dir / "throughput_comparison.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print()
    print(f"Wrote {raw_path}")
    print(f"Wrote {markdown_path}")
    print("\n".join(markdown_lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    print(
        "experiment\tphase\tproposals\toverrides\tk8_requests"
        "\tk8_output_tokens\tk8_draft_tokens\tk8_efficiency"
        "\tdynamic_batches\tmixed_batches\tk4_verify_calls\tlong_verify_calls"
    )
    dynamic_probe: dict[str, float] | None = None
    for experiment_dir in sorted(path for path in args.results_dir.iterdir() if path.is_dir()):
        if not (experiment_dir / "metrics_startup.prom").exists():
            continue
        snapshots = {
            name: read_snapshot(experiment_dir / f"metrics_{name}.prom")
            for name in SNAPSHOTS
        }
        phase_pairs = [
            ("startup", "after_warmup", "warmup"),
            ("after_warmup", "after_k8_probe", "k8_probe"),
        ]
        previous_name = "after_k8_probe"
        measurement_snapshots = sorted(
            path
            for path in experiment_dir.glob("metrics_after_measurement_bs*.prom")
            if re.fullmatch(r"metrics_after_measurement_bs\d+\.prom", path.name)
        )
        if measurement_snapshots:
            for snapshot_path in measurement_snapshots:
                snapshot_name = snapshot_path.stem.removeprefix("metrics_")
                phase_name = snapshot_name.removeprefix("after_measurement_")
                snapshots[snapshot_name] = read_snapshot(snapshot_path)
                phase_pairs.append((previous_name, snapshot_name, phase_name))
                previous_name = snapshot_name
        elif (experiment_dir / "metrics_after_measurement.prom").exists():
            # Keep compatibility with results produced by the first script.
            phase_pairs.append(("after_k8_probe", "after_measurement", "measurement"))

        for before_name, after_name, phase in phase_pairs:
            delta = subtract(snapshots[after_name], snapshots[before_name])
            k8_draft = delta["sglang:dynamic_k8_draft_token_total"]
            k8_efficiency = (
                delta["sglang:dynamic_k8_output_token_total"] / k8_draft
                if k8_draft
                else 0.0
            )
            print(
                f"{experiment_dir.name}\t{phase}\t"
                f"{delta['sglang:suffix_proposal_total']:.0f}\t"
                f"{delta['sglang:suffix_override_total']:.0f}\t"
                f"{delta['sglang:dynamic_k8_request_total']:.0f}\t"
                f"{delta['sglang:dynamic_k8_output_token_total']:.0f}\t"
                f"{k8_draft:.0f}\t{k8_efficiency:.3f}\t"
                f"{delta['sglang:dynamic_k_verify_batch_total']:.0f}\t"
                f"{delta['sglang:dynamic_k_mixed_verify_batch_total']:.0f}\t"
                f"{delta['sglang:dynamic_k_normal_verify_call_total']:.0f}\t"
                f"{delta['sglang:dynamic_k_long_verify_call_total']:.0f}"
            )
            if experiment_dir.name == "dynamic_k4_k8" and phase == "k8_probe":
                dynamic_probe = delta

    if dynamic_probe is None:
        write_throughput_comparison(args.results_dir)
        return
    request_count = dynamic_probe["sglang:dynamic_k8_request_total"]
    draft_tokens = dynamic_probe["sglang:dynamic_k8_draft_token_total"]
    output_tokens = dynamic_probe["sglang:dynamic_k8_output_token_total"]
    efficiency = output_tokens / draft_tokens if draft_tokens else 0.0
    print()
    if request_count == 0:
        print("VERDICT: K=8 never triggered in the low-concurrency probe.")
    else:
        print(
            "VERDICT: K=8 triggered for "
            f"{request_count:.0f} requests; committed-token efficiency={efficiency:.3f}."
        )
        print(
            "Compare dynamic_k4_k8/k8_probe.log against "
            "suffix_static_k4/k8_probe.log for end-to-end benefit."
        )
    write_throughput_comparison(args.results_dir)


if __name__ == "__main__":
    main()

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
)
SNAPSHOTS = ("startup", "after_warmup", "after_k8_probe", "after_measurement")
LABEL_RE = re.compile(r'tp_rank="([^"]+)"')


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    print(
        "experiment\tphase\tproposals\toverrides\tk8_requests"
        "\tk8_output_tokens\tk8_draft_tokens\tk8_efficiency"
    )
    dynamic_probe: dict[str, float] | None = None
    for experiment_dir in sorted(path for path in args.results_dir.iterdir() if path.is_dir()):
        if not (experiment_dir / "metrics_startup.prom").exists():
            continue
        snapshots = {
            name: read_snapshot(experiment_dir / f"metrics_{name}.prom")
            for name in SNAPSHOTS
        }
        for before_name, after_name, phase in (
            ("startup", "after_warmup", "warmup"),
            ("after_warmup", "after_k8_probe", "k8_probe"),
            ("after_k8_probe", "after_measurement", "measurement"),
        ):
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
                f"{k8_draft:.0f}\t{k8_efficiency:.3f}"
            )
            if experiment_dir.name == "dynamic_k4_k8" and phase == "k8_probe":
                dynamic_probe = delta

    if dynamic_probe is None:
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


if __name__ == "__main__":
    main()

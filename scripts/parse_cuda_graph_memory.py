#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict


STAGE_PATTERNS = (
    ("target_verify", re.compile(r"Capture cuda graph end.*mem usage=([0-9.]+) GB.*avail mem=([0-9.]+) GB")),
    ("draft_decode", re.compile(r"Capture draft cuda graph end.*mem usage=([0-9.]+) GB.*avail mem=([0-9.]+) GB")),
    ("draft_extend", re.compile(r"Capture draft extend cuda graph end.*mem usage=([0-9.]+) GB.*avail mem=([0-9.]+) GB")),
)
FINAL_AVAIL_RE = re.compile(r"max_total_num_tokens=.*available_(?:gpu_)?mem=([0-9.]+) GB")


def mean(values):
    return sum(values) / len(values) if values else 0.0


def parse_log(path):
    stages = defaultdict(list)
    final_avail = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for stage, pattern in STAGE_PATTERNS:
                match = pattern.search(line)
                if match:
                    stages[stage].append(
                        {
                            "mem_usage": float(match.group(1)),
                            "avail_mem": float(match.group(2)),
                        }
                    )
                    break

            match = FINAL_AVAIL_RE.search(line)
            if match:
                final_avail = float(match.group(1))

    target = mean([x["mem_usage"] for x in stages["target_verify"]])
    draft = mean([x["mem_usage"] for x in stages["draft_decode"]])
    draft_extend = mean([x["mem_usage"] for x in stages["draft_extend"]])
    total = target + draft + draft_extend
    return {
        "target_verify": target,
        "draft_decode": draft,
        "draft_extend": draft_extend,
        "total_cuda_graph": total,
        "final_available": final_avail,
        "target_samples": len(stages["target_verify"]),
        "draft_samples": len(stages["draft_decode"]),
        "draft_extend_samples": len(stages["draft_extend"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parse SGLang CUDA graph memory usage from launch logs."
    )
    parser.add_argument("logs", nargs="+", help="Log file paths")
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print CSV instead of a markdown table.",
    )
    args = parser.parse_args()

    rows = []
    for path in args.logs:
        row = parse_log(path)
        row["log"] = path
        rows.append(row)

    if args.csv:
        print(
            "log,target_verify_gb,draft_decode_gb,draft_extend_gb,total_cuda_graph_gb,final_available_gb"
        )
        for row in rows:
            final_available = (
                f"{row['final_available']:.2f}"
                if row["final_available"] is not None
                else ""
            )
            print(
                f"{row['log']},{row['target_verify']:.2f},{row['draft_decode']:.2f},"
                f"{row['draft_extend']:.2f},{row['total_cuda_graph']:.2f},{final_available}"
            )
        return

    print(
        "| Log | Target Verify | Draft Decode | Draft Extend | Total Graph | Final Available |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        final_available = (
            f"{row['final_available']:.2f} GB"
            if row["final_available"] is not None
            else "N/A"
        )
        print(
            f"| `{row['log']}` | {row['target_verify']:.2f} GB | "
            f"{row['draft_decode']:.2f} GB | {row['draft_extend']:.2f} GB | "
            f"{row['total_cuda_graph']:.2f} GB | {final_available} |"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Send a prompt-only JSONL dataset to /generate and persist exact outputs."""

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import requests


def get_prompt(record: dict[str, Any]) -> str:
    if isinstance(record.get("text"), str):
        return record["text"]
    if isinstance(record.get("prompt"), str):
        return record["prompt"]
    if isinstance(record.get("turns"), list) and all(
        isinstance(turn, str) for turn in record["turns"]
    ):
        return "\n".join(record["turns"])
    if isinstance(record.get("question"), str):
        return record["question"]
    raise ValueError("record needs text, prompt, turns, or question")


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines()[:limit]:
        prompts.append(get_prompt(json.loads(line)))
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def generate(
    base_url: str, prompt_index: int, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=1800,
    )
    response.raise_for_status()
    output = response.json()
    if isinstance(output, list):
        output = output[0]
    if "error" in output:
        raise RuntimeError(output["error"])
    return {
        "index": prompt_index,
        "output_ids": output.get("output_ids"),
        "text": output.get("text"),
        "meta_info": output.get("meta_info", {}),
    }


def run(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.dataset_path, args.num_prompts)
    outputs: list[dict[str, Any] | None] = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        futures = {
            pool.submit(
                generate, args.base_url, index, prompt, args.max_new_tokens
            ): index
            for index, prompt in enumerate(prompts)
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            index = futures[future]
            outputs[index] = future.result()
            print(f"completed {completed}/{len(prompts)}", end="\r", flush=True)
    print()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for output in outputs:
            assert output is not None
            file.write(json.dumps(output, ensure_ascii=False) + "\n")


def compare(args: argparse.Namespace) -> None:
    reference = [json.loads(line) for line in args.reference.read_text(encoding="utf-8").splitlines()]
    candidate = [json.loads(line) for line in args.candidate.read_text(encoding="utf-8").splitlines()]
    mismatches = []
    first_difference = None
    for index, (left, right) in enumerate(zip(reference, candidate)):
        left_value = left.get("output_ids") or left.get("text")
        right_value = right.get("output_ids") or right.get("text")
        if left_value != right_value:
            mismatches.append(index)
            if first_difference is None:
                if isinstance(left_value, list) and isinstance(right_value, list):
                    first_token = next(
                        (
                            position
                            for position, (left_token, right_token) in enumerate(
                                zip(left_value, right_value)
                            )
                            if left_token != right_token
                        ),
                        min(len(left_value), len(right_value)),
                    )
                    first_difference = (
                        f"- First difference: request {index}, output token "
                        f"{first_token}; {args.reference_label}="
                        f"{left_value[first_token:first_token + 1]}, "
                        f"{args.candidate_label}="
                        f"{right_value[first_token:first_token + 1]}\n"
                    )
                else:
                    first_difference = f"- First difference: request {index} text differs\n"
    if len(reference) != len(candidate):
        mismatches.extend(range(min(len(reference), len(candidate)), max(len(reference), len(candidate))))
    report = (
        "# Greedy Output Consistency\n\n"
        f"- Compared requests: {min(len(reference), len(candidate))}\n"
        f"- Mismatches: {len(mismatches)}\n"
        + (
            f"- First mismatch indices: {mismatches[:20]}\n{first_difference}"
            if mismatches
            else "- Result: PASS\n"
        )
    )
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    if mismatches:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--dataset-path", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--num-prompts", type=int, default=200)
    run_parser.add_argument("--max-concurrency", type=int, default=10)
    run_parser.add_argument("--max-new-tokens", type=int, default=512)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)
    compare_parser.add_argument("--reference-label", default="static")
    compare_parser.add_argument("--candidate-label", default="ragged")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()

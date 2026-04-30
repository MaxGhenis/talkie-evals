"""Command line interface for Talkie evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

from talkie_evals.constants import DEFAULT_SEED, MODEL_REVISIONS
from talkie_evals.io import read_json, safe_name, timestamp, write_json
from talkie_evals.summary import markdown_table


def _add_common_remote_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        default="",
        help="JSON output path. Defaults to results/<eval>_<models>_<timestamp>.json.",
    )


def _run_arithmetic(args: argparse.Namespace) -> None:
    from talkie_evals.modal_app import app, run_arithmetic_eval

    with modal.enable_output(), app.run():
        result = run_arithmetic_eval.remote(
            model_names=args.model_names,
            task_names=args.task_names,
            sample_size=args.sample_size,
            seed=args.seed,
            log_examples=args.log_examples,
        )

    output = args.output or (
        f"results/arithmetic_eval_{safe_name(args.model_names)}_{timestamp()}.json"
    )
    path = write_json(result, output)
    print(f"Wrote {path}")
    print()
    print(markdown_table(result))


def _run_gsm8k(args: argparse.Namespace) -> None:
    from talkie_evals.modal_app import app, run_gsm8k_eval

    with modal.enable_output(), app.run():
        result = run_gsm8k_eval.remote(
            model_name=args.model_name,
            sample_size=args.sample_size,
            seed=args.seed,
            few_shot=args.few_shot,
            condition_names=args.condition_names,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    output = args.output or (
        f"results/gsm8k_eval_{safe_name(args.model_name)}_{timestamp()}.json"
    )
    path = write_json(result, output)
    print(f"Wrote {path}")
    print()
    print(markdown_table(result))


def _summarize(args: argparse.Namespace) -> None:
    result = read_json(args.path)
    print(markdown_table(result))


def _provenance(_: argparse.Namespace) -> None:
    from talkie_evals.constants import (
        ARITHMETIC_REVISION,
        GSM8K_REVISION,
        MODAL_PIP_PACKAGES,
        TALKIE_GIT_REVISION,
    )

    print(f"Talkie package git revision: {TALKIE_GIT_REVISION}")
    print(f"EleutherAI/arithmetic revision: {ARITHMETIC_REVISION}")
    print(f"openai/gsm8k revision: {GSM8K_REVISION}")
    print("Model revisions:")
    for model, revision in MODEL_REVISIONS.items():
        print(f"  {model}: {revision}")
    print("Modal image packages:")
    for package in MODAL_PIP_PACKAGES:
        print(f"  {package}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="talkie-evals",
        description="Run reproducible Talkie numeracy evaluations on Modal CUDA GPUs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    arithmetic = subparsers.add_parser(
        "arithmetic",
        help="Run EleutherAI/OpenAI arithmetic loglikelihood tasks.",
    )
    arithmetic.add_argument(
        "--model-names",
        default="talkie-1930-13b-base",
        help="Comma-separated Talkie model names.",
    )
    arithmetic.add_argument(
        "--task-names",
        default="",
        help="Comma-separated arithmetic task names. Defaults to all 10 tasks.",
    )
    arithmetic.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Rows per task. Use 0 for all 2,000 rows per task.",
    )
    arithmetic.add_argument("--log-examples", type=int, default=20)
    _add_common_remote_args(arithmetic)
    arithmetic.set_defaults(func=_run_arithmetic)

    gsm8k = subparsers.add_parser("gsm8k", help="Run GSM8K generation probes.")
    gsm8k.add_argument("--model-name", default="talkie-1930-13b-it")
    gsm8k.add_argument("--sample-size", type=int, default=50)
    gsm8k.add_argument("--few-shot", type=int, default=4)
    gsm8k.add_argument(
        "--condition-names",
        default="",
        help=(
            "Comma-separated subset of zero_shot_direct, zero_shot_cot, "
            "and <few-shot>_shot_cot."
        ),
    )
    gsm8k.add_argument("--temperature", type=float, default=0.18)
    gsm8k.add_argument("--top-k", type=int, default=20)
    _add_common_remote_args(gsm8k)
    gsm8k.set_defaults(func=_run_gsm8k)

    summarize = subparsers.add_parser("summarize", help="Print a Markdown table.")
    summarize.add_argument("path", type=Path)
    summarize.set_defaults(func=_summarize)

    provenance = subparsers.add_parser("provenance", help="Print pinned inputs.")
    provenance.set_defaults(func=_provenance)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

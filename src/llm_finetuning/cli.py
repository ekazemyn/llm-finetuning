"""Command-line entry point for reproducible project workflows."""

from __future__ import annotations

import argparse
import json

from llm_finetuning.benchmark import benchmark
from llm_finetuning.config import load_config
from llm_finetuning.data import prepare_finqa
from llm_finetuning.evaluation import evaluate
from llm_finetuning.experiments import ablate, compare
from llm_finetuning.preflight import run_preflight
from llm_finetuning.training import train


def _csv_values(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-finetune")
    parser.add_argument("--config", default="configs/experiment.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("prepare", help="Download, normalize, and persist FinQA splits")
    preflight_parser = commands.add_parser(
        "preflight", help="Check runtime and backend readiness without loading model weights"
    )
    preflight_parser.add_argument("--method", choices=("baseline", "lora", "qlora"))
    preflight_parser.add_argument("--backend", choices=("transformers", "vllm"))
    preflight_parser.add_argument("--quantization")
    train_parser = commands.add_parser("train", help="Train a LoRA or QLoRA adapter")
    train_parser.add_argument("--method", choices=("lora", "qlora"))
    train_parser.add_argument("--rank", type=int, choices=(8, 16, 32))
    train_parser.add_argument("--resume-from-checkpoint")

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate a base model or adapter")
    evaluate_parser.add_argument(
        "--method", choices=("baseline", "lora", "qlora"), default="baseline"
    )
    evaluate_parser.add_argument("--model-path")
    evaluate_parser.add_argument("--split", choices=("validation", "test"), default="test")

    compare_parser = commands.add_parser("compare", help="Compare baseline, LoRA, and QLoRA")
    compare_parser.add_argument("--methods", default="baseline,lora,qlora")

    ablate_parser = commands.add_parser("ablate", help="Run controlled rank and seed ablations")
    ablate_parser.add_argument("--methods", default="lora,qlora")
    ablate_parser.add_argument("--ranks", default="8,16,32")
    ablate_parser.add_argument("--seeds", default=None)

    benchmark_parser = commands.add_parser(
        "benchmark", help="Benchmark Transformers or vLLM inference"
    )
    benchmark_parser.add_argument("--backend", choices=("transformers", "vllm"), required=True)
    benchmark_parser.add_argument(
        "--method", choices=("baseline", "lora", "qlora"), default="baseline"
    )
    benchmark_parser.add_argument("--model-path")
    benchmark_parser.add_argument("--quantization", default="none")
    benchmark_parser.add_argument("--batch-sizes", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config).raw
    if args.command == "prepare":
        result = {"prepared_data": str(prepare_finqa(config))}
    elif args.command == "preflight":
        result = run_preflight(
            config,
            method=args.method,
            backend=args.backend,
            quantization=args.quantization,
        )
    elif args.command == "train":
        result = train(
            config,
            method=args.method,
            rank=args.rank,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    elif args.command == "evaluate":
        result = evaluate(config, method=args.method, model_path=args.model_path, split=args.split)
    elif args.command == "compare":
        result = compare(config, methods=_csv_values(args.methods, str))
    elif args.command == "ablate":
        seeds = _csv_values(args.seeds, int) if args.seeds else None
        result = ablate(
            config,
            methods=_csv_values(args.methods, str),
            ranks=_csv_values(args.ranks, int),
            seeds=seeds,
        )
    elif args.command == "benchmark":
        batch_sizes = _csv_values(args.batch_sizes, int) if args.batch_sizes else None
        result = benchmark(
            config,
            backend=args.backend,
            method=args.method,
            model_path=args.model_path,
            quantization=args.quantization,
            batch_sizes=list(batch_sizes) if batch_sizes else None,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2, default=str))
    if args.command == "preflight" and not result["ok"]:
        raise SystemExit(2)

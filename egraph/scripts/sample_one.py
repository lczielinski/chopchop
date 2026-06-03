"""Generate many equivalent programs for ONE benchmark and print the distinct ones.

Usage:
    python -m egraph.scripts.sample_one lerp.egglog --model llama7b --samples 20 --temp 0.8
"""

import argparse

from core.rewrite import rewriter
from llm.run_llm import Config, LanguageModelRunner, ModelConfig

from .run import BENCHMARKS_DIR, build_checker, load_benchmark, load_file

MODELS = {
    "llama13b": "codellama/CodeLlama-13b-Instruct-hf",
    "llama7b": "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate all equivalent programs for ONE benchmark."
    )
    parser.add_argument("benchmark", help="benchmark filename, e.g. lerp.egglog")
    parser.add_argument("--model", choices=MODELS, default="llama7b")
    parser.add_argument(
        "--samples", type=int, default=20, help="number of generation attempts"
    )
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()

    original, source = load_benchmark(args.benchmark)
    checker = build_checker(source)
    context = load_file(f"{BENCHMARKS_DIR}/context.md")
    prompt = f"The original program is:\n{original}"

    runner = LanguageModelRunner(ModelConfig(model_id=MODELS[args.model]))
    config = Config(temperature=args.temp, repetition_penalty=1.2)

    print(f"Benchmark : {args.benchmark}")
    print(f"Original  : {original}")
    print(f"Sampling  : {args.samples}x  model={args.model}  temp={args.temp}\n")

    seen: dict[str, int] = {}
    for i in range(args.samples):
        info = runner.run(config, prompt, context, realizability_checker=checker)
        ok = checker.realizable(info.output, True) if info.llm_finished else False
        out = info.output.strip()
        status = "ok " if ok else "BAD"
        marker = "new" if (ok and out not in seen) else "   "
        print(f"[{i + 1:>3}/{args.samples}] {status} {marker}")
        if ok and out not in seen:
            seen[out] = len(seen) + 1
        rewriter.clear()

    print(f"\n=== {len(seen)} distinct equivalent programs ===\n")
    for prog, n in seen.items():
        print(f"--- program {n} ---")
        print(prog)
        print()


if __name__ == "__main__":
    main()

"""Generate many equivalent programs for ONE benchmark and print the distinct ones.

Usage:
    PYTHONUNBUFFERED=1 uv run python -u -m egraph.sample_one lerp.egglog \
    --model llama13b --target 6 --samples 80 --temp 1.0 --rep-penalty 1.0 --max-tries 25 \
    --output egraph/data/lerp_programs.txt
"""

import argparse
import random
import re
from typing import Tuple

import numpy as np
import torch

from core.rewrite import rewriter
from llm.realizability import RealizabilityChecker
from llm.run_llm import Config, LanguageModelRunner, ModelConfig

from .egraph import egraph_from_egglog
from .let import let_equivalence, let_grammar, let_lexer_spec

# make everything deterministic
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

BENCHMARKS_DIR = "egraph/benchmarks"
LET_EGGLOG_PATH = "egraph/let.egglog"

MODELS = {
    "llama13b": "codellama/CodeLlama-13b-Instruct-hf",
    "llama7b": "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
}


def load_file(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


def load_benchmark(name: str) -> Tuple[str, str]:
    content = load_file(f"{BENCHMARKS_DIR}/{name}")
    source = load_file(LET_EGGLOG_PATH)

    assert content.startswith(";; ")
    original = content.splitlines()[0][3:]
    source += content + "\n(run 100)"
    return original, source


def build_checker(source: str) -> RealizabilityChecker:
    egraph = egraph_from_egglog(source, "start", "Math")
    vars = re.findall(r'Var\s*"([^"]+)"', source)
    return RealizabilityChecker(
        lambda term: let_equivalence(egraph, term, frozenset(vars)),
        let_grammar,
        let_lexer_spec,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate all equivalent programs for ONE benchmark."
    )
    parser.add_argument("benchmark", help="benchmark filename, e.g. lerp.egglog")
    parser.add_argument("--model", choices=MODELS, default="llama7b")
    parser.add_argument(
        "--samples", type=int, default=20, help="max number of generation attempts"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="stop early once this many DISTINCT programs are found",
    )
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument(
        "--rep-penalty",
        type=float,
        default=1.2,
        help="repetition penalty (paper default 1.2); lower toward 1.0 for "
        "token-heavy programs like quadratic that reuse vars/operators",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="hard cap on tokens per program (0 = unlimited)",
    )
    parser.add_argument(
        "--max-stall",
        type=int,
        default=8,
        help="abort a generation after this many tokens add no non-whitespace "
        "content (0 = off); stops the model flooding whitespace",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=40,
        help="abort after this many rejected guesses at one position (0 = off); "
        "stops endless churn when the model has a complete program but won't stop",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="write distinct programs to this file"
    )
    args = parser.parse_args()

    original, source = load_benchmark(args.benchmark)
    checker = build_checker(source)
    context = load_file(f"{BENCHMARKS_DIR}/context.md")
    prompt = f"The original program is:\n{original}"

    runner = LanguageModelRunner(ModelConfig(model_id=MODELS[args.model]))
    config = Config(
        temperature=args.temp,
        repetition_penalty=args.rep_penalty,
        verbose=True,
        max_new_tokens=args.max_tokens,
        max_stall=args.max_stall,
        max_tries=args.max_tries,
    )

    print(f"Benchmark : {args.benchmark}", flush=True)
    print(f"Original  : {original}", flush=True)
    print(
        f"Sampling  : up to {args.samples}x  target={args.target}  "
        f"model={args.model}  temp={args.temp}\n",
        flush=True,
    )

    seen: dict[str, int] = {}
    for i in range(args.samples):
        info = runner.run(config, prompt, context, realizability_checker=checker)
        ok = checker.realizable(info.output, True) if info.llm_finished else False
        out = info.output.strip()
        is_new = ok and out not in seen
        status = "ok " if ok else "BAD"
        marker = "NEW" if is_new else "   "
        print(f"[{i + 1:>3}/{args.samples}] {status} {marker}  (distinct: {len(seen) + int(is_new)})", flush=True)
        if is_new:
            seen[out] = len(seen) + 1
            print(f"--- program {seen[out]} ---\n{out}\n", flush=True)
        rewriter.clear()
        if args.target is not None and len(seen) >= args.target:
            print(f"reached target of {args.target} distinct programs.", flush=True)
            break

    print(f"\n=== {len(seen)} distinct equivalent programs ===\n", flush=True)
    for prog, n in seen.items():
        print(f"--- program {n} ---", flush=True)
        print(prog, flush=True)
        print(flush=True)

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"# benchmark: {args.benchmark}\n")
            f.write(f"# original: {original}\n")
            f.write(f"# {len(seen)} distinct equivalent programs\n\n")
            for prog, n in seen.items():
                f.write(f"--- program {n} ---\n{prog}\n\n")
        print(f"wrote {len(seen)} programs to {args.output}", flush=True)


if __name__ == "__main__":
    main()

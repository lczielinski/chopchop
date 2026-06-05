"""Generate equivalent programs for ONE target by LLM rejection sampling.

The model proposes whole programs; each is kept only if it is *realizable* under
the chosen language spec (grammatically complete and egglog-equivalent to the
target). Distinct accepted programs are printed and optionally written to a file.

Usage:
    OPENROUTER_API_KEY=... PYTHONUNBUFFERED=1 python -u -m egraph.sample \
        --lang let lerp.egglog --model opus --target 6 --samples 80 --temp 1.0 \
        --output egraph/data/lerp_programs.txt
"""

import argparse
import os

from llm.run_llm import Config, LanguageModelRunner, ModelConfig

from .language import LanguageSpec, load_target, rejection_sample
from .let import build_let_spec

BENCHMARKS_DIR = "egraph/benchmarks"

# Registry of available language specs (built lazily so unused languages don't
# pay import/parse costs).
SPECS: dict[str, callable] = {
    "let": build_let_spec,
}

# Short model names -> OpenRouter slugs.
MODELS = {
    "opus": "anthropic/claude-opus-4.8",
    "qwen-coder": "qwen/qwen-2.5-coder-32b-instruct",
    "deepseek": "deepseek/deepseek-coder",
}


def resolve_target_path(arg: str) -> str:
    """Accept either a full path or a bare benchmark filename under
    egraph/benchmarks/."""
    if os.path.exists(arg):
        return arg
    candidate = os.path.join(BENCHMARKS_DIR, arg)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"benchmark not found: {arg} (also tried {candidate})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate equivalent programs for ONE target via rejection sampling."
    )
    parser.add_argument(
        "target", help="benchmark filename (under egraph/benchmarks/) or a full path"
    )
    parser.add_argument("--lang", choices=SPECS, default="let", help="language spec")
    parser.add_argument("--model", choices=MODELS, default="opus")
    parser.add_argument(
        "--samples", type=int, default=20, help="max number of generation attempts"
    )
    parser.add_argument(
        "--target",
        dest="target_count",
        type=int,
        default=None,
        help="stop early once this many DISTINCT programs are found",
    )
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="hard cap on tokens per generated program",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="write distinct programs to this file"
    )
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="print each rejected completion and why it failed "
        "(no completion / does-not-parse / not-provably-equivalent)",
    )
    parser.add_argument(
        "--no-diverse",
        dest="diversify",
        action="store_false",
        help="disable feeding already-found programs back into the prompt to "
        "steer toward new, distinct programs (on by default)",
    )
    args = parser.parse_args()

    spec: LanguageSpec = SPECS[args.lang]()
    target = load_target(resolve_target_path(args.target))

    runner = LanguageModelRunner(ModelConfig(model_id=MODELS[args.model]))
    config = Config(
        temperature=args.temp,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
        verbose=True,
    )

    print(f"Language  : {spec.name}", flush=True)
    print(f"Target    : {args.target}", flush=True)
    print(f"Original  : {target.original}", flush=True)
    print(
        f"Sampling  : up to {args.samples}x  target={args.target_count}  "
        f"model={args.model} ({MODELS[args.model]})  temp={args.temp}\n",
        flush=True,
    )

    seen = rejection_sample(
        spec,
        target,
        runner,
        config,
        args.samples,
        args.target_count,
        show_rejected=args.show_rejected,
        diversify=args.diversify,
    )

    print(f"\n=== {len(seen)} distinct equivalent programs ===\n", flush=True)
    for prog, n in seen.items():
        print(f"--- program {n} ---", flush=True)
        print(prog, flush=True)
        print(flush=True)

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"# benchmark: {args.target}\n")
            f.write(f"# original: {target.original}\n")
            f.write(f"# {len(seen)} distinct equivalent programs\n\n")
            for prog, n in seen.items():
                f.write(f"--- program {n} ---\n{prog}\n\n")
        print(f"wrote {len(seen)} programs to {args.output}", flush=True)


if __name__ == "__main__":
    main()

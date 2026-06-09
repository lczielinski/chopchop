"""Generate code that is numerically equivalent to a reference program.

For a benchmark (a reference program plus egglog equivalence rules), this builds an
egraph-constrained realizability checker and uses it to constrain LLM decoding, so every
sampled program is provably equivalent to the reference under the rewrite rules.
"""

import argparse
import random
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import torch

from core.rewrite import rewriter
from llm.realizability import RealizabilityChecker
from llm.run_llm import Config, LanguageModelRunner, ModelConfig

from .egraph import (
    egraph_from_egglog,
    in_egraph,
)
from .fpcore import (
    SATURATION_RUNS,
    fpcore_equivalence,
    fpcore_grammar,
    fpcore_lexer_spec,
)

EGRAPH_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = EGRAPH_DIR / "benchmarks"
LET_EGGLOG_PATH = EGRAPH_DIR / "let.egglog"


class BenchmarkResult(TypedDict):
    benchmark: str
    reference: str
    programs: list[str]


VALID_MODELS = {
    # strong open-weight code models (sizes are bf16 weight footprints)
    "qwen14b": "Qwen/Qwen2.5-Coder-14B-Instruct",  # ~29 GB
    "qwen7b": "Qwen/Qwen2.5-Coder-7B-Instruct",  # ~15 GB
    "codestral": "mistralai/Codestral-22B-v0.1",  # ~44 GB (gated on HF)
    "deepseek-v2": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",  # ~31 GB (MoE)
    # smaller originals
    "llama13b": "codellama/CodeLlama-13b-Instruct-hf",
    "llama7b": "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
}


def load_file(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8")


def get_benchmark_names() -> list[str]:
    return sorted(path.name for path in BENCHMARKS_DIR.glob("*.egglog"))


def load_benchmark(name: str) -> tuple[str, str]:
    """Return the reference program and the full egglog source for a benchmark.

    The first line of a benchmark file is the human-readable reference program (a `;;`
    comment); the rest is appended to the base rules in let.egglog.
    """
    content = load_file(BENCHMARKS_DIR / name)
    source = load_file(LET_EGGLOG_PATH)

    if not content.startswith(";; "):
        raise ValueError(f"Benchmark {name!r} must start with a ';; ' reference line.")
    original = content.splitlines()[0][3:]
    source += content + f"\n(run {SATURATION_RUNS})"
    return original, source


def canonical(program: str) -> str:
    """Canonical form for distinctness checks: just collapse whitespace.

    Programs are plain arithmetic s-expressions (no `let` bindings), so two outputs are
    "the same" only if they differ purely in formatting.
    """
    return re.sub(r"\s+", " ", program).strip()


# Guards against degenerate cyclic-rule towers during decoding (see RealizabilityChecker).
# The deepest benchmark nests ~9 levels, so 18 leaves headroom for real rewrites; legitimate
# checks are sub-second, so a 5s cap only catches runaway ones; the timeout is skipped for
# shallow prefixes, which are legitimately slow but realizable.
MAX_DEPTH = 18
CHECK_TIMEOUT = 5.0
TIMEOUT_MIN_DEPTH = 5


def build_checker(source: str) -> RealizabilityChecker:
    """Build the egraph-constrained checker for a benchmark's egglog source."""
    egraph = egraph_from_egglog(source, "start", "Math")
    # Pre-warm the cached e-graph index so it is not charged to the first token.
    in_egraph(egraph)
    return RealizabilityChecker(
        lambda term: fpcore_equivalence(egraph, term),
        fpcore_grammar,
        fpcore_lexer_spec,
        max_depth=MAX_DEPTH,
        timeout=CHECK_TIMEOUT,
        timeout_min_depth=TIMEOUT_MIN_DEPTH,
    )


def build_prompt(
    original: str,
    prior: list[str],
) -> str:
    """Prompt for one attempt.

    The programs already produced are listed back to the model, with an
    instruction to produce another that is algebraically equivalent but evaluates
    with different floating-point rounding (a genuinely different rewrite, not a
    reordering of commutative operands)."""
    prompt = (
        f"The original program is:\n{original}\n\n"
        "Produce one complete FPCore program that is algebraically equivalent to the "
        "original but evaluates with different floating-point behavior. Prefer a real "
        "rounding-changing rewrite, not a commutative reordering."
    )
    if prior:
        listed = "\n".join(prior)
        prompt += (
            "\n\nYou have already produced these equivalent programs:\n"
            f"{listed}\n\n"
            "Produce another program that is algebraically equivalent to the original "
            "but evaluates with different floating-point behavior. Avoid minor variants "
            "of the programs already produced; choose a different structural rewrite "
            "family when possible. Useful rounding-changing rewrites include: re-associate "
            "a sum or product, rewrite a division as multiplication by a reciprocal "
            "(`(/ x y)` -> `(* x (/ 1 y))`), split a fraction over a sum or difference, "
            "split a quotient of products, expand a squared variable (`(pow v 2)` -> "
            "`(* v v)`), distribute a product over a sum or difference, or rationalize a "
            "`(+ (- b) (sqrt d))` numerator by its conjugate. Keep sums written as sums in "
            "the same orientation and do not factor; do not merely reorder the operands "
            "of a commutative operator."
        )
    return prompt


def run_benchmark(
    name: str,
    get_runner: Callable[[], LanguageModelRunner],
    config: Config,
    context: str,
    num_programs: int,
    max_tries: int,
    stream: bool,
) -> BenchmarkResult:
    """Generate up to `num_programs` distinct equivalent programs for one benchmark.

    Samples full programs (up to `max_tries` attempts) and keeps the distinct ones the
    egraph checker confirms are equivalent to the reference. Every generated program is
    equivalent by construction; the re-check is a final guard.
    """
    original, source = load_benchmark(name)
    checker = build_checker(source)

    print(f"\n=== {name} ===")
    print(f"reference: {original}")

    programs: list[str] = []
    seen: set[str] = set()  # canonical forms of accepted programs

    def try_accept(program: str) -> str:
        key = canonical(program)
        if key in seen:
            return "duplicate"
        if not checker.realizable(program, True):
            return "not equivalent"
        seen.add(key)
        programs.append(program)
        return f"accepted ({len(programs)}/{num_programs})"

    attempts = 0
    while len(programs) < num_programs and attempts < max_tries:
        attempts += 1
        print(
            f"\n[attempt {attempts}/{max_tries}] ",
            end="",
            flush=True,
        )
        prompt = build_prompt(original, programs)
        run_info = get_runner().run(
            config,
            prompt,
            context,
            realizability_checker=checker,
            stream=stream,
        )
        if not stream:
            print(run_info.output, end="", flush=True)

        if not run_info.llm_finished:
            reason = (
                "timed out" if run_info.timed_out else "too long, no equivalent program"
            )
            print(f"  -> did not finish ({reason})")
        else:
            print(f"  -> {try_accept(run_info.output)}")

    print(f"\nGenerated {len(programs)} distinct program(s) in {attempts} attempt(s):")
    for i, program in enumerate(programs, 1):
        print(f"  {i}. {program}")
    return {"benchmark": name, "reference": original, "programs": programs}


def resolve_benchmarks(name: str | None) -> list[str]:
    names = get_benchmark_names()
    if name is None:
        return names
    if not name.endswith(".egglog"):
        name += ".egglog"
    if name not in names:
        raise SystemExit(f"Unknown benchmark {name!r}. Available: {', '.join(names)}")
    return [name]


def format_settings(
    args: argparse.Namespace,
    model_config: ModelConfig,
    config: Config,
    now: datetime,
) -> str:
    """Header block recording the settings used for a run."""
    return "\n".join(
        [
            "# ChopChop egraph run",
            f"# timestamp: {now:%Y-%m-%d %H:%M:%S}",
            f"# model: {args.model} ({model_config.model_id})",
            f"# device: {model_config.device}",
            f"# dtype: {model_config.dtype}",
            f"# temperature: {config.temperature}",
            f"# top_p: {config.top_p}",
            f"# repetition_penalty: {config.repetition_penalty}",
            f"# max_new_tokens: {config.max_new_tokens}",
            f"# max_token_tries: {config.max_token_tries}",
            f"# num_programs: {args.num_programs}",
            f"# max_tries: {args.max_tries}",
        ]
    )


def format_block(result: BenchmarkResult) -> str:
    """Render one benchmark's reference and accepted programs as text."""
    lines = [f"=== {result['benchmark']} ===", f"reference: {result['reference']}", ""]
    if not result["programs"]:
        lines.append("(no equivalent program found)")
    for i, program in enumerate(result["programs"], 1):
        lines.append(f"{i}.")
        lines.append(program)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate code numerically equivalent to a reference program "
        "using egraph-constrained decoding."
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark to run (e.g. quadratic.egglog). If omitted, runs all benchmarks.",
    )
    parser.add_argument(
        "--model",
        choices=VALID_MODELS.keys(),
        default="qwen14b",
        help="Which model to use (default: qwen14b).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8). Higher gives more varied attempts, which "
        "helps the model land on a recognized rewrite.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling cutoff (default: 0.9). Use 1.0 to disable tail filtering.",
    )
    parser.add_argument(
        "--num-programs",
        "-n",
        type=int,
        default=1,
        help="Number of distinct equivalent programs to generate per benchmark (default: 1).",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=25,
        help="Maximum generation attempts per benchmark (default: 25). The accepted-rewrite "
        "class is narrow, so more attempts materially raise the hit rate.",
    )
    parser.add_argument(
        "--max-token-tries",
        type=int,
        default=256,
        help="Abort one LLM attempt after this many rejected token proposals at the same "
        "prefix (default: 256).",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream tokens as they are generated (default: on; use --no-stream to disable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for per-run output files (default: outputs/).",
    )
    args = parser.parse_args()

    # make sampling deterministic
    torch.manual_seed(0)
    random.seed(0)

    model_config = ModelConfig(model_id=VALID_MODELS[args.model])
    runner: LanguageModelRunner | None = None

    def get_runner() -> LanguageModelRunner:
        nonlocal runner
        if runner is None:
            runner = LanguageModelRunner(model_config)
        return runner

    config = Config(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.2,
        max_token_tries=args.max_token_tries,
    )
    context = load_file(BENCHMARKS_DIR / "context.md")

    now = datetime.now()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"run_{now:%Y%m%d_%H%M%S}.txt"
    header = format_settings(args, model_config, config, now)

    results = []
    for name in resolve_benchmarks(args.benchmark):
        results.append(
            run_benchmark(
                name,
                get_runner,
                config,
                context,
                args.num_programs,
                args.max_tries,
                args.stream,
            )
        )
        # rewrite the file after each benchmark so partial results survive an interruption
        blocks = "\n".join(format_block(r) for r in results)
        output_path.write_text(f"{header}\n\n{blocks}", encoding="utf-8")
        rewriter.clear()

    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()

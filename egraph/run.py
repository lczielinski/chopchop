"""Generate code that is numerically equivalent to a reference program.

For a benchmark (a reference program plus egglog equivalence rules), this builds an
egraph-constrained realizability checker and uses it to constrain LLM decoding, so every
sampled program is provably equivalent to the reference under the rewrite rules.
"""

import argparse
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch

from core.rewrite import rewriter
from llm.realizability import RealizabilityChecker
from llm.run_llm import Config, LanguageModelRunner, ModelConfig

from .egraph import egraph_from_egglog
from .let import let_equivalence, let_grammar, let_lexer_spec

BENCHMARKS_DIR = "egraph/benchmarks"
LET_EGGLOG_PATH = "egraph/let.egglog"

VALID_MODELS = {
    "llama13b": "codellama/CodeLlama-13b-Instruct-hf",
    "llama7b": "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
}


def load_file(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


def get_benchmark_names() -> list[str]:
    return sorted(f.name for f in Path(BENCHMARKS_DIR).glob("*.egglog"))


def load_benchmark(name: str) -> Tuple[str, str]:
    """Return the reference program and the full egglog source for a benchmark.

    The first line of a benchmark file is the human-readable reference program (a `;;`
    comment); the rest is appended to the base rules in let.egglog.
    """
    content = load_file(f"{BENCHMARKS_DIR}/{name}")
    source = load_file(LET_EGGLOG_PATH)

    assert content.startswith(";; ")
    original = content.splitlines()[0][3:]
    source += content + "\n(run 100)"
    return original, source


def canonical(program: str) -> str:
    """Canonical form for distinctness checks.

    Collapses whitespace and renames let-bound variables to positional placeholders, so
    programs that differ only by binding names or formatting (e.g. `let d = E in sqrt d`
    vs `let distance = E in sqrt distance`) compare equal. Genuinely different refactorings
    keep different canonical forms. (Bindings are top-level and names unique in this
    grammar, so a token-level rename is sufficient.)
    """
    text = re.sub(r"\s+", " ", program).strip()
    bound = re.findall(r"\blet\s+([A-Za-z_]\w*)\s*=", text)
    renames = {name: f"$v{i}" for i, name in enumerate(bound)}
    return re.sub(r"\b[A-Za-z_]\w*\b", lambda m: renames.get(m.group(), m.group()), text)


def build_checker(source: str) -> RealizabilityChecker:
    """Build the egraph-constrained checker for a benchmark's egglog source."""
    egraph = egraph_from_egglog(source, "start", "Math")
    vars = re.findall(r'Var\s*"([^"]+)"', source)
    return RealizabilityChecker(
        lambda term: let_equivalence(egraph, term, frozenset(vars)),
        let_grammar,
        let_lexer_spec,
    )


def run_benchmark(
    name: str,
    runner: LanguageModelRunner,
    config: Config,
    context: str,
    num_programs: int,
    max_tries: int,
    stream: bool,
) -> dict:
    """Generate up to `num_programs` distinct equivalent programs for one benchmark.

    Samples full programs (up to `max_tries` attempts) and keeps the distinct ones the
    egraph checker confirms are equivalent to the reference. Every generated program is
    equivalent by construction; the re-check is a final guard.
    """
    original, source = load_benchmark(name)
    checker = build_checker(source)
    prompt = f"The original program is:\n{original}"

    print(f"\n=== {name} ===")
    print(f"reference: {original}")

    programs: list[str] = []
    seen: set[str] = set()  # canonical forms of accepted programs
    attempts = 0
    while len(programs) < num_programs and attempts < max_tries:
        attempts += 1
        print(f"\n[attempt {attempts}/{max_tries}] ", end="", flush=True)
        run_info = runner.run(
            config, prompt, context, realizability_checker=checker, stream=stream
        )
        if not stream:
            print(run_info.output, end="", flush=True)

        if not run_info.llm_finished:
            reason = "timed out" if run_info.timed_out else "too long, no equivalent program"
            print(f"  -> did not finish ({reason})")
        elif not checker.realizable(run_info.output, True):
            print("  -> not equivalent")
        elif canonical(run_info.output) in seen:
            print("  -> duplicate")
        else:
            seen.add(canonical(run_info.output))
            programs.append(run_info.output)
            print(f"  -> accepted ({len(programs)}/{num_programs})")

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
        raise SystemExit(
            f"Unknown benchmark {name!r}. Available: {', '.join(names)}"
        )
    return [name]


def format_settings(args, model_config: ModelConfig, config: Config, now) -> str:
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
            f"# num_programs: {args.num_programs}",
            f"# max_tries: {args.max_tries}",
        ]
    )


def format_block(result: dict) -> str:
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
        default="llama7b",
        help="Which model to use (default: llama7b).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (default: 0.5). Use >0 to get distinct programs.",
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
        default=10,
        help="Maximum generation attempts per benchmark (default: 10).",
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
    runner = LanguageModelRunner(model_config)
    config = Config(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.2,
    )
    context = load_file(f"{BENCHMARKS_DIR}/context.md")

    now = datetime.now()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"run_{now:%Y%m%d_%H%M%S}.txt"
    header = format_settings(args, model_config, config, now)

    results = []
    for name in resolve_benchmarks(args.benchmark):
        results.append(
            run_benchmark(
                name,
                runner,
                config,
                context,
                args.num_programs,
                args.max_tries,
                args.stream,
            )
        )
        # rewrite the file after each benchmark so partial results survive an interruption
        blocks = "\n".join(format_block(r) for r in results)
        output_path.write_text(f"{header}\n\n{blocks}")
        rewriter.clear()

    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()

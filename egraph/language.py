"""A general LLM rejection-sampling framework.

A :class:`LanguageSpec` bundles everything needed to sample programs in some
language: a grammar (a `.lark` source plus the Python constructor classes its
actions reference), a set of egglog rules defining program equivalence, a system
prompt, and a constraint that decides which programs are accepted.

Given a :class:`LanguageSpec` and a :class:`Target` (one program to refactor),
:func:`rejection_sample` repeatedly asks the model for a whole program and keeps
the distinct ones that are *realizable* — i.e. grammatically complete and
egglog-equivalent to the target.
"""

from dataclasses import dataclass, field
from typing import Callable

from egglog.bindings import EGraph

from core.grammar import Application, TreeGrammar
from core.lark.from_lark import parse_attribute_grammar
from core.rewrite import rewriter
from llm.realizability import RealizabilityChecker
from llm.run_llm import Config, LanguageModelRunner

from .egraph import egraph_from_egglog, in_egraph

Constraint = Callable[[TreeGrammar], TreeGrammar]
ConstraintFactory = Callable[[EGraph, str], Constraint]


def _default_constraint_factory(egraph: EGraph, source: str) -> Constraint:
    """The generic equivalence constraint: the program must lie in the target's
    egglog equivalence class."""
    return in_egraph(egraph)


@dataclass(frozen=True)
class LanguageSpec:
    """Everything that defines one language / equivalence problem."""

    name: str
    grammar_source: str  # `.lark` text
    constructors: list[type[Application]]  # AST classes the grammar references
    start_symbol: str  # grammar start rule, e.g. "let"
    egglog_rules: str  # datatype + rewrite rules (the `(run N)` is added per target)
    egglog_start: str  # name of the start symbol in a target, e.g. "start"
    egglog_start_type: str  # its datatype, e.g. "Math"
    context: str  # system prompt
    constraint_factory: ConstraintFactory = field(
        default=_default_constraint_factory
    )
    saturation_depth: int = 100  # egglog `(run N)` depth
    diversify_hint: str = ""  # extra instruction appended when asking for a
    # program different from those already found (see `rejection_sample`)


@dataclass(frozen=True)
class Target:
    """One program to find equivalents of."""

    original: str  # human-readable original program (shown to the model)
    egglog: str  # egglog source defining the `start` symbol


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence (``` or ```lang … ```), which
    models often add despite being asked for bare code. The grammar parses plain
    programs, so fenced output would otherwise be rejected as unparseable."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()[1:]  # drop the opening ``` / ```lang line
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]  # drop the closing fence
    return "\n".join(lines).strip()


def load_target(path: str) -> Target:
    """Load a benchmark file: a `;; <original>` header line followed by egglog
    defining the target program."""
    with open(path, "r") as f:
        content = f.read()
    assert content.startswith(";; "), (
        f"benchmark {path} must start with a ';; <original program>' header"
    )
    original = content.splitlines()[0][3:]
    return Target(original=original, egglog=content)


def build_checker(spec: LanguageSpec, target: Target) -> RealizabilityChecker:
    """Build the realizability checker (the accept/reject predicate) for a
    target under a language spec."""
    lexer_spec, grammar = parse_attribute_grammar(
        spec.constructors, spec.grammar_source, spec.start_symbol
    ).build_parser()
    source = f"{spec.egglog_rules}\n{target.egglog}\n(run {spec.saturation_depth})"
    egraph = egraph_from_egglog(source, spec.egglog_start, spec.egglog_start_type)
    constraint = spec.constraint_factory(egraph, source)
    return RealizabilityChecker(constraint, grammar, lexer_spec)


def build_grammar_checker(spec: LanguageSpec) -> RealizabilityChecker:
    """A grammar-only checker (identity constraint): tests whether output is a
    complete, parseable program, ignoring equivalence. Used to explain why a
    program was rejected (bad syntax vs. not provably equivalent)."""
    lexer_spec, grammar = parse_attribute_grammar(
        spec.constructors, spec.grammar_source, spec.start_symbol
    ).build_parser()
    return RealizabilityChecker(lambda g: g, grammar, lexer_spec)


def rejection_sample(
    spec: LanguageSpec,
    target: Target,
    runner: LanguageModelRunner,
    config: Config,
    samples: int,
    target_count: int | None = None,
    show_rejected: bool = False,
    diversify: bool = True,
) -> dict[str, int]:
    """Generate whole programs and keep the distinct realizable ones.

    Returns a mapping from each distinct accepted program to its 1-based index.
    When ``show_rejected`` is set, each rejected completion is printed along with
    why it failed (no completion / does-not-parse / parses-but-not-equivalent).
    When ``diversify`` is set, the programs found so far are fed back into the
    prompt and the model is asked for a *different* one — this is the main lever
    for broadening the set of distinct programs.
    """
    checker = build_checker(spec, target)
    grammar_checker = build_grammar_checker(spec) if show_rejected else None
    base_prompt = f"The original program is:\n{target.original}"

    seen: dict[str, int] = {}
    for i in range(samples):
        prompt = base_prompt
        if diversify and seen:
            already = "\n\n".join(f"{n}.\n{prog}" for prog, n in seen.items())
            prompt = (
                f"{base_prompt}\n\n"
                f"You have already produced these equivalent programs:\n\n"
                f"{already}\n\n"
                f"Produce a NEW equivalent program that is meaningfully "
                f"different from every one above."
            )
            if spec.diversify_hint:
                prompt += f" {spec.diversify_hint}"
        info = runner.run(config, prompt, spec.context)
        out = _strip_code_fence(info.output)
        ok = checker.realizable(out, True) if info.llm_finished else False
        is_new = ok and out not in seen
        status = "ok " if ok else "BAD"
        marker = "NEW" if is_new else "   "
        print(
            f"[{i + 1:>3}/{samples}] {status} {marker}  "
            f"(distinct: {len(seen) + int(is_new)})",
            flush=True,
        )
        if is_new:
            seen[out] = len(seen) + 1
            print(f"--- program {seen[out]} ---\n{out}\n", flush=True)
        elif not ok and grammar_checker is not None:
            if not info.llm_finished:
                reason = "no completion (request failed or hit token cap)"
            elif grammar_checker.realizable(out, True):
                reason = "parses but NOT provably equivalent (add egglog rules?)"
            else:
                reason = "does NOT parse (bad syntax / stray prose)"
            print(f"      rejected: {reason}\n      {out!r}", flush=True)

        # Clear the rewrite-system memoization between attempts.
        rewriter.clear()

        if target_count is not None and len(seen) >= target_count:
            print(f"reached target of {target_count} distinct programs.", flush=True)
            break

    return seen

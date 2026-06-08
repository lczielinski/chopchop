from core.rewrite import rewrite
from core.grammar import TreeGrammar, EmptySet, Application, Union, ASTLeaf
from core.lark.from_lark import parse_attribute_grammar
from importlib.resources import files
from .egraph import EGraph, in_egraph
from .fpcore_abstract_syntax import (
    FPCore,
    Var,
    Num,
    constructors,
)


fpcore_source = files(__package__).joinpath("fpcore.lark").read_text()
fpcore_lexer_spec, fpcore_grammar = parse_attribute_grammar(
    constructors, fpcore_source, "fpcore"
).build_parser()

# Iterations of equality saturation to run. The conjugate-rationalization rules in
# let.egglog (which enable the Citardauq form) make saturation NON-terminating: the
# equivalence class keeps growing and `(run 100)` explodes (minutes, then OOM). Capping
# at a few iterations bounds the e-graph while still reaching deep rewrites — the
# Citardauq derivation is found in 6 iterations; the blow-up starts around 12. This
# trades completeness (some valid-but-deep equivalences may be missed → rejected, never
# wrongly accepted) for termination. See egraph/probe.py for the measurements.
SATURATION_RUNS = 8


def expr_to_egglog(expr: TreeGrammar) -> str:
    """Translate a concrete arithmetic expression into an egglog s-expression.

    Only the let-free arithmetic fragment reaches this function: the ``FPCore``
    wrapper and ``let`` bindings are peeled off by ``fpcore_equivalence`` before
    any term is handed to egglog. The arithmetic node names (Add, Sub, Mul, Div,
    Neg, Sqrt, Pow) match the egglog ``Math`` datatype, so they translate
    generically.
    """
    match expr:
        case Var(ASTLeaf(prefix=name)):
            return f'(Var "{name}")'
        case Num(ASTLeaf(prefix=name)):
            return f"(Num {name})"
        case Application(children):
            egglog_children = " ".join(expr_to_egglog(child) for child in children)
            return f"({expr.constructor} {egglog_children})"
        case _:
            raise ValueError(f"Unable to process expression: {expr}")


@rewrite
def fpcore_equivalence(egraph: EGraph, t: TreeGrammar) -> TreeGrammar:
    """Prune a FPCore program space down to programs equivalent to the reference.

    The ``FPCore`` wrapper is peeled off (its argument list is left unconstrained)
    and the arithmetic body is intersected with the egraph.
    """
    match t:
        case EmptySet():
            return EmptySet()
        case Union(children):
            return Union.of(fpcore_equivalence(egraph, child) for child in children)
        case FPCore(args, body):
            # The declared arguments are free; only the body is constrained.
            return FPCore.of(args, fpcore_equivalence(egraph, body))
        case _:
            return in_egraph(egraph)(t)

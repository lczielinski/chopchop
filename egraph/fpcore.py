from core.rewrite import rewrite
from core.grammar import TreeGrammar, EmptySet, Application, Union, ASTLeaf, as_tree
from core.lark.from_lark import parse_attribute_grammar
from typing import Optional
from functools import lru_cache
from importlib.resources import files
from .egraph import EGraph, in_egraph
from .fpcore_abstract_syntax import (
    FPCore,
    Let,
    Bindings,
    Binding,
    Var,
    Num,
    constructors,
)


fpcore_source = files(__package__).joinpath("fpcore.lark").read_text()
fpcore_lexer_spec, fpcore_grammar = parse_attribute_grammar(
    constructors, fpcore_source, "fpcore"
).build_parser()


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


@lru_cache(maxsize=None)
def update_egraph(
    egraph: EGraph, var: TreeGrammar, value: TreeGrammar, saturation_depth=100
) -> EGraph:
    """Return a new egraph that additionally knows ``value`` is equal to ``var``.

    This is how a ``let`` binding is folded in: we add a rewrite teaching the
    egraph that occurrences of the bound expression collapse onto the variable,
    then re-saturate.
    """
    new_egraph = EGraph(record=True)
    ran_commands = egraph.commands()
    assert ran_commands is not None, "got EGraph with record=False"
    lines = [
        line
        for line in ran_commands.splitlines()
        if not line.startswith("(run-schedule")
    ]
    new_egraph.run_program(*new_egraph.parse_program("\n".join(lines)))

    # build egglog rewrite: the bound value rewrites to the variable
    var_egglog = expr_to_egglog(var)
    value_egglog = expr_to_egglog(value)
    rewrite_str = f"(rewrite {value_egglog} {var_egglog})"

    # run the commands and saturate the egraph
    saturate_str = f"(run {saturation_depth})"
    new_commands = new_egraph.parse_program(rewrite_str + "\n" + saturate_str)
    new_egraph.run_program(*new_commands)
    return new_egraph


@rewrite
def fpcore_equivalence(
    egraph: EGraph, t: TreeGrammar, used_names: Optional[frozenset[str]] = None
) -> TreeGrammar:
    """Prune a FPCore program space down to programs equivalent to the reference.

    The ``FPCore`` wrapper is peeled off (its argument list is left unconstrained),
    each ``let``/``let*`` binding is folded into the egraph, and the remaining
    arithmetic body is intersected with the egraph.
    """
    if used_names is None:
        used_names = frozenset()
    match t:
        case EmptySet():
            return EmptySet()
        case Union(children):
            return Union.of(
                fpcore_equivalence(egraph, child, used_names) for child in children
            )
        case FPCore(args, body):
            # The declared arguments are free; only the body is constrained.
            return FPCore.of(args, fpcore_equivalence(egraph, body, used_names))
        case Let(bindings, body):
            bindings_tree = as_tree(bindings)
            if bindings_tree is None:
                # bindings not yet fully decoded; defer the constraint
                return t
            current_egraph = egraph
            names = used_names
            node: TreeGrammar = bindings_tree
            # `bindings` is a non-empty list: a cons (Bindings) for >1 binding,
            # or a bare Binding for the last/only one.
            while True:
                match node:
                    case Bindings(
                        Binding(Var(ASTLeaf(prefix=name)) as var, value), rest
                    ):
                        if name in names:
                            return EmptySet()
                        current_egraph = update_egraph(current_egraph, var, value)
                        names = names | {name}
                        node = rest
                    case Binding(Var(ASTLeaf(prefix=name)) as var, value):
                        if name in names:
                            return EmptySet()
                        current_egraph = update_egraph(current_egraph, var, value)
                        names = names | {name}
                        break
                    case _:
                        return EmptySet()
            return fpcore_equivalence(current_egraph, body, names)
        case _:
            return in_egraph(egraph)(t)

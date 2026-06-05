import re
from core.rewrite import rewrite
from core.grammar import TreeGrammar, EmptySet, Application, Union, ASTLeaf, as_tree
from core.lark.from_lark import parse_attribute_grammar
from typing import Optional
from .egraph import EGraph, in_egraph
from functools import lru_cache
from importlib.resources import files
from .language import LanguageSpec
from .let_abstract_syntax import Let, Var, Num, constructors


let_source = files(__package__).joinpath("let.lark").read_text()
let_lexer_spec, let_grammar = parse_attribute_grammar(
    constructors, let_source, "let"
).build_parser()
_, code_block_grammar = parse_attribute_grammar(
    constructors, let_source, "codeblock"
).build_parser()


def expr_to_egglog(expr: TreeGrammar) -> str:
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
    egraph: EGraph, binding: TreeGrammar, expr: TreeGrammar, saturation_depth=100
) -> EGraph:
    new_egraph = EGraph(record=True)
    ran_commands = egraph.commands()
    assert ran_commands is not None, "got EGraph with record=False"
    lines = [
        line
        for line in ran_commands.splitlines()
        if not line.startswith("(run-schedule")
    ]
    new_egraph.run_program(*new_egraph.parse_program("\n".join(lines)))

    # build egglog rewrite
    binding_egglog = expr_to_egglog(binding)
    expr_egglog = expr_to_egglog(expr)
    rewrite_str = f"(rewrite {expr_egglog} {binding_egglog})"

    # run the commands and saturate the egraph
    saturate_str = f"(run {saturation_depth})"
    new_commands = new_egraph.parse_program(rewrite_str + "\n" + saturate_str)
    new_egraph.run_program(*new_commands)
    return new_egraph


@rewrite
def let_equivalence(
    egraph: EGraph, t: TreeGrammar, used_names: Optional[frozenset[str]] = None
) -> TreeGrammar:
    if used_names is None:
        used_names = frozenset()
    match t:
        case EmptySet():
            return EmptySet()
        case Union(children):
            return Union.of(
                let_equivalence(egraph, child, used_names) for child in children
            )
        case Let(var, binding, expr):
            var_tree = as_tree(var)
            binding_tree = as_tree(binding)
            match var_tree:
                case Var(ASTLeaf(prefix=name)):
                    if name in used_names:
                        return EmptySet()
                    if binding_tree:
                        return let_equivalence(
                            update_egraph(egraph, var_tree, binding_tree),
                            expr,
                            used_names | {name},
                        )
            return t
        case _:
            return in_egraph(egraph)(t)


def _let_constraint_factory(egraph: EGraph, source: str):
    """Equivalence under the `let` language: respects let-binding semantics and
    forbids reusing variable names already defined in the target program."""
    used_names = frozenset(re.findall(r'Var\s*"([^"]+)"', source))
    return lambda term: let_equivalence(egraph, term, used_names)


def build_let_spec() -> LanguageSpec:
    """The reference :class:`LanguageSpec`: a small functional language with
    arithmetic and let-bindings."""
    return LanguageSpec(
        name="let",
        grammar_source=let_source,
        constructors=constructors,
        start_symbol="let",
        egglog_rules=files(__package__).joinpath("let.egglog").read_text(),
        egglog_start="start",
        egglog_start_type="Math",
        context=files(f"{__package__}.benchmarks")
        .joinpath("context.md")
        .read_text(),
        constraint_factory=_let_constraint_factory,
        diversify_hint=(
            "Explore a structurally different refactoring: introduce or remove "
            "let-bindings, name different subexpressions, reorder commutative "
            "operations (+ and *), or factor/inline shared terms. Output only code."
        ),
    )

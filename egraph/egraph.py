import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable

from egglog.bindings import EGraph

from core.grammar import Application, TreeGrammar, EmptySet, Union, ASTLeaf
from core.rewrite import rewrite

# egglog routes Rust logs through Python's logging (pyo3-log). It warns about `let` globals
# that are not prefixed with `$`; our benchmarks intentionally use plain names, so quiet
# warning-level messages from egglog while leaving errors visible.
logging.getLogger("egglog").setLevel(logging.ERROR)

START_RELATION = "__start__"  # dummy relation for the start symbol


@dataclass(frozen=True)
class ENode:
    op: str
    children: tuple[str, ...]


EClassMapping = dict[str, set[ENode]]  # maps eclass names to contained enodes


def root_and_eclass_mapping(egraph: EGraph) -> tuple[str, EClassMapping]:
    """
    Extracts the root eclass and a mapping of eclasses to their contained ENodes.
    """
    # Hack to work around egglog python not providing a way to iterate over eclasses.
    egraph_json = json.loads(egraph.serialize([]).to_json())
    nodes = egraph_json["nodes"]
    root_eclass: str | None = None
    eclasses: defaultdict[str, set[ENode]] = defaultdict(set)

    for node_data in nodes.values():
        eclass, op = node_data["eclass"], node_data["op"]
        if op.startswith('"') and op.endswith('"'):
            op = op[1:-1]
            # TODO: this is a hack to unescape variable names
            # should take formating function describing conversion
            # from egglog asts to our asts
        children_eclasses = tuple(
            nodes[child]["eclass"] for child in node_data["children"]
        )
        if op == START_RELATION:
            root_eclass = children_eclasses[0]
        else:
            eclasses[eclass].add(ENode(op, children_eclasses))
    if root_eclass is None:
        raise ValueError("No start relation found in the egraph.")
    return root_eclass, dict(eclasses)


@lru_cache(maxsize=None)
def in_egraph(egraph: EGraph) -> Callable[[TreeGrammar], TreeGrammar]:
    """
    Given an egraph, returns a predicate on TreeGrammars that computes the intersection
    of the grammar with the egraph.
    """
    root_eclass, eclasses = root_and_eclass_mapping(egraph)

    @rewrite
    def in_eclass(eclass: str, t: TreeGrammar) -> TreeGrammar:
        match t:
            case EmptySet():
                return EmptySet()
            case Union(children):
                return Union.of(in_eclass(eclass, child) for child in children)
            case ASTLeaf(prefix=prefix, is_complete=True):
                matches_constant = any(
                    enode.op == prefix and not enode.children
                    for enode in eclasses.get(eclass, ())
                )
                return t if matches_constant else EmptySet()
            case ASTLeaf(prefix=prefix, token_regex=token_regex, is_complete=False):
                matches_constant = any(
                    not enode.children
                    and enode.op.startswith(prefix)
                    and token_regex.fullmatch(enode.op, partial=True)
                    for enode in eclasses.get(eclass, ())
                )
                return t if matches_constant else EmptySet()
            case Application(children):
                matches = []
                for enode in eclasses.get(eclass, ()):
                    if t.constructor != enode.op or len(enode.children) != len(
                        children
                    ):
                        continue
                    matches.append(
                        t.of(
                            [
                                in_eclass(child_eclass, child)
                                for child_eclass, child in zip(enode.children, children)
                            ],
                        )
                    )
                return Union.of(matches)
            case _:
                raise ValueError

    return partial(in_eclass, root_eclass)


def egraph_from_egglog(egglog_source: str, start: str, start_type: str) -> EGraph:
    if "(run" not in egglog_source:
        raise ValueError("egglog source must contain a `(run ...)` command.")
    egglog_source += f"\n(relation {START_RELATION} ({start_type}))"
    egglog_source += f"\n({START_RELATION} {start})"
    egraph = EGraph(record=True)
    commands = egraph.parse_program(egglog_source)
    egraph.run_program(*commands)
    return egraph

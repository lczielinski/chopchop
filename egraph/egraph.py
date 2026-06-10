import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable

import networkx as nx
from egglog.bindings import EGraph

from core.grammar import Application, TreeGrammar, EmptySet, Union, ASTLeaf
from core.rewrite import rewrite

logging.getLogger("egglog").setLevel(logging.ERROR)
START_RELATION = "__start__"


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


def strip_identity_enodes(eclasses: EClassMapping) -> EClassMapping:
    """Remove identity-padding enodes: (* 1 x), (+ x 0), (- x 0), (- 0 x), (/ x 1),
    and equal-operand (- x x) / (/ x x)
    """
    const_cache: dict[tuple[str, str], bool] = {}

    def is_const(eclass: str, literal: str) -> bool:
        key = (eclass, literal)
        if key not in const_cache:
            const_cache[key] = any(
                enode.op == "Num"
                and any(
                    leaf.op == literal and not leaf.children
                    for leaf in eclasses.get(enode.children[0], ())
                )
                for enode in eclasses.get(eclass, ())
                if enode.children
            )
        return const_cache[key]

    def is_identity(enode: ENode) -> bool:
        if len(enode.children) != 2:
            return False
        a, b = enode.children
        match enode.op:
            case "Mul":
                # 1 is an identity, 0 an absorber
                return (
                    is_const(a, "1")
                    or is_const(b, "1")
                    or is_const(a, "0")
                    or is_const(b, "0")
                )
            case "Add":
                return is_const(a, "0") or is_const(b, "0")
            case "Sub":
                return a == b or is_const(a, "0") or is_const(b, "0")
            case "Div":
                return a == b or is_const(b, "1") or is_const(a, "0")
        return False

    stripped = {
        eclass: {enode for enode in enodes if not is_identity(enode)}
        for eclass, enodes in eclasses.items()
    }

    # Second pass: break spelling cycles (x as (/ (* a x) a), (- (- x)), ...),
    # which otherwise let a stuck model nest equivalent wrappers forever. No cycle
    # consists entirely of minimal-depth enodes (depths strictly decrease along
    # them), so dropping non-minimal enodes with a child in their own SCC leaves
    # the index acyclic while keeping every acyclic spelling.
    min_depth: dict[str, float] = {eclass: float("inf") for eclass in stripped}

    def enode_depth(enode: ENode) -> float:
        return 1 + max(
            (min_depth.get(child, float("inf")) for child in enode.children),
            default=0,
        )

    changed = True
    while changed:
        changed = False
        for eclass, enodes in stripped.items():
            best = min((enode_depth(e) for e in enodes), default=float("inf"))
            if best < min_depth[eclass]:
                min_depth[eclass] = best
                changed = True

    graph = nx.DiGraph()
    graph.add_nodes_from(stripped)
    for eclass, enodes in stripped.items():
        graph.add_edges_from(
            (eclass, child) for enode in enodes for child in enode.children
        )
    scc_of: dict[str, int] = {}
    for i, component in enumerate(nx.strongly_connected_components(graph)):
        for node in component:
            scc_of[node] = i

    return {
        eclass: {
            enode
            for enode in enodes
            if enode_depth(enode) <= min_depth[eclass]  # minimal witnesses stay
            or all(scc_of.get(child) != scc_of[eclass] for child in enode.children)
        }
        for eclass, enodes in stripped.items()
    }


@lru_cache(maxsize=None)
def in_egraph(egraph: EGraph) -> Callable[[TreeGrammar], TreeGrammar]:
    """
    Given an egraph, returns a predicate on TreeGrammars that computes the intersection
    of the grammar with the egraph.
    """
    root_eclass, eclasses = root_and_eclass_mapping(egraph)
    eclasses = strip_identity_enodes(eclasses)

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

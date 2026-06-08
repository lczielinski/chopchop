import json
import logging
import re
from collections.abc import Iterable
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import product
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


@dataclass(frozen=True)
class ExtractedExpression:
    """A bounded, rendered expression from the root eclass."""

    text: str
    size: int
    depth: int
    score: float


@dataclass(frozen=True)
class _Candidate:
    text: str
    size: int
    depth: int
    ops: frozenset[str]
    root_op: str


EClassMapping = dict[str, set[ENode]]  # maps eclass names to contained enodes

_BINARY_FPCORE_OPS = {
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
    "Pow": "pow",
}
_UNARY_FPCORE_OPS = {
    "Neg": "-",
    "Sqrt": "sqrt",
}
_MATH_OPS = frozenset(_BINARY_FPCORE_OPS | _UNARY_FPCORE_OPS)


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


def interesting_equivalent_expressions(
    egraph: EGraph,
    *,
    limit: int = 6,
    max_depth: int = 12,
    max_size: int = 80,
    cheap_per_eclass: int = 12,
    interesting_per_eclass: int = 12,
) -> list[ExtractedExpression]:
    """Extract nontrivial root-equivalent FPCore bodies from the saturated egraph.

    This is intentionally a bounded, heuristic extractor, not a complete enumerator. It
    keeps a few small representatives for each eclass so larger expressions can still be
    assembled, plus a few "interesting" representatives that tend to expose
    rounding-changing rewrites such as distribution, reciprocal forms, quotient splits,
    and conjugate rationalizations. The bounds keep extraction separate from equality
    saturation, so asking for targets does not enlarge the egraph used by decoding.
    """
    if limit <= 0 or max_depth < 0 or max_size <= 0:
        return []

    root_eclass, eclasses = root_and_eclass_mapping(egraph)

    @lru_cache(maxsize=None)
    def candidates_for(eclass: str, depth_left: int) -> tuple[_Candidate, ...]:
        if depth_left < 0:
            return ()

        candidates: list[_Candidate] = []
        for enode in eclasses.get(eclass, ()):
            if not enode.children:
                # The FPCore grammar has no negative integer literals; render a folded
                # negative constant as a unary negation so the checker can lex it.
                if re.fullmatch(r"-\d+", enode.op):
                    text, size, depth = f"(- {enode.op[1:]})", 2, 1
                else:
                    text, size, depth = enode.op, 1, 0
                candidates.append(
                    _Candidate(
                        text=text,
                        size=size,
                        depth=depth,
                        ops=frozenset(),
                        root_op=enode.op,
                    )
                )
                continue

            if depth_left == 0:
                continue

            child_candidates = [
                candidates_for(child, depth_left - 1) for child in enode.children
            ]
            if any(not choices for choices in child_candidates):
                continue

            for children in product(*child_candidates):
                size = 1 + sum(child.size for child in children)
                if size > max_size:
                    continue
                depth = 1 + max(child.depth for child in children)
                rendered = _render_fpcore(enode.op, children)
                if rendered is None:
                    continue
                ops = frozenset().union(*(child.ops for child in children))
                if enode.op in _MATH_OPS:
                    ops |= frozenset((enode.op,))
                candidates.append(
                    _Candidate(
                        text=rendered,
                        size=size,
                        depth=depth,
                        ops=ops,
                        root_op=enode.op,
                    )
                )

        return _prune_candidates(
            candidates,
            cheap_limit=cheap_per_eclass,
            interesting_limit=interesting_per_eclass,
        )

    # Enumerate only as deep as this benchmark needs: the shallowest member of the root
    # eclass plus a small margin (capped at max_depth). Keeps shallow benchmarks fast while
    # letting deep references (e.g. variance, ~9 levels) still be reachable.
    root_min_depth = _eclass_min_depth(eclasses, root_eclass)
    working_depth = max_depth if root_min_depth is None else min(
        max_depth, max(8, root_min_depth + 2)
    )
    root_candidates = list(candidates_for(root_eclass, working_depth))
    if not root_candidates:
        return []

    min_size = min(candidate.size for candidate in root_candidates)
    selected: list[ExtractedExpression] = []
    seen: set[str] = set()
    seen_signatures: set[str] = set()
    for candidate in sorted(
        root_candidates,
        key=lambda candidate: (
            -_target_score(candidate, min_size),
            candidate.size,
            candidate.text,
        ),
    ):
        if candidate.text in seen:
            continue
        seen.add(candidate.text)

        signature = _structural_signature(candidate.text)
        if signature in seen_signatures:
            continue
        if candidate.size > min(max_size, 60):
            continue
        if _identity_noise(candidate.text) > 2:
            continue

        score = _target_score(candidate, min_size)
        if score <= 0:
            continue
        selected.append(
            ExtractedExpression(
                text=candidate.text,
                size=candidate.size,
                depth=candidate.depth,
                score=score,
            )
        )
        seen_signatures.add(signature)
        if len(selected) >= limit:
            break

    return selected


def _eclass_min_depth(eclasses: EClassMapping, root: str) -> int | None:
    """Shallowest expression depth representable from `root` (leaf depth 0), or None.

    A cheap fixpoint over the e-graph (no enumeration), used to size the extraction depth
    to the benchmark instead of using one global bound.
    """
    depth: dict[str, int] = {}
    for _ in range(len(eclasses) + 1):
        changed = False
        for eclass, enodes in eclasses.items():
            best: int | None = depth.get(eclass)
            for enode in enodes:
                if not enode.children:
                    node_depth: int | None = 0
                else:
                    child_depths = [depth.get(c) for c in enode.children]
                    if any(d is None for d in child_depths):
                        continue
                    node_depth = 1 + max(d for d in child_depths if d is not None)
                if node_depth is not None and (best is None or node_depth < best):
                    best = node_depth
            if best is not None and best != depth.get(eclass):
                depth[eclass] = best
                changed = True
        if not changed:
            break
    return depth.get(root)


def _render_fpcore(op: str, children: tuple[_Candidate, ...]) -> str | None:
    child_text = tuple(child.text for child in children)
    if op in {"Num", "Var"}:
        return child_text[0] if len(child_text) == 1 else None
    if op in _UNARY_FPCORE_OPS:
        if len(child_text) != 1:
            return None
        return f"({_UNARY_FPCORE_OPS[op]} {child_text[0]})"
    if op in _BINARY_FPCORE_OPS:
        if len(child_text) != 2:
            return None
        return f"({_BINARY_FPCORE_OPS[op]} {child_text[0]} {child_text[1]})"
    return None


def _prune_candidates(
    candidates: Iterable[_Candidate],
    *,
    cheap_limit: int,
    interesting_limit: int,
) -> tuple[_Candidate, ...]:
    by_text: dict[str, _Candidate] = {}
    for candidate in candidates:
        current = by_text.get(candidate.text)
        if current is None or (candidate.size, candidate.depth) < (
            current.size,
            current.depth,
        ):
            by_text[candidate.text] = candidate

    pool = list(by_text.values())
    selected: dict[str, _Candidate] = {}
    for candidate in sorted(pool, key=lambda c: (c.size, c.depth, c.text))[
        :cheap_limit
    ]:
        selected[candidate.text] = candidate
    for candidate in sorted(
        pool,
        key=lambda c: (-_candidate_score(c), c.size, c.depth, c.text),
    )[:interesting_limit]:
        selected[candidate.text] = candidate
    return tuple(selected.values())


def _target_score(candidate: _Candidate, min_size: int) -> float:
    score = _candidate_score(candidate)
    score += 1.5 * max(candidate.size - min_size, 0)
    if candidate.size > 60:
        score -= 1.5 * (candidate.size - 60)
    return score


def _candidate_score(candidate: _Candidate) -> float:
    text = candidate.text
    score = float(candidate.size)
    score += 5.0 * len(candidate.ops & frozenset(("Add", "Sub", "Mul", "Div")))
    score += 2.0 * len(candidate.ops & frozenset(("Neg", "Pow", "Sqrt")))

    if candidate.root_op in {"Add", "Sub", "Mul", "Div"}:
        score += 8.0
    if candidate.root_op == "Div" and candidate.ops & frozenset(("Add", "Sub", "Mul")):
        score += 8.0
    if "(/ 1 " in text:
        score += 6.0
    if "(/ (+" in text or "(/ (-" in text or "(/ (*" in text:
        score += 5.0
    if "(sqrt " in text and candidate.root_op == "Div":
        score += 8.0
    if "(sqrt " in text and "(- (- " in text:
        score += 8.0

    score -= 10.0 * _identity_noise(text)
    score -= 6.0 * max(candidate.depth - 9, 0)
    return score


def _identity_noise(text: str) -> int:
    patterns = (
        r"\(\+ 0\b",
        r"\(\+ [^()]+ 0\)",
        r"\(- 0\b",
        r"\(\* 0\b",
        r"\(\* [^()]+ 0\)",
        r"\(\* 1\b",
        r"\(\* [^()]+ 1\)",
        r"\(/ 1 \(/ 1",
        r"\(/ 1 \(\* 1",
        r"\(/ -?\d+ -?\d+\)",
    )
    return (
        sum(len(re.findall(pattern, text)) for pattern in patterns)
        + text.count(" 0)")
        + text.count(" 1)")
    )


def _structural_signature(text: str) -> str:
    """Canonicalize cosmetic target variants for diversity filtering."""
    tokens = re.findall(r"\(|\)|[^\s()]+", text)

    def parse(index: int) -> tuple[object, int]:
        if index >= len(tokens):
            raise ValueError("unexpected end of expression")

        token = tokens[index]
        if token != "(":
            return token, index + 1

        op = tokens[index + 1]
        index += 2
        children = []
        while index < len(tokens) and tokens[index] != ")":
            child, index = parse(index)
            children.append(child)
        if index >= len(tokens):
            raise ValueError("unterminated expression")
        return _canonical_node(op, tuple(children)), index + 1

    try:
        tree, index = parse(0)
    except (IndexError, ValueError):
        return text
    if index != len(tokens):
        return text
    return repr(tree)


def _neg(node: object) -> object:
    """Canonical negation, using only FP-identical rewrites: -(-x)=x, -(a+b)=(-a)+(-b)."""
    if isinstance(node, tuple) and len(node) == 2 and node[0] == "neg":
        return node[1]
    if isinstance(node, tuple) and len(node) == 2 and node[0] == "+":
        return ("+", tuple(sorted((_neg(c) for c in node[1]), key=repr)))
    return ("neg", node)


def _sign_canon(node: object) -> object:
    """Representative of {node, -node}; the sign of a squared base is irrelevant."""
    return min((node, _neg(node)), key=repr)


def _canonical_node(op: str, children: tuple[object, ...]) -> object:
    # a - b == a + (-b) (FP-identical), so route subtraction through addition and let the
    # commutative sort below unify a+(-b) with b's-first orderings; unary minus becomes neg.
    if op == "-" and len(children) == 2:
        return _canonical_node("+", (children[0], _neg(children[1])))
    if op == "-" and len(children) == 1:
        return _neg(children[0])

    if op in {"+", "*"}:
        flattened = []
        for child in children:
            if (
                isinstance(child, tuple)
                and len(child) == 2
                and child[0] == op
                and isinstance(child[1], tuple)
            ):
                flattened.extend(child[1])
            else:
                flattened.append(child)
        children = tuple(sorted(flattened, key=repr))

    # x^2: the squared base's sign is irrelevant, so collapse (a-b)^2 with (b-a)^2.
    if op == "pow" and len(children) == 2 and children[1] == "2":
        return ("pow2", _sign_canon(children[0]))
    if op == "*" and len(children) == 2 and children[0] == children[1]:
        return ("pow2", _sign_canon(children[0]))
    return (op, children)


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

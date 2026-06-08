"""Fast, LLM-free probe of the equivalence checker.

Builds the same RealizabilityChecker used in decoding and tests whether
hand-written FPCore programs are accepted as equivalent to a benchmark's
reference. Lets us measure what the rule set admits without loading a model.

    uv run python -m egraph.probe quadratic
"""

import os
import re
import sys
import time

from .run import build_checker, load_benchmark


def check(checker, program: str) -> tuple[bool, float]:
    t0 = time.time()
    ok = checker.realizable(program, True)
    return ok, time.time() - t0


# Hand-written candidates per benchmark: (label, program, expected_equivalent)
CANDIDATES = {
    "quadratic": [
        # the reference itself, written in FPCore
        ("reference", "(FPCore (a b c) (/ (+ (- b) (sqrt (- (pow b 2) (* (* 4 a) c)))) (* 2 a)))", True),
        # pow b 2 -> b*b  (needs Pow<->Mul rule)
        ("pow->mul", "(FPCore (a b c) (/ (+ (- b) (sqrt (- (* b b) (* (* 4 a) c)))) (* 2 a)))", True),
        # factor 4ac differently: 4*(a*c) vs (4*a)*c  (assoc, already supported)
        ("reassoc-4ac", "(FPCore (a b c) (/ (+ (- b) (sqrt (- (pow b 2) (* 4 (* a c))))) (* 2 a)))", True),
        # div as mult by reciprocal of (2a)  (already supported)
        ("recip", "(FPCore (a b c) (* (+ (- b) (sqrt (- (pow b 2) (* (* 4 a) c)))) (/ 1 (* 2 a))))", True),
        # NOT equivalent: wrong sign on b
        ("wrong-sign", "(FPCore (a b c) (/ (+ b (sqrt (- (pow b 2) (* (* 4 a) c)))) (* 2 a)))", False),
        # Citardauq (conjugate) form: 2c / (-b - sqrt(b^2-4ac)). Equal in exact reals,
        # numerically distinct. Needs multiply-by-conjugate, not a ring axiom.
        ("citardauq", "(FPCore (a b c) (/ (* 2 c) (- (- b) (sqrt (- (pow b 2) (* (* 4 a) c))))))", True),
        # adversarial: the OTHER root, 2c/(-b + sqrt D). NOT equal to the reference root.
        ("citardauq-wrong", "(FPCore (a b c) (/ (* 2 c) (+ (- b) (sqrt (- (pow b 2) (* (* 4 a) c))))))", False),
    ],
    "distance": [
        ("reference", "(FPCore (x1 x2 y1 y2) (sqrt (+ (pow (- x1 x2) 2) (pow (- y1 y2) 2))))", True),
        ("pow->mul", "(FPCore (x1 x2 y1 y2) (sqrt (+ (* (- x1 x2) (- x1 x2)) (* (- y1 y2) (- y1 y2)))))", True),
        # expand (x1-x2)^2 = x1^2 - 2 x1 x2 + x2^2  (needs distribution of mul over sub)
        ("expand-sq", "(FPCore (x1 x2 y1 y2) (sqrt (+ (+ (- (pow x1 2) (* (* 2 x1) x2)) (pow x2 2)) (pow (- y1 y2) 2))))", True),
    ],
    "lerp": [
        ("reference", "(FPCore (start end scale) (+ start (* (- end start) scale)))", True),
        # distribute: start + end*scale - start*scale
        ("distribute", "(FPCore (start end scale) (+ start (- (* end scale) (* start scale))))", True),
        # factor form: start*(1-scale) + end*scale  -- needs 1, identity; expect maybe False
        ("factor-1mscale", "(FPCore (start end scale) (+ (* start (- 1 scale)) (* end scale)))", True),
    ],
}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "quadratic"
    _, source = load_benchmark(name + ".egglog")
    # CQ_RUNS lets us cap saturation depth to study the explosion/coverage tradeoff.
    runs = os.environ.get("CQ_RUNS")
    if runs:
        source = re.sub(r"\(run \d+\)", f"(run {runs})", source)
        print(f"(saturation capped at run {runs})")
    t0 = time.time()
    checker = build_checker(source)
    build_t = time.time() - t0
    print(f"=== {name} === (checker built in {build_t:.2f}s)")
    for label, program, expected in CANDIDATES.get(name, []):
        ok, dt = check(checker, program)
        mark = "OK " if ok == expected else "XX "
        print(f"  {mark} {label:16s} accepted={ok!s:5s} expected={expected!s:5s} ({dt:.2f}s)")


if __name__ == "__main__":
    main()

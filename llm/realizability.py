import signal
from functools import reduce
from core.grammar import is_nonempty, TreeGrammar
from core.parser import D, Choice, Parser, delta, image
from core.lexing.lexing import LexerSpec
from typing import Callable


class _CheckTimeout(Exception):
    pass


class RealizabilityChecker:
    def __init__(
        self,
        constraint: Callable[[TreeGrammar], TreeGrammar],
        initial_parser: Parser,
        lexerspec: LexerSpec,
        max_depth: int | None = None,
        timeout: float | None = None,
        timeout_min_depth: int = 0,
    ):
        self.constraint = constraint
        self.parser = initial_parser
        self.lexerspec = lexerspec
        # Per-check wall-clock cap (seconds). Cyclic equivalence rules (reciprocal,
        # squaring) let a stuck model build towers like `(* (/ (* (/ ...)))` whose
        # realizability check explores the cycle and runs for tens of seconds, even at
        # modest depth — within the depth range of legitimate programs, so a depth cap
        # alone can't separate them. Legitimate checks are sub-second once the e-graph
        # index is warm, so aborting any check that exceeds this bound (and rejecting the
        # token) stops the runaway without blocking real programs. Uses SIGALRM, so it only
        # applies on the main thread (the decode loop); off-thread it is a no-op.
        # The timeout is applied ONLY to prefixes nesting at least `timeout_min_depth` deep:
        # shallow/wide-open prefixes (e.g. the empty body) are legitimately slow but return
        # True, so timing them out would wrongly reject good tokens; deep prefixes are where
        # the degenerate towers live, and legitimate deep checks are sub-second.
        self.timeout = timeout
        self.timeout_min_depth = timeout_min_depth
        # Optional cap on s-expression nesting depth. Equivalence rules that form cycles
        # (e.g. Div <-> Mul-by-reciprocal, or square <-> pow) make arbitrarily deep towers
        # like `(* (/ (* (/ ...))))` realizable, so the constraint never prunes them and a
        # stuck model nests forever (running to max_new_tokens, with the per-token check
        # slowing as it deepens). Real equivalent programs are shallow; rejecting anything
        # past this bound stops the runaway cheaply, before the e-graph work.
        self.max_depth = max_depth

    def _max_depth(self, prefix: str) -> int:
        depth = mx = 0
        for ch in prefix:
            if ch in "([":
                depth += 1
                mx = max(mx, depth)
            elif ch in ")]":
                depth -= 1
        return mx

    def _realizable(self, prefix: str, final: bool) -> bool:
        lexes = self.lexerspec.lex(prefix, final)
        prefix_space = Choice.of(
            reduce(lambda parser, token: D(token, parser), lex, self.parser)
            for lex in lexes
        )
        if final:
            prefix_space = delta(prefix_space)

        constrained_prefix_space = self.constraint(image(prefix_space))
        return is_nonempty(constrained_prefix_space)

    def realizable(self, prefix: str, final: bool = False) -> bool:
        """
        Checks if a prefix is realizable.
        If final is True, the prefix must be a complete program.
        """
        depth = self._max_depth(prefix)
        if self.max_depth is not None and depth > self.max_depth:
            return False
        if self.timeout is None or depth < self.timeout_min_depth:
            return self._realizable(prefix, final)
        # Abort a check that runs too long (a degenerate cyclic-tower prefix) and treat the
        # token as not realizable. SIGALRM only fires on the main thread; if we're not on it
        # (no-op handler install fails), fall back to running without a timeout.
        def _on_timeout(signum, frame):
            raise _CheckTimeout()

        try:
            old_handler = signal.signal(signal.SIGALRM, _on_timeout)
        except ValueError:
            return self._realizable(prefix, final)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            return self._realizable(prefix, final)
        except _CheckTimeout:
            return False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

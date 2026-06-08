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
        # Guards against degenerate "towers" — cyclic rules (e.g. Div <-> Mul-by-reciprocal)
        # let a stuck model nest `(* (/ (* (/ ...))))` forever, staying realizable.
        # max_depth: reject anything nesting past this (cheap, before the e-graph work).
        # timeout: abort a check exceeding this many seconds (SIGALRM, main thread only),
        #   applied only at depth >= timeout_min_depth so the legitimately-slow but shallow
        #   wide-open prefixes aren't wrongly rejected.
        self.max_depth = max_depth
        self.timeout = timeout
        self.timeout_min_depth = timeout_min_depth

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

        def _on_timeout(signum, frame):
            raise _CheckTimeout()

        try:
            old_handler = signal.signal(signal.SIGALRM, _on_timeout)
        except ValueError:  # not on the main thread; run without a timeout
            return self._realizable(prefix, final)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            return self._realizable(prefix, final)
        except _CheckTimeout:
            return False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

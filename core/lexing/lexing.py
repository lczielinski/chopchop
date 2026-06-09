from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable

import regex

from .lru_cache import LRUCache
from .token import Token

IGNORE = "RESERVED_IGNORE_SORT_TITLE"


@dataclass(frozen=True)
class LexerSpec:
    tokens: frozenset[Token]
    ignore_regex: regex.Pattern = regex.compile(r"^(?!)$")
    lexical_cache: LRUCache[str, LexerState] = field(
        default_factory=lambda: LRUCache(128),
        compare=False,
        repr=False,
    )

    def __hash__(self) -> int:
        return hash((self.tokens, self.ignore_regex.pattern))

    def get_lexemes(self) -> Iterable[Token]:
        yield from self.tokens
        yield Token(IGNORE, self.ignore_regex)

    def lex(self, inp: str, final: bool = True) -> set[tuple[Token, ...]]:
        lstate = self.compute_lexer_state(inp)
        if final:
            lstate = lstate.finalize()
        lstate = lstate.remove_ignorable_tokens()
        return lstate.get_partial_lexes()

    def compute_lexer_state(self, inp: str) -> LexerState:
        # Reuse the lex of the longest prefix in the cache.
        lstate = LexerState()
        start_idx = 0
        for i in range(len(inp), 0, -1):
            cached = self.lexical_cache.get(inp[:i])
            if cached is not None:
                lstate = cached
                start_idx = i
                break

        # Lex the new part of the input and cache every prefix we compute.
        for idx, char in enumerate(inp[start_idx:], start_idx):
            lstate = lstate.extend_lexer_state(char, self)
            lstate.remove_nonmaximal_munch()
            self.lexical_cache.put(inp[: idx + 1], lstate)
        return lstate


@dataclass
class LexerState:
    prefix: tuple[Token, ...] = field(default_factory=tuple)
    continuations: set[tuple[Token, ...]] = field(default_factory=lambda: {()})

    def get_partial_lexes(self) -> set[tuple[Token, ...]]:
        return {tuple(self.prefix) + cont for cont in self.continuations}

    def finalize(self) -> LexerState:
        """Complete every completable partial lex and discard the others."""
        if self.continuations:
            continuations = {
                c[:-1] + (c[-1].complete(),)
                for c in self.continuations
                if c and c[-1].nullable()
            }
            return LexerState(self.prefix, continuations)
        return self

    def extend_lexer_state(self, char: str, lexerspec: LexerSpec) -> LexerState:
        """Extend the lexer state by lexing one more character."""
        new_continuations: set[tuple[Token, ...]] = set()
        for state in self.continuations:
            if len(state) == 0:
                for lexeme in lexerspec.get_lexemes():
                    derived = lexeme.extend(char)
                    if derived.nonempty():
                        new_continuations.add((derived,))
            else:
                if state[-1].nullable():
                    for lexeme in lexerspec.get_lexemes():
                        derived = lexeme.extend(char)
                        if derived.nonempty():
                            new_continuations.add(
                                (state[:-1] + (state[-1].complete(), derived))
                            )
                if state[-1].extend(char).nonempty():
                    new_continuations.add(state[:-1] + (state[-1].extend(char),))
        return LexerState(self.prefix, new_continuations)

    def remove_nonmaximal_munch(self) -> None:
        """Remove continuations that violate maximal munch."""
        self.continuations = {
            state
            for state in self.continuations
            if not any(
                _violates_maximal_munch(state, other)
                for other in self.continuations
                if other != state
            )
        }

    def remove_ignorable_tokens(self) -> LexerState:
        """Remove ignorable tokens (e.g., whitespace)."""
        continuations = {
            tuple(filter(lambda x: x.token_type != IGNORE, state))
            for state in self.continuations
        }
        return LexerState(self.prefix, continuations)


def _violates_maximal_munch(
    candidate: tuple[Token, ...],
    other: tuple[Token, ...],
) -> bool:
    for idx in range(min(len(candidate), len(other))):
        if candidate[idx] != other[idx]:
            return (
                len(candidate[idx].prefix) < len(other[idx].prefix)
                and other[idx].nullable()
            )
    return False

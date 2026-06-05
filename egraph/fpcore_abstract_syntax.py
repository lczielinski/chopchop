"""Abstract syntax for a small subset of FPCore 2.0.

We support a deliberately tiny fragment (see ``fpcore.lark``):

    (FPCore (x y ...) body)

where ``body`` is either an arithmetic expression or a (possibly nested)
``let``/``let*`` whose binding values are arithmetic expressions:

    (let  ([v e] ...) body)
    (let* ([v e] ...) body)

Arithmetic expressions are ``+ - * /``, unary negation, ``sqrt``, ``pow``,
integer literals, and variables.

The structural nodes (``FPCore``, ``Let``, the ``Args``/``Bindings`` cons-lists,
``Binding``) are peeled off by the equivalence pruner and never reach egglog.
The arithmetic nodes (``Add``, ``Sub``, ``Mul``, ``Div``, ``Neg``, ``Sqrt``,
``Pow``, ``Var``, ``Num``) share their names with the egglog ``Math`` datatype,
so ``expr_to_egglog`` can translate them generically.
"""

from core.grammar import Application, Unary, Binary


# --- structural nodes (peeled off before egglog) ---


class FPCore(Binary):  # (args, body)
    ...


class Args(Binary):  # cons cell of a >1 argument list (head id, tail args)
    ...


class Let(Binary):  # (bindings, body)
    ...


class Bindings(Binary):  # cons cell of a >1 binding list (head binding, tail)
    ...


class Binding(Binary):  # one [var value] pair
    ...


# --- arithmetic nodes (translated to egglog by name) ---


class Var(Unary): ...


class Num(Unary): ...


class Neg(Unary): ...


class Sqrt(Unary): ...


class Add(Binary): ...


class Sub(Binary): ...


class Mul(Binary): ...


class Div(Binary): ...


class Pow(Binary): ...


constructors: list[type[Application]] = [
    FPCore,
    Args,
    Let,
    Bindings,
    Binding,
    Var,
    Num,
    Neg,
    Sqrt,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
]

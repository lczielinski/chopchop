# ChopChop
A programmable constrained decoder for semantic properties (e.g., type safety, program equivalence up to rewrite, simple static analyses).
Users encode semantic constraints as pruners over AST-like "program spaces". ChopChop then automatically constrains the sampling of autoregressive LLMs to produce constraint-satisfying output. 
A comprehensive overview appears in [our paper](https://doi.org/10.1145/3776708).

# Installation
### Requirements
- Python 3.12+.
- [uv](https://docs.astral.sh/uv/) for dependency management.
### Instructions
1. Clone the repository:
```bash
git clone https://github.com/timothytmzhou/chopchop.git
cd chopchop
```

2. Create a virtual environment and install dependencies:
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

3. Verify installation succeeded by importing the package:
```bash
uv run python -c "import egraph.run"
```

# Usage
ChopChop requires a grammar, abstract syntax, and zero or more pruner(s).
Grammars are written in the following Lark-like format:
```
<!-- Definitions of terminals. -->
NUM: /[0-9]+/
WS: /\s+/

<!-- Nonterminals and their productions. -->
start: expr ";"
expr: expr "+" num {Add} 
    | num
num: NUM {Num}

<!-- Ignore whitespace -->
%ignore WS
```
Each annotation in braces gives the name of the AST node to be constructed when the production rule fires.


Abstract syntax is specified by subclassing `Application`, a generic superclass for internal AST nodes.
For example, the Add node in the grammar above can be written as
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Add(Application):
    left: TreeGrammar
    right: TreeGrammar
```

At runtime, a cyclic graph representing the set of possible ASTs is constructed.
We expose these graphs to users as `TreeGrammar` objects, which they should think of as infinite AST-like trees that have special `Union` and `EmptySet` nodes to represent a union of sets of ASTs and the empty set of ASTs respectively.
To manipulate these objects, users write pruners, which are functions that map `TreeGrammar`s to `TreeGrammar`s by removing undesirable programs.
For example,
```python
from core.grammar import TreeGrammar, Union, EmptySet, as_tree
from core.lexing.token import Token
from core.rewrite import rewrite

@rewrite
def sum_of_evens(t: TreeGrammar) -> TreeGrammar:
    """Remove ASTs that contain odd integers."""
    match t:
        case Union(children):
            return Union.of(sum_of_evens(c) for c in children)
        case Num(arg):
            token = as_tree(arg)
            match token:
                case Token(is_complete=True, prefix=prefix) if int(prefix) % 2 == 1:
                    return EmptySet()
                case _:
                    return t
        case Add(left, right):
            return Add(sum_of_evens(left), sum_of_evens(right))
        case _:
            return EmptySet()
```
Note that the pruner does not explicitly worry about cycles.
The `@rewrite` annotation lifts the pruner to our cyclic data structures.
However, users should avoid writing pruners where the set of distinct recursive invocations will not reach a fixpoint if run on a cyclic graph, e.g., by passing around a counter.

Finally, a user bundles the information into a realizability checker which can be used to constrain LLM calls.
```python
# Grammar & Abstract Syntax
grammar_source = files(__package__).joinpath("my_grammar.lark").read_text()
ast_constructors: list[type[Application]] = [Add, Num]

# Extract grammatical information
start_lexer_spec, start_grammar = parse_attribute_grammar(
    ast_constructors, grammar_source, "start"
).build_parser()

# Build RealizabilityChecker
checker = RealizabilityChecker(
    sum_of_evens,
    start_grammar,
    start_lexer_spec,
)

# Set up LLM and run it on a prompt
model_config = ModelConfig(model_id='codellama/CodeLlama-7b-Instruct-hf')
model_runner = LanguageModelRunner(model_config=model_config)
out = model_runner.run(
    Config(),
    "Write a sum of your favorite integers.",
    "You are a helpful assistant.",
    realizability_checker=checker,
)
```

A complete example is the egraph equivalence experiment in `egraph`.

# Running the egraph experiment
The egraph experiment generates code that is numerically equivalent to a reference program.
Each benchmark in `egraph/benchmarks/` pairs a reference program with egglog rewrite rules;
`egraph/let.egglog` holds the base algebraic rules. Decoding is constrained so that every
generated program is provably equivalent to the reference under those rules.

Generated programs are written in a small subset of [FPCore 2.0](https://fptalks.org/spec/fpcore-2.0.html):

```
(FPCore (a b c)
  (/ (+ (- b) (sqrt (- (* b b) (* (* 4 a) c)))) (* 2 a)))
```

The subset (see `egraph/fpcore.lark`) is: an `(FPCore (args...) body)` header where `body` is a
single arithmetic expression built from the operators `+ - * /` (binary) and `-` (unary
negation), `sqrt`, integer literals, and variables. There is no exponentiation operator (write
a square as a product, e.g. `(* b b)`), no generic function application, and no `let` bindings —
only these numeric operators — so the benchmarks are all numeric expressions.

`egraph/fpcore.py` translates this syntax to egglog (`expr_to_egglog`) and contains the
equivalence pruner (`fpcore_equivalence`): it peels off the `FPCore` wrapper and intersects the
body with the reference's egraph. Reference programs and generated programs share the one
`Math` datatype in `egraph/let.egglog`, whose constructor names (`Add`, `Mul`, `Pow`, `Sqrt`,
…) match the FPCore AST node names, so `expr_to_egglog` translates generically.

Run a single benchmark (from the repository root):
```bash
uv run python -m egraph.run --benchmark quadratic.egglog
```
Tokens stream to the terminal as they are decoded. The model runs on the Apple Silicon GPU
(MPS) in bfloat16 by default. Omit `--benchmark` to run all benchmarks. Useful options:
- `--model NAME` (default: `qwen14b`) — one of `qwen14b`, `qwen7b`, `codestral`,
  `deepseek-v2`, `llama13b`, `llama7b`, `deepseek`
- `--temperature FLOAT` (default: `0.8`)
- `--min-p FLOAT` — min-p sampling cutoff (default: `0` = off). Keeps tokens with probability
  at least `min_p` times the top token's; adapts per position, so it permits tail exploration
  at genuine forks while pruning garbage. For exploration runs prefer
  `--temperature 1.3 --min-p 0.05`: the realizability checker rejects bad tokens
  anyway, so high temperature is much safer here than in unconstrained generation
- `--num-programs N` / `-n N` — generate `N` distinct equivalent programs (default: `1`)
- `--max-tries N` — cap LLM generation attempts per benchmark (default: `25`)
- `--max-token-tries N` — abort one LLM attempt after `N` rejected token proposals at the same prefix (default: `256`)
- `--no-stream` — disable live token streaming
- `--output-dir DIR` — folder for per-run output files; each run writes a new timestamped `.txt` with the settings at the top (default: `outputs/`)

For example, to generate 3 distinct programs equivalent to the quadratic formula, allowing up
to 20 attempts:
```bash
uv run python -m egraph.run --benchmark quadratic.egglog -n 3 --max-tries 20
```

The LLM generates each program on its own, constrained only by the egraph-backed
realizability checker — it is never handed a target shape extracted from the e-graph. To
encourage additional distinct programs, the prompt lists the programs already produced for
the benchmark and asks for one with different floating-point behavior from a different
structural rewrite family when possible. Accepted programs are deduplicated by a canonical
form (whitespace normalized), and more coarsely by a rewrite-family signature (root
operator plus operator multiset), so formatting changes, commutative reorders, and sign
shuffles are not counted as distinct.

The e-graph index used for the intersection is pruned before decoding
(`strip_identity_enodes`): identity padding like `(* 1 ..)` and `(+ 0 ..)` is removed,
and spelling cycles are broken so the index is acyclic — a stuck model cannot nest
equivalent wrappers forever, while every acyclic rewrite survives.

Because constrained decoding filters tokens but never up-weights rare branches, later
attempts cycle a forced-divergence fork (`diverge@opK` in the attempt header): the attempt
may not follow an already-accepted program through its K-th operator token, so it must
branch onto a different — still provably equivalent — structure at an earlier fork.

### Models and memory
Models are loaded with `transformers` in bfloat16 onto MPS, so the whole model must fit in
unified memory (bf16 footprints: `qwen7b` ~15 GB, `deepseek-v2` ~31 GB, `qwen14b` ~29 GB,
`codestral` ~44 GB). On a 64 GB machine, `qwen14b` is the recommended default; `codestral`
fits with less headroom. Notes:
- `codestral` is gated on Hugging Face — accept its license on the model page, then
  authenticate with `uv run hf auth login` (or set `HF_TOKEN` in your environment).
- A 32B model (~64 GB in bf16) does **not** fit alongside activations on 64 GB; it would need
  a quantized backend (e.g. MLX), which the current fp16/bf16 transformers+MPS path doesn't do.
- If MPS hits an unimplemented op in bf16, run with `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall
  back to CPU for those ops.

# Repository Organization
- **`core`** — the backend of the tool (constructing and manipulating prefix spaces).
- **`llm`** — running LLMs and interfacing an LLM with a realizability checker.
- **`egraph`** — the equivalence case study: the FPCore grammar (`fpcore.lark`), its abstract
  syntax (`fpcore_abstract_syntax.py`), the egglog translation + equivalence pruner
  (`fpcore.py`) backed by egglog rewrite rules (`let.egglog`), benchmark programs in
  `benchmarks/`, and `run.py` to run it.

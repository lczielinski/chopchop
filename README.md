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
from core.grammar import TreeGrammar, Union, EmptySet, Token

@rewrite
def sum_of_evens(t: TreeGrammar) -> TreeGrammar:
  """Remove ASTs that contain even integers."""
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
The `@rewrite` annotation lifts the pruner to our cyclic datastructures.
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
  (/ (+ (- b) (sqrt (- (pow b 2) (* (* 4 a) c)))) (* 2 a)))
```

The subset (see `egraph/fpcore.lark`) is: an `(FPCore (args...) body)` header where `body` is a
single arithmetic expression built from the operators `+ - * /` (binary) and `-` (unary
negation), `sqrt` and `pow`, integer literals, and variables. There is no generic function
application and no `let` bindings — only these numeric operators — so the benchmarks are all
numeric expressions.

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
- `--top-p FLOAT` — nucleus sampling cutoff (default: `0.9`; `1.0` disables tail filtering)
- `--num-programs N` / `-n N` — generate `N` distinct equivalent programs (default: `1`)
- `--max-tries N` — cap LLM generation attempts per benchmark (default: `25`)
- `--no-stream` — disable live token streaming
- `--no-egraph-targets` — disable bounded target extraction from the saturated e-graph
- `--num-egraph-targets N` — number of nontrivial e-graph target bodies to extract per benchmark (default: `6`)
- `--output-dir DIR` — folder for per-run output files; each run writes a new timestamped `.txt` with the settings at the top (default: `outputs/`)

For example, to generate 3 distinct programs equivalent to the quadratic formula, allowing up
to 20 attempts:
```bash
uv run python -m egraph.run --benchmark quadratic.egglog -n 3 --max-tries 20
```

Before LLM decoding, the runner now extracts a bounded, structurally diverse set of
nontrivial equivalent bodies from the root e-class. Those targets can contribute programs
directly (for example, the Citardauq form of the quadratic formula) and any remaining targets
are rotated into prompts as rewrite-system-discovered shapes. This is generic over the
saturated e-graph rather than benchmark-specific prompting. To encourage additional distinct
programs, the prompt lists the programs already produced for the benchmark and asks for one
with different floating-point behavior from a different structural rewrite family when
possible. Accepted programs are deduplicated by a canonical form (whitespace normalized), so
trivial formatting changes are not counted as distinct.

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

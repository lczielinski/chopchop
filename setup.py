from setuptools import find_packages, setup

setup(
    name="chopchop",
    version="1.0",
    description=(
        "A programmable constrained decoder for semantic properties over "
        "AST-like program spaces."
    ),
    packages=find_packages(),
    package_data={
        "core": ["lark/*.lark"],
        "egraph": ["*.egglog", "*.lark", "benchmarks/*.egglog", "benchmarks/*.md"],
    },
)

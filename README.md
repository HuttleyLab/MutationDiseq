[![Coverage Status](https://coveralls.io/repos/github/HuttleyLab/MutationDiseq/badge.svg)](https://coveralls.io/github/HuttleyLab/MutationDiseq)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# mdeq: a tool for analysing mutation disequilibrium

A manuscript describing the methods will be made available as a preprint soon.

## Installation

```
$ pip install mdeq
```

> **Note**
> `accupy` is an optional package for the most numerically accurate routines involving matrices (e.g. dot products). These routines are explicitly employed for computing the nabla related statistics and are used if this package is installed. Unfortunately it is not easily installed and (at the time of writing) is not under active development.

## The available commands

<!-- [[[cog
import cog
from mdeq import main
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, ["--help"])
help = result.output.replace("Usage: main", "Usage: mdeq")
cog.out(
    "```\n{}\n```".format(help)
)
]]] -->
```
Usage: mdeq [OPTIONS] COMMAND [ARGS]...

  mdeq: mutation disequilibrium analysis tools.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  tui              Open Textual TUI.
  prep             pre-process alignment data.
  make-adjacent    makes sqlitedb of adjacent alignment records.
  toe              test of existence of mutation equilibrium.
  teop             between branch equivalence of mutation process test
  aeop             between loci equivalence of mutation process test
  convergence      estimates convergence towards mutation equilibrium.
  make-controls    simulate negative and positive controls
  db-summary       displays summary information about a db
  extract-pvalues  extracts p-values from TOE sqlitedb results
  extract-nabla-c  extracts nabla-c from convergence sqlitedb results
  slide            generate window sized sub-alignments.

```
<!-- [[[end]]] -->

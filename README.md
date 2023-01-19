# mdeq: a tool for analysing mutation disequilibrium

## Installation

Create a conda environment, install flit and the git version of cogent3. Then do a developer install of this tool, again using flit  

## Running the tests

```
$ pytest -n auto
```

This runs in parallel, greatly speeding things up.

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
  aeop             between loci equivalence of mutation process test
  convergence      estimates convergence towards mutation equilibrium.
  db-summary       displays summary information about a db
  extract-pvalues  extracts p-values from TOE sqlitedb results
  make-adjacent    makes sqlitedb of adjacent alignment records.
  make-controls    simulate negative and positive controls
  prep             pre-process alignment data.
  slide            generate window sized sub-alignments.
  teop             between branch equivalence of mutation process test
  toe              test of existence of mutation equilibrium.

```
<!-- [[[end]]] -->

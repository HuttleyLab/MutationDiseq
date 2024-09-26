# mdeq: a tool for analysing mutation disequilibrium

## Installation

Create a conda environment, install flit and the git version of cogent3. Then do a developer install of this tool, again using flit  

### Installing `accupy`

For the most numerically accurate results you will need to install `accupy`. This is just a `pip install`, but it requires you have the `Eigen` library installed.

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
  tui                  Open Textual TUI.
  prep                 pre-process alignment data.
  make-adjacent        makes sqlitedb of adjacent alignment records.
  toe                  test of existence of mutation equilibrium.
  teop                 between branch equivalence of mutation process test
  aeop                 between loci equivalence of mutation process test
  convergence          estimates convergence towards mutation equilibrium.
  make-controls        simulate negative and positive controls
  db-summary           displays summary information about a db
  extract-pvalues      extracts p-values from TOE sqlitedb results
  extract-delta-nabla  extracts delta-nabla from convergence sqlitedb results
  slide                generate window sized sub-alignments.

```
<!-- [[[end]]] -->

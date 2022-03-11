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
  aeop           test of equivalence of mutation equilibrium between...
  convergence    uses output from toe to generate delta_nabla.
  make-adjacent  makes tinydb of adjacent alignment records.
  make-controls  simulate negative and positive controls
  teop           test of equivalence of mutation equilibrium between branches.
  toe            test of existence of mutation equilibrium.

```
<!-- [[[end]]] -->

"""mdeq: mutation disequilibrium analysis tools"""

# following line to stop automatic threading by numpy
from . import _block_threading  # isort: skip  # make sure this stays at the top
from warnings import filterwarnings

from . import (
    model as _model,  # to ensure registration of define substitution models
)


__version__ = "2021.12.20"

filterwarnings("ignore", "Not using MPI")
filterwarnings("ignore", "Unexpected warning from scipy")
filterwarnings("ignore", "using slow exponentiator")
filterwarnings("ignore", ".*decreased to keep within bounds")

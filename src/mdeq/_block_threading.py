"""this module is to prevent over-subscription of CPUs when using codon models.

It arises because BLAS, relied on by numpy for linalg, auto-threads for
some problems
"""

import os


__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

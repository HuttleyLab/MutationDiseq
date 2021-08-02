from cogent3.app import evo, io
from cogent3.app.composable import (
    ALIGNED_TYPE,
    BOOTSTRAP_RESULT_TYPE,
    RESULT_TYPE,
    SERIALISABLE_TYPE,
    ComposableHypothesis,
    NotCompleted,
)
from cogent3.app.result import bootstrap_result, generic_result
from cogent3.util import misc, parallel

from kath_library.model import GN_sm, GS_instance, GS_sm
from kath_library.stationary_pi import OscillatingPiException

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]


def create_bootstrap_app(num_reps=5, discrete_edges=None):
    """
    wrapper of cogent3.app.evo.bootstrap with hypothesis of GS as the null and GN as the alternate
    """

    GS = GS_sm(discrete_edges)
    GN = GN_sm(discrete_edges)

    hyp = evo.hypothesis(GS, GN, sequential=False)
    bstrap = evo.bootstrap(hyp, num_reps)

    return bstrap


class confidence_interval(ComposableHypothesis):
    """
    Parametric bootstrap to give confidence intervals for a provided statistic.
    Returns a confindence_interval_result.

    Vendored with alterations from cogent3.evo.app.bootstrap
    """

    _input_types = (ALIGNED_TYPE, SERIALISABLE_TYPE)
    _output_types = SERIALISABLE_TYPE
    _data_types = ("ArrayAlignment", "Alignment")

    def __init__(self, stat_func, num_reps, parallel=False, verbose=False):
        super(confidence_interval, self).__init__(
            input_types=self._input_types,
            output_types=self._output_types,
            data_types=self._data_types,
        )
        self.stat_func = stat_func
        self._num_reps = num_reps
        self._verbose = verbose
        self._parallel = parallel
        self.func = self.run

    def fit_sim(self, rep_num):
        sim_aln = self.alt_params.simulate_alignment()
        sim_aln.info.source = "%s - simalign %d" % (self._inpath, rep_num)
        sim_aln.info.fg_edge = self.fg_edge

        try:
            sym_model_fit = self.alt(sim_aln)
            sym_result = self.stat_func(sym_model_fit)
        except ValueError:
            sym_result = None
        return sym_result, sym_model_fit

    def run(self, aln):
        result = generic_result(aln.info.source)

        self.fg_edge = aln.info.fg_edge
        self.bg_edges = list({self.fg_edge} ^ set(aln.names))

        GN = GN_sm(discrete_edges=self.bg_edges)

        self.alt = GN
        self.alt_params = self.alt(aln)
        try:
            obs = self.stat_func(self.alt_params)
        except OscillatingPiException as err:
            obs = NotCompleted(
                "ERROR - OscillatingPiException", str(self.stat_func), err.args[0]
            )
            return obs
        result["observed"] = obs
        self._inpath = aln.info.source

        map_fun = map if not self._parallel else parallel.imap
        sym_results = [r for r in map_fun(self.fit_sim, range(self._num_reps)) if r]
        null = {}
        for i, (sym_result, sym_model_fit) in enumerate(sym_results):
            if not sym_result:
                continue
            null[f"sym_{i+1}-result"] = sym_result
            null[f"sym_{i+1}-model_fit"] = sym_model_fit

        result.update(dict(null))
        return result


def null_distribution(gen_result):
    return [
        gen_result[k].to_rich_dict()["items"][0][1]
        for k in gen_result.keys()
        if k[-6:] == "result"
    ]


def null_models(gen_result):
    return [gen_result[k] for k in gen_result.keys() if k[-3:] == "fit"]


def plot_null_dist(gen_result):
    import numpy as np
    import plotly.express as px

    return px.histogram(np.array(gen_result.null_dist))

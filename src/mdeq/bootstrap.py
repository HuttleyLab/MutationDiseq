from copy import deepcopy

from cogent3.app import evo
from cogent3.app.composable import (
    ALIGNED_TYPE,
    BOOTSTRAP_RESULT_TYPE,
    RESULT_TYPE,
    SERIALISABLE_TYPE,
    ComposableHypothesis,
    NotCompleted,
    user_function,
)
from cogent3.app.result import bootstrap_result, generic_result
from cogent3.util import deserialise

from mdeq.model import GN_sm, GS_sm
from mdeq.stationary_pi import OscillatingPiException


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]


_aln_key = "alignment"


def _reconstitute_collection(data):
    """injects a top-level alignment into all individual model dicts"""
    aln = data.pop(_aln_key)
    # inject alignment into each model dict
    for _, mr in data["items"]:
        for _, m in mr["items"]:
            m[_aln_key] = deepcopy(aln)
    return data


@deserialise.register_deserialiser("compact_bootstrap_result")
def deserialise_compact(data):
    """returns a model_collection_result"""
    result_obj = compact_bootstrap_result(**data["result_construction"])

    for key, item in data["items"]:
        item = _reconstitute_collection(item)
        item = deserialise.deserialise_object(item)
        result_obj[key] = item

    return result_obj


def _eliminated_redundant_aln_in_place(hyp_result):
    """eliminates multiple definitions of alignment

    Parameters
    ----------
    hyp_result : hypothesis_result
        has two models, split_codons disallowed

    Returns
    -------
    dict
        alignment is moved to the top-level
    """
    aln = None
    for _, item in hyp_result["items"]:
        # this is a single hypothesis result will be individual
        r = item["items"][0][1].pop(_aln_key)
        if aln:
            assert aln == r, "mismatched alignments!"
        aln = r
    hyp_result[_aln_key] = aln
    return


class compact_bootstrap_result(bootstrap_result):
    """removes redundant alignments from individual model results"""

    def to_rich_dict(self):
        rd = super(self.__class__, self).to_rich_dict()
        # dict is modified within _eliminated_redundant_aln_in_place(
        for item in rd["items"]:
            _eliminated_redundant_aln_in_place(item[1])

        return rd


class bootstrap(ComposableHypothesis):
    """
    Parametric bootstrap for a provided hypothesis.
    Only returns the LR for the boostrapped models (to avoid overloading memory for use on nci)
    Returns a generic_result
    """

    _input_types = (ALIGNED_TYPE, SERIALISABLE_TYPE)
    _output_types = (RESULT_TYPE, BOOTSTRAP_RESULT_TYPE, SERIALISABLE_TYPE)
    _data_types = ("ArrayAlignment", "Alignment")

    def __init__(self, hyp, num_reps, verbose=False):
        super(bootstrap, self).__init__(
            input_types=self._input_types,
            output_types=self._output_types,
            data_types=self._data_types,
        )
        self._formatted_params()
        self._hyp = hyp
        self._num_reps = num_reps
        self._verbose = verbose
        self.func = self.run

    def _fit_sim(self, rep_num):
        sim_aln = self._null.simulate_alignment()
        sim_aln.info.source = "%s - simalign %d" % (self._inpath, rep_num)
        return self._hyp(sim_aln)

    def run(self, aln):
        result = compact_bootstrap_result(aln.info.source)
        try:
            obs = self._hyp(aln)
        except ValueError as err:
            result = NotCompleted("ERROR", str(self._hyp), err.args[0])
            return result

        result.observed = obs
        self._null = obs.null
        self._inpath = aln.info.source

        sim_results = [r for r in map(self._fit_sim, range(self._num_reps)) if r]
        for sim_result in sim_results:
            result.add_to_null(sim_result)

        return result


def create_bootstrap_app(num_reps=100, discrete_edges=None):
    """
    wrapper of cogent3.app.evo.bootstrap with hypothesis of GS as the null and GN as the alternate
    """

    GS = GS_sm(discrete_edges)
    GN = GN_sm(discrete_edges)

    hyp = evo.hypothesis(GS, GN, sequential=False)
    return bootstrap(hyp, num_reps)


def estimate_pval(result):
    obs = result.observed.LR
    num_ge = sum(v >= obs for v in result.null_dist)
    return num_ge / len(result.null_dist)


class confidence_interval(ComposableHypothesis):
    """
    Parametric bootstrap to give confidence intervals for a provided statistic.
    Returns a confindence_interval_result.

    Fits a General Nucleotide (GN) model to derive statistics.

    Vendored with alterations from cogent3.evo.app.bootstrap
    """

    _input_types = (ALIGNED_TYPE, SERIALISABLE_TYPE)
    _output_types = SERIALISABLE_TYPE
    _data_types = ("ArrayAlignment", "Alignment")

    def __init__(self, stat_func, num_reps, verbose=False):
        super(confidence_interval, self).__init__(
            input_types=self._input_types,
            output_types=self._output_types,
            data_types=self._data_types,
        )
        self.stat_func = stat_func
        self._num_reps = num_reps
        self._verbose = verbose
        self.func = self.run

    def fit_sim(self, rep_num):
        sim_aln = self.alt_params.simulate_alignment()
        sim_aln.info.source = "%s - simalign %d" % (self._inpath, rep_num)
        sim_aln.info.fg_edge = self.fg_edge

        try:
            sim_model_fit = self.alt(sim_aln)
            sim_result = self.stat_func(sim_model_fit)
        except ValueError:
            sim_result = None
        return sim_result, sim_model_fit

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
        result["observed-model_fit"] = self.alt_params

        self._inpath = aln.info.source

        sim_results = [r for r in map(self.fit_sim, range(self._num_reps)) if r]
        null = {}
        for i, (sim_result, sim_model_fit) in enumerate(sim_results):
            null[f"sim_{i+1}-result"] = sim_result
            null[f"sim_{i+1}-model_fit"] = sim_model_fit

        result.update(dict(null))
        return result

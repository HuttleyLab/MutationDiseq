from copy import deepcopy
from json import loads
from zlib import compress, decompress

from cogent3.app import evo
from cogent3.app.composable import (
    ALIGNED_TYPE,
    BOOTSTRAP_RESULT_TYPE,
    RESULT_TYPE,
    SERIALISABLE_TYPE,
    ComposableHypothesis,
    NotCompleted,
    appify,
)
from cogent3.app.result import bootstrap_result
from cogent3.util import deserialise
from tqdm import tqdm

from mdeq.model import GN_sm, GS_sm
from mdeq.toe import ALT_TOE, NULL_TOE, toe_on_edge


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]


_aln_key = "alignment"


def _reconstitute_collection(data):
    """injects a top-level alignment into all individual model dicts."""
    aln = data.pop(_aln_key)
    # inject alignment into each model dict
    for _, mr in data["items"]:
        for _, m in mr["items"]:
            m[_aln_key] = deepcopy(aln)
    return data


@deserialise.register_deserialiser("compact_bootstrap_result")
def deserialise_compact(data):
    """returns a compact_bootstrap_result."""
    result_obj = compact_bootstrap_result(**data["result_construction"])

    for key, item in data["items"]:
        item = _reconstitute_collection(item)
        item = deserialise.deserialise_object(item)
        result_obj[key] = item

    return result_obj


def _eliminated_redundant_aln_in_place(hyp_result):
    """eliminates multiple definitions of alignment.

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
    """removes redundant alignments from individual model results."""

    def __setitem__(self, key, data):
        # ignore validation checks, put compressed json straight
        # into self._store
        # NOTE: json requires less memory and is faster than using pickle
        self._store[key] = compress(data.to_json().encode("utf8"))

    def __getitem__(self, key):
        # decompress the values on the fly
        rd = loads(decompress(self._store[key]))
        return deserialise.deserialise_object(rd)

    def to_rich_dict(self):
        rd = super(self.__class__, self).to_rich_dict()
        # dict is modified within _eliminated_redundant_aln_in_place(
        for item in rd["items"]:
            _eliminated_redundant_aln_in_place(item[1])

        return rd

    @property
    def pvalue(self):
        obs = self.observed.get_hypothesis_result(NULL_TOE, ALT_TOE).LR
        if obs < 0:  # not optimised correctly?
            return 1.0

        size_valid = 0
        num_ge = 0
        for k, v in self.items():
            v = v.get_hypothesis_result(NULL_TOE, ALT_TOE)
            if k == "observed" or v.LR < 0:
                continue

            size_valid += 1
            if v.LR >= obs:
                num_ge += 1

        if size_valid == 0:
            return 1.0

        return num_ge / size_valid


class bootstrap(ComposableHypothesis):
    """Parametric bootstrap for a provided hypothesis.

    Only returns the LR for the boostrapped models (to avoid overloading
    memory for use on nci) Returns a generic_result
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

    def run(self, aln):
        result = compact_bootstrap_result(aln.info.source)
        try:
            obs = self._hyp(aln)
        except ValueError as err:
            return NotCompleted("ERROR", str(self._hyp), err.args[0])

        if isinstance(obs, NotCompleted):
            return obs

        result.observed = obs
        self._null = obs[NULL_TOE]
        self._inpath = aln.info.source

        series = range(self._num_reps)
        if self._verbose:
            series = tqdm(series)

        for i in series:
            sim_aln = self._null.simulate_alignment()
            sim_aln.info.update(aln.info)
            sim_aln.info.source = f"{self._inpath} - simalign {i}"
            sim_result = self._hyp(sim_aln)
            if not sim_result:
                continue

            result.add_to_null(sim_result)
            del sim_result

        return result


# todo reconcile usage and overlap between this and bootstrap_toe
def create_bootstrap_app(
    tree=None, num_reps=100, discrete_edges=None, opt_args=None, verbose=False
):
    """wrapper of cogent3.app.evo.bootstrap with hypothesis of GSN as the null
    and GN as the alternate."""

    GS = GS_sm(tree=tree, discrete_edges=discrete_edges, opt_args=opt_args)
    GN = GN_sm(tree=tree, discrete_edges=discrete_edges, opt_args=opt_args)

    hyp = evo.hypothesis(GS, GN, sequential=False)
    return bootstrap(hyp, num_reps, verbose=verbose)


@appify(
    (SERIALISABLE_TYPE, ALIGNED_TYPE),
    (RESULT_TYPE, BOOTSTRAP_RESULT_TYPE, SERIALISABLE_TYPE),
)
def bootstrap_toe(
    aln, tree=None, num_reps=100, sequential=False, opt_args=None, verbose=False
):
    """dynamically constructs a bootstrap app and performs the toe."""
    if isinstance(aln, NotCompleted):
        return aln
    hyp = toe_on_edge(
        aln, tree=tree, with_gtr=False, sequential=sequential, opt_args=opt_args
    )
    bstrapper = bootstrap(hyp, num_reps, verbose=verbose)
    return bstrapper(aln)

import json
from copy import deepcopy

from cogent3.app import evo
from cogent3.app import io as io_app
from cogent3.app.composable import NotCompleted, define_app, get_unique_id
from cogent3.app.result import bootstrap_result
from cogent3.app.typing import AlignedSeqsType, SerialisableType
from cogent3.util import deserialise, union_dict
from rich.progress import track

from mdeq.model import GN_sm, GS_sm
from mdeq.toe import ALT_TOE, NULL_TOE, test_of_existence

_aln_key = "alignment"


def _reconstitute_collection(data):
    """injects a top-level alignment into all individual model dicts."""
    if _aln_key not in data:
        return data
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
        result_obj[key] = item

    return result_obj


def deserialise_single_hyp(data: dict):
    """
    custom deserialiser for compact bootstrap model_collection_results

    Parameters
    ----------
    data : dict
        will contain a single top level 'alignment' key

    Returns
    -------
    model_collection_result
    """
    from cogent3.util.deserialise import deserialise_object

    from mdeq.utils import CompressedValue

    data = CompressedValue(data["data"]).as_primitive
    _reconstitute_collection(data)
    return deserialise_object(data)


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


class compact_bootstrap_result(bootstrap_result):
    """removes redundant alignments from individual model results."""

    compress = io_app.pickle_it() + io_app.compress()
    decompress = io_app.decompress() + io_app.unpickle_it()

    def __setitem__(self, key, data):
        # ignore validation checks, put compressed json straight
        # into self._store
        # NOTE: json requires less memory and is faster than using pickle
        # I create an intermediate dict that contains top level statistics (LR, df, pvalue)
        # so we can delay decompression
        if isinstance(data, dict):
            data = union_dict.UnionDict(**data)
            self._store[key] = data
            return

        if hasattr(data, "get_hypothesis_result"):
            hyp = data.get_hypothesis_result(NULL_TOE, ALT_TOE)

        rd = data.to_rich_dict()
        _eliminated_redundant_aln_in_place(rd)

        data = union_dict.UnionDict(
            LR=hyp.LR,
            df=hyp.df,
            pvalue=hyp.pvalue,
            data=self.compress(rd),
        )

        self._store[key] = data

    def __getitem__(self, key):
        # decompress the values on the fly
        return self._store[key]

    def to_rich_dict(self, to_json=False):
        rd = super(self.__class__, self).to_rich_dict()
        if not to_json:
            # this for the sqlitedb version where values can be binary
            return rd

        # decompress values if fit new slitedb storeage pattern for this
        # result type
        for item in rd["items"]:
            if "data" in item[1]:
                item[1]["data"] = self.decompress(item[1]["data"])

        return rd

    def to_json(self):
        return json.dumps(self.to_rich_dict(to_json=True))

    @property
    def pvalue(self):
        obs = self.observed.LR
        if obs < 0:  # not optimised correctly?
            return 1.0

        # subtract 1 for the observed
        size_valid = -1
        num_ge = -1
        for v in self.values():
            if v.LR < 0:
                continue

            size_valid += 1
            if obs <= v.LR:
                num_ge += 1

        if size_valid == 0:
            return 1.0

        return num_ge / size_valid

    def deserialised_values(self):
        """inflate all values"""
        from cogent3.util.deserialise import deserialise_object

        from mdeq.utils import CompressedValue

        for k, v in self._store.items():
            if "data" in v:
                v = CompressedValue(v["data"]).as_primitive
                v = _reconstitute_collection(v)
            self._store[k] = deserialise_object(v)


@define_app
class bootstrap:
    """Parametric bootstrap for a provided hypothesis.

    Only returns the LR for the boostrapped models (to avoid overloading
    memory for use on nci) Returns a generic_result
    """

    def __init__(self, hyp, num_reps, verbose=False):
        self._hyp = hyp
        self._num_reps = num_reps
        self._verbose = verbose

    def main(
        self,
        aln: AlignedSeqsType,
    ) -> SerialisableType | compact_bootstrap_result:
        result = compact_bootstrap_result(get_unique_id(aln))
        try:
            obs = self._hyp(aln)
        except ValueError as err:
            return NotCompleted("ERROR", str(self._hyp), err.args[0])

        if isinstance(obs, NotCompleted):
            return obs

        result.observed = obs
        self._null = obs[NULL_TOE]
        self._inpath = get_unique_id(aln)

        series = range(self._num_reps)
        if self._verbose:
            series = track(series)

        for i in series:
            sim_aln = self._null.simulate_alignment()
            sim_aln.info.update(aln.info)
            sim_aln.source = f"{self._inpath} - simalign {i}"
            sim_result = self._hyp(sim_aln)
            if not sim_result:
                continue

            result.add_to_null(sim_result)
            del sim_result

        return result


# TODO reconcile usage and overlap between this and bootstrap_toe
def create_bootstrap_app(
    tree=None,
    just_continuous=False,
    num_reps=100,
    discrete_edges=None,
    opt_args=None,
    verbose=False,
):
    """wrapper of cogent3.app.evo.bootstrap with hypothesis of GSN as the null
    and GN as the alternate."""
    if just_continuous:
        discrete_edges = None
    GS = GS_sm(tree=tree, discrete_edges=discrete_edges, opt_args=opt_args)
    GN = GN_sm(tree=tree, discrete_edges=discrete_edges, opt_args=opt_args)

    hyp = evo.hypothesis(GS, GN, sequential=False)
    return bootstrap(hyp, num_reps, verbose=verbose)


@define_app
def bootstrap_toe(
    aln: AlignedSeqsType,
    tree=None,
    just_continuous=False,
    num_reps=100,
    sequential=False,
    opt_args=None,
    verbose=False,
) -> SerialisableType | compact_bootstrap_result:
    """dynamically constructs a bootstrap app and performs the toe."""
    if isinstance(aln, NotCompleted):
        return aln
    hyp = test_of_existence(
        aln,
        just_continuous=just_continuous,
        tree=tree,
        with_gtr=False,
        sequential=sequential,
        opt_args=opt_args,
    )
    bstrapper = bootstrap(hyp, num_reps, verbose=verbose)
    return bstrapper(aln)

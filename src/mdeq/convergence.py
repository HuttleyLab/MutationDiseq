import json

from dataclasses import dataclass
from functools import lru_cache, singledispatch
from types import NoneType

from accupy import fdot as dot
from cogent3.app.composable import SERIALISABLE_TYPE, user_function
from cogent3.app.result import generic_result
from numpy import array, std
from numpy.linalg import eig, norm
from cogent3.util import deserialise
from cogent3.util.misc import get_object_provenance
from scipy.linalg import expm

from mdeq.model import GN_sm, GS_sm
from mdeq.utils.utils import get_foreground, get_pi_0, get_pi_tip


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Ben Kaehler", "Gavin Huttley"]


def unit_stationary_Q(pi_0: ndarray, Q: ndarray):
    """returns Q with sum(pi_i Q[i,i]) == -1 given pi_0"""
    indices = diag_indices(Q.shape[0])
    scalar = -sum(pi_0 * Q[indices])
    Q /= scalar
    scalar = -sum(pi_0 * Q[indices])
    assert allclose(scalar, 1.0)
    return Q





def convergence(pi_0, Q, t):
    """a measure of how fast pi(t) is changing."""
    pi_deriv = dot(pi_0, dot(Q, expm(Q * t)))
    return norm(pi_deriv)


def _get_convergence_mc(mc):
    """Wrapper function to return convergence estimate from a model collection
    that includes a GN fit.
@dataclass(eq=True)
class delta_nabla:
    obs_nabla: float
    null_nabla: tuple[float]
    size_null: int = None
    source: str = None

    Returns a generic_result
    """

    alt = mc["mcr"]["GN"]

    bg_edges = alt.lf.to_rich_dict()["likelihood_construction"]["discrete_edges"]
    (fg_edge,) = set(bg_edges) ^ set(alt.alignment.names)
    def __post_init__(self):
        if len(self.null_nabla) <= 1:
            raise ValueError("len null distribution must be > 1")

    Q = alt.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_0(alt)
    t = alt.lf.get_param_value("length", edge=fg_edge)
        self.null_nabla = tuple(self.null_nabla)
        self.size_null = len(self.null_nabla)

    observed_conv = convergence(pi, Q, t)
    def __hash__(self):
        return id((self.obs_nabla, self.null_nabla))

    null = mc["mcr"]["GSN"]
    GN = GN_sm(bg_edges)
    neutral_convs = []
    @property
    @lru_cache()
    def mean_null(self):
        return mean(self.null_nabla)

    while len(neutral_convs) < 10:
        i = len(neutral_convs) + 1
        sim_aln = null.lf.simulate_alignment()
        sim_aln.info.source = "%s - simalign %d" % (mc.source, i)
        sim_aln.info.fg_edge = fg_edge
    @property
    @lru_cache()
    def std_null(self):
        return std(self.null_nabla, ddof=1)

        try:
            sim_model_fit = GN(sim_aln)
            sim_Q = sim_model_fit.lf.get_rate_matrix_for_edge(
                fg_edge, calibrated=False
            ).to_array()
            sim_pi = get_pi_0(sim_model_fit)
            sim_t = sim_model_fit.lf.get_param_value("length", edge=fg_edge)

            sim_conv = convergence(sim_pi, sim_Q, sim_t)
            neutral_convs.append(sim_conv)
        except ValueError:
            sim_result = None
    @property
    @lru_cache()
    def delta_nabla(self):
        """returns observed nabla minus mean of the null nabla distribution"""
        return self.obs_nabla - self.mean_null

    neutral_mean = array(neutral_convs).mean()
    neutral_std = std(array(neutral_convs), ddof=1)

    conv_normalised = (observed_conv - neutral_mean) / neutral_std
    def to_rich_dict(self):
        return {
            "obs_nabla": self.obs_nabla,
            "null_nabla": self.null_nabla,
            "size_null": self.size_null,
            "type": get_object_provenance(self),
            "source": self.source,
        }

    result = generic_result(source=mc.source)
    result.update(
        [
            ("convergence_normalised", conv_normalised),
            ("convergence", observed_conv),
            ("fg_edge", fg_edge),
            ("source", mc.source),
            ("model_fit", alt),
        ]
    )
    def to_json(self):
        return json.dumps(self.to_rich_dict())

    return result
    @classmethod
    def from_json(cls, data):
        """constructor from json data"""
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict):
        """constructor from dict data"""
        data.pop("type", None)
        return cls(**data)


get_convergence_mc = user_function(
    _get_convergence_mc, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)
@deserialise.register_deserialiser("delta_nabla")
def deserialise_delta_nabla(data: dict):
    """recreates delta_nabla instance from dict"""
    return delta_nabla.from_dict(data)


def _get_convergence(gn_sm, opt_args=None):
    """Wrapper function to return convergence estimate from a non-stationary
    model fit.

    Returns a generic_result
    """
    opt_args = opt_args or {}
    opt_args = {"max_restarts": 5, "tolerance": 1e-8, **opt_args}

    bg_edges = gn_sm.lf.to_rich_dict()["likelihood_construction"]["discrete_edges"]
    (fg_edge,) = set(bg_edges) ^ set(gn_sm.alignment.names)

    Q = gn_sm.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_0(gn_sm)
    t = gn_sm.lf.get_param_value("length", edge=fg_edge)

    conv = convergence(pi, Q, t)

    GS = GS_sm(bg_edges, opt_args=opt_args)
    GN = GN_sm(bg_edges, opt_args=opt_args)

    null = GN(gn_sm.alignment)
    neutral_convs = []
@singledispatch
def get_nabla(fg_edge, gn_result=None, time_delta=None, wrt_nstat=False) -> float:
    """returns the convergence statistic from a model_result object

    while len(neutral_convs) < 10:
        i = len(neutral_convs) + 1
        sim_aln = null.lf.simulate_alignment()
        sim_aln.info.source = "simalign %d" % (i)
        sim_aln.info.fg_edge = fg_edge

        try:
            sim_model_fit = GN(sim_aln)
            sim_Q = sim_model_fit.lf.get_rate_matrix_for_edge(
                fg_edge, calibrated=False
            ).to_array()
            sim_pi = get_pi_0(sim_model_fit)
            sim_t = sim_model_fit.lf.get_param_value("length", edge=fg_edge)

            sim_conv = convergence(sim_pi, sim_Q, sim_t)
            neutral_convs.append(sim_conv)
        except ValueError:
            sim_result = None

    neutral_mean = array(neutral_convs).mean()
    neutral_std = std(array(neutral_convs), ddof=1)

    conv_normalised = (conv - neutral_mean) / neutral_std

    result = generic_result(source=gn_sm.source)
    result.update([("convergence", conv), ("convergence_normalised", conv_normalised)])

    return result


get_convergence = user_function(
    _get_convergence, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)


def _get_convergence_bstrap(result):
    """Wrapper function to return convergence estimate from a generic_result
    generated from a bootstrap app..

    Returns a generic_result
    """

    alt = result["observed"].alt
    null = result["observed"].null

    bg_edges = alt.lf.to_rich_dict()["likelihood_construction"]["discrete_edges"]
    (fg_edge,) = set(bg_edges) ^ set(alt.alignment.names)

    Q = alt.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_0(alt)
    t = alt.lf.get_param_value("length", edge=fg_edge)

    conv = convergence(pi, Q, t)

    GN = GN_sm(bg_edges)
    neutral_convs = []

    while len(neutral_convs) < 10:
        i = len(neutral_convs) + 1
        sim_aln = null.lf.simulate_alignment()
        sim_aln.info.source = "%s - simalign %d" % (result.source, i)
        sim_aln.info.fg_edge = fg_edge

        try:
            sim_model_fit = GN(sim_aln)
            sim_Q = sim_model_fit.lf.get_rate_matrix_for_edge(
                fg_edge, calibrated=False
            ).to_array()
            sim_pi = get_pi_0(sim_model_fit)
            sim_t = sim_model_fit.lf.get_param_value("length", edge=fg_edge)

            sim_conv = convergence(sim_pi, sim_Q, sim_t)
            neutral_convs.append(sim_conv)
        except ValueError:
            sim_result = None

    neutral_mean = array(neutral_convs).mean()
    neutral_std = std(array(neutral_convs), ddof=1)

    conv_normalised = (conv - neutral_mean) / neutral_std

    result = generic_result(source=result.source)
    result.update(
        [
            ("convergence_normalised", conv_normalised),
            ("convergence", conv),
            ("fg_edge", fg_edge),
            ("source", result.source),
        ]
    )

    return result


get_convergence_bstrap = user_function(
    _get_convergence_bstrap,
    input_types=SERIALISABLE_TYPE,
    output_types=SERIALISABLE_TYPE,
)

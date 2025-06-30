import json
from dataclasses import dataclass
from functools import singledispatch
from types import NoneType
from typing import ForwardRef

from cogent3 import make_table
from cogent3.app.composable import NotCompleted, define_app
from cogent3.app.data_store import DataStoreABC
from cogent3.app.typing import SerialisableType
from cogent3.maths.matrix_exponential_integration import expected_number_subs
from cogent3.recalculation.scope import InvalidScopeError
from cogent3.util import deserialise
from cogent3.util.misc import get_object_provenance
from numpy import allclose, diag_indices, mean, ndarray, std
from numpy.linalg import norm
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

from mdeq.numeric import dot, sum
from mdeq.toe import ALT_TOE
from mdeq.utils import SerialisableMixin, load_from_sqldb


def unit_stationary_Q(pi_0: ndarray, Q: ndarray):
    """returns Q with sum(pi_i Q[i,i]) == -1 given pi_0."""
    indices = diag_indices(Q.shape[0])
    scalar = -sum(pi_0 * Q[indices])
    Q /= scalar
    scalar = -sum(pi_0 * Q[indices])
    assert allclose(scalar, 1.0)
    return Q


def unit_nonstationary_Q(pi_0: ndarray, Q: ndarray) -> ndarray:
    """returns Q with ENS==1 given pi_0."""
    result = minimize_scalar(
        lambda x: (1.0 - expected_number_subs(pi_0, Q, x)) ** 2,
        method="brent",
        tol=1e-8,
    )
    if not result.success:
        raise RuntimeError(result.message)

    Q *= result.x
    return Q


def convergence(pi_0: ndarray, Q: ndarray, t: float, wrt_nstat=False) -> float:
    """measure of how fast pi is changing.

    Parameters
    ----------
    pi_0 : 1D array
        a valid probability vector representing the initial state freqs
    Q : 2D array
        a valid rate matrix
    t : float
        tau, scalar of unit-Q for the non-stationary process
    wrt_nstat : bool
        If True, Q is calibrated such that as a non-stationary process,
        i.e. ENS(pi_0, Q) == 1, otherwise it will calibrated as a stationary
        process, i.e. -sum(pi_0_j * Q[j, j])==1

    Returns
    -------
    float

    Notes
    -----
    See Kaehler et al (2015) Systematic Biology for details
    """
    # make sure Q is scaled relative to pi_0
    Q = unit_nonstationary_Q(pi_0, Q) if wrt_nstat else unit_stationary_Q(pi_0, Q)
    pi_deriv = dot(pi_0, dot(Q, expm(Q * t)))
    return norm(pi_deriv)


@dataclass(eq=True)
class nabla_c(SerialisableMixin):
    obs_nabla: float
    null_nabla: tuple[float]
    fg_edge: str
    size_null: int | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if len(self.null_nabla) <= 1:
            msg = "len null distribution must be > 1"
            raise ValueError(msg)

        self.null_nabla = tuple(self.null_nabla)
        self.size_null = len(self.null_nabla)

    def __hash__(self) -> int:
        return id((self.source, self.fg_edge, self.obs_nabla, self.null_nabla))

    @property
    def mean_null(self):
        return mean(self.null_nabla)

    @property
    def std_null(self):
        return std(self.null_nabla, ddof=1)

    @property
    def nabla_c(self):
        """returns observed nabla minus mean of the null nabla distribution."""
        return self.obs_nabla - self.mean_null

    def to_json(self) -> str:
        return json.dumps(self.to_rich_dict())


@deserialise.register_deserialiser(
    get_object_provenance(nabla_c),
    "mdeq.convergence.delta_nabla",
)
def deserialise_nabla_c(data: dict) -> nabla_c:
    """recreates nabla_c instance from dict."""
    return nabla_c.from_dict(data)


@singledispatch
def get_nabla(
    fg_edge,
    gn_result=None,
    time_delta=None,
    wrt_nstat=False,
) -> tuple[str, float]:
    """returns the convergence statistic from a model_result object.

    Parameters
    ----------
    fg_edge : str or None
        a designated edge to compute nabla for, by default None. If not
        specified, identifies the single edge that is continuous time.
        In the latter case, we assume a time-homogeneous process and
        derive nabla from a rate matrix calibrated for a given expected
        number of substitutions. If all edges are continuous-time,
        employs the time_delta.
    gn_result : model_result
        for a general Markov nucleotide model
    time_delta : float, optional
        when fg_edge is None, interpreted as the scalar to multiply the
        calibrated matrix default 0.1
    wrt_nstat : bool
        argument to convergence function, triggers calibration for
        a non-stationary process

    Notes
    -----
    Either all edges are continuous-time or one edge is continuous time.
    """
    raise NotImplementedError("Implement get_nabla")


@get_nabla.register(str)
def _(fg_edge, gn_result=None, time_delta=None, wrt_nstat=False) -> tuple[str, float]:
    # this is triggered when a fg_edge IS specified as a str
    # we infer the time_delta from that edge if it wasn't provided
    assert isinstance(time_delta, (NoneType, float))
    pi_0 = gn_result.lf.get_param_value("mprobs")
    time_delta = time_delta or gn_result.lf.get_param_value("length", edge=fg_edge)
    Q = gn_result.lf.get_rate_matrix_for_edge(fg_edge).to_array()
    return fg_edge, convergence(pi_0, Q, time_delta, wrt_nstat=wrt_nstat)


@get_nabla.register(NoneType)
def _(fg_edge, gn_result=None, time_delta=None, wrt_nstat=False) -> tuple[str, float]:
    # this is triggered when fg_edge is None (i.e. not specified), in which
    # case we need to identify whether there's a clear fg_edge candidate or
    # select an edge as a representative
    # TODO assumes time-homogeneous process
    assert isinstance(time_delta, (NoneType, float))
    tree = gn_result.lf.tree
    names = [e.name for e in tree.get_edge_vector(include_root=False)]
    num_Q = 0
    for name in names:
        try:
            # is this edge modelled using discrete-time process
            gn_result.lf.get_param_value("dpsubs", edge=name)
        except (InvalidScopeError, KeyError):
            # continuous time model
            fg_edge = name
            num_Q += 1

    if num_Q == 0:
        raise ValueError("at least one edge must be continuous-time")

    if num_Q == 1:
        # we identified a single continuous-time fg_edge, so we want length
        # from that, which is done in the other function
        time_delta = None

    if num_Q > 1:
        if num_Q != len(names):
            raise NotImplementedError(
                f"either one or all {len(names)} edges are continuous-time, not {num_Q}",
            )
        # use a tip name, to facilitate testing
        fg_edge = tree.get_tip_names()[0]
        time_delta = time_delta or 0.1  # we want a set length

    return get_nabla(
        fg_edge,
        gn_result=gn_result,
        time_delta=time_delta,
        wrt_nstat=wrt_nstat,
    )


def get_nabla_c(
    obs_result,
    sim_results,
    fg_edge=None,
    wrt_nstat=False,
) -> nabla_c:
    """returns the adjusted nabla statistic.

    Parameters
    ----------
    obs_result : model_result
        resulting from fitting a GN model to observed data
    sim_results : series of model_result objects
        each one is from fitting a GN model to data generated under the
        null (GNS). The latter model having been fit to the observed data.
    fg_edge : str or None
        a designated edge to compute nabla for, by default None.
    wrt_nstat : bool
        If True, Q is calibrated such that as a non-stationary process,
        i.e. ENS(pi_0, Q) == 1, otherwise it will calibrated as a stationary
        process, i.e. -sum(pi_0_j * Q[j, j])==1

    """
    kwargs = dict(wrt_nstat=wrt_nstat)
    fg_edge, obs_nabla = get_nabla(fg_edge, gn_result=obs_result, **kwargs)
    sim_nabla = tuple(get_nabla(fg_edge, gn_result=r, **kwargs)[1] for r in sim_results)
    return nabla_c(obs_nabla, tuple(sim_nabla), fg_edge, source=obs_result.source)


@define_app
def bootstrap_to_nabla(
    result: ForwardRef("compact_bootstrap_result"),
    fg_edge=None,
    wrt_nstat=False,
) -> nabla_c | SerialisableType:
    """returns delta nabla stats from bootstrap result."""
    from mdeq.bootstrap import deserialise_single_hyp

    if isinstance(result, NotCompleted):
        return result

    null_results = []
    obs_result = result.observed
    if "data" in obs_result:
        obs_result = deserialise_single_hyp(obs_result)

    obs_result = obs_result[ALT_TOE]

    for k, v in result.items():
        if k == "observed":
            continue
        if "data" in v:
            v = deserialise_single_hyp(v)
        null_results.append(v[ALT_TOE])

    return get_nabla_c(
        obs_result,
        null_results,
        fg_edge=fg_edge,
        wrt_nstat=wrt_nstat,
    )


def nabla_c_table(dstore: DataStoreABC) -> "Table":
    """returns the centered nabla statistics from a convergence type."""
    loader = load_from_sqldb()
    rows = []
    for m in dstore.completed:
        r = loader(m)
        rows.append((r.source, r.nabla_c, r.std_null))
    return make_table(header=["source", "nabla_c", "std"], data=rows)

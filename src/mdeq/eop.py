from typing import ForwardRef

from cogent3 import get_model, make_tree
from cogent3.app import evo
from cogent3.app import result as c3_result
from cogent3.app.composable import NotCompleted, define_app, get_unique_id
from cogent3.app.typing import (
    AlignedSeqsType,
    HypothesisResultType,
    SerialisableType,
)

from mdeq.model import RATE_PARAM_UPPER
from mdeq.utils import get_foreground

NULL_AEOP = "GN-aeop-null"
ALT_AEOP = "GN-aeop-alt"

NULL_TEOP = "GN-teop-null"
ALT_TEOP = "GN-teop-alt"


@define_app
class adjacent_eop:
    def __init__(self, tree=None, opt_args=None, share_mprobs=True, time_het=None):
        opt_args = opt_args or {}
        self._opt_args = {
            "max_restarts": 5,
            "tolerance": 1e-8,
            "show_progress": False,
            **opt_args,
        }
        self._tree = tree
        self._share_mprobs = share_mprobs
        time_het = time_het or {}
        self._time_het = {"upper": RATE_PARAM_UPPER, **time_het}

    def _background_edges(self, data):
        selected_foreground = [get_foreground(e) for e in data.elements]
        if len(set(selected_foreground)) != 1:
            return NotCompleted(
                "ERROR",
                self,
                f"inconsistent foreground edges {selected_foreground}",
                source=data,
            )

        fg_edge = selected_foreground[0]
        if fg_edge is None:
            return None

        return list({fg_edge} ^ set(data.elements[0].names))

    def main(
        self,
        data: ForwardRef("grouped"),
        *args,
        **kwargs,
    ) -> HypothesisResultType | SerialisableType:
        """fits multiple adjacent loci in."""
        # TODO is it possible to get the param rules from each locus in null
        # if so, they could then be applied to the corresponding alternate
        bg_edges = self._background_edges(data)
        if isinstance(bg_edges, NotCompleted):
            return bg_edges

        aligns = {}
        for i, e in enumerate(data.elements):
            n = e.info.get("name", f"locus-{i}")
            aligns[n] = e

        names = list(aligns)
        if self._tree is None:
            assert len(data.elements[0].names) == 3, (
                f"need tree specified for {len(data.elements)} seqs"
            )
            tree = make_tree(tip_names=data.elements[0].names)
        else:
            tree = self._tree

        null = get_model("GN", optimise_motif_probs=True)
        lf = null.make_likelihood_function(
            tree,
            loci=names,
            discrete_edges=bg_edges,
            expm="pade",
        )
        lf.set_time_heterogeneity(**self._time_het)
        lf.set_alignment([aligns[k] for k in names])
        if self._share_mprobs:
            lf.set_param_rule("mprobs", is_independent=False)
        lf.optimise(**self._opt_args)
        lf.name = "null"

        null_result = c3_result.model_result(source=get_unique_id(data))
        null_result["null"] = lf
        # each alignment fit separately under alt
        alt_results = c3_result.model_result(source=get_unique_id(data))
        alt = get_model("GN", optimise_motif_probs=True)
        for locus, aln in aligns.items():
            lf = alt.make_likelihood_function(
                tree,
                discrete_edges=bg_edges,
                expm="pade",
            )
            lf.set_alignment(aln)
            lf.set_time_heterogeneity(**self._time_het)
            lf.optimise(**self._opt_args)
            lf.name = aln.info.name
            alt_results[locus] = lf

        combined = c3_result.hypothesis_result(NULL_AEOP, source=get_unique_id(data))
        combined[NULL_AEOP] = null_result
        combined[ALT_AEOP] = alt_results
        return combined


@define_app
class temporal_eop:
    def __init__(self, edge_names, tree=None, opt_args=None):
        """performs the temporal Equivalence of Process LRT

        Parameters
        ----------
        edge_names : list[str]
            names of edges whose EOP is to be assessed
        tree : str, TreeNode, None
            newick formatted, a  cogent3 TreeNode instance or None. If None,
            the alignments MUST only have 3 sequences and a star tree will
            be consructed.
        opt_args : dict
            arguments for the numerical optimisations step
        """
        opt_args = opt_args or {}
        self._opt_args = {
            "max_restarts": 5,
            "tolerance": 1e-8,
            "show_progress": False,
            **opt_args,
        }
        assert not isinstance(edge_names, str) and len(edge_names) > 1, (
            "must specify > 1 edge name"
        )
        self._edge_names = edge_names
        self._tree = tree
        self._hyp = None

    def _get_app(self, aln):
        if self._tree is None:
            assert len(aln.names) == 3
            self._tree = make_tree(tip_names=aln.names)
        assert set(self._tree.get_tip_names()) == set(aln.names)
        if self._hyp is None:
            null = evo.model(
                "GN",
                time_het=[
                    dict(
                        edges=self._edge_names,
                        is_independent=False,
                        upper=RATE_PARAM_UPPER,
                    ),
                ],
                name=NULL_TEOP,
                opt_args=self._opt_args,
                upper=RATE_PARAM_UPPER,
                optimise_motif_probs=True,
            )
            alt = evo.model(
                "GN",
                name=ALT_TEOP,
                opt_args=self._opt_args,
                time_het=[
                    dict(
                        edges=self._edge_names,
                        is_independent=True,
                        upper=RATE_PARAM_UPPER,
                    ),
                ],
                upper=RATE_PARAM_UPPER,
                optimise_motif_probs=True,
            )
            self._hyp = evo.hypothesis(null, alt)
        return self._hyp

    def main(
        self,
        data: AlignedSeqsType,
        *args,
        **kwargs,
    ) -> HypothesisResultType | SerialisableType:
        app = self._get_app(data)
        return app(data)

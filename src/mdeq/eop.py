from cogent3 import get_model, make_tree
from cogent3.app import evo
from cogent3.app import result as c3_result
from cogent3.app.composable import (
    SERIALISABLE_TYPE,
    ComposableAligned,
    NotCompleted,
)

from mdeq.utils import get_foreground


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]

NULL_AEOP = "null"
ALT_AEOP = "alt"

NULL_TEOP = "GN-teop"
ALT_TEOP = "GN"


class adjacent_eop(ComposableAligned):
    def __init__(self, tree=None, opt_args=None, share_mprobs=True):
        super(adjacent_eop, self).__init__(
            data_types="grouped",
            input_types=SERIALISABLE_TYPE,
            output_types=SERIALISABLE_TYPE,
        )
        opt_args = opt_args or {}
        self._opt_args = {
            "max_restarts": 5,
            "tolerance": 1e-8,
            "show_progress": False,
            **opt_args,
        }
        self._tree = tree
        self._share_mprobs = share_mprobs
        self.func = self.fit

    def _background_edges(self, data):
        selected_foreground = [get_foreground(e) for e in data.elements]
        if len(set(selected_foreground)) != 1:
            return NotCompleted(
                "ERROR",
                self,
                f"inconsistent foreground edges {selected_foreground}",
                source=data.source,
            )

        fg_edge = selected_foreground[0]
        if fg_edge is None:
            return None

        return list({fg_edge} ^ set(data.elements[0].names))

    def fit(self, data, *args, **kwargs):
        """fits multiple adjacent loci in."""
        # todo is it possible to get the param rules from each locus in null
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
            assert (
                len(data.elements[0].names) == 3
            ), f"need tree specified for {len(data.elements)} seqs"
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
        lf.set_alignment([aligns[k] for k in names])
        if self._share_mprobs:
            lf.set_param_rule("mprobs", is_independent=False)
        lf.optimise(**self._opt_args)
        lf.name = "null"

        null_result = c3_result.model_result(source=data.source)
        null_result["null"] = lf
        # each alignment fit separately under alt
        alt_results = c3_result.model_result(source=data.source)
        alt = get_model("GN", optimise_motif_probs=True)
        for locus, aln in aligns.items():
            lf = alt.make_likelihood_function(
                tree,
                discrete_edges=bg_edges,
                expm="pade",
            )
            lf.set_alignment(aln)
            lf.optimise(**self._opt_args)
            lf.name = aln.info.name
            alt_results[locus] = lf

        combined = c3_result.hypothesis_result("null", source=data.source)
        combined[NULL_AEOP] = null_result
        combined[ALT_AEOP] = alt_results
        return combined


class temporal_eop(ComposableAligned):
    def __init__(self, edge_names, tree=None, opt_args=None):
        super(temporal_eop, self).__init__(
            input_types=SERIALISABLE_TYPE,
            output_types=SERIALISABLE_TYPE,
        )
        opt_args = opt_args or {}
        self._opt_args = {
            "max_restarts": 5,
            "tolerance": 1e-8,
            "show_progress": False,
            **opt_args,
        }
        assert (
            not isinstance(edge_names, str) and len(edge_names) > 1
        ), "must specify > 1 edge name"
        self._edge_names = edge_names
        self._tree = tree
        self._hyp = None

        self.func = self.fit

    def _get_app(self, aln):
        if self._tree is None:
            assert len(aln.names) == 3
            self._tree = make_tree(tip_names=aln.names)
        assert set(self._tree.get_tip_names()) == set(aln.names)
        if self._hyp is None:
            null = evo.model(
                "GN",
                time_het=[
                    dict(edges=self._edge_names, is_independent=False, upper=100)
                ],
                name=NULL_TEOP,
                opt_args=self._opt_args,
            )
            alt = evo.model(
                "GN",
                name=ALT_TEOP,
                opt_args=self._opt_args,
                time_het=[dict(edges=self._edge_names, is_independent=True, upper=100)],
            )
            self._hyp = evo.hypothesis(null, alt)
        return self._hyp

    def fit(self, data, *args, **kwargs):
        app = self._get_app(data)
        return app(data)

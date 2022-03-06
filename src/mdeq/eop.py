from typing import Sequence

import numpy

from accupy import fsum
from cogent3 import get_model, make_table, make_tree
from cogent3.app import evo
from cogent3.app import result as c3_result
from cogent3.app.composable import ComposableAligned, NotCompleted
from cogent3.maths.stats import chisqprob

from mdeq.utils.numeric_utils import fix_rounding_error


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]

from mdeq.utils.utils import get_foreground


class adjacent_eop(ComposableAligned):
    def __init__(self, tree=None, opt_args=None):
        super(adjacent_eop, self).__init__(data_types="grouped_data")
        opt_args = opt_args or {}
        self._opt_args = {
            "max_restarts": 5,
            "tolerance": 1e-8,
            "show_progress": False,
            **opt_args,
        }
        self._tree = tree
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
        """fits multiple adjacent loci in"""
        bg_edges = self._background_edges(data)
        if isinstance(bg_edges, NotCompleted):
            return bg_edges

        aligns = {e.info.name: e for e in data.elements}
        names = list(aligns)
        if self._tree is None:
            assert (
                len(data.elements[0].names) == 3
            ), f"need tree specified for {len(data.elements)} seqs"
            tree = make_tree(tip_names=data.elements[0].names)
        else:
            tree = self._tree

        null = get_model("GSN", optimise_motif_probs=True)
        lf = null.make_likelihood_function(
            tree,
            loci=names,
            discrete_edges=bg_edges,
            expm="pade",
        )
        lf.set_alignment([aligns[k] for k in names])
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
        combined["null"] = null_result
        combined["alt"] = alt_results
        return combined


# todo usage example
class edge_EOP:
    def __init__(self, locus, edges, mod="GN"):
        assert len(edges) == 2
        for edge in edges:
            assert (
                edge in locus.names
            ), "edges must correspond to taxa in the given alignment"
        self.mod = mod
        self.null_lf = self.get_null_lf(edges, locus)
        self.alt_lf = self.get_alt_lf(locus)
        self.LRT = self.get_LRT_stats()
        self.LR = self.get_LR()

    def get_null_lf(self, edges, locus):
        null_mod = evo.model(
            self.mod, time_het=[dict(edges=edges, is_independent=False, upper=200)]
        )
        result = null_mod(locus)
        lf = result.lf
        lf.optimise()
        return lf

    def get_alt_lf(self, locus):
        alt_mod = evo.model(self.mod, time_het="max")
        result = alt_mod(locus)
        lf = result.lf
        lf.optimise()
        return lf

    def get_LRT_stats(self):
        null = self.null_lf.lnL
        alt = self.alt_lf.lnL
        df = self.alt_lf.nfp - self.null_lf.nfp

        LR = 2 * fsum(numpy.array([alt, -null]))
        LR = fix_rounding_error(LR, round_error=1e-1)

        table = make_table(
            header=["LR", "df", "p"],
            rows=[[LR, df, chisqprob(LR, df)]],
            digits=2,
            space=3,
        )
        return table

    def get_LR(self):
        return self.LRT.to_dict(flatten=True)[(0, "LR")]

import numpy
from accupy import fsum
from cogent3 import get_model, make_table, make_tree
from cogent3.app import evo
from cogent3.maths.stats import chisqprob

from mdeq.jsd import get_jsd
from mdeq.utils.numeric_utils import fix_rounding_error

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]


class adjacent_EOP:
    def __init__(self, loci, fg, mod="GN"):
        self.loci = {f"aln{index+1}": locus for index, locus in enumerate(loci)}
        self.mod = mod

        bg_edges = list({fg} ^ set(self.loci["aln1"].names))

        self.edges = {"fg_edge": fg, "bg_edges": bg_edges}
        self.null_lf = self.get_null_lf()
        self.alt_lf = self.get_alt_lf()
        self.LRT = self.get_LRT_stats()
        self.LR = self.get_LR()

    def get_null_lf(self):
        sm = get_model(self.mod, optimise_motif_probs=True)
        tree = make_tree(tip_names=list(self.loci.values())[0].names)
        names = list(self.loci.keys())
        null_lf = sm.make_likelihood_function(
            tree,
            loci=names,
            discrete_edges=self.edges["bg_edges"],
            expm="pade",
        )
        null_lf.set_alignment([self.loci[k] for k in names])
        null_lf.set_param_rule("mprobs", is_independent=False)
        null_lf.optimise(max_restarts=5, tolerance=1e-8, show_progress=False)

        return null_lf

    def get_alt_lf(self):
        lfs = {}

        for locus, aln in self.loci.items():
            sm = get_model(self.mod, optimise_motif_probs=True)
            tree = make_tree(tip_names=list(self.loci.values())[0].names)
            lf = sm.make_likelihood_function(
                tree,
                discrete_edges=self.edges["bg_edges"],
                expm="pade",
            )
            lf.set_alignment(aln)
            lf.optimise(max_restarts=5, tolerance=1e-8, show_progress=False)
            lfs[locus] = lf

        return lfs

    def get_LRT_stats(self):
        alt = fsum(numpy.array([l.lnL for l in self.alt_lf.values()]))
        null = self.null_lf.lnL
        df = sum([l.nfp for l in self.alt_lf.values()]) - self.null_lf.nfp

        LR = 2 * fsum(numpy.array([alt, -null]))
        LR = fix_rounding_error(LR, round_error=1e-6)

        table = make_table(
            header=["LR", "df", "p"],
            rows=[[LR, df, chisqprob(LR, df)]],
            digits=2,
            space=3,
        )
        return table

    def get_LR(self):
        return self.LRT.to_dict(flatten=True)[(0, "LR")]

    def get_relative_entropies(self):
        rel_entropies = {}

        for locus, lf in self.alt_lf.items():
            fll = self.null_lf.get_full_length_likelihoods(locus=locus)
            lfll = numpy.log(fll).sum()
            alt = lf.lnL
            length = len(self.loci[locus])
            rel_entropy = ((2 * (alt - lfll)) / self.LR) / 2 * length
            rel_entropies[locus] = rel_entropy
        return rel_entropies


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

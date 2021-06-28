import numpy
from accupy import fsum
from cogent3 import get_model, make_table, make_tree
from cogent3.maths.stats import chisqprob

from kath_library.jsd import get_jsd

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]
__version__ = "2021.06.28"
__maintainer__ = "Katherine Caley"
__email__ = "katherine.caley@anu.edu.au"
__status__ = "develop"


class EOP:
    def __init__(self, loci, mod="GN"):
        self.loci = self.get_loci(loci)
        self.mod = mod
        self.edges = self.get_fg_ang_bg()
        self.null_lf = self.get_null_lf()
        self.alt_lfs = self.get_alt_lfs()
        self.LRT = self.get_LRT_stats()
        self.LR = self.get_LR()

    def get_loci(self, loci):
        named_loci = {}
        for index, locus in enumerate(loci):
            named_loci[f"aln{index+1}"] = locus
        return named_loci

    def get_fg_ang_bg(self):

        if self.loci["aln1"].info.fg_edge is None:
            fg, _, _ = get_jsd(self.loci["aln1"])
            for locus in self.loci.values():
                locus.info["fg_edge"] = fg
        else:
            fg = self.loci["aln1"].info.fg_edge

        bg_edges = list({fg} ^ set(self.loci["aln1"].names))
        dict = {"fg_edge": fg, "bg_edges": bg_edges}

        return dict

    def get_null_lf(self):
        sm = get_model(self.mod, optimise_motif_probs=True)
        tree = make_tree(tip_names=list(self.loci.values())[0].names)
        null_lf = sm.make_likelihood_function(
            tree,
            loci=list(self.loci.keys()),
            discrete_edges=self.edges["bg_edges"],
            expm="pade",
        )
        null_lf.set_alignment(list(self.loci.values()))
        null_lf.set_param_rule("mprobs", is_independent=False)
        null_lf.optimise(max_restarts=5, tolerance=1e-8, show_progress=False)

        return null_lf

    def get_alt_lfs(self):
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
        alt = fsum(numpy.array([l.lnL for l in self.alt_lfs.values()]))
        null = self.null_lf.lnL
        df = sum([l.nfp for l in self.alt_lfs.values()]) - self.null_lf.nfp

        LR = 2 * fsum(numpy.array([alt, -null]))

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

        for locus, lf in self.alt_lfs.items():
            fll = self.null_lf.get_full_length_likelihoods(locus=locus)
            lfll = numpy.log(fll).sum()
            alt = lf.lnL
            length = len(self.loci[locus])
            rel_entropy = ((2 * (alt - lfll)) / self.LR) / 2 * length
            rel_entropies[locus] = rel_entropy
        return rel_entropies

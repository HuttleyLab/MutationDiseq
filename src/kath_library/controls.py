from cogent3.app import io
from cogent3.app.composable import RAISE, SERIALISABLE_TYPE, user_function
from cogent3.app.result import generic_result
from cogent3.util.deserialise import deserialise_object

from kath_library.eop import adjacent_EOP


def _expected_dists_eop(alns):
    aln_1 = deserialise_object(alns["aln_1"])
    aln_2 = deserialise_object(alns["aln_2"])

    fg_edge = "Human"  # todo: hard-coded because I am in a rush :(
    eop = adjacent_EOP([aln_1, aln_2], fg=fg_edge)
    obs_eop_stats = eop.get_LRT_stats()

    null_lf = eop.null_lf
    aln1, aln2 = null_lf.locus_names

    null_aln_1 = null_lf.simulate_alignment(locus=aln1)
    null_aln_2 = null_lf.simulate_alignment(locus=aln2)

    null_eop = adjacent_EOP([null_aln_1, null_aln_2], fg=fg_edge)
    null_eop_stats = null_eop.get_LRT_stats()

    aln_lf_1 = eop.alt_lf[aln1]
    aln_lf_2 = eop.alt_lf[aln2]

    alt_aln_1 = aln_lf_1.simulate_alignment()
    alt_aln_2 = aln_lf_2.simulate_alignment()

    alt_eop = adjacent_EOP([alt_aln_1, alt_aln_2], fg=fg_edge)
    alt_eop_stats = alt_eop.get_LRT_stats()

    result = generic_result(source=alns.source)

    result.update(
        [
            ("obs_eop", obs_eop_stats),
            ("null_eop", null_eop_stats),
            ("alt_eop", alt_eop_stats),
        ]
    )
    return result


expected_dists_eop = user_function(
    _expected_dists_eop,
    input_types=SERIALISABLE_TYPE,
    output_types=SERIALISABLE_TYPE,
)

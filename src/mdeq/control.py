import pathlib

from typing import Union

from cogent3 import ArrayAlignment
from cogent3.app.composable import (
    ALIGNED_TYPE,
    RESULT_TYPE,
    SERIALISABLE_TYPE,
    Composable,
)
from cogent3.app.result import bootstrap_result, generic_result, model_result

from mdeq.adjacent import grouped, make_identifier


__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]
__version__ = "2021.12.20"


class select_model_result:
    """selects the specified model name from a bootstrap result object"""

    def __init__(self, model_name):
        self._name = model_name

    def __call__(self, result):
        result.deserialised_values()
        if isinstance(result, model_result):
            return result

        if isinstance(result, bootstrap_result):
            return result.observed[self._name]

        return result[self._name]


class control_generator(Composable):
    def __init__(self, model_selector):
        super(control_generator, self).__init__(
            input_types=(
                SERIALISABLE_TYPE,
                RESULT_TYPE,
            ),
            output_types=(SERIALISABLE_TYPE, ALIGNED_TYPE),
        )
        self._select_model = model_selector
        self.func = self.gen

    def _from_single_model_single_locus(self, result) -> ArrayAlignment:
        source = pathlib.Path(result.source).stem
        # conventional model object
        model = self._select_model(result)
        sim = model.lf.simulate_alignment()
        sim.info.source = source
        return sim

    def _from_single_model_multi_locus(self, result) -> grouped:
        # this is an adjacent EOP null, so has multiple alignments for one lf
        # aeop modelling works via grouped data, so we need to bundle the
        # simulated alignments into a grouped instance
        model = self._select_model(result)
        source = pathlib.Path(result.source).stem
        locus_names = model.lf.locus_names[:]
        names = []
        alns = []
        for name in locus_names:
            n = name
            names.append(n)
            sim = model.lf.simulate_alignment(locus=name)
            sim.info.name = n
            alns.append(sim)

        r = grouped(names, source=source)
        r.elements = alns
        return r

    def _from_multi_model_multi_locus(self, result) -> grouped:
        # this is an adjacent EOP alt, so has a separate model
        # instance for each alignment
        # aeop modelling works via grouped data, so we need to bundle the
        # simulated alignments into a grouped instance

        model = self._select_model(result)
        source = pathlib.Path(result.source).stem
        sims = []
        ids = []
        for name, lf in model.items():
            ids.append(name)
            sim = lf.simulate_alignment()
            sim.info.source = ids[-1]
            sims.append(sim)
        result = grouped(ids, source=source)
        result.elements = sims
        return result

    def gen(self, result) -> Union[ArrayAlignment, grouped]:
        # this function will only be called on the first result object,
        # it establishes the appropriate method to set for the data
        # and assigns that to self.func, which the Composable architecture
        # invokes
        model = self._select_model(result)
        if len(model) > 1:
            self.func = self._from_multi_model_multi_locus
            return self.func(result)

        if len(model.lf.locus_names) > 1:
            self.func = self._from_single_model_multi_locus
            return self.func(result)

        self.func = self._from_single_model_single_locus
        return self.func(result)

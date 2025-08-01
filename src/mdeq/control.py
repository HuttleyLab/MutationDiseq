import pathlib
from random import Random
from typing import ForwardRef, Union

from cogent3.app.composable import define_app
from cogent3.app.result import bootstrap_result, model_result
from cogent3.app.typing import HypothesisResultType, SerialisableType
from cogent3.core.alignment import Alignment
from cogent3.util.deserialise import deserialise_object

from mdeq.adjacent import grouped
from mdeq.bootstrap import _reconstitute_collection
from mdeq.utils import CompressedValue


class select_model_result:
    """selects the specified model name from a bootstrap result object"""

    def __init__(self, model_name):
        self._name = model_name

    def __call__(self, result):
        if isinstance(result, model_result):
            return result

        if isinstance(result, bootstrap_result):
            if "data" in result.observed:
                # only decompress the instance required
                data = CompressedValue(result.observed["data"]).as_primitive
                data = _reconstitute_collection(data)
                result = deserialise_object(data)
            else:
                result = result.observed

        return result[self._name]


@define_app
class control_generator:
    def __init__(self, model_selector, seed=None):
        self._select_model = model_selector
        self.rng = Random()
        self.rng.seed(a=seed)

    def _from_single_model_single_locus(self, result) -> Alignment:
        source = pathlib.Path(result.source).stem
        # conventional model object
        model = self._select_model(result)
        if isinstance(model.lf, dict):
            model.deserialise_values()

        sim = model.lf.simulate_alignment(random_series=self.rng)
        sim.source = source
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
            sim = model.lf.simulate_alignment(random_series=self.rng, locus=name)
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
        model.deserialised_values()
        source = pathlib.Path(result.source).stem
        sims = []
        ids = []
        for name, lf in model.items():
            ids.append(name)
            sim = lf.simulate_alignment()
            sim.source = ids[-1]
            sims.append(sim)
        result = grouped(ids, source=source)
        result.elements = sims
        return result

    T = Union[Alignment, grouped, SerialisableType]

    def main(
        self,
        result: HypothesisResultType | ForwardRef("compact_bootstrap_result"),
    ) -> T:
        # this function will only be called on the first result object,
        # it establishes the appropriate method to set for the data
        # and assigns that to self.main, which the Composable architecture
        # invokes
        model = self._select_model(result)
        if len(model) > 1:
            self.main = self._from_multi_model_multi_locus
            return self.main(result)

        if len(model.lf.locus_names) > 1:
            self.main = self._from_single_model_multi_locus
            return self.main(result)

        self.main = self._from_single_model_single_locus
        return self.main(result)

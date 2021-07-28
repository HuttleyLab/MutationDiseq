import json
from importlib import import_module

from cogent3.app.result import generic_result
from cogent3.util.deserialise import deserialise_object
from cogent3.util.misc import open_, path_exists

__author__ = "Katherine Caley"
__credits__ = ["Gavin Huttley", "Katherine Caley"]


def _get_class(provenance):
    """
    vendored from cogent3.util.deserialise._get_class - thanks Gavin!
    """
    index = provenance.rfind(".")
    assert index > 0
    klass = provenance[index + 1 :]
    nc = "NotCompleted"
    klass = nc if nc in klass else klass
    mod = import_module(provenance[:index])
    klass = getattr(mod, klass)
    return klass


class confidence_interval_result(generic_result):
    _type = "bootstrap_result"
    _item_types = ("generic_result", "model_collection_result")

    def __init__(self, source=None):
        super(confidence_interval_result, self).__init__(source)
        self._construction_kwargs = dict(source=source)

    @property
    def observed(self):
        """the results for the observed data"""
        return self["observed"]

    @observed.setter
    def observed(self, data):
        self.update(dict(observed=data))

    def add_to_null(self, data):
        """add results for a synthetic data set"""
        size = len(self)
        self[size + 1] = data

    @property
    def null_dist(self):
        """returns the statistics of interest corresponding to the synthetic data"""
        return [self[k].to_rich_dict()["items"][0][1] for k in self if k != "observed"]


def deserialise_confidence_interval_result(data):
    """
    vendored with alterationgs from cogent3.util.deserialise- thanks Gavin!
    """
    if path_exists(data):
        with open_(data) as infile:
            data = json.load(infile)

    if type(data) is str:
        data = json.loads(data)

    type_ = data.get("type", None)
    if type_ is None:
        return data

    """returns a result object"""
    data.pop("version", None)
    klass = _get_class(data.pop("type"))
    kwargs = data.pop("result_construction")
    result = klass(**kwargs)
    if "items" in data:
        items = data.pop("items")
    else:
        # retain support for the old style result serialisation
        items = data.items()
    for key, value in items:
        # only deserialise the result object, other attributes loaded as
        # required
        if type(value) == dict and "app.result" in str(value.get("type")):
            value = deserialise_object(value)
        try:
            result[key] = value
        except TypeError:
            result[tuple(key)] = value
    return result

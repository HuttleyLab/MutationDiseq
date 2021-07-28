from cogent3.app.result import generic_result

__author__ = "Katherine Caley"
__credits__ = ["Gavin Huttley", "Katherine Caley"]


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

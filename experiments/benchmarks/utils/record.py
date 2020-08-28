"""Tools to store benchmark results"""
from math import sqrt
import pandas as pd
from dictwrapper import DictWrapper


class Record(DictWrapper):
    """Dictionary-like object that can be converted to a pandas dataframe row"""


class EvaluationRecord(Record):
    """Dictionary-like object holding a value and a std"""

    def __init__(self, *, value, value_std=0., **kwargs):
        """"""
        store = {
            "value": value,
            "value_std": value_std,
        }

        store.update(kwargs)
        super(EvaluationRecord, self).__init__(store)


class ComparisonRecord(Record):
    """Dictionary-like object with a truth value to store the result of the comparison
    between to integral estimations
    """

    def __init__(self, *, value, target, value_std=0., target_std=0., **kwargs):
        """"""
        store = {
            "value": value,
            "target": target,
            "value_std": value_std,
            "target_std": target_std,
        }

        store.update(kwargs)
        super(ComparisonRecord, self).__init__(store)

        if "match" not in self:
            if "sigma_cutoff" not in self:
                self["sigma_cutoff"] = 2
            difference_uncertainty = sqrt(self["value_std"] ** 2 + self["target_std"] ** 2)
            if difference_uncertainty == 0.:
                self["match"] = (self["value"] == self["target"])
            else:
                self["match"] = (abs(self["value"] - self["target"]) / difference_uncertainty <= self["sigma_cutoff"])

        assert isinstance(self["match"], bool)

    def __bool__(self):
        return self["match"]

    def __repr__(self):
        return self.store.__repr__()

from better_abc import abstract_attribute
from ..general_flow import GeneralFlow

class AnalyticFlow(GeneralFlow):
    def __init__(self,d):
        super(AnalyticFlow, self).__init__(d=d)
        self.flow_ = abstract_attribute()

    def flow(self, x):
        return self.flow_(x.detach())

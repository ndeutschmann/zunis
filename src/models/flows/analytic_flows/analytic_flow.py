from better_abc import abstract_attribute
from ..general_flow import GeneralFlow

class AnalyticFlow(GeneralFlow):
    def __init__(self,d):
        super(AnalyticFlow, self).__init__(d=d)
        self.flow_ = abstract_attribute()

    def flow(self, x):
        return self.flow_(x)


class InvertibleAnalyticFlow(GeneralFlow):
    def __init__(self,d):
        super(InvertibleAnalyticFlow, self).__init__(d=d)

        # I'd like to have these but because of how PyTorch registers modules
        # This does not work
        #self.flow_ = abstract_attribute()
        #self.inverse_flow_ = abstract_attribute()

        self.inverse = False

    def runs_forward(self):
        """Check the running mode: True if forward and False if backward/inverse"""
        return not self.inverse

    def invert(self):
        self.inverse = not self.inverse

    def flow(self, x):
        if self.inverse:
            return self.inverse_flow_.flow(x)

        return self.flow_.flow(x)

    def transform_and_compute_jacobian(self, xj):
        if self.inverse:
            return self.backward_flow.transform_and_compute_jacobian(xj)

        return self.forward_flow.transform_and_compute_jacobian(xj)

"""Implementation of the abstract GeneralFlow class
Most generic variable transformation:
- takes in a point x and -log(PDF(x))
- outputs a transformed point y and - log(PDF(y)) = - log(PDF(x)) + log(dy/dx)

Reminder:

dx p(x) = dy q(y) = dx dy/dx q(y)
=> q(y) = p(x)/(dy/dx)
=> -log q(y) = -log p(x) + log dy/dx
"""
import torch
from better_abc import ABC,abstract_attribute,abstractmethod


class GeneralFlow(torch.nn.Module,ABC):
    """General abstract class for flows"""

    def __init__(self, *, d):
        super(GeneralFlow, self).__init__()
        self.d = d

    flow = abstract_attribute()

    @abstractmethod
    def transform_and_compute_jacobian(self, xj):
        pass

    def forward(self, xj):
        if self.training:
            assert xj.shape[1] == self.d+1
            return self.transform_and_compute_jacobian(xj)
        else:
            assert xj.shape[1] == self.d
            return self.flow(xj)





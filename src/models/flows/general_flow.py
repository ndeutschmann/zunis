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
from better_abc import ABC, abstract_attribute, abstractmethod


class GeneralFlow(torch.nn.Module,ABC):
    """General abstract class for flows"""

    def __init__(self, *, d):
        super(GeneralFlow, self).__init__()
        self.d = d

    @abstractmethod
    def flow(self, x):
        """Transform the batch of points x  with shape (...,d)
        This is an abstract method that should be overriden"""
        pass

    @abstractmethod
    def transform_and_compute_jacobian(self, xj):
        """Compute the flow transformation and its Jacobian simulatenously on
        xj with xj.shape == (...,d+1)

        This is an abstract method that should be overriden
        """
        pass

    def forward(self, xj):
        """Compute the flow transformation on some input xj
        - In training mode, xj.shape == (:,d+1)
        and the last dimension is the log-inverse PDF of x[:,:-1]
        - In eval mode,, xj.shape == (:,d)
        and no jacobian is passed: pure sampling mode.
        """
        if self.training:
            assert xj.shape[-1] == self.d+1
            return self.transform_and_compute_jacobian(xj)
        else:
            assert xj.shape[-1] == self.d
            return self.flow(xj)





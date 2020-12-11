"""Abstract classes for integrands"""
from abc import abstractmethod

from better_abc import ABC
import vegas
import torch

class Integrand(ABC):
    """Abstract class to define integrands for testing the integration library"""

    def __init__(self, d, *args, **kwargs):
        self.d = d

    @abstractmethod
    def evaluate_integrand(self, x):
        """Compute the value of the integrand on a batch of points"""

    def __call__(self, x):
        """Compute the value of the integrand on a batch of points"""
        assert len(x.shape) == 2, f"Shape mismatch, expected (*, {self.d})"
        assert x.shape[1] == self.d, f"Shape mismatch, expected (*, {self.d}), got {tuple(x.shape)}"
        return self.evaluate_integrand(x)

    def vegas(self, device=torch.device("cpu")):
        """Turn this integrand into a vegas batch integrand"""
        @vegas.batchintegrand
        def vself(x):
            return self(torch.tensor(x).to(device)).detach().cpu().numpy()

        return vself


class KnownIntegrand(Integrand, ABC):
    """Abstract class for integrands with a known integral value"""

    @abstractmethod
    def integral(self):
        """Compute the true value of the integral"""

    def compare_absolute(self, estimated_integral):
        """Compute the absolute difference between an integral and the true value"""
        return abs(self.integral() - estimated_integral)

    def compare_relative(self, estimated_integral):
        """Compute the relative difference between an integral estimation and the true value"""
        return self.compare_absolute(estimated_integral) / (0.5 * (abs(self.integral()) + abs(estimated_integral)))

    def check(self, estimated_integral, tolerance=1.e-3, method="relative"):
        """Check the value of an estimation of the integral value

        Parameters
        ----------
        estimated_integral: float
        tolerance: float
        method: {"relative", "absolute"}

        Returns
        -------
            bool
        """
        if method == "relative":
            return self.compare_relative(estimated_integral) <= tolerance
        elif method == "absolute":
            return self.compare_absolute(estimated_integral) <= tolerance
        else:
            raise ValueError("Only accepted methods are 'relative' and 'absolute")


class RegulatedIntegrand(Integrand):
    """Integrand deriving from another one and adding a small constant
    regulating factor

    This is a Mixin class. Have a look at the following references:
    `[1]`_ `[2]`_

    .. _[1]: https://stackoverflow.com/a/50465583
    .. _[2]: https://rhettinger.wordpress.com/2011/05/26/super-considered-super/
    """

    def __init__(self, reg, *args, **kwargs):
        self.reg = reg
        super().__init__(*args, **kwargs)

    def evaluate_integrand(self, x):
        return super().evaluate_integrand(x) + self.reg


class RegulatedKnownIntegrand(RegulatedIntegrand, KnownIntegrand):
    """Integrand deriving from another known one and adding a small constant
    regulating factor"""

    def integral(self):
        return super().integral() + self.reg


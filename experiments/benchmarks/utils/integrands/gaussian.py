from math import pi, sqrt
import torch

from utils.integrands import sanitize_variable
from utils.integrands.abstract import Integrand, RegulatedIntegrand, KnownIntegrand


class DiagonalGaussianIntegrand(Integrand):
    """N-dimensional gaussian with a diagonal covariance matrix"""

    def __init__(self, d, mu=0.5, s=0.1, norm=1., device=None, *args, **kwargs):
        """

        Parameters
        ----------
        mu : float or torch.Tensor
            Mean of the gaussian. Either a scalar or a vector of size d
        s: float or torch.Tensor
            Standard deviation of the gaussian. Either a scalar or a vector of size d
        norm: float or torch.Tensor
            Prefactor of the gaussian. Must be a scalar.
        device: torch.device
            Default device where the parameters are stored

        Notes
        -----
        Correct value in 2D with standard params: 0.031415898(81)
        """
        super(DiagonalGaussianIntegrand, self).__init__(d)
        self.mu = sanitize_variable(mu, device)
        self.s = sanitize_variable(s, device)
        self.norm = sanitize_variable(norm, device)
        self.default_device = device

        assert len(self.mu.shape) == 0 or tuple(self.mu.shape) == (d,)
        assert len(self.s.shape) == 0 or tuple(self.s.shape) == (d,)
        assert len(self.norm.shape) == 0

    def evaluate_integrand(self, x):
        """Compute the gaussian

        The parameters of the gaussian are sent to the device of the input

        Parameters
        ----------
        x: torch.Tensor
            Batch of points of size ```(*,d)```

        Returns
        -------
            torch.Tensor
        """
        return self.norm * torch.exp(-((x - self.mu.to(x.device)) / self.s.to(x.device)).square().sum(axis=1))


class RegulatedDiagonalGaussianIntegrand(RegulatedIntegrand, DiagonalGaussianIntegrand):
    """N-dimensional regulated gaussian with a diagonal covariance matrix"""

    def __init__(self, d, mu=0.5, s=0.1, norm=1., reg=1.e-6, device=None, *args, **kwargs):
        """

        Parameters
        ----------
        mu : float or torch.Tensor
            Mean of the gaussian. Either a scalar or a vector of size d
        s: float or torch.Tensor
            Standard deviation of the gaussian. Either a scalar or a vector of size d
        norm: float or torch.Tensor
            Prefactor of the gaussian. Must be a scalar.
        reg: float
            regularization constant
        device: torch.device
            Default device where the parameters are stored

        Notes
        -----
        Correct value in 2D with standard params: 0.031416898(81)
        """
        super().__init__(reg, d, mu=mu, s=s, norm=norm, device=device)


class KnownGaussianIntegrand(KnownIntegrand, DiagonalGaussianIntegrand):
    """N-dimensional guassian integrand whose integral over the unit hypercube is provided
    using the Error Function as implemented in pytorch"""

    @staticmethod
    def erf(x):
        """Returns the integral of exp(-x^2) between 0 and 1."""
        return torch.erf(x)*sqrt(pi)/2.

    def integral(self):
        """Compute the 1D integral of a gaussian over the unit interval [0,1]"""
        if len(self.s.shape) == 0:
            sigma = self.s.repeat(self.d)
        else:
            sigma = self.s
        if len(self.mu.shape) == 0:
            mu = self.mu.repeat(self.d)
        else:
            mu = self.mu

        return (sigma * (self.erf(mu/sigma) + self.erf((1-mu)/sigma)) * self.norm).prod()


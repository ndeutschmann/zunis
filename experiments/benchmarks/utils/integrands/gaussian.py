import torch

from utils.integrands import sanitize_variable
from utils.integrands.abstract import Integrand


class DiagonalGaussianIntegrand(Integrand):
    """N-dimensional gaussian with a diagonal covariance matrix"""

    def __init__(self, d, mu=0.5, s=0.1, norm=1., device=None):
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
            Batch of points of size (*,d)

        Returns
        -------
            torch.Tensor
        """
        return self.norm * torch.exp(-((x - self.mu.to(x.device)) / self.s.to(x.device)).square().sum(axis=1))


class CamelIntegrand(Integrand):
    """Camel function: two gaussian peaks on the hyperdiagonal of the unit hypercube
    at points (0.25, ..., 0.25) and (0.75, ..., 0.75).
    """

    def __init__(self, d, s1=0.1, norm1=1, s2=0.2, norm2=2, device=None):
        """
        Parameters
        ----------
        s1: float or torch.Tensor
            std of the first gaussian. Either a scalar or a vector of size d
        s2: float or torch.Tensor
            std of the second gaussian. Either a scalar or a vector of size d
        norm1: float or torch.Tensor
            normalization of the first gaussian. Must be a scalar.
        norm2: float or torch.Tensor
            normalization of the second gaussian. Must be a scalar.
        device: default device on which to run the computation
        """
        super(CamelIntegrand, self).__init__(d)
        self.s1 = sanitize_variable(s1, device)
        self.norm1 = sanitize_variable(norm1, device)
        self.s2 = sanitize_variable(s2, device)
        self.norm2 = sanitize_variable(norm2, device)
        self.default_device = device

        self.hump1 = DiagonalGaussianIntegrand(d, 0.25, s1, norm1, device)
        self.hump2 = DiagonalGaussianIntegrand(d, 0.75, s2, norm2, device)

    def evaluate_integrand(self, x):
        """Compute the camel function for a batch of points

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.hump1.evaluate_integrand(x) + self.hump2.evaluate_integrand(x)

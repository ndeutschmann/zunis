"""Pretty, low dimensional examples where we clearly beat VEGAS"""

import torch
from math import pi
from utils.integrands import sanitize_variable
from utils.integrands.abstract import Integrand, RegulatedIntegrand


class CircleLineIntegrand(Integrand):
    """Classic failure mode of VEGAS: function with an enhancement along
    a circle and a line on the diagonal, preventing a straightforward analytic
    change of variable

    This is a generalization with a n-sphere and maximal diagonal pattern"""

    def __init__(self, d, r=0.3, sig=0.05, device=None, *args, **kwargs):
        super(CircleLineIntegrand, self).__init__(d)
        self.sig = sanitize_variable(sig, device)
        self.r = sanitize_variable(r, device)
        self.default_device = device

    def evaluate_integrand(self, x):
        """Compute the value function

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
            torch.Tensor
        """
        sig = self.sig.to(x.device)
        r = self.r.to(x.device)

        return torch.clamp(
            torch.exp(-
                      (((x - 0.5) ** 2).sum(dim=-1).sqrt() - r) ** 2 / sig ** 2
                      )
            +
            torch.exp(-
                      torch.sum(
                          (x[:, 0].unsqueeze(-1) - x) ** 2 / sig ** 2,
                          dim=-1)
                      )
            , 0., .9)


class SineLineIntegrand(Integrand):
    """2D function enhanced on the graph y = sin(2*pi*f*x)"""

    def __init__(self, d, sig, f=2., device=None, *args, **kwargs):
        """

        Parameters
        ----------
        sig: float
            thickness of the line
        f: float
            period of the sinus
        """
        super(SineLineIntegrand, self).__init__(d=d)
        self.sig = sanitize_variable(sig, device)
        self.f = sanitize_variable(f, device)

    def evaluate_integrand(self, x):
        """Compute the integrand as a gaussian whose argument is the characteristic equation of the graph"""

        sig = self.sig.to(x.device)
        f = self.f.to(x.device)

        return torch.exp(- (2 * x[:, 1] - (1 + torch.cos(2 * pi * f * x[:, 0]))) ** 2 / sig ** 2)


class SineIntegrand(Integrand):
    """Sinusoidal function with a wave vector along the maximal hyperdiagonal"""

    def __init__(self, d, f=1., offset=1., device=None, *args, **kwargs):
        """

        Parameters
        ----------
        f: float
            period of the sinus
        offset: float
        """

        super(SineIntegrand, self).__init__(d=d)
        self.offset = sanitize_variable(offset, device)
        self.f = sanitize_variable(f, device)

    def evaluate_integrand(self, x):
        """Define a wave vector k with the dimensionality of x and all entries 1 and return cst+cos(f*k.x) """

        offset = self.offset.to(x.device)
        f = self.f.to(x.device)
        phase = f * x.sum(dim=-1)
        return offset + torch.cos(phase) ** 2

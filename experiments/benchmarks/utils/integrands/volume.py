from abc import abstractmethod
from math import pi

from better_abc import ABC
from scipy.special import gamma

from utils.integrands import sanitize_variable
from utils.integrands.abstract import Integrand, KnownIntegrand, RegulatedKnownIntegrand


class VolumeIntegrand(Integrand, ABC):
    """Abstract class to define integrands that are 1 on a characteristic subspace and 0 outside, defined
    through an inequality f(x) >= 0"""

    @abstractmethod
    def inequality(self, x):
        """The characteristic inequality of the volume we want to measure

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
            torch.BoolTensor
        """

    def evaluate_integrand(self, x):
        """Compute the value of the integrand on a batch of points by checking their inequality
        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
            torch.Tensor
        """
        return self.inequality(x).to(x.dtype)


class HyperrectangleVolumeIntegrand(VolumeIntegrand, KnownIntegrand):
    """Characteristic function of an hyperrectangle defined by cutting the
    unit hypercube along one axis. Its characteristic inequality is x[i] <= frac
    where 0. < frac < 1.
    """

    def __init__(self, d, split_dim=0, frac=0.5, *args, **kwargs):
        """

        Parameters
        ----------
        d: int
        split_dim: int
        frac: float
        """
        super(HyperrectangleVolumeIntegrand, self).__init__(d=d)
        assert 0 <= split_dim < d
        assert 0. <= frac <= 1.

        self.split_dim = split_dim
        self.frac = frac

    def integral(self):
        """The integral is the length of the non-unit side of the hyperrectangle"""
        return self.frac

    def inequality(self, x):
        """Check if we are in the hyperrectangle"""
        return x[:, self.split_dim] < self.frac


class HypersphereVolumeIntegrand(VolumeIntegrand, KnownIntegrand):
    """Characteristic function of an hypersphere. The hypersphere must fit in the unit hypercube fully"""

    def __init__(self, d, r, c, device=None, *args, **kwargs):
        """

        Parameters
        ----------
        d: int
        r: float
        c: torch.Tensor or float
        """
        super(HypersphereVolumeIntegrand, self).__init__(d=d)
        self.r = r
        self.c = sanitize_variable(c, device=device)
        assert (self.r > 0), "The radius must be positive"
        assert len(self.c.shape) == 0 or tuple(self.c.shape) == (d,), "The center is either a number or a d-vector"
        assert ((self.c - self.r) >= 0.).all().item(), "The full hypersphere must fit in the unit hypercube"
        assert ((self.c + self.r) <= 1.).all().item(), "The full hypersphere must fit in the unit hypercube"

    def inequality(self, x):
        """Check if the points are in the hypersphere"""
        return ((x - self.c) ** 2).sum(dim=1).sqrt() <= self.r

    def integral(self):
        """Compute the volume of the hypersphere in d dimensions"""
        return float((self.r ** self.d) * (pi ** (self.d / 2.)) / gamma(self.d / 2. + 1))


class RegulatedHyperSphereIntegrand(RegulatedKnownIntegrand, HypersphereVolumeIntegrand):
    """Characteristic function of an hypersphere with a small regulating factor.
    The hypersphere must fit in the unit hypercube fully"""

    def __init__(self, d, r, c, reg, device=None, *args, **kwargs):
        super().__init__(reg, d, r, c, device=device)


class RegulatedHyperSphericalCamel(KnownIntegrand):
    """Camel function consisting of the sum of two regulated hyperspherical
    volume functions with centers at (0.25, ..., 0.25) and (0.75, ..., 0.75)"""

    def __init__(self, d, r1=0.24, r2=0.24, reg=1.e-6, device=None, *args, **kwargs):
        super(RegulatedHyperSphericalCamel, self).__init__(d=d)
        self.hump1 = RegulatedHyperSphereIntegrand(d, r=r1, c=0.25, reg=reg, device=device)
        self.hump2 = RegulatedHyperSphereIntegrand(d, r=r2, c=0.75, reg=reg, device=device)

    def integral(self):
        return self.hump1.integral() + self.hump2.integral()

    def evaluate_integrand(self, x):
        return self.hump1.evaluate_integrand(x) + self.hump2.evaluate_integrand(x)


class RegulatedSymmetricHyperSphericalCamel(RegulatedHyperSphericalCamel):
    """Camel function consisting of the sum of two regulated hyperspherical
        volume functions with centers at (0.25, ..., 0.25) and (0.75, ..., 0.75)
        with identical radii"""
    def __init__(self, d, r=0.24, reg=1.e-6, device=None, *args, **kwargs):
        super(RegulatedSymmetricHyperSphericalCamel, self).__init__(
            d=d,
            r1=r,
            r2=r,
            reg=reg,
            device=device,
            *args,
            **kwargs
        )

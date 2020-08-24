"""Abstract class for coupling cells"""
import torch
from better_abc import abstract_attribute

from zunis.models.flows.general_flow import GeneralFlow
from .transforms import InvertibleTransform


def not_list(l):
    """Return the element wise negation of a list of booleans"""
    assert all([isinstance(it, bool) for it in l])
    return [not it for it in l]


class GeneralCouplingCell(GeneralFlow):
    """Abstract class for coupling cell"""

    def __init__(self, *, d, transform, mask):
        """
        Parameters
        ----------
        d: int
            dimension of the space
        transform: function or :py:class:`InvertibleTransform <zunis.models.flows.coupling_cells.transforms.InvertibleTransform>`
            bijective variable transform that maps the unit hypercube (in `d-sum(mask)` dimensions) to itself
            given some parameters. It should have the same signature as
            :py:meth:`InvertibleTransform.forward <zunis.models.flows.coupling_cells.transforms.InvertibleTransform.forward>`
        mask: list of bool
            indicates which variables are passed (y_N) through and which are transformed (y_M)
        """
        super(GeneralCouplingCell, self).__init__(d=d)

        self.mask = mask + [False]
        self.mask_complement = not_list(mask) + [False]
        self.transform = transform
        self.T = abstract_attribute()

    def transform_and_compute_jacobian(self, yj):
        """Apply the variable change on a batch of points y and compute the jacobian"""
        y_n = yj[..., self.mask]
        y_m = yj[..., self.mask_complement]
        log_j = yj[..., -1]

        x = torch.zeros_like(yj).to(yj.device)
        x[..., self.mask] = y_n
        x[..., self.mask_complement], log_jy = self.transform(y_m, self.T(y_n), compute_jacobian=True)
        x[..., -1] = log_j + log_jy
        return x

    def flow(self, y):
        """Apply the variable change on a batch of points y"""
        y_n = y[..., self.mask[:-1]]
        y_m = y[..., self.mask_complement[:-1]]

        x = torch.zeros_like(y).to(y.device)
        x[..., self.mask] = y_n
        x[..., self.mask_complement], _ = self.transform(y_m, self.T(y_n), compute_jacobian=False)

        return x


class InvertibleCouplingCell(GeneralCouplingCell):
    """An invertible coupling cell works exactly the same as a general coupling cell
    but its transform admits and inverse transformation.
    """

    def __init__(self, *, d, transform, mask):
        """
        Parameters
        ----------
        d: int
            dimension of the space
        transform: :py:class:`InvertibleTransform <zunis.models.flows.coupling_cells.transforms.InvertibleTransform>`
            bijective variable transform that maps the unit hypercube (in `d-sum(mask)` dimensions) to itself
            given some parameters.
        mask: list of bool
            indicates which variables are passed (y_N) through and which are transformed (y_M)"""
        assert isinstance(transform, InvertibleTransform)
        super(InvertibleCouplingCell, self).__init__(d=d, transform=transform, mask=mask)

    def invert(self):
        """Change the running mode from forward to inverse (invert the change of variable)"""
        self.transform.invert()

    def runs_forward(self):
        """Check the running mode: True if forward and False if backward/inverse"""
        return not self.transform.inverse

"""Abstract class for coupling cells"""
import torch
from better_abc import abstract_attribute

from src.models.flows.general_flow import GeneralFlow

def not_list(l):
    """Return the element wise negation of a list of booleans"""
    assert all([isinstance(it,bool) for it in l])
    return [not it for it in l]


class GeneralCouplingCell(GeneralFlow):
    """Abstract class for coupling cell"""
    def __init__(self, *, d, transform, mask):
        """Coupling cell instantiation:
        - mask: boolean Tensor that indicates which variables are passed (y_N) through
        and which are transformed (y_M)
        - transform: function that takes three arguments
            - y_M: the variables that are transformed element wise
            - t: a set of parameters for the transformation
            - compute_jacobian: a bool deciding whether the jacobian should be computed
        and returns a 2-tuple
            - x_M: the transformed variable batch
            - log_j: the log-Jacobian batch if compute_jacobian, None otherwise
        - T the function that computes the parameters T from y_N

        NB:
        """
        super(GeneralCouplingCell, self).__init__(d=d)

        self.mask = mask+[False]
        self.mask_complement = not_list(mask)+[False]
        self.transform = transform
        self.T = abstract_attribute()

    def transform_and_compute_jacobian(self, yj):
        """Apply the variable change on a batch of points y and compute the jacobian"""
        y_n = yj[..., self.mask]
        y_m = yj[..., self.mask_complement]
        log_j = yj[..., -1]

        x = torch.zeros_like(yj).to(yj.device)
        x[..., self.mask] = y_n
        x[..., self.mask_complement], log_jy = self.transform(y_m, self.T(y_n))
        x[..., -1] = log_j + log_jy
        return x

    def flow(self, y):
        """Apply the variable change on a batch of points y"""
        y_n = y[..., self.mask[:-1]]
        y_m = y[..., self.mask_complement[:-1]]

        x = torch.zeros_like(y).to(y.device)
        x[..., self.mask] = y_n
        x[..., self.mask_complement] = self.transform(y_m, self.T(y_n))

        return x

"""Implementatin of realNVP: a coupling cell with an affine transform"""

import torch
from .general_coupling import InvertibleCouplingCell
from src.models.layers.trainable import ArbitraryShapeRectangularDNN
from .transforms import InvertibleTransform


def element_wise_affine(x, st, compute_jacobian=True):
    """Transform x element-wise through an affine function y = exp(s)*x + t
    where s = st[...,0] and t = st[...,1] with s.shape == x.shape == t.shape

    The Jacobian for this transformation is the coordinate-wise product of the scaling factors
    J = prod(es[...,i],i)
    """
    es = torch.exp(st[..., 0])
    t = st[..., 1]
    logj = None
    if compute_jacobian:
        logj = torch.sum(torch.log(es), dim=-1)

    return es*x + t, logj


def inverse_element_wise_affine(x, st, compute_jacobian=True):
    """Transform x element-wise through an affine function y = exp(-s)*(x - t)
    where s = st[...,0] and t = st[...,1] with s.shape == x.shape == t.shape
    This is the inverse of `element_wise_affine` above for the same set of parameters st

    The Jacobian for this transformation is the coordinate-wise product of the scaling factors
    J = prod(es[...,i],i)
    """
    es = torch.exp(-st[..., 0])
    t = st[..., 1]
    logj = None
    if compute_jacobian:
        logj = torch.sum(torch.log(es), dim=-1)

    return es*(x - t), logj


class ElementWiseAffineTransform(InvertibleTransform):
    def forward(self, y, st, compute_jacobian=True):
        return element_wise_affine(y, st, compute_jacobian=compute_jacobian)

    def backward(self, y, st, compute_jacobian=True):
        return inverse_element_wise_affine(y, st, compute_jacobian=compute_jacobian)


class GeneralRealNVP(InvertibleCouplingCell):
    """Coupling cell based on affine transforms without a specific parameter prediction function"""
    def __init__(self, *, d, mask):
        super(GeneralRealNVP, self).__init__(d=d, transform=ElementWiseAffineTransform(), mask=mask)


class FakeT(torch.nn.Module):
    """Fake neural net for the test FakeRealNVP that returns a trainable constant"""
    def __init__(self, s=0., t=0.):
        super(FakeT, self).__init__()
        self.st = torch.nn.Parameter(torch.tensor([s, t]),requires_grad=True)

    def forward(self,x):
        return self.st


class FakeRealNVP(GeneralRealNVP):
    """Fake real NVP with position independent affine parameters
    for basic debugging and training tests
    """
    def __init__(self, *, d, mask, s=0., t=0.):
        super(FakeRealNVP, self).__init__(d=d, mask=mask)
        self.T = FakeT(s, t)


class RealNVP(GeneralRealNVP):
    def __init__(self, *, d, mask,
                 d_hidden,
                 n_hidden,
                 input_activation=None,
                 hidden_activation=torch.nn.ReLU,
                 output_activation=None,
                 use_batch_norm=False):
    
        super(RealNVP, self).__init__(d=d,mask=mask)

        d_in = sum(mask)
        d_out = d - d_in

        self.T = ArbitraryShapeRectangularDNN(d_in=d_in,
                                              out_shape=(d_out,2),
                                              d_hidden=d_hidden,
                                              n_hidden=n_hidden,
                                              input_activation=input_activation,
                                              hidden_activation=hidden_activation,
                                              output_activation=output_activation,
                                              use_batch_norm=use_batch_norm)




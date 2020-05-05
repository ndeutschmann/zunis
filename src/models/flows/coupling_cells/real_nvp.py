"""Implementatin of realNVP: a coupling cell with an affine transform"""

import torch
from .general_coupling import GeneralCouplingCell
from src.models.layers.trainable import create_rectangular_dnn

def element_wise_affine(x,st,compute_jacobian=True):
    """Transform x element-wise through an affine function y = exp(s)*x + t
    where s = st[...,0] and t = st[...,1] with s.shape == x.shape == t.shape

    The Jacobian for this transformation is the coordinate-wise produc of the scaling factors
    J = prod(es[...,i],i)
    """
    es = torch.exp(st[..., 0])
    t = st[..., 1]
    logj = None
    if compute_jacobian:
        torch.sum(torch.log(es),dim=-1)

    return es*x + t, logj


class GeneralRealNVP(GeneralCouplingCell):
    """Coupling cell based on affine transforms without a specific parameter prediction function"""
    def __init__(self, *, d, mask):
        super(GeneralRealNVP, self).__init__(d=d, transform=element_wise_affine, mask=mask)

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

        self.T = create_rectangular_dnn(d_in=d_in,
                                        d_out=d_out,
                                        d_hidden=d_hidden,
                                        n_hidden=n_hidden,
                                        input_activation=input_activation,
                                        hidden_activation=torch.nn.ReLU,
                                        output_activation=output_activation,
                                        use_batch_norm=use_batch_norm)



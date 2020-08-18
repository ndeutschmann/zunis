"""This module implements an abstract class for flows defined as the application of a
differentiable pytorch function on an input point. In train mode, the last entry of the
input is the logarithm of the inverse PDF at that point.
"""
import torch
from better_abc import abstract_attribute
from zunis.models.flows.general_flow import GeneralFlow


class GeneralBackpropJacobianFlow(GeneralFlow):
    """Abstract class for a flow defined as a general differentiable Pytorch function
    This implements the universal transform_and_compute_jacobian behavior:
    y = flow(x), - log(j(y)) = - log(j(x)) + log(det(dy_i/dx_j))
    The jacobian is computed naively by using autograd.
    """
    def __init__(self, *, d):
        super(GeneralBackpropJacobianFlow, self).__init__(d=d)
        # A backprop Jacobian flow must implement an attribute flow_
        # which computes the transformation in a differentiable way
        self.flow_ = abstract_attribute()

    def flow(self, x):
        """For a backprop_flow, the transformation is just a pytorch module"""
        return self.flow_(x)

    def transform_and_compute_jacobian(self, xj):
        """Compute the flow transformation and its Jacobian using pytorch.autograd"""
        x = xj[:, :self.d].detach()
        log_j = xj[:, -1]

        x.requires_grad = True
        y = self.flow_(x)

        n_batch = xj.shape[0]

        jx = torch.zeros(n_batch, self.d, self.d).to(log_j.device)
        directions = torch.eye(self.d).to(log_j).unsqueeze(0).repeat(n_batch, 1, 1)

        for i in range(self.d):
            jx[:, i, :] = torch.autograd.grad(y, x, directions[:, i, :],
                                              allow_unused=True, create_graph=True, retain_graph=True)[0]
        x.requires_grad = False
        x.grad = None

        log_det_j = torch.log(torch.abs(torch.det(jx)))
        return torch.cat([y.detach(), (log_j + log_det_j).unsqueeze(1)], 1)

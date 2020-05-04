"""Simple flows with an easily computable jacobian - useful for tests and checks"""
import torch
from .general_backprop_j_flow import GeneralBackpropJacobianFlow


class LinearFlow(GeneralBackpropJacobianFlow):
    """Flow transformation as a pure matrix multiplication
    Expected jacobian: det(self.flow.weight)"""
    def __init__(self, *, d):
        super(LinearFlow, self).__init__(d=d)

        self.flow_ = torch.nn.Linear(d, d, bias=False)

    def weight_init_identity_(self,std=None):
        """Initialize weights as eye + normal(0,std)
        Essentially realizing a resnet-like layer
        """
        if std is None:
            std_ = 0.1/self.d
        else:
            std_ = std
        torch.nn.init.normal_(self.flow_.weight, std=std_)
        self.flow_.weight.data += torch.eye(self.d).to(self.flow_.weight.data.device)


class SigmoidFlow(GeneralBackpropJacobianFlow):
    """Flow transformation as an element-wise sigmoid application
    Expected jacobian: sigmoid(x)*(1-sigmoid(x))"""
    def __init__(self, *, d):
        super(SigmoidFlow, self).__init__(d=d)

        layers = [torch.nn.Sigmoid()]
        self.flow_ = (torch.nn.Sequential(*layers))

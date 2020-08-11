"""Neural network as a flow"""
import torch
from .general_backprop_j_flow import GeneralBackpropJacobianFlow
from zunis.models.layers.activations import NormBiTanh


class NNFlow(GeneralBackpropJacobianFlow):
    """Flow defined as a neural network mapping R^d to R^d.
    Bijectivity is not guaranteed but likely: each linear layer *can* have rank >= d
    and each activation layer is bijective.

    With weights taken randomly from any reasonable definition (flat Glorot, normal Glorot),
    the probability that the matrices are less than maximal rank is 0 so the transformation is
    nearly always bijective.

    The layers are organized as follows:
    R^d -[Linear(d,dh), activation]-> R^dh -[(Linear(dh,dh),activation)*nh]-> R^dh -[Linear(dh,d)]-> R^d
    """
    def __init__(self, *, d, nh, dh, activation_layer_class=NormBiTanh, batch_norm=False):
        super(GeneralBackpropJacobianFlow, self).__init__(d=d)
        self.nh = nh
        self.dh = dh

        assert d <= dh, "The hidden dimension must be larger than the in/out dimension for bijectivity"

        ll = torch.nn.Linear(d, dh, bias=False)
#        torch.nn.init.normal_(ll.weight, std=0.1 / d / nh)
#        ll.weight.data += torch.eye(dh, d).to(ll.weight.data.device)
        layers = [
            ll,
            activation_layer_class(),
        ]
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(dh))

        for i in range(nh):
            ll = torch.nn.Linear(dh, dh, bias=False)
#            torch.nn.init.normal_(ll.weight, std=0.1 / d / nh)
#            ll.weight.data += torch.eye(dh).to(ll.weight.data.device)
            layers.append(ll)
            layers.append(activation_layer_class())
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(dh))

        ll = torch.nn.Linear(dh, d, bias=False)
#        torch.nn.init.normal_(ll.weight, std=0.1)
#        ll.weight.data += 5. * torch.eye(d, dh).to(ll.weight.data.device)
        layers.append(ll)
        layers.append(activation_layer_class())

        self.flow_ = (torch.nn.Sequential(*layers))

    def weight_init_identity_(self,std=None):
        """Initialize weights as eye + normal(0,std)
        Essentially realizing a resnet-like layer
        """
        if std is None:
            std_ = 0.1 / self.d / self.nh
        else:
            std_ = std
        for layer in self.flow_:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, std=std_)
                layer.weight.data += torch.eye(self.d).to(layer.weight.data.device)

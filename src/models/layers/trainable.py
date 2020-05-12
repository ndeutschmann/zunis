"""Trainable layers
"""
import torch

class OverallAffineLayer(torch.nn.Module):
    """Learnable overall affine transformation
    f(x) = alpha x + delta
    """

    def __init__(self, alpha=10., delta=0.):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.delta = torch.nn.Parameter(torch.tensor(delta), requires_grad=True)

    def forward(self, input):
        """Output of the OverallAffineLayer"""
        return input * self.alpha + self.delta


def create_rectangular_dnn(
        *,
        d_in,
        d_out,
        d_hidden,
        n_hidden,
        input_activation=None,
        hidden_activation=torch.nn.ReLU,
        output_activation=None,
        use_batch_norm=False):
        
        layers = []
        if input_activation is not None:
            layers.append(input_activation())
        layers.append(torch.nn.Linear(d_in,d_hidden))
        layers.append(hidden_activation())
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(d_hidden))

        for i in range(n_hidden):
            layers.append(torch.nn.Linear(d_hidden, d_hidden))
            layers.append(hidden_activation())
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(d_hidden))

        layers.append(torch.nn.Linear(d_hidden, d_out))

        if output_activation is not None:
            layers.append(output_activation())

        return torch.nn.Sequential(*layers)


class ArbitraryShapeRectangularDNN(torch.nn.Module):
    """Rectangular DNN with the output layer reshaped to a given shape"""
    def __init__(self, *,
                 d_in,
                 out_shape,
                 d_hidden,
                 n_hidden,
                 input_activation=None,
                 hidden_activation=torch.nn.ReLU,
                 output_activation=None,
                 use_batch_norm=False):

        super(ArbitraryShapeRectangularDNN, self).__init__()
        self.out_shape = out_shape

        d_out = 1
        for d in out_shape:
            d_out *= d

        self.nn = create_rectangular_dnn(d_in=d_in,
                                         d_out=d_out,
                                         d_hidden=d_hidden,
                                         n_hidden=n_hidden,
                                         input_activation=input_activation,
                                         hidden_activation=hidden_activation,
                                         output_activation=output_activation,
                                         use_batch_norm=use_batch_norm)

    def forward(self, x):
        return self.nn(x).view(*(x.shape[:-1]), *self.out_shape)


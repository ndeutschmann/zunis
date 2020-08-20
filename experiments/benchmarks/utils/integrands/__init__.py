"""Functions defined on the unit hypercube for the purpose of integration tests"""
from numbers import Number
import torch
import numpy as np


def sanitize_variable(x, device=None):
    """Prepare input variable for a pytorch function:
    if it is a python numerical variable or a numpy array, then cast it to a tensor
    if it is is a tensor, make a copy to make sure the function cannot be altered from outside

    Parameters
    ----------
    x: float or np.ndarray or torch.Tensor
    device: None or torch.device

    Returns
    -------
        torch.Tensor

    """
    if isinstance(x, Number) or isinstance(x, np.ndarray):
        if device is None:
            device = torch.device("cpu")
        x = torch.tensor(x).to(device)
    else:
        assert isinstance(x, torch.Tensor), "Only numerical types, numpy arrays and torch Tensor accepted"
        x = x.clone().detach()
        if device is not None:
            x = x.to(device)
    return x



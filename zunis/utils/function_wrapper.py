"""Function wrappers to provide API-compatible functions"""

import torch


def wrap_numpy_hypercube_batch_function(f):
    """Take a function that evaluates on numpy batches with values in the unit hypercube
    and return a function that evaluates on pytorch batches in the unit hypercube
    """
    def torchf(x):
        npx = x.detach().cpu().numpy()
        npfx = f(npx)
        return torch.tensor(npfx, device=x.device)

    return torchf


def wrap_numpy_compact_batch_function(f, dimensions):
    """Take a function that evaluates on numpy batches with values in compact intervals provided
    as a list of shape (d,2) where each element is the pair (lower,upper) of interval boundaries.
    and return a function that evaluates on pytorch batches in the unit hypercube,
    weighted by the proper Jacobian factor to preserve integrals.
    """
    tdim = torch.tensor(dimensions)
    assert tdim.shape[1] == 2, "argument dimensions is expected to have shape (N,2)"
    assert torch.all(tdim[:, 1] > tdim[:, 0]), "Each dimension is expected to be (a,b) with a<b"
    starts = tdim[:, 0]
    lengths = tdim[:, 1] - tdim[:, 0]
    jac = torch.prod(lengths).cpu().item()

    def torchf(x):
        npx = (x * lengths.to(x.device) + starts.to(x.device)).detach().cpu().numpy()
        npfx = f(npx)
        return (torch.tensor(npfx) * jac).to(x.device)

    return torchf


def wrap_compact_arguments_function(f, dimensions):
    """Take a function that evaluates on a sequence of arguments with values in compact intervals provided
    as a list of shape (d,2) where each element is the pair (lower,upper) of interval boundaries and
    return a function that evaluates on pytorch batches in the unit hypercube,
    weighted by the proper Jacobian factor to preserve integrals.

    Explicitly: f(x_1,x_2,...,x_N) where x_i are numbers in [dimensions[i][0], dimensions[i][1]] returns a single float.
    """
    tdim = torch.tensor(dimensions)
    assert tdim.shape[1] == 2, "argument dimensions is expected to have shape (N,2)"
    assert torch.all(tdim[:, 1] > tdim[:, 0]), "Each dimension is expected to be (a,b) with a<b"
    starts = tdim[:, 0]
    lengths = tdim[:, 1] - tdim[:, 0]
    jac = torch.prod(lengths).item()

    def torchf(x):
        lxs = (x * lengths.to(x.device) + starts.to(x.device)).detach().cpu().tolist()
        fxs = torch.zeros(x.shape[0], device=x.device)
        for i, lx in enumerate(lxs):
            fxs[i] = f(*lx)
        return fxs * jac

    return torchf


def wrap_hypercube_arguments_function(f):
    """Take a function that evaluates on a sequence of arguments with values in the unit hypercube and
        return a function that evaluates on pytorch batches in the unit hypercube,
        weighted by the proper Jacobian factor to preserve integrals.

        Explicitly: f(x_1,x_2,...,x_N) where x_i are numbers in [0, 1] returns a single float.
        """
    def torchf(x):
        lxs = x.detach().cpu().tolist()
        fxs = torch.zeros(x.shape[0], device=x.device)
        for i, lx in enumerate(lxs):
            fxs[i] = f(*lx)
        return fxs

    return torchf

"""Implementation of the piecewie linear coupling cell
This means that the *variable transform* is piecewise-linear.
"""
import torch

third_dimension_softmax = torch.nn.Softmax(dim=2)


def piecewise_linear_transform(x, q_tilde, compute_jacobian=True):
    """Apply an element-wise piecewise-linear transformation to some variables

    Parameters
    ----------
    x : torch.Tensor
        a tensor with shape (N,k) where N is the batch dimension while k is the
        dimension of the variable space. This variable span the k-dimensional unit
        hypercube

    q_tilde: torch.Tensor
        is a tensor with shape (N,k,b) where b is the number of bins.
        This contains the un-normalized heights of the bins of the piecewise-constant PDF for dimension k,
        i.e. q_tilde lives in all of R and we don't impose a constraint on their sum yet.
        Normalization is imposed in this function using softmax.

    compute_jacobian : bool, optional
        determines whether the jacobian should be compute or None is returned

    Returns
    -------
    tuple of torch.Tensor
        pair `(y,h)`.
        - `y` is a tensor with shape (N,k) living in the k-dimensional unit hypercube
        - `j` is the jacobian of the transformation with shape (N,) if compute_jacobian==True, else None.
    """

    j = None

    # TODO do a bottom-up assesment of how we handle the differentiability of variables

    # Compute the bin width w
    N, k, b = q_tilde.shape
    Nx, kx = x.shape
    assert N == Nx and k == kx, "Shape mismatch"

    w = 1. / b

    # Compute the normalized bin heights by applying a softmax function on the bin dimension
    q = 1. / w * third_dimension_softmax(q_tilde)

    # x is in the mx-th bin: x \in [0,1],
    # mx \in [[0,b-1]], so we clamp away the case x == 1
    mx = torch.clamp(torch.floor(b * x), 0, b - 1).to(torch.long)
    # We compute the output variable in-place
    out = x - mx * w  # alpha (element of [0.,w], the position of x in its bin

    # Multiply by the slope
    # q has shape (N,k,b), mxu = mx.unsqueeze(-1) has shape (N,k) with entries that are a b-index
    # gather defines slope[i, j, k] = q[i, j, mxu[i, j, k]] with k taking only 0 as a value
    # i.e. we say slope[i, j] = q[i, j, mx [i, j]]
    slopes = torch.gather(q, 2, mx.unsqueeze(-1)).squeeze(-1)
    # We do not ensure the differentiability of the *points*
    # Only of the jacobians.
    # It is therefore much cheaper to detach
    out = out * slopes.detach()
    # The jacobian is the product of the slopes in all dimensions
    if compute_jacobian:
        j = torch.prod(slopes, 1)
    del slopes

    # Compute the integral over the left-bins.
    # 1. Compute all integrals: cumulative sum of bin height * bin weight.
    # We want that index i contains the cumsum *strictly to the left* so we shift by 1
    # leaving the first entry null, which is achieved with a roll and assignment
    q_left_integrals = torch.roll(torch.cumsum(q, 2) * w, 1, 2)
    q_left_integrals[:, :, 0] = 0

    # 2. Access the correct index to get the left integral of each point and add it to our transformation
    out = out + torch.gather(q_left_integrals, 2, mx.unsqueeze(-1)).squeeze(-1)

    return out, j


def piecewise_linear_inverse_transform(y, q_tilde, compute_jacobian=True):
    """
    Apply the inverse of an element-wise piecewise-linear transformation to some variables

    Parameters
    ----------
    y : torch.Tensor
        a tensor with shape (N,k) where N is the batch dimension while k is the
        dimension of the variable space. This variable span the k-dimensional unit
        hypercube

    q_tilde: torch.Tensor
        is a tensor with shape (N,k,b) where b is the number of bins.
        This contains the un-normalized heights of the bins of the piecewise-constant PDF for dimension k,
        i.e. q_tilde lives in all of R and we don't impose a constraint on their sum yet.
        Normalization is imposed in this function using softmax.

    compute_jacobian : bool, optional
        determines whether the jacobian should be compute or None is returned

    Returns
    -------
    tuple of torch.Tensor
        pair `(x,h)`.
        - `x` is a tensor with shape (N,k) living in the k-dimensional unit hypercube
        - `j` is the jacobian of the transformation with shape (N,) if compute_jacobian==True, else None.
    """

    j = None

    # TODO do a bottom-up assesment of how we handle the differentiability of variables

    # Compute the bin width w
    N, k, b = q_tilde.shape
    Ny, ky = y.shape
    assert N == Ny and k == ky, "Shape mismatch"

    w = 1. / b

    # Compute the normalized bin heights by applying a softmax function on the bin dimension
    q = 1. / w * third_dimension_softmax(q_tilde)

    # Compute the integral over the left-bins in the forward transform.
    # 1. Compute all integrals: cumulative sum of bin height * bin weight.
    # We want that index i contains the cumsum *strictly to the left* so we shift by 1
    # leaving the first entry null, which is achieved with a roll and assignment
    q_left_integrals = torch.roll(torch.cumsum(q, 2) * w, 1, 2)
    q_left_integrals[:, :, 0] = 0

    # We can figure out which bin each y belongs to by finding the smallest bin such that
    # y - q_left_integral is positive

    edges = y.unsqueeze(-1) - q_left_integrals
    # y and q_left_integrals are between 0 and 1 so that their difference is at most 1.
    # By setting the negative values to 2., we know that the smallest value left
    # is the smallest positive
    edges[edges < 0] = 2.
    edges = torch.argmin(edges, dim=2)

    # Gather the left integrals at each edge. See comment about gathering in q_left_integrals
    # for the unsqueeze
    q_left_integrals = q_left_integrals.gather(2, edges.unsqueeze(-1))

    # Gather the slope at each edge.
    q = q.gather(2, edges.unsqueeze(-1))

    # Build the output
    x = (y - q_left_integrals) / q + edges * w

    # Prepare the jacobian
    j = None
    if compute_jacobian:
        j = 1. / torch.prod(q, 2)

    return x, j

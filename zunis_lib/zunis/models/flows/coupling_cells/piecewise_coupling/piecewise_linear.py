"""Implementation of the piecewise linear coupling cell
This means that the *variable transform* is piecewise-linear.
"""
import torch

from ..transforms import InvertibleTransform
from ..general_coupling import InvertibleCouplingCell
from zunis.utils.exceptions import AvertedCUDARuntimeError
from zunis.models.layers.trainable import ArbitraryShapeRectangularDNN
from zunis.models.utils import Reshift

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

    logj = None

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
    # Need special error handling because trying to index with mx
    # if it contains nans will lock the GPU. (device-side assert triggered)
    if torch.any(torch.isnan(mx)).item() or torch.any(mx < 0) or torch.any(mx >= b):
        raise AvertedCUDARuntimeError("NaN detected in PWLinear bin indexing")

    # We compute the output variable in-place
    out = x - mx * w  # alpha (element of [0.,w], the position of x in its bin

    # Multiply by the slope
    # q has shape (N,k,b), mxu = mx.unsqueeze(-1) has shape (N,k) with entries that are a b-index
    # gather defines slope[i, j, k] = q[i, j, mxu[i, j, k]] with k taking only 0 as a value
    # i.e. we say slope[i, j] = q[i, j, mx [i, j]]
    slopes = torch.gather(q, 2, mx.unsqueeze(-1)).squeeze(-1)
    out = out * slopes
    # The jacobian is the product of the slopes in all dimensions
    if compute_jacobian:
        logj = torch.log(torch.prod(slopes, 1))

    del slopes

    # Compute the integral over the left-bins.
    # 1. Compute all integrals: cumulative sum of bin height * bin weight.
    # We want that index i contains the cumsum *strictly to the left* so we shift by 1
    # leaving the first entry null, which is achieved with a roll and assignment
    q_left_integrals = torch.roll(torch.cumsum(q, 2) * w, 1, 2)
    q_left_integrals[:, :, 0] = 0

    # 2. Access the correct index to get the left integral of each point and add it to our transformation
    out = out + torch.gather(q_left_integrals, 2, mx.unsqueeze(-1)).squeeze(-1)

    # Regularization: points must be strictly within the unit hypercube
    # Use the dtype information from pytorch
    eps = torch.finfo(out.dtype).eps
    out = out.clamp(
        min=eps,
        max=1. - eps
    )

    return out, logj


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

    edges = (y.unsqueeze(-1) - q_left_integrals).detach()
    # y and q_left_integrals are between 0 and 1 so that their difference is at most 1.
    # By setting the negative values to 2., we know that the smallest value left
    # is the smallest positive
    edges[edges < 0] = 2.
    edges = torch.clamp(torch.argmin(edges, dim=2), 0, b - 1).to(torch.long)

    # Need special error handling because trying to index with mx
    # if it contains nans will lock the GPU. (device-side assert triggered)
    if torch.any(torch.isnan(edges)).item() or torch.any(edges < 0) or torch.any(edges >= b):
        raise AvertedCUDARuntimeError("NaN detected in PWLinear bin indexing")

    # Gather the left integrals at each edge. See comment about gathering in q_left_integrals
    # for the unsqueeze
    q_left_integrals = q_left_integrals.gather(2, edges.unsqueeze(-1)).squeeze(-1)

    # Gather the slope at each edge.
    q = q.gather(2, edges.unsqueeze(-1)).squeeze(-1)

    # Build the output
    x = (y - q_left_integrals) / q + edges * w

    # Regularization: points must be strictly within the unit hypercube
    # Use the dtype information from pytorch
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(
        min=eps,
        max=1. - eps
    )

    # Prepare the jacobian
    logj = None
    if compute_jacobian:
        logj = - torch.log(torch.prod(q, 1))
    return x.detach(), logj


class ElementWisePWLinearTransform(InvertibleTransform):
    """Invertible piecewise-linear transformations over the unit hypercube

    Implements a batched bijective transformation `h` from the d-dimensional unit hypercube to itself,
    in an element-wise fashion (each coordinate transformed independently)

    In each direction, the bijection is a piecewise-linear transform with b bins
    where the forward transform has evenly spaced bins. The transformation in each bin is
    actually an affine transformation. The slopes for each direction and each point in the batch
    are given by an unormalized tensor `q_tilde`. This input is softmax-normalized such that
    1. h(0) = 0
    2. h(1) = 1
    3. h is monotonous
    4. h is continuous

    for which knowing the slopes in each bin is sufficient (when the abuse of language "linear")

    Conditions 1. to 3. ensure the transformation is a bijection and therefore invertible
    The inverse is also an element-wise, piece-wise linear transformation,
    but, of course, with variable input bin sizes (and fixed output bin sizes).
    """

    backward = staticmethod(piecewise_linear_transform)
    forward = staticmethod(piecewise_linear_inverse_transform)


class GeneralPWLinearCoupling(InvertibleCouplingCell):
    """Abstract class implementing a coupling cell based on PW linear transformations

    A specific way to predict the parameters of the transform must be implemented
    in child classes.
    """

    def __init__(self, *, d, mask):
        """Generator for the abstract class GeneralPWLinearCoupling

        Parameters
        ----------
        d: int
            dimension of the space
        mask: list of bool
            variable mask which variables are transformed (False)
            or used as parameters of the transform (True)

        """
        super(GeneralPWLinearCoupling, self).__init__(d=d, transform=ElementWisePWLinearTransform(), mask=mask)


class PWLinearCoupling(GeneralPWLinearCoupling):
    """Piece-wise Linear coupling

    Coupling cell using an element-wise piece-wise linear transformation as a change of
    variables. The transverse neural network is a rectangular dense neural network

    Notes:
        Transformation used:
        `zunis.models.flows.coupling_cells.piecewise_coupling.piecewise_linear.ElementWisePWLinearTransform`
        Neural network used:
        zunis.models.layers.trainable.ArbitraryShapeRectangularDNN
    """

    def __init__(self, *, d, mask,
                 n_bins=10,
                 d_hidden=256,
                 n_hidden=8,
                 input_activation=Reshift,
                 hidden_activation=torch.nn.LeakyReLU,
                 output_activation=None,
                 use_batch_norm=False):
        """
        Generator for PWLinearCoupling

        Parameters
        ----------
        d: int
        mask: list of bool
            variable mask: which dimension are transformed (False) and which are not (True)
        n_bins: int
            number of bins in each dimensions
        d_hidden: int
            dimension of the hidden layers of the DNN
        n_hidden: int
            number of hidden layers in the DNN
        input_activation: optional
            pytorch activation function before feeding into the DNN.
            must be a callable generator without arguments (i.e. a classname or a function)
        hidden_activation: optional
            pytorch activation function between hidden layers of the DNN.
            must be a callable generator without arguments (i.e. a classname or a function)
        output_activation: optional
            pytorch activation function at the output of the DNN.
            must be a callable generator without arguments (i.e. a classname or a function)
        use_batch_norm: bool
            whether batch normalization should be used in the DNN.
        """

        super(PWLinearCoupling, self).__init__(d=d, mask=mask)

        d_in = sum(mask)
        d_out = d - d_in

        self.T = ArbitraryShapeRectangularDNN(d_in=d_in,
                                              out_shape=(d_out, n_bins),
                                              d_hidden=d_hidden,
                                              n_hidden=n_hidden,
                                              input_activation=input_activation,
                                              hidden_activation=hidden_activation,
                                              output_activation=output_activation,
                                              use_batch_norm=use_batch_norm)

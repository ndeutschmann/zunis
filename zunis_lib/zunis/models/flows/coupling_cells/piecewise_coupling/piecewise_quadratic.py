"""Implementation of the piecewise quadratic coupling cell
This means that the *variable transform* is piecewise-quadratic.
"""
import torch
import numpy as np

from ..transforms import InvertibleTransform
from ..general_coupling import InvertibleCouplingCell
from zunis.utils.exceptions import AvertedCUDARuntimeError
from zunis.models.layers.trainable import ArbitraryShapeRectangularDNN
from zunis.models.utils import Reshift

third_dimension_softmax = torch.nn.Softmax(dim=2)

def modified_softmax (v,w):
    v=torch.exp(v)
    vsum=torch.cumsum(v, axis=-1)
    vnorms=torch.cumsum(torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w),axis=-1)
    vnorms_tot=vnorms[:, :, -1].clone() 
    return torch.div(v,torch.unsqueeze(vnorms_tot,axis=-1)) 


def piecewise_quadratic_transform(x, wv_tilde, compute_jacobian=True):
    """Apply an element-wise piecewise-quadratic transformation to some variables

    Parameters
    ----------
    x : torch.Tensor
        a tensor with shape (N,k) where N is the batch dimension while k is the
        dimension of the variable space. This variable spans the k-dimensional unit
        hypercube

    wv_tilde: torch.Tensor
        is a tensor with shape (N,k,2b+1) where b is the number of bins.
        This contains the un-normalized widths and heights of the bins of the piecewise-constant 
        PDF for dimension k,
        i.e. q_tilde lives in all of R and we don't impose a constraint on their sum yet.
        Normalization is imposed in this function using a modified softmax.

    compute_jacobian : bool, optional
        determines whether the jacobian should be compute or None is returned

    Returns
    -------
    tuple of torch.Tensor
        pair `(y,j)`.
        - `y` is a tensor with shape (N,k) living in the k-dimensional unit hypercube
        - `j` is the jacobian of the transformation with shape (N,) if compute_jacobian==True, else None.
    """
    logj = None

    # TODO do a bottom-up assesment of how we handle the differentiability of variables
    
    v_tilde=wv_tilde[:,:,:int(np.ceil(wv_tilde.shape[2]/2))]
    w_tilde=wv_tilde[:,:,v_tilde.shape[2]:]
    N, k, b = w_tilde.shape
    Nx, kx = x.shape
    assert N == Nx and k == kx, "Shape mismatch"
    
    w=torch.exp(w_tilde)
    wsum = torch.cumsum(w, axis=-1) 
    wnorms = torch.unsqueeze(wsum[:, :, -1], axis=-1) 
    w = w/wnorms
    wsum=wsum/wnorms
    wsum_shift=torch.cat((torch.zeros([wsum.shape[0],wsum.shape[1],1]).to(wsum.device, wsum.dtype),wsum),axis=-1)
    
    v=modified_softmax(v_tilde, w)
    
    #tensor of shape (N,k,b+1) with 0 entry if x is smaller than the cumulated w and 1 if it is bigger
    #this is used to find the bin with the number mx in which x lies; for this, the sum of the bin 
    #widths w has to be smaller to the left and bigger to the right
    finder=torch.where(wsum>torch.unsqueeze(x,axis=-1),torch.zeros_like(wsum),torch.ones_like(wsum))
    eps = torch.finfo(wsum.dtype).eps
    #the bin number can be extracted by finding the last index for which finder is nonzero. As wsum 
    #is increasing, this can be found by searching for the maximum entry of finder*wsum. In order to 
    #get the right result when x is in the first bin and finder is everywhere zero, a small first entry 
    #is added
    mx=torch.unsqueeze(  #we need to unsqueeze for later operations
        torch.argmax( #we search for the maximum in order to find the last bin for which x was greater than wsum
            torch.cat((torch.ones([wsum.shape[0],wsum.shape[1],1]).to(wsum.device, wsum.dtype)*eps,finder*wsum),
                      axis=-1),  #we add an offset to ensure that if x is in the first bin, a maximal argument is found 
            axis=-1), 
        axis=-1)
   
    # x is in the mx-th bin: x \in [0,1],
    # mx \in [[0,b-1]], so we clamp away the case x == 1
    mx = torch.clamp(mx, 0, b - 1).to(torch.long)
    # Need special error handling because trying to index with mx
    # if it contains nans will lock the GPU. (device-side assert triggered)
    if torch.any(torch.isnan(mx)).item() or torch.any(mx < 0) or torch.any(mx >= b):
        raise AvertedCUDARuntimeError("NaN detected in PWQuad bin indexing")
    
    # alpha (element of [0.,1], the position of x in its bin)
    # gather collects the cumulated with of all bins until the one in which x lies
    # alpha=(x- Sum_(k=0)^(i-1) w_k)/w_b for x in bin b
    alphas=torch.div((x-torch.squeeze(torch.gather(wsum_shift,-1,mx),axis=-1)),
                         torch.squeeze(torch.gather(w,-1,mx),axis=-1))
    
    #vw_i= (v_i+1 - v_i)w_i/2 where i is the bin index
    vw=torch.cat((torch.zeros([v.shape[0],v.shape[1],1]).to(wsum.device, wsum.dtype),
                                  torch.cumsum(torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w),axis=-1)),axis=-1)
    
    #quadratic term
    out_1=torch.mul((alphas**2)/2,torch.squeeze(torch.mul(torch.gather(v,-1, mx+1)-torch.gather(v,-1, mx),
                                                        torch.gather(w,-1,mx)),axis=-1))
    
    #linear term
    out_2=torch.mul(torch.mul(alphas,torch.squeeze(torch.gather(v,-1,mx),axis=-1)),
                            torch.squeeze(torch.gather(w,-1,mx),axis=-1))
    
    #constant
    out_3= torch.squeeze(torch.gather(vw,-1,mx),axis=-1)
    
    
    out=out_1+out_2+out_3
    
    #the derivative of this transformation is the linear interpolation between v_i-1 and v_i at alpha
    #the jacobian is the product of all linear interpolations
    if compute_jacobian:
        logj=torch.squeeze(
            torch.log(torch.unsqueeze(
                    torch.prod(#we need to take the product over all transformed dimensions
                        torch.lerp(torch.squeeze(torch.gather(v,-1,mx),axis=-1),
                                   torch.squeeze(torch.gather(v,-1,mx+1),axis=-1),alphas),
                        #linear extrapolation between alpha, mx and mx+1
                    axis=-1),
                axis=-1)),
            axis=-1)
       
    # Regularization: points must be strictly within the unit hypercube
    # Use the dtype information from pytorch
    eps = torch.finfo(out.dtype).eps
    out = out.clamp(
        min=eps,
        max=1. - eps
    )
    return out, logj


def piecewise_quadratic_inverse_transform(y, wv_tilde, compute_jacobian=True):
    """
    Apply the inverse of an element-wise piecewise-linear transformation to some variables

    Parameters
    ----------
    y : torch.Tensor
        a tensor with shape (N,k) where N is the batch dimension while k is the
        dimension of the variable space. This variable span the k-dimensional unit
        hypercube

    wv_tilde: torch.Tensor
        is a tensor with shape (N,k,2b+1) where b is the number of bins.
        This contains the un-normalized widths and heights of the bins of the piecewise-constant PDF for dimension k,
        i.e. q_tilde lives in all of R and we don't impose a constraint on their sum yet.
        Normalization is imposed in this function using a modified softmax.
        

    compute_jacobian : bool, optional
        determines whether the jacobian should be compute or None is returned

    Returns
    -------
    tuple of torch.Tensor
        pair `(x,j)`.
        - `x` is a tensor with shape (N,k) living in the k-dimensional unit hypercube
        - `j` is the jacobian of the transformation with shape (N,) if compute_jacobian==True, else None.
    """
    
    logj = None

    # TODO do a bottom-up assesment of how we handle the differentiability of variables
    
    v_tilde=wv_tilde[:,:,:int(np.ceil(wv_tilde.shape[2]/2))]
    w_tilde=wv_tilde[:,:,v_tilde.shape[2]:]
    N, k, b = w_tilde.shape
    
    Nx, kx = y.shape
    assert N == Nx and k == kx, "Shape mismatch"
    
    w=torch.exp(w_tilde)
    wsum = torch.cumsum(w, axis=-1) 
    wnorms = torch.unsqueeze(wsum[:, :, -1], axis=-1) 
    w = w/wnorms
    wsum=wsum/wnorms
    wsum_shift=torch.cat((torch.zeros([wsum.shape[0],wsum.shape[1],1]).to(wsum.device, wsum.dtype),wsum),axis=-1)
    
    v=modified_softmax(v_tilde, w)
    
    #need to find the bin number for each of the y/x
    #-> find the last bin such that y is greater than the constant of the quadratic equation
    
    #vw_i= (v_i+1 - v_i)w_i/2 where i is the bin index. VW is the constant of the quadratic equation
    vw=torch.cat((torch.zeros([v.shape[0],v.shape[1],1]).to(wsum.device, wsum.dtype),
                                  torch.cumsum(torch.mul((v[:,:,:-1]+v[:,:,1:])/2,w),axis=-1)),axis=-1)
    # finder is contains 1 where y is smaller then the constant and 0 if it is greater
    finder=torch.where(vw>torch.unsqueeze(y,axis=-1),torch.zeros_like(vw),torch.ones_like(vw))
    eps = torch.finfo(vw.dtype).eps
    #the bin number can be extracted by finding the last index for which finder is nonzero. As vw 
    #is increasing, this can be found by searching for the maximum entry of finder*vw. In order to 
    #get the right result when y is in the first bin and finder is everywhere zero, a small first entry 
    #is added and mx is reduced by one to account for the shift.
    mx=torch.unsqueeze(
        torch.argmax(#we search for the maximum in order to find the last bin for which y was greater than vw
            torch.cat((torch.ones([vw.shape[0],vw.shape[1],1]).to(vw.device, vw.dtype)*eps,finder*(vw+1)),axis=-1),
            axis=-1), #we add an offset to ensure that if x is in the first bin, a maximal argument is found
        axis=-1)-1 # we substract -1 to account for the offset
    
    # x is in the mx-th bin: x \in [0,1],
    # mx \in [[0,b-1]], so we clamp away the case x == 1
    edges = torch.clamp(mx, 0, b - 1).to(torch.long)
    
    # Need special error handling because trying to index with mx
    # if it contains nans will lock the GPU. (device-side assert triggered)
    if torch.any(torch.isnan(edges)).item() or torch.any(edges < 0) or torch.any(edges >= b):
        raise AvertedCUDARuntimeError("NaN detected in PWQuad bin indexing")
    
    #solve quadratic equation
    
    #prefactor of quadratic term
    a=torch.squeeze(torch.mul(torch.gather(v,-1, edges+1)-torch.gather(v,-1, edges),
                                                        torch.gather(w,-1,edges)),axis=-1)
    #prefactor of linear term
    b=torch.mul(torch.squeeze(torch.gather(v,-1,edges),axis=-1),torch.squeeze(torch.gather(w,-1,edges),axis=-1))
    #constant - y
    c= torch.squeeze(torch.gather(vw,-1,edges),axis=-1)-y
    
    #ensure that division by zero is taken care of
    eps = torch.finfo(a.dtype).eps
    a=torch.where(torch.abs(a)<eps,eps*torch.ones_like(a),a)
    
    d = (b**2) - (2*a*c)
    
    assert not torch.any(d<0), "Value error in PWQuad inversion"
    assert not torch.any(a==0), "Value error in PWQuad inversion, a==0"
    
    # find two solutions
    sol1 = (-b-torch.sqrt(d))/(a)
    sol2 = (-b+torch.sqrt(d))/(a)
    
    # choose solution which is in the allowed range
   
    sol=torch.where((sol1>=0)&(sol1<1), sol1, sol2)
    
    if torch.any(torch.isnan(sol)).item():
        raise AvertedCUDARuntimeError("NaN detected in PWQuad inversion")
    
    eps = torch.finfo(sol.dtype).eps
    
    
    sol = sol.clamp(
        min=eps,
        max=1. - eps
    )
    
    #the solution is the relative position inside the bin. This can be
    #converted to the absolute position by adding the sum of the bin widths up to this bin
    x=torch.mul(torch.squeeze(torch.gather(w,-1,edges),axis=-1), sol)+torch.squeeze(torch.gather(wsum_shift,-1,edges),axis=-1)
    
    eps = torch.finfo(x.dtype).eps
    
    x = x.clamp(
        min=eps,
        max=1. - eps
    )
    
    if compute_jacobian:
        logj =-torch.squeeze(torch.log(
            torch.unsqueeze(torch.prod(#we have to take the product of the jacobian of all dimensions
                torch.lerp(torch.squeeze(torch.gather(v,-1,edges),axis=-1),torch.squeeze(torch.gather(v,-1,edges+1),
                                                                                         axis=-1),sol),
                axis=-1), #linear extrapolation between sol, edges and edges+1 gives the jacobian of the forward transformation. The prefactor of -1 is the log of the jacobian of the inverse
            axis=-1)),
        axis=-1)
       
    return x.detach(), logj


class ElementWisePWQuadraticTransform(InvertibleTransform):
    """Invertible piecewise-quadratic transformations over the unit hypercube

    Implements a batched bijective transformation `h` from the d-dimensional unit hypercube to itself,
    in an element-wise fashion (each coordinate transformed independently)

    In each direction, the bijection is a piecewise-quadratic transform with b bins
    where the forward transform has bins with adjustable width. The transformation in each bin is
    then a quadratic spline. The network predicts the bin width w_tilde and the vertex height v_tilde of the
    derivative of the transform for each direction and each point of the batch. They are normalized such that: 
    1. h(0) = 0
    2. h(1) = 1
    3. h is monotonous
    4. h is continuous

    Conditions 1. to 3. ensure the transformation is a bijection and therefore invertible.
    The inverse is also an element-wise, piece-wise quadratic transformation.
    """

    backward = staticmethod(piecewise_quadratic_transform)
    forward = staticmethod(piecewise_quadratic_inverse_transform)


class GeneralPWQuadraticCoupling(InvertibleCouplingCell):
    """Abstract class implementing a coupling cell based on PW quadratic transformations

    A specific way to predict the parameters of the transform must be implemented
    in child classes.
    """

    def __init__(self, *, d, mask):
        """Generator for the abstract class GeneralPWQuadraticCoupling

        Parameters
        ----------
        d: int
            dimension of the space
        mask: list of bool
            variable mask which variables are transformed (False)
            or used as parameters of the transform (True)

        """
        super(GeneralPWQuadraticCoupling, self).__init__(d=d, transform=ElementWisePWQuadraticTransform(), mask=mask)


class PWQuadraticCoupling(GeneralPWQuadraticCoupling):
    """Piece-wise Quadratic coupling

    Coupling cell using an element-wise piece-wise quadratic transformation as a change of
    variables. The transverse neural network is a rectangular dense neural network.

    Notes:
        Transformation used:
        `zunis.models.flows.coupling_cells.piecewise_coupling.piecewise_quadratic.ElementWisePWQuadraticTransform`
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
        Generator for PWQuadraticCoupling

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

        super(PWQuadraticCoupling, self).__init__(d=d, mask=mask)

        d_in = sum(mask)
        d_out = d - d_in

        self.T = ArbitraryShapeRectangularDNN(d_in=d_in,
                                              out_shape=(d_out, 2*n_bins+1),
                                              d_hidden=d_hidden,
                                              n_hidden=n_hidden,
                                              input_activation=input_activation,
                                              hidden_activation=hidden_activation,
                                              output_activation=output_activation,
                                              use_batch_norm=use_batch_norm)

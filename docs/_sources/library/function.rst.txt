:todo:

Integrand functions
###################

The ZüNIS library is a tool to compute integrals and therefore functions are a central element of its API.
The goal here is to be as agnostic possible as to which functions can be integrated and they are indeed always
treated as a black box. In particular they do not need to be differentiable, run on a specific device, on a
specific thread, etc.

The specifications we enforce are:

1. integrals are always computed over a d-dimensional unit hypercube
2. a function is a callable Python object
3. input and output are provided by batch

In specific terms, the input will always be a :code:`torch.Tensor` object :code:`x` with shape :math:`(N, d)` and values between 0 and 1,
and the output is expected to be a :code:`torch.Tensor` object :code:`y` with shape :math:`(N,)`, such that :code:`y[i] = f(x[i])`

Basic example
#############

The most basic usage of this library is using the default
:py:func:`Integrator <zunis.integration.default_integrator.Integrator>` API
as follows

.. code-block:: python

    import torch
    from zunis.integration import Integrator

    device = torch.device("cuda")

    d = 2

    def f(x):
        return x[:,0]**2 + x[:,1]**2

    integrator = Integrator(d=d,f=f,device=device)
    result, uncertainty, history = integrator.integrate()

The function `f` is integrated over the `d`-dimensional unit hypercube and

* takes `torch.Tensor` batched inputs with shape `(N,d)` for arbitrary batch size `N` on `device`
* returns `torch.Tensor` batched inputs with shape `(N,)` for arbitrary batch size `N` on `device`

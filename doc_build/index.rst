.. ZuNIS Documentation

###################
ZüNIS documentation
###################


ZüNIS (Zürich Neural Importance Sampling) a work-in-progress Pytorch-based library for Monte-Carlo integration
based on Neural imporance sampling `[1]`_, developed at ETH Zürich.
In simple terms, we use artificial intelligence to compute integrals faster.

The goal is to provide a flexible library to integrate black-box functions with a level of automation comparable
to the VEGAS Library `[2]`_, while using state-of-the-art methods that go around the limitations of existing tools.

.. _[1]: https://arxiv.org/abs/1808.03856
.. _[2]: https://pypi.org/project/vegas/

.. _overview:


Get Started
***********

Do you need to compute an integral *right now* and cannot wait?

1. go to the :doc:`Installation <library/installation>` page
2. have a look at our :doc:`Basic Example <library/basic-example>`


API Overview
************

The ZüNIS library provides three level of abstractions, to allow both high-level and fine-grained control:

1. :doc:`Integrators <library/integrator>` are the highest level of abstraction and control function integration strategies.
They can automate trainer and flow creation.

2. :doc:`Trainers <library/trainer>` are one level below and steer model training through loss functions, optimizers, sampling etc.
They can automate flow creation.

3. :doc:`Normalizing Flows <library/flow>` are neural-network-based bijections from the unit hypercube to itself. They are the
actual trainable sampling mechanism that we use to sample points for Monte Carlo integration.



Functions
^^^^^^^^^

The ZüNIS library is a tool to compute integrals and therefore functions are a central element of its API.
The goal here is to be as agnostic possible as to which functions can be integrated and they are indeed always
treated as a black box. In particular they do not need to be differentiable, run on a specific device, on a
specific thread, etc.

The specifications we enforce are:

1. integrals are always computed over a d-dimensional unit hypercube
2. a function is a callable Python object
3. input and output are provided by batch
4. the output must be positive [TEMPORARY]

In specific terms, the input will always be a :code:`torch.Tensor` object :code:`x` with shape :math:`(N, d)` and values between 0 and 1,
and the output is expected to be a :code:`torch.Tensor` object :code:`y` with shape :math:`(N,)`, such that :code:`y[i] = f(x[i])`


Importance sampling
*******************

ZüNIS is a tool to compute integrals by `importance sampling`_ Monte Carlo estimation. This means that we have a
function :math:`f` defined over some multi-dimensional space :math:`\Omega` and we want to compute

.. math::

    I = \int_\Omega dx f(x)

The importance sampling approach is based on the observation that
for any non-zero probability distribution function :math:`p` over :math:`\Omega`,

.. math::
    I = \underset{x \sim p(x) } {\mathbb{E}}\frac{f(x)}{p(x)}

We can therefore define an estimator for :math:`I` by sampling :math:`N` points from :math:`\Omega`.
The standard deviation of this estimator :math:`\hat{I}_N` is

.. math::
    \sigma\left[\hat{I}_N\right] = \frac{1}{\sqrt{N}}\left(\underset{x \sim p(x)}{\sigma}\left[  \frac{f(x)}{p(x)}\right]\right)

and the name of the game is to find a :math:`p(x)` that minimizes this quantity in order to minimize the number of times
we need to sample the function :math:`f` to attain a given uncertainty on our integral estimation.

If this seems like a problem that machine learning should be able to solve, you are indeed onto something.

.. _importance sampling: https://en.wikipedia.org/wiki/Monte_Carlo_integration#Importance_sampling


.. toctree::
    :maxdepth: 1
    :hidden:

    Homepage <self>

.. toctree::
    :caption: Library
    :maxdepth: 1
    :hidden:

    Installation <library/installation>
    Basic example <library/basic-example>
    Concepts <library/concepts>

.. toctree::
    :caption: Background
    :maxdepth: 1
    :hidden:

    Neural Importance Sampling <background/nis>
    Training strategies <background/training>

.. toctree::
    :caption: Documentation
    :maxdepth: 1
    :hidden:

    API Documentation <api/zunis>
    Module Hierarchy <py-modindex>
    Symbol Index <genindex>


.. toctree::
    :caption: Info
    :maxdepth: 1
    :hidden:

    References <references>
    About <about>

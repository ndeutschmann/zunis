Concepts
########

.. toctree::
    :hidden:
    :maxdepth: 1

    Integrators <integrator>
    Trainers <trainer>
    Normalizing flows <flow>
    Integrand functions <function>

The ZüNIS library provides three level of abstractions, to allow both high-level and fine-grained control:

1. :doc:`Integrators <integrator>` are the highest level of abstraction and control function integration strategies.
They can automate trainer and flow creation.

2. :doc:`Trainers <trainer>` are one level below and steer model training through loss functions, optimizers, sampling etc.
They can automate flow creation.

3. :doc:`Normalizing Flows <flow>` are neural-network-based bijections from the unit hypercube to itself. They are the
actual trainable sampling mechanism that we use to sample points for Monte Carlo integration.

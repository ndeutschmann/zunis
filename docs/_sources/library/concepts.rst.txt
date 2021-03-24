Concepts
########

.. toctree::
    :hidden:

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

An Integrator contains a Trainer, which contains a Flow. Each level of abstraction can either be instantiated by
providing it it with an explicit object of the lower level, or using an API which builds the lower-level constructs
automatically.

.. image:: /_static/img/Abstractions.png
    :width: 80%
ZüNIS Tutorials
###############

.. toctree::
    :hidden:

    Configuration files <tutorial/config>
    Training without integrating <tutorial/nointeg>
    Training on a fixed sample <tutorial/preeval>
    Sampling points <tutorial/sample>
    Integrating in R^d <tutorial/Rd>
    Inverting normalizing flows <tutorial/invert>
    Custom coupling cells <tutorial/coupling>

These pages are intended to teach how to perform specific tasks using the ZüNIS library without focussing on explaining why and how things are implemented.
More detailed explanations can be found in the :doc:`Background <../background/nis>` and :doc:`Concepts <concepts>` sections.

.. rubric:: :doc:`How to use a configuration file <tutorial/config>`

Configuration files can be used to specify fine-grained configuration of integrators and their subparts. They also provide a good way to track parameters for reproducibility.

.. rubric:: :doc:`How to train without integrating <tutorial/nointeg>`

ZüNIS implements normalizing flows and facilities for training them to learn the distribution induced by a non-normalized function.
We show here how to train models outside of the importance sampling context, which can be useful for other applications.

.. rubric:: :doc:`How to train on a pre-evaluated sample <tutorial/preeval>`

In some applications, evaluating a function is extremely costly or needs to be performed in a specific environment. We show here how to decouple
sampling and function evaluation from training a model, which can be used for example to tune parameters in a data efficient way.

.. rubric:: :doc:`How to sample from a trained model <tutorial/sample>`

If a model has been trained in advance, it can be used to sample points from the learned distribution.
This can be useful for unweighting distributions as well as for evaluating integrals in cases where function evaluation is not easily interfaceable with ZüNIS.

.. rubric:: :doc:`How to invert a normalizing flow <tutorial/invert>`

For some applications, it is necessary to invert a trained model. We show here how this can be done in a directly within the framework of ZüNIS.

.. rubric:: :doc:`How to use custom coupling cells <tutorial/coupling>`

In the case that a user wants to extend the range of available transformations for the normalizing flows, it is possible to define custom coupling cells and to use them instead of the implemented transformations.
This tutorial shows the necessary steps to define a coupling cells and how to include it into a model.

.. rubric:: :doc:`How to integrate in R^d <tutorial/Rd>`

ZüNIS accepts only functions defined on the unit hypercube. Of course, many integration problems are setup in different spaces.
We show here how to perform a change of variables to reformulate infinite volume integrals as unit hypercube integrals.

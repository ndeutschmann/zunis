.. pytorch_flows documentation master file, created by
   sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ZüNIS documentation
==============================================


This is a work-in-progress library to provide importance sampling Monte-Carlo integration tools based on
Neural imporance sampling `[1]`_. This method uses normalzing flows to optimally sample
an integrand function in order to evaluate its (multi-dimensional) integral.

The goal is to provide a flexible library to integrate black-box functions for which classical methods such as VEGAS do
not work well due to an unknown or complicated structure which prevents the typical variable change and multi-channelling
tricks.

.. _[1]: https://arxiv.org/abs/1808.03856

Contents:
---------
.. toctree::
   :maxdepth: 2

   getting-started
   api/zunis



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
 

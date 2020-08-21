Installation
############


Using `pip`
===============

As the library is not yet fully mature, we have not released it to the Python Package Index (PyPI).
You can nevertheless install it with pip from this repository as follows:

.. code-block:: bash

    pip install 'git+https://github.com/ndeutschmann/zunis#egg=zunis&subdirectory=zunis_lib'

Setting up a development environment
====================================

If you would like to contribute to the library, run the benchmarks or try the examples,
the easiest is to clone this repository directly and install the extended requirements:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/ndeutschmann/zunis.git ./zunis
    # Create a virtual environment (recommended)
    python3.7 -m venv  zunis_venv
    source ./zunis_venv/bin/activate
    pip install --upgrade pip
    # Install the requirements
    cd ./zunis
    pip install -r requirements.txt
    # Run one benchmark (GPU highly recommended)
    cd ./experiments/benchmarks
    python benchmark_hypersphere.py


GPU Support
===========

ZüNIS depends on PyTorch and can therefore run on GPUs - which is strongly recommended for real-life applications.
Refer to the PyTorch `documentation`_ on how to setup your system to enable PyTorch's CUDA support.

.. _documentation: https://pytorch.org/get-started/
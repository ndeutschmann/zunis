Installation
############


Using `pip`
===============

The library is available on PyPI:

.. code-block:: bash

    pip install zunis

The latest version can be installed directly from GitHub:

.. code-block:: bash

    pip install 'git+https://github.com/zunis-anonymous/zunis#egg=zunis&subdirectory=zunis_lib'

Setting up a development environment
====================================

If you would like to contribute to the library, run the benchmarks or try the examples,
the easiest is to clone this repository directly and install the extended requirements:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/zunis-anonymous/zunis.git ./zunis
    # Create a virtual environment (recommended)
    python3.8 -m venv  zunis_venv
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
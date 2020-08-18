"""Piecewise coupling cells
Coupling cells whose transform f is defined as the primitive of a function g
such that
* f(0) = 0
* f(1) = 1
* g(y,t^N) > 0
* g is a piecewise-simple function

e.g.: g can be a positive, piecewise constant function, whose step heights are
the outputs of a neural network taking the y^N as input, properly normalized to
integrate to 1.

Throughout this sub-package, "PW" stands for piecewise.
"""
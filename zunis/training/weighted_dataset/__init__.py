"""This training module provides functions to train invertible flow models on datasets
drawn from a known distribution, which does not correspond to the target of our flow model,
which we know as a black box function

In other words, we have a set of (x,p(x),q(x)), such that
- x lives in the target space
- x ~ p(x)
- q is a black box function
- we want to train a flow model F such that F(z) ~ q(F(z)) when we sample z in the latent space

Note that typical cases where a flow is trained on an experimental dataset
- q = p
- p is unknown
In which case one can maximize the likelihood of the dataset under the flow model.
"""
"""Optimization of invertible flows in the weighted dataset problem using the DKL loss

Reminder: we have a dataset (x,p(x),f(x)) such that
- x ~ p(x)
- we want to learn a model that draws points according to f(x),
which is positive, and known up to normalization

We can estimate the expected un-normalized log-likelihood of a random datapoint x~f(x)
under the pdf of our flow model q(x) as

lL(f,q) = ∫ dx f(x) łog(q(x)) = E(f(x)/p(x) log(q(x)), x~p(x))

This is the same as minimizing the DKL between φ(x)=f(x)/∫ dx f(x) and q(x):

D_{KL}(φ|q) = ∫ dx φ(x) łog(φ(x)/q(x)) = - lL(f,q)*λ + η

where λ>0 and η are constants that are independent of the parameters of our flow q so maximizing the
estimated log-likelihood is the same as minimizing the DKL. In either case, we can only do it up to a constant
and use the loss lL(f,q) as a proxy

NB: An important point for importance sampling.
In importance sampling, a natural loss is the variance of the integrand. One can however note that
the optimum for both the variance loss and the DKL/ML is when the flow reproduces the target distribution.
"""
import logging
import torch
from .weighted_dataset_trainer import BasicTrainer, BasicStatefulTrainer

logger = logging.getLogger(__name__)


def weighted_dkl_loss(fx, px, logqx):
    return - torch.mean(fx * logqx / px)


class BasicDKLTrainer(BasicTrainer):
    def __init__(self, flow, latent_prior):
        super(BasicDKLTrainer, self).__init__(flow, latent_prior)
        self.loss = weighted_dkl_loss


class BasicStatefulDKLTrainer(BasicStatefulTrainer):
    def __init__(self, flow, latent_prior, **kwargs):
        super(BasicStatefulDKLTrainer, self).__init__(flow, latent_prior, **kwargs)
        self.loss = weighted_dkl_loss

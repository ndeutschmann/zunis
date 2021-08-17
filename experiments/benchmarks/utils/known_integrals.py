import torch
import numpy as np
from random import shuffle
from utils.integral_validation import Sampler, evaluate_integral
from utils.integrands.abstract import KnownIntegrand


class FakeKnownSampler(Sampler):
    """Fake sampler for using :class:`utils.integrands.abstract.KnownIntegrand` with
    :class:`utils.integral_validation.evaluate_integral`"""

    def sample(self, f, n_batch=2, *args, **kwargs):
        """Sampling method for known integrands. We model it by a constant function over the unit hypercube
        whose value is the integral value.

        Parameters
        ----------
        f: KnownIntegrand
        n_batch: int

        Returns
        -------
            tuple of torch.Tensor
            x,px,fx: points, pdfs, function values
        """

        fx = f.integral()
        x = torch.zeros(n_batch).uniform_().to(fx.device)
        px = torch.ones(n_batch).to(fx.device)

        return x, px, fx


def evaluate_known_integral(f, n_batch=2):
    """Validate a known integral using a VEGAS integrator as a sampler

    Parameters
    ----------
    f: utils.integrands.KnownIntegrand
    n_batch: int

    Returns
    -------
        utils.record.EvaluationRecord
    """
    sampler = FakeKnownSampler()

    return evaluate_integral(f, sampler, n_batch=n_batch)

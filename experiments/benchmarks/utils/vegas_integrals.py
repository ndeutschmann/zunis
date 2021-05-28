import torch
import numpy as np
from random import shuffle
from utils.integral_validation import Sampler, evaluate_integral


class VegasSampler(Sampler):
    def __init__(self, integrator, integrand, train=True, n_survey_steps=10, n_batch=100000):
        """

        Parameters
        ----------
        integrator
        train
        n_survey_steps
        n_batch
        """

        self.integrator = integrator
        self.point_iterator = None
        self.actual_n_batch = 0
        self.integrand = integrand

        if train:
            self.train_integrator(n_survey_steps, n_batch)

    def reset_point_iterator(self):
        self.actual_n_batch = len(list(self.integrator.random()))
        self.point_iterator = self.integrator.random()

    def get_point(self):
        """Sample a single point from the vegas integrator with point pdfs normalized to 1.

        Yields
        -------
            x, px: tuple of float
                point and its pdf
        """
        if self.point_iterator is None:
            self.reset_point_iterator()
        try:
            x, wx = next(self.point_iterator)
        except StopIteration:
            self.reset_point_iterator()
            x, wx = next(self.point_iterator)
        # x is originally a view: map it to an array
        # furthermore the C backend of vegas.Integrator.random
        # reuses the same location in memory to store points: we need to copy
        x = np.asarray(x).copy()
        wx = float(np.asarray(wx)) * self.actual_n_batch
        return x, 1 / wx

    def train_integrator(self, n_survey_steps, n_batch):
        """Train the integrator before sampling

        Parameters
        ----------
        n_survey_steps: int
            if train is `True`, how many survey steps to use for training
        n_batch: int
            maximum number of function evaluations per survey step
        """
        self.integrator(self.integrand, nitn=n_survey_steps, neval=n_batch)
        # integrating changes how points are sampled, the iterator should be reset
        self.reset_point_iterator()

    def sample(self, f, n_batch=10000, *args, **kwargs):
        """

        Parameters
        ----------
        f: batch callable
            function mapping numpy arrays to numpy arrays
        n_batch
        args
        kwargs

        Returns
        -------

        """
        xs = []
        pxs = []
        while len(xs) <= n_batch:
            xi, pxi = self.get_point()
            xs.append(xi)
            pxs.append(pxi)
        x = np.array(xs)
        px = torch.tensor(pxs)
        fx = f(x)
        x = torch.tensor(x)

        return x, px, fx


def evaluate_integral_vegas(f, integrator, n_batch=10000, train=True, n_survey_steps=10, n_batch_survey=10000):
    """Validate a known integral using a VEGAS integrator as a sampler

    Parameters
    ----------
    f: utils.integrands.KnownIntegrand
    integrator: zunis.integration.adaptive_survey_integrator.AdaptiveSurveyIntegrator
    n_batch: int
    train: bool
        whether to train the integrator using `integrator.survey`
    n_survey_steps: int or None
        positional `integrator.survey` argument
    n_batch_survey: int or None


    Returns
    -------
        utils.record.EvaluationRecord
    """

    if n_survey_steps is None:
        n_survey_steps = 10
    if n_batch_survey is None:
        n_batch_survey = 10000

    sampler = VegasSampler(integrator, f, train=train, n_survey_steps=n_survey_steps, n_batch=n_batch_survey)
    sampler.reset_point_iterator()

    return evaluate_integral(f, sampler, n_batch)

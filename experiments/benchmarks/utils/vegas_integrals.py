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
        self.integrand = integrand
        self.n_batch = n_batch
        self.n_batches = None
        self.n_eval = None

        self.reset_point_iterator(n_batch=n_batch)

        if train:
            self.train_integrator(n_survey_steps, n_batch)

    def reset_point_iterator(self, n_batch):
        self.integrator.set(neval=n_batch)
        self.n_batch = n_batch
        self.n_batches = len(list(self.integrator.random_batch()))
        self.n_eval = len(next(self.integrator.random_batch())[0])

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
        approx
        args
        kwargs

        Returns
        -------

        """
        if n_batch != self.n_batch:
            self.reset_point_iterator(n_batch)

        x, wx = self.integrator.sample_batch()
        # x is originally a view: map it to an array
        # furthermore the C backend of vegas.Integrator.random
        # reuses the same location in memory to store points: we need to copy
        x = np.asarray(x).copy()
        # Point weights are normalized so that the sum of weights is the volume
        # We use PDFs: the mean of PDFs is the volume.
        # We need to divide by the number of points sampled by the integrator.sample_batch() function
        wx = float(np.asarray(wx)) * self.n_eval * self.n_batches

        px = 1 / torch.tensor(wx)
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

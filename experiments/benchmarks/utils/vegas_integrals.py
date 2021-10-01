import torch
import numpy as np
from utils.integral_validation import Sampler, evaluate_integral, evaluate_integral_stratified


class VegasSampler(Sampler):
    def __init__(self, integrator, integrand, train=True, n_survey_steps=10, n_batch=100000, stratified=False):
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
        self.stratified = stratified
        # 1e9 is the default value for stratified sampling in VEGAS
        max_nhcube = int(1e9) if stratified else 1

        self.integrator.set(neval=n_batch, max_nhcube=max_nhcube)

        if train:
            self.train_integrator(n_survey_steps, n_batch)

    def train_integrator(self, n_survey_steps, n_batch):
        """Train the integrator before sampling

        Parameters
        ----------
        n_survey_steps: int
            if train is `True`, how many survey steps to use for training
        n_batch: int
            maximum number of function evaluations per survey step
        """
        max_nhcube = int(1e9) if self.stratified else 1
        self.integrator(self.integrand, nitn=n_survey_steps, neval=n_batch, max_nhcube=max_nhcube)
        # integrating changes how points are sampled, the iterator should be reset
        self.integrator.set(neval=n_batch)

    def sample(self, f, n_batch=10000, *args, **kwargs):
        if self.stratified:
            return self.sample_stratified(f, n_batch=n_batch)
        else:
            return self.sample_non_stratified(f, n_batch=n_batch)

    def sample_stratified(self, f, n_batch):
        if n_batch != self.n_batch:
            self.n_batch = n_batch
            self.integrator.set(neval=n_batch)

        gen = self.integrator.random_batch(yield_hcube=True)
        x, wx, hc = next(gen)
        x = np.asarray(x).copy()
        wx = np.asarray(wx).copy()
        hc = np.asarray(hc).copy()

        for x_batch, wx_batch, hc_batch in gen:
            x_batch = np.asarray(x_batch).copy()
            wx_batch = np.asarray(wx_batch).copy()
            hc_batch = np.asarray(hc_batch).copy()
            x = np.concatenate((x, x_batch), axis=0)
            wx = np.concatenate((wx, wx_batch), axis=0)
            hc = np.concatenate((hc, hc_batch), axis=0)

        fx = f(x)

        return x, (wx, hc), fx

    def sample_non_stratified(self, f, n_batch=10000, *args, **kwargs):
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

        if n_batch != self.n_batch:
            self.n_batch = n_batch
            self.integrator.set(neval=n_batch)

        x, wx = next(self.integrator.random_batch())
        # x is originally a view: map it to an array
        # furthermore the C backend of vegas.Integrator.random
        # reuses the same location in memory to store points: we need to copy
        x = np.asarray(x).copy()
        # Point weights are normalized so that the sum of weights is the volume
        # We use PDFs: the mean of PDFs is the volume.
        # We need to divide by the number of points sampled by the integrator.sample_batch() function
        n_eval = x.shape[0]
        wx = np.asarray(wx).copy() * n_eval

        px = 1 / torch.tensor(wx, dtype=torch.float32)
        fx = f(x)
        x = torch.tensor(x, dtype=torch.float32)

        return x, px, fx


def evaluate_integral_vegas(f, integrator, n_batch=10000, train=True, n_survey_steps=10, n_batch_survey=10000,
                            stratified=False):
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
    stratified: bool
        whether to use VEGAS stratified sampling

    Returns
    -------
        utils.record.EvaluationRecord
    """

    if n_survey_steps is None:
        n_survey_steps = 10
    if n_batch_survey is None:
        n_batch_survey = 10000

    sampler = VegasSampler(integrator, f, train=train, n_survey_steps=n_survey_steps, n_batch=n_batch_survey,
                           stratified=stratified)

    if stratified:
        return evaluate_integral_stratified(f, sampler, n_batch)
    else:
        return evaluate_integral(f, sampler, n_batch)

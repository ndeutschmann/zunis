from utils.integral_validation import Sampler
import torch


class VegasSampler(Sampler):
    def __init__(self, integrator, integrand, train=True, n_survey_steps=10, n_batch=100000):
        """

        Parameters
        ----------
        integrator
        train
        n_survey_steps
        survey_args
        """

        self.integrator = integrator
        self.integrand = integrand

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

        self.integrator(self.integrand, nitn=n_survey_steps, neval=n_batch)


    def sample(self, f, n_batch=10000, *args, **kwargs):
        """

        Parameters
        ----------
        f
        n_batch
        args
        kwargs

        Returns
        -------

        """
        self.integrator.nhcube_batch = n_batch
        x, px = self.integrator.random_batch()
        x = torch.tensor(x)
        px = torch.tensor(px)
        fx = f(x)
        return x, px, fx

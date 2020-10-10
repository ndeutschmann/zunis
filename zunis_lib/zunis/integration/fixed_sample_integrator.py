"""Integrator that does not sample points during the training phase but uses a fixed dataset of points"""
import torch

from .base_integrator import BaseIntegrator


class FixedSampleSurveyIntegrator(BaseIntegrator):
    """Integrator that trains its model during the survey phase using a pre-computed sample provided externally"""

    def __init__(self, f, trainer, sample=None, n_iter=None, n_iter_survey=1, n_iter_refine=10,
                 n_points=None, n_points_survey=None, n_points_refine=10000, use_survey=False,
                 verbosity=None, trainer_verbosity=None, **kwargs):
        """

        Parameters
        ----------
        f: callable
            ZuNIS-compatible function
        trainer: BasicTrainer
            trainer object used to perform the survey
        sample: tuple of torch.Tensor
            (x, fx, px): target-space point batch drawn from some PDF p, function value batch, PDF value batch p(x)
        n_iter: int
            number of iterations (used for both survey and  refine unless specified)
        n_iter_survey: int
            number of iterations for survey
        n_iter_refine: int
            number of iterations for refine
        n_points: int
            number of points for both survey and refine unless specified
        n_points_survey: int
            number of points for survey
        n_points_refine: int
            number of points for refine
        use_survey: bool
            whether to use the integral estimations from the survey phase. This makes error estimation formally
            incorrect since samples from the refine depend on the survey training, but these correlation can be negligible
            in some cases.
        verbosity: int
            level of verbosity for the integrator-level logger
        trainer_verbosity: int
            level of verbosity for the trainer-level logger
        kwargs
        """
        super(FixedSampleSurveyIntegrator, self).__init__(f=f,
                                                          trainer=trainer,
                                                          trainer_verbosity=trainer_verbosity,
                                                          n_iter=n_iter,
                                                          n_iter_survey=n_iter_survey,
                                                          n_iter_refine=n_iter_refine,
                                                          n_points=n_points,
                                                          n_points_survey=n_points_survey,
                                                          n_points_refine=n_points_refine,
                                                          use_survey=use_survey,
                                                          verbosity=verbosity,
                                                          **kwargs)

        self.sample = sample

    def sample_survey(self, n_points=None, **kwargs):
        """Sample points from the internally stored sample

        Parameters
        ----------
        n_points: int, None
            size of the batch to select from the sample
        kwargs

        Returns
        -------
            tuple of torch.Tensor
                (x,px,fx): sampled points, sampling distribution PDF values, function values

        """
        assert self.sample is not None, "The training sample must be instantiated before starting the survey"

        if n_points is None:
            n_points = self.n_points_survey

        if n_points is None:
            return self.sample

        x, px, fx = self.sample

        sample_size = x.shape[0]
        assert n_points <= sample_size

        # Sample n_points indices randomly among the sample_size options with equal probability
        indices = torch.multinomial(torch.ones(sample_size), n_points)
        return x[indices], px[indices], fx[indices]
import torch

# TODO: change where the method we use from this class is
from zunis.integration.base_integrator import BaseIntegrator
from zunis.training.weighted_dataset.weighted_dataset_trainer import BasicStatefulTrainer
from zunis.models.flows.sampling import UniformSampler


class PosteriorSurveySamplingIntegrator(BaseIntegrator):
    """Integrator using a target space posterior to sample during survey:
    Takes a trainer object at initialization and samples from a given distribution in target space
    during the survey phase. The function to integrate is given at initialization.
    """

    def __init__(self, f, trainer, posterior, n_iter=10, n_iter_survey=None, n_iter_refine=None,
                 n_points=100000, n_points_survey=None, n_points_refine=None, use_survey=False,
                 verbosity=None, trainer_verbosity=None, **kwargs):
        super(PosteriorSurveySamplingIntegrator, self).__init__(f=f,
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

        self.posterior = posterior

    def sample_survey(self, *, n_points=None, f=None, **kwargs):
        """Sample points from target space distribution"""
        # TODO, change where this method is /!\
        if n_points is None:
            n_points = self.n_points_survey
        if f is None:
            f = self.f
        return BasicStatefulTrainer.generate_target_batch_from_posterior(n_points, f, self.posterior)


class FlatSurveySamplingIntegrator(PosteriorSurveySamplingIntegrator):
    def __init__(self, f, trainer, d, n_iter=10, n_iter_survey=None, n_iter_refine=None,
                 n_points=100000, n_points_survey=None, n_points_refine=None, use_survey=False,
                 device=torch.device("cpu"), verbosity=2, trainer_verbosity=1, **kwargs):
        posterior = UniformSampler(d=d, device=device)
        super(FlatSurveySamplingIntegrator, self).__init__(f=f,
                                                           trainer=trainer,
                                                           d=d,
                                                           posterior=posterior,
                                                           n_iter=n_iter,
                                                           n_iter_survey=n_iter_survey,
                                                           n_iter_refine=n_iter_refine,
                                                           n_points=n_points,
                                                           n_points_survey=n_points_survey,
                                                           n_points_refine=n_points_refine,
                                                           use_survey=use_survey,
                                                           verbosity=verbosity,
                                                           trainer_verbosity=trainer_verbosity,
                                                           **kwargs)
        self.d = d

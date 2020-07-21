import logging
import torch
from src.integration.flat_survey_integrator import FlatSurveySamplingIntegrator
from src.training.weighted_dataset.generators import create_dkl_trainer

integration_logger = logging.getLogger(__name__)


class DefaultIntegrator(FlatSurveySamplingIntegrator):
    """Default integrator for out-of-the-box usage.
    Basic usage: provide a function f and a dimensionality d.
    The function must accept batches of points (torch.Tensor) of shape (*,d)
    in the unit hypercube and output tensors of shape (*,).

    Methods
    -------
    integrate(n_survey_steps=10, n_refine_steps=10, **kwargs)
        Compute the integral of the function using neural importance sampling

    """

    reset_trainer = create_dkl_trainer

    def __init__(self, f, d, n_iter=10, n_iter_survey=None, n_iter_refine=None,
                 n_points=100000, n_points_survey=None, n_points_refine=None, use_survey=False,
                 device=torch.device("cpu"), verbosity=2, trainer_verbosity=1, n_epochs=10, minibatch_size=None,
                 lr=1.e-4, model_params=None, **kwargs):

        if minibatch_size is None:
            if n_points_survey is None:
                minibatch_size = n_points // 10
            else:
                minibatch_size = n_points_survey // 10

        super(DefaultIntegrator, self).__init__(f=f,
                                                d=d,
                                                trainer=self.reset_trainer(d, device=device,
                                                                           n_epochs=n_epochs,
                                                                           minibatch_size=minibatch_size,
                                                                           lr=lr,
                                                                           model_params=model_params),
                                                n_iter=n_iter,
                                                n_iter_survey=n_iter_survey,
                                                n_iter_refine=n_iter_refine,
                                                n_points=n_points,
                                                n_points_survey=n_points_survey,
                                                n_points_refine=n_points_refine,
                                                use_survey=use_survey,
                                                verbosity=verbosity,
                                                trainer_verbosity=trainer_verbosity,
                                                device=device,
                                                **kwargs)

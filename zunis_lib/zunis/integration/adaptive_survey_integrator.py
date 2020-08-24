from better_abc import abstractmethod
import torch
from .flat_survey_integrator import FlatSurveySamplingIntegrator


class AdaptiveSurveyIntegrator(FlatSurveySamplingIntegrator):
    """Adaptive integrator based on a separation between survey and sampling

    Survey:
        Sample points and spend some time training the model

        Sampling is done in two phases:

        1. Sample in the target space (input space of the integrand) using a uniform distribution
        2. Sample in the latent space and use the model to sample point in the target space


        The switch between the two phases is performed based on a test method - abstract here - that checks
        whether the flat distribution does a better job of estimating the loss than the flat distribution

    Refine:
        Sample points using the trained model and evaluate the integral
    """

    def __init__(self, f, trainer, d, n_iter=10, n_iter_survey=None, n_iter_refine=None,
                 n_points=100000, n_points_survey=None, n_points_refine=None, use_survey=False,
                 device=torch.device("cpu"), verbosity=2, trainer_verbosity=1, **kwargs):
        super(AdaptiveSurveyIntegrator, self).__init__(f,
                                                       trainer,
                                                       d,
                                                       n_iter=n_iter,
                                                       n_iter_survey=n_iter_survey,
                                                       n_iter_refine=n_iter_refine,
                                                       n_points=n_points,
                                                       n_points_survey=n_points_survey,
                                                       n_points_refine=n_points_refine,
                                                       use_survey=use_survey,
                                                       device=device,
                                                       verbosity=verbosity,
                                                       trainer_verbosity=trainer_verbosity,
                                                       **kwargs)

        # As long as the survey switch condition (see abstractmethod below) is False,
        # This is false
        self.sample_forward = False

    @abstractmethod
    def survey_switch_condition(self):
        """Boolean valued method that checks if it is time to switch between sampling uniformly
        and using the model"""

    def sample_survey(self, *, n_points=None, f=None, **kwargs):

        # Starting mode: sample from the flat distribution in target space
        if not self.sample_forward:
            return super(AdaptiveSurveyIntegrator, self).sample_survey(n_points=n_points, f=f, **kwargs)

        # When the model is trained enough, sample from it
        if n_points is None:
            n_points = self.n_points_survey
        if f is None:
            f = self.f

        xj = self.model_trainer.sample_forward(n_points)
        x = xj[:, :-1]
        px = torch.exp(- xj[:, -1])
        fx = f(x)

        return x, px, fx

    def process_survey_step(self, sample, integral, integral_var, training_record, **kwargs):
        super(AdaptiveSurveyIntegrator, self).process_survey_step(sample, integral, integral_var, training_record,
                                                                  **kwargs)
        if (not self.sample_forward) and self.survey_switch_condition():
            self.logger.info("Switching sampling mode")
            self.sample_forward = True

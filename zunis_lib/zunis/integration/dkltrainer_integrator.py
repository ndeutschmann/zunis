"""Survey/Refine integrator based on training models with a DKL trainer"""
import torch
from .adaptive_survey_integrator import AdaptiveSurveyIntegrator
from zunis.training.weighted_dataset.dkl_training import BasicStatefulDKLTrainer


class DKLAdaptiveSurveyIntegrator(AdaptiveSurveyIntegrator):
    """Survey/Refine adaptive integrator based on the DKL loss. The loss is the D_KL distance between
    the PDF from a flow model and an un-normalized function, up to non-trainable terms

    Explicitly:

    L(f,q) = - \int dx f(x) log(q(x))

    This integrator is adaptive in the sense that survey batches are sampled from the flat distribution in
    the target space (the domain of f and q) until the learned q distribution is a better approximation of
    the normalized target function f than the flat distribution. Since our target space is the unit hypercube,
    this is easy:

    L(f, uniform) = 0.

    So as soon as the loss is negative, we sample from the flow instead of the uniform distribution.
    """

    def __init__(self, f, trainer, d, n_iter=10, n_iter_survey=None, n_iter_refine=None,
                 n_points=100000, n_points_survey=None, n_points_refine=None, use_survey=False,
                 device=torch.device("cpu"), verbosity=2, trainer_verbosity=1, **kwargs):

        super(DKLAdaptiveSurveyIntegrator, self).__init__(f,
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

    def survey_switch_condition(self):
        """Check if the loss is negative. This test is used to switch from uniform sampling
        to sampling from the flow in the survey phase.

        The loss is the distance between the target function and the flow PDF. Since the distance between
        the target function and the uniform function, a negative loss indicates that flow is doing better.
        """
        return self.model_trainer.record["loss"] < 0

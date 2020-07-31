"""High level interface for integration"""
import torch

from .flat_survey_integrator import FlatSurveySamplingIntegrator
from .dkltrainer_integrator import DKLAdaptiveSurveyIntegrator

from src.training.weighted_dataset.stateful_trainer import StatefulTrainer


def Integrator(f, d, survey_strategy="flat", n_iter=10, n_iter_survey=None, n_iter_refine=None,
               n_points=100000, n_points_survey=None, n_points_refine=None, use_survey=False,
               device=torch.device("cpu"), verbosity=2, trainer_verbosity=1,
               loss="variance", flow="pwlinear", trainer=None, trainer_options=None, flow_options=None):
    """High level integration API

    This is a factory method that instantiates the relevant Integrator subclass based on the options

    Parameters
    ----------
    f
    d
    survey_strategy
    n_iter
    n_iter_survey
    n_iter_refine
    n_points
    n_points_survey
    n_points_refine
    use_survey
    device
    verbosity
    trainer_verbosity
    loss
    flow
    trainer
    trainer_options
    flow_options

    Returns
    -------

    """
    if trainer is None:
        if trainer_options is None:
            trainer_options = dict()
        trainer = StatefulTrainer(d=d, loss=loss, flow=flow, device=device, flow_options=flow_options,
                                  **trainer_options)

    if survey_strategy == "flat":
        return FlatSurveySamplingIntegrator(f=f, trainer=trainer, d=d, n_iter=n_iter, n_iter_survey=n_iter_survey,
                                            n_iter_refine=n_iter_refine,
                                            n_points=n_points, n_points_survey=n_points_survey,
                                            n_points_refine=n_points_refine, use_survey=use_survey,
                                            device=torch.device("cpu"), verbosity=verbosity,
                                            trainer_verbosity=trainer_verbosity)

    if survey_strategy == "adaptive_dkl":
        if trainer is None:
            assert loss == "dkl", "The adaptive DKL strategy must be used with the DKL loss"
        return DKLAdaptiveSurveyIntegrator(f=f, trainer=trainer, d=d, n_iter=n_iter, n_iter_survey=n_iter_survey,
                                           n_iter_refine=n_iter_refine,
                                           n_points=100000, n_points_survey=n_points_survey,
                                           n_points_refine=n_points_refine, use_survey=use_survey,
                                           device=torch.device("cpu"), verbosity=verbosity,
                                           trainer_verbosity=trainer_verbosity)

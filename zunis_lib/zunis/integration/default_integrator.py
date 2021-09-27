"""High level interface for integration"""
import torch
import logging

from .flat_survey_integrator import FlatSurveySamplingIntegrator
from .dkltrainer_integrator import DKLAdaptiveSurveyIntegrator
from .adaptive_survey_integrator import ForwardSurveySamplingIntegrator, VarianceAdaptiveSurveyIntegrator
from .fixed_sample_integrator import FixedSampleSurveyIntegrator

from zunis.training.weighted_dataset.stateful_trainer import StatefulTrainer

logger = logging.getLogger(__name__)


def Integrator(f, d, survey_strategy="forward_flat_init", n_iter=10, n_iter_survey=None, n_iter_refine=None,
               n_points=100000, n_points_survey=None, n_points_refine=None, use_survey=False,
               device=torch.device("cpu"), verbosity=None, trainer_verbosity=None,
               loss="dkl", flow="pwquad", trainer=None, trainer_options=None, flow_options=None):
    """High level integration API

    This is a factory method that instantiates the relevant Integrator subclass based on the options

    Parameters
    ----------
    f: function
        the function to integrate
    d: int
        dimensionality of the integration space
    survey_strategy: str
        how points are sampled during the survey step: one of `'flat'`, `'forward'`, `'adaptive_dkl'`, `'adaptive_variance'`,
        `'forward_flat_init'`, `'fixed_sample'`
    n_iter: int
        general number of iterations - ignored for survey/refine if n_iter_survey/n_inter_refine is set
    n_iter_survey: int
        number of iterations for the survey stage
    n_iter_refine: int
        number of iterations for the refine stage
    n_points:
        general number of points per iteration - ignored for survey/refine if n_points_survey/n_points_refine is set
    n_points_survey: int
        number of points per iteration for the survey stage
    n_points_refine: int
        number of points per iteration for the refine stage
    use_survey: bool
        whether to use the points generated during the survey to compute the final integral
        not recommended due to uncontrolled correlations in error estimates
    device:
        pytorch device on which to run
    verbosity: int
        verbosity level of the integrator
    trainer_verbosity:
        verbosity level of the trainer
    loss: str or function
        loss function used by the trainer
    flow: str or flow model
        normalizing flow model to use for importance sampling
    trainer: None or :py:class:`zunis.training.weighted_dataset.weighted_dataset_trainer.BasicStatefulTrainer`
        optional argument to provide a full trainer object - overrides any other trainer or model setup argument
    trainer_options: dict
        dictionary of options to pass to the :py:class:`zunis.training.weighted_dataset.stateful_trainer.StatefulTrainer`
        created if `trainer` is None
    flow_options: dict
        dictionary of options to pass to the :py:class:`zunis.models.flows.sequential.repeated_cell.RepeatedCellFlow`
        created by the stateful trainer if `trainer` is None

    Returns
    -------
       subclass of :py:class:`zunis.models.flows.sequential.repeated_cell.RepeatedCellFlow`
    """
    if trainer is None:
        if trainer_options is None:
            trainer_options = dict()
        trainer = StatefulTrainer(d=d, loss=loss, flow=flow, device=device, flow_options=flow_options,
                                  **trainer_options)

    # TODO work out the issue for the variance training with RealNVP
    if loss == "variance" and flow == "realnvp":
        logger.warning(f"======================================================")
        logger.warning(f"Using loss {loss} and flow {flow}. This usually fails.")
        logger.warning(f"======================================================")

    if survey_strategy == "flat":
        return FlatSurveySamplingIntegrator(f=f, trainer=trainer, d=d, n_iter=n_iter, n_iter_survey=n_iter_survey,
                                            n_iter_refine=n_iter_refine,
                                            n_points=n_points, n_points_survey=n_points_survey,
                                            n_points_refine=n_points_refine, use_survey=use_survey,
                                            device=device, verbosity=verbosity,
                                            trainer_verbosity=trainer_verbosity)

    if survey_strategy == "adaptive_dkl":
        if trainer is None:
            assert loss == "dkl", "The adaptive DKL strategy must be used with the DKL loss"
        return DKLAdaptiveSurveyIntegrator(f=f, trainer=trainer, d=d, n_iter=n_iter, n_iter_survey=n_iter_survey,
                                           n_iter_refine=n_iter_refine,
                                           n_points=n_points, n_points_survey=n_points_survey,
                                           n_points_refine=n_points_refine, use_survey=use_survey,
                                           device=device, verbosity=verbosity,
                                           trainer_verbosity=trainer_verbosity)

    if survey_strategy == "adaptive_variance":
        if trainer is None:
            assert loss == "variance", "The adaptive variance strategy must be used with the variance loss"
        return VarianceAdaptiveSurveyIntegrator(f=f, trainer=trainer, d=d, n_iter=n_iter, n_iter_survey=n_iter_survey,
                                                n_iter_refine=n_iter_refine,
                                                n_points=n_points, n_points_survey=n_points_survey,
                                                n_points_refine=n_points_refine, use_survey=use_survey,
                                                device=device, verbosity=verbosity,
                                                trainer_verbosity=trainer_verbosity)

    if survey_strategy == 'forward':
        return ForwardSurveySamplingIntegrator(f=f, trainer=trainer, d=d, n_iter=n_iter, n_iter_survey=n_iter_survey,
                                               n_iter_refine=n_iter_refine,
                                               n_points=n_points, n_points_survey=n_points_survey,
                                               n_points_refine=n_points_refine, use_survey=use_survey,
                                               device=device, verbosity=verbosity,
                                               trainer_verbosity=trainer_verbosity,
                                               sample_flat_once=False)

    if survey_strategy == 'forward_flat_init':
        return ForwardSurveySamplingIntegrator(f=f, trainer=trainer, d=d, n_iter=n_iter, n_iter_survey=n_iter_survey,
                                               n_iter_refine=n_iter_refine,
                                               n_points=n_points, n_points_survey=n_points_survey,
                                               n_points_refine=n_points_refine, use_survey=use_survey,
                                               device=device, verbosity=verbosity,
                                               trainer_verbosity=trainer_verbosity,
                                               sample_flat_once=True)

    if survey_strategy == 'fixed_sample':
        return FixedSampleSurveyIntegrator(f=f, trainer=trainer, d=d, n_iter=n_iter, n_iter_survey=n_iter_survey,
                                           n_iter_refine=n_iter_refine,
                                           n_points=n_points, n_points_survey=n_points_survey,
                                           n_points_refine=n_points_refine, use_survey=use_survey,
                                           device=device, verbosity=verbosity,
                                           trainer_verbosity=trainer_verbosity)

    raise ValueError("""No valid survey strategy was provided. Allowed strategies are:
     ['flat', 'adaptive_dkl', 'adaptive_variance', 'forward', 'forward_flat_init', 'fixed_sample']""")

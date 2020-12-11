"""Facilities to compute, estimate and benchmark integrals performed with Integrator objects"""

from utils.integral_validation import Sampler, validate_integral, evaluate_integral


class IntegratorSampler(Sampler):
    """Sampler using an Integrator's sample_refine method"""

    def __init__(self, integrator, train=True, n_survey_steps=None, survey_args=None):
        """

        Parameters
        ----------
        integrator: zunis.integration.flat_survey_integrator.PosteriorSurveySamplingIntegrator
        train: bool
            whether to train the integrator by running `integrator.survey`
        n_survey_steps: int, None
            if train is `True`, how many survey steps to use for training. If None, use the configuration
            set at the creation of the Integrator
        survey_args: None or dict
            if train is `True`, additional survey options to pass to `integrator.survey`
        """

        self.integrator = integrator

        if train:
            self.train_integrator(n_survey_steps, survey_args)

    def train_integrator(self, n_survey_steps=None, survey_args=None):
        """Train the integrator before sampling

        Parameters
        ----------
        n_survey_steps: int
            if train is `True`, how many survey steps to use for training
        survey_args: None or dict
            if train is `True`, additional survey options to pass to `integrator.survey`"""

        if survey_args is None:
            survey_args = dict()

        self.integrator.survey(n_survey_steps, **survey_args)

    def sample(self, f, n_batch=10000, *args, **kwargs):
        """

        Parameters
        ----------
        f: utils.integrands.Integrand
        n_batch: int

        Returns
        -------
            tuple of torch.Tensor
            x,px,fx: points, pdfs, function values
        """

        x, px, fx = self.integrator.sample_refine(n_points=n_batch, f=f)
        return x, px, fx

    def get_history(self):
        """Get the integrator's training history

        Returns
        -------
            pandas.Dataframe
        """
        return self.integrator.integration_history


def validate_integral_integrator(f, integrator, n_batch=10000, sigma_cutoff=2, train=True, n_survey_steps=None,
                                 survey_args=None, keep_history=False):
    """Validate a known integral using an integrator as a sampler

    Parameters
    ----------
    f: utils.integrands.KnownIntegrand
    integrator: zunis.integration.adaptive_survey_integrator.AdaptiveSurveyIntegrator
    n_batch: int
    sigma_cutoff: int
    train: bool
        whether to train the integrator using `integrator.survey`
    n_survey_steps: int
        positional `integrator.survey` argument
    survey_args: None or dict
        optional keyword-argument dictionary for `integrator.survey`
    keep_history: bool
        whether to keep track of the training history

    Returns
    -------
        utils.record.ComparisonRecord
    """
    sampler = IntegratorSampler(integrator, train=train, n_survey_steps=n_survey_steps, survey_args=survey_args)
    return validate_integral(f, sampler, n_batch, sigma_cutoff, keep_history=keep_history)


def evaluate_integral_integrator(f, integrator, n_batch=10000, train=True, n_survey_steps=None, survey_args=None,
                                 keep_history=False):
    """Validate a known integral using an integrator as a sampler

    Parameters
    ----------
    f: utils.integrands.KnownIntegrand
    integrator: zunis.integration.flat_survey_integrator.PosteriorSurveySamplingIntegrator
    n_batch: int
    train: bool
        whether to train the integrator using `integrator.survey`
    n_survey_steps: int, optional
        positional `integrator.survey` argument. Use the attribute n_iter_survey set at instantiation if not
        specified.
    survey_args: None or dict
        optional keyword-argument dictionary for `integrator.survey`
    keep_history: bool
        whether to keep track of the training history

    Returns
    -------
        utils.record.EvaluationRecord
    """
    sampler = IntegratorSampler(integrator, train=train, n_survey_steps=n_survey_steps, survey_args=survey_args)

    return evaluate_integral(f, sampler, n_batch, keep_history=keep_history)

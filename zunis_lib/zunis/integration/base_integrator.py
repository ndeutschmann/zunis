"""Implementation of basic integrator functions"""
import numpy as np
import pandas as pd
import torch
from zunis.integration.integratorAPI import SurveyRefineIntegratorAPI
from zunis.training.weighted_dataset.weighted_dataset_trainer import BasicTrainer


class BaseIntegrator(SurveyRefineIntegratorAPI):
    """Base abstract class that implements common functionality"""

    @staticmethod
    def empty_history():
        """Create an empty history object"""
        return pd.DataFrame({
            "integral": pd.Series([], dtype="float"),
            "error": pd.Series([], dtype="float"),
            "n_points": pd.Series([], dtype="int"),
            "phase": pd.Series([], dtype="str")
        })

    def __init__(self, f, trainer, n_iter=10, n_iter_survey=None, n_iter_refine=None,
                 n_points=100000, n_points_survey=None, n_points_refine=None, use_survey=False,
                 verbosity=None, trainer_verbosity=None, **kwargs):
        """

        Parameters
        ----------
        f: function
            the function to integrate
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
        verbosity: int
            verbosity level of the integrator
        """
        super(BaseIntegrator, self).__init__(verbosity=verbosity, **kwargs)
        self.f = f

        self.n_iter_survey = n_iter_survey if n_iter_survey is not None else n_iter
        self.n_iter_refine = n_iter_refine if n_iter_refine is not None else n_iter
        self.n_points_survey = n_points_survey if n_points_survey is not None else n_points
        self.n_points_refine = n_points_refine if n_points_refine is not None else n_points

        self.use_survey = use_survey


        assert isinstance(trainer, BasicTrainer), "This integrator relies on the BasicTrainer API"
        self.model_trainer = trainer
        self.model_trainer.set_verbosity(trainer_verbosity)

        self.integration_history = self.empty_history()

    def initialize(self, **kwargs):
        self.integration_history = self.empty_history()

    def initialize_survey(self, **kwargs):
        pass

    def initialize_refine(self, **kwargs):
        pass

    def sample_refine(self, *, n_points=None, f=None, **kwargs):
        if n_points is None:
            n_points = self.n_points_refine
        if f is None:
            f = self.f

        xj = self.model_trainer.sample_forward(n_points)
        x = xj[:, :-1]
        px = torch.exp(-xj[:, -1])
        fx = f(x)

        return x, px, fx

    def process_survey_step(self, sample, integral, integral_var, training_record, **kwargs):
        x, px, fx = sample
        n_points = x.shape[0]
        self.integration_history = self.integration_history.append(
            {"integral": integral,
             "error": (integral_var / n_points) ** 0.5,
             "n_points": n_points,
             "phase": "survey",
             "training record": training_record},
            ignore_index=True
        )
        self.logger.info(f"Integral: {integral:.3e} +/- {(integral_var / n_points) ** 0.5:.3e}")

    def process_refine_step(self, sample, integral, integral_var, **kwargs):
        x, px, fx = sample
        n_points = x.shape[0]
        self.integration_history = self.integration_history.append(
            {"integral": integral,
             "error": (integral_var / n_points) ** 0.5,
             "n_points": n_points,
             "phase": "refine"},
            ignore_index=True
        )

        self.logger.info(f"Integral: {integral:.3e} +/- {(integral_var / n_points) ** 0.5:.3e}")

    def finalize_survey(self, **kwargs):
        pass

    def finalize_refine(self, **kwargs):
        pass

    def finalize_integration(self, use_survey=None, **kwargs):
        if use_survey is None:
            use_survey = self.use_survey

        if use_survey:
            data = self.integration_history
        else:
            data = self.integration_history.loc[self.integration_history["phase"] == "refine"]

        result = (data["integral"] * data["n_points"]).sum()
        result /= data["n_points"].sum()

        error = np.sqrt(((data["error"] * data["n_points"]) ** 2).sum() / (data["n_points"].sum()) ** 2)

        self.logger.info(f"Final result: {float(result):.5e} +/- {float(error):.5e}")

        return float(result), float(error), self.integration_history

    def survey(self, n_survey_steps=None, **kwargs):
        if n_survey_steps is None:
            n_survey_steps = self.n_iter_survey
        super(BaseIntegrator, self).survey(n_survey_steps=n_survey_steps, **kwargs)

    def refine(self, n_refine_steps=None, **kwargs):
        if n_refine_steps is None:
            n_refine_steps = self.n_iter_refine
        super(BaseIntegrator, self).refine(n_refine_steps=n_refine_steps, **kwargs)

    def integrate(self, n_survey_steps=None, n_refine_steps=None, **kwargs):
        """Perform the integration"""
        return super(BaseIntegrator, self).integrate(n_survey_steps=n_survey_steps,
                                                                 n_refine_steps=n_refine_steps, **kwargs)

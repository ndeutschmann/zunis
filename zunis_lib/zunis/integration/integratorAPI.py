import logging
from better_abc import ABC, abstractmethod, abstract_attribute
import torch

from zunis.utils.logger import set_verbosity as set_verbosity_fct
from zunis.utils.exceptions import TrainingInterruption

class SurveyRefineIntegratorAPI(ABC):
    """API specification for an integrator that performs a number of survey steps in which
    a flow model is trained and then a number of refine steps in which sampling is done through the model
    The interaction with the model is done through the GenericTrainerAPI
    """

    set_verbosity = set_verbosity_fct

    def __init__(self, *args, verbosity=None, **kwargs):
        self.model_trainer = abstract_attribute()
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__ + ":" + hex(id(self)))
        self.set_verbosity(verbosity)

    @abstractmethod
    def initialize(self, **kwargs):
        """Intialization before the whole integration process"""

    @abstractmethod
    def initialize_refine(self, **kwargs):
        """Initialization before the survey phase"""

    @abstractmethod
    def initialize_survey(self, **kwargs):
        """Initialization before the survey phase"""

    @abstractmethod
    def sample_survey(self, **kwargs):
        """Sample points for a survey step"""

    @abstractmethod
    def sample_refine(self, **kwargs):
        """Sample points for a refine step"""

    @abstractmethod
    def process_survey_step(self, sample, integral, integral_var, **kwargs):
        """Process the result of a survey step"""

    @abstractmethod
    def process_refine_step(self, sample, integral, integral_var, **kwargs):
        """Process the result of a refine step"""

    @abstractmethod
    def finalize_survey(self, **kwargs):
        """Perform the final operations of the survey phase"""

    @abstractmethod
    def finalize_refine(self, **kwargs):
        """Perform the final operations of the refine phase"""

    @abstractmethod
    def finalize_integration(self, **kwargs):
        """Perform the final operations of the whole integration"""

    def format_arguments(self, **kwargs):
        """Format keyword arguments passed to the train function in a suitable way
        In this generic API definition, the kwargs must be passed in the correct format
        however for a specific realization, options should be made accessible through keywords
        which will then be sorted into the right structure.
        """
        return kwargs

    def survey_step(self, **kwargs):
        """Basic survey step: sample points, estimate the integral, its error, train model

        possible keyword arguments:
            sampling_args: dict
            training_args: dict
        """

        try:
            sampling_args = kwargs["sampling_args"]
        except KeyError:
            sampling_args = dict()

        try:
            training_args = kwargs["training_args"]
        except KeyError:
            training_args = dict()

        x, px, fx = self.sample_survey(**sampling_args)
        integral_var, integral = torch.var_mean(fx / px)
        integral = integral.cpu().item()
        integral_var = integral_var.cpu().item()

        training_record = self.model_trainer.train_on_batch(x, px, fx, **training_args)

        self.process_survey_step((x, px, fx), integral, integral_var, training_record=training_record)

    def refine_step(self, **kwargs):
        """Basic refine step: sample points, estimate the integral, its error, train model

        possible keyword arguments:
            sampling_args: dict
        """

        try:
            sampling_args = kwargs["sampling_args"]
        except KeyError:
            sampling_args = dict()

        x, px, fx = self.sample_refine(**sampling_args)
        integral_var, integral = torch.var_mean(fx / px)
        integral = integral.cpu().item()
        integral_var = integral_var.cpu().item()

        self.process_refine_step((x, px, fx), integral, integral_var)

    def survey(self, n_survey_steps=10, **kwargs):
        """Perform a survey phase of integration

        possible keyword arguments:
            trainer_config_args: dict
            survey_step_args: dict
            finalize_survey_args: dict

        """
        self.logger.info("Initializing the survey phase")

        try:
            trainer_config_args = kwargs["trainer_config_args"]
            self.model_trainer.set_config(**trainer_config_args)
        except KeyError:
            pass

        try:
            survey_step_args = kwargs["survey_step_args"]
        except KeyError:
            survey_step_args = dict()

        try:
            initialize_survey_args = kwargs["initialize_survey_args"]
        except KeyError:
            initialize_survey_args = dict()

        try:
            finalize_survey_args = kwargs["finalize_survey_args"]
        except KeyError:
            finalize_survey_args = dict()

        self.initialize_survey(**initialize_survey_args)

        self.logger.info("Starting the survey phase")
        try:
            for i in range(n_survey_steps):
                self.survey_step(**survey_step_args)
        except TrainingInterruption as e:
            self.logger.debug(" "*72)
            self.logger.debug("="*72)
            self.logger.debug(" "*20 + "/!\ This message is important" + " "*23)
            self.logger.debug("="*72)
            self.logger.exception("Survey interrupted by a training interruption")
            self.logger.error("A working checkpoint was loaded successfully and integration can go on.")

        self.logger.info("Finalizing the survey phase")
        self.finalize_survey(**finalize_survey_args)

    def refine(self, n_refine_steps=10, **kwargs):
        """Perform the refine phase of integration

        Possible keyword arguments:

        trainer_config_args: dict
        refine_step_args: dict
        finalize_refine_args: dict
        """

        self.logger.info("Initializing the refine phase")

        try:
            trainer_config_args = kwargs["trainer_config_args"]
            self.model_trainer.set_config(**trainer_config_args)
        except KeyError:
            pass

        try:
            refine_step_args = kwargs["refine_step_args"]
        except KeyError:
            refine_step_args = dict()

        try:
            initialize_refine_args = kwargs["initialize_refine_args"]
        except KeyError:
            initialize_refine_args = dict()

        try:
            finalize_refine_args = kwargs["finalize_refine_args"]
        except KeyError:
            finalize_refine_args = dict()

        self.initialize_refine(**initialize_refine_args)

        self.logger.info("Starting the refine phase")
        for i in range(n_refine_steps):
            self.refine_step(**refine_step_args)

        self.logger.info("Finalizing the refine phase")
        self.finalize_refine(**finalize_refine_args)

    def integrate(self, n_survey_steps=10, n_refine_steps=10, **kwargs):
        self.logger.info("Starting integration")

        formated_kwargs = self.format_arguments(**kwargs)

        try:
            survey_args = formated_kwargs["survey_args"]
        except KeyError:
            survey_args = dict()

        try:
            refine_args = formated_kwargs["refine_args"]
        except KeyError:
            refine_args = dict()

        try:
            finalize_args = formated_kwargs["finalize_args"]
        except KeyError:
            finalize_args = dict()

        self.survey(n_survey_steps, **survey_args)
        self.refine(n_refine_steps, **refine_args)

        return self.finalize_integration(**finalize_args)

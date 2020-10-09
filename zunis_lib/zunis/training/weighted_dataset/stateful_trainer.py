"""Main weighted dataset stateful trainer API class"""
from functools import partial
import torch

from .weighted_dataset_trainer import BasicStatefulTrainer
from zunis.models.flows.sequential.repeated_cell import RepeatedCellFlow
from .dkl_training import weighted_dkl_loss
from .variance_training import weighted_variance_loss
from zunis.models.flows.sampling import UniformSampler, FactorizedGaussianSampler, FactorizedFlowSampler


class StatefulTrainer(BasicStatefulTrainer):
    """High-level API for stateful trainers using weighted datasets
    (dataset consisting of tuples of point, function value, point pdf).
    """

    loss_map = {
        "dkl": weighted_dkl_loss,
        "variance": weighted_variance_loss
    }
    """Dictionary for the string-based API to define the loss function used in training"""

    default_flow_priors = {
        "realnvp": FactorizedGaussianSampler,
        "pwlinear": UniformSampler,
        "pwquad": UniformSampler}
    """Dictionary for the string-based API to define the distribution of the data in latent space based on
    the choice of coupling cell"""

    flow_priors = {
        "gaussian": FactorizedGaussianSampler,
        "uniform": UniformSampler
    }
    """Dictionary for the string-based API to define the distribution of the data in latent space"""

    def __init__(self, d, loss="variance", flow="pwlinear", flow_prior=None, flow_options=None, prior_options=None,
                 device=torch.device("cpu"), n_epochs=10, optim=None, **kwargs):
        """

        Parameters
        ----------
        d: int
            dimensionality of the space
        loss: str or function
            loss function. If this argument is a string, it is mapped to a function using
            :py:attr:`zunis.training.weighted_dataset.stateful_trainer.loss_map`
        flow: str or :py:class:`zunis.models.flows.general_flow.GeneralFlow`
            if this variable is a string, it is a cell key used in :py:class:`zunis.models.flows.sequential.repeated_cell.RepeatedCellFlow`
            otherwise it can be an actual flow model
        flow_prior: None or str or :py:class:`zunis.models.flows.sampling.FactorizedFlowSampler`
            PDF used for sampling latent space. If None (default) then use the "natural choice" defined
            in the class variable :py:attr:`zunis.training.weighted_dataset.stateful_trainer.StatefulTrainer.default_flow_priors`
            A string argument will be mapped using :py:attr:`zunis.training.weighted_dataset.stateful_trainer.StatefulTrainer.flow_priors`
        flow_options: None or dict
            options to be passed to the :py:class:`zunis.models.flows.sequential.repeated_cell.RepeatedCellFlow` model if
            `flow` is a string
        prior_options: None or dict
            options to be passed to the latent prior constructor if a "natural choice" prior is used
            i.e. if `flow_prior` is `None` or a `str`
        device:
            device on which to run the model and the sampling
        n_epochs: int
            number of epochs per batch of data during training
        optim: None or torch.optim.Optimizer sublcass
            optimizer to use for training. If none, default Adam is used
        """

        # The loss function is either a string that points to the loss_map class dictionary or an actual loss function
        if isinstance(loss, str):
            loss = self.loss_map[loss]

        # The prior has three options:
        #
        # - None: if we use the string-based interface to define a RepeatedCellFlow, use the associated default prior
        # in `self.default_flow_priors`
        #
        # - string: use `self.flow_priors`
        #
        # - an actual prior object to be used as-is
        if prior_options is None:
            prior_options = dict()

        if flow_prior is None:
            assert isinstance(flow,
                              str), "You must specify a flow if you are not using the string-based model interface"
            flow_prior = self.default_flow_priors[flow](d=d, device=device, **prior_options)
        elif isinstance(flow_prior, str):
            flow_prior = self.flow_priors[flow_prior](d=d, device=device, **prior_options)
        else:
            assert isinstance(flow_prior,
                              FactorizedFlowSampler), \
                "The flow prior must be either None, a string or a FactorizedFlowSampler"

        # Two options for flows: either use the RepeatedCellFlow interface or pass an actual flow object
        if isinstance(flow, str):
            if flow_options is None:
                flow_options = dict()
            flow = RepeatedCellFlow(d=d, cell=flow, **flow_options).to(device)

        if optim is None:
            optim_ = torch.optim.Adam(flow.parameters())
        else:
            optim_ = optim(flow.parameters())

        super(StatefulTrainer, self).__init__(flow, flow_prior, n_epochs=n_epochs, optim=optim_, **kwargs)
        self.loss = loss

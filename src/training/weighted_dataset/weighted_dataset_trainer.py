import logging
import torch
from math import isfinite, ceil
from better_abc import ABC, abstractmethod, abstract_attribute
from .training_record import TrainingRecord

logger = logging.getLogger(__name__)


class InvalidLossError(ValueError):
    """Value error that indicates something is wrong with a loss value specifically"""
    pass


class GenericTrainerAPI(ABC):
    """Weighted dataset trainer API definition

    The goal of this API specification is to interact with the Integrator
    """

    @abstractmethod
    def train_on_batch(self, x, px, fx, **kwargs):
        """Training function

        Arguments:
            - x: batch of points in target space sampled from a distribution p(x)
            - px: the corresponding batch of p(x) values
            - fx: the (un-normalized) target weights
            - kwargs: trainer-specific options. This overrides the config

        Notes:
            This is not enforced at the level of this API but it is intended that kwargs leads to a call of
            set_config and therefore that these options are saved - at least by default.
        """

    @abstractmethod
    def set_config(self, **kwargs):
        """Set training configuration options"""

    @abstractmethod
    def get_config(self, **kwargs):
        """Get the current training configuration"""


class BasicTrainer(ABC):
    """Basic trainer implementation: sample points in target space from a fixed distribution and train over
    a fixed number of epochs on each batch, over a fixed number of batches.

    This is the implementation of all training facilities for this training mode at a low level: no automation,
    tracking, checkpointing etc is performed. No state or history is conserved.

    Rationale: the training implementation should be independent from the tracking and from the API
    """

    def __init__(self, flow, latent_prior):
        self.flow = flow
        self.latent_prior = latent_prior
        self.loss = abstract_attribute()

    def sample_forward(self, n_points):
        if self.flow.inverse:
            self.flow.invert()

        points = self.latent_prior(n_points)
        with torch.no_grad():
            points = self.flow(points)[:, :-1]
        return points

    def process_loss(self, loss):
        if not isfinite(loss):
            raise InvalidLossError(f"loss value {loss} is not valid")
        logger.debug(f"Loss: {loss:.3e}")
        return False

    def handle_invalid_loss(self, error):
        raise

    def compute_loss_no_grad(self, x, px, fx):
        if not self.flow.inverse:
            self.flow.invert()

        with torch.no_grad():
            xj = torch.cat(x, torch.ones(x.shape[0], 1), dim=1)

            zj = self.flow(xj)
            z = zj[:, :-1]
            logqx = zj[:, -1] + self.latent_prior.log_prob(z)
            loss = self.loss(fx, px, logqx)

        return loss.cpu().item()

    def train_step_on_target_minibatch(self, x, px, fx, optim):
        """Perform one training set on a minibatch

        Parameters:
        -----------
            x: batch of points in target space sampled from some PDF p(x)
            px: values of p(x) for the batch
            fx: values of the target (un-normalized) PDF
            optim: pytorch optimizer class
        """
        if not self.flow.inverse:
            self.flow.invert()

        xj = torch.cat([x, torch.zeros(x.shape[0], 1).to(x.device)], dim=1)

        zj = self.flow(xj)
        z = zj[:, :-1]
        logqx = zj[:, -1] + self.latent_prior.log_prob(z)
        optim.zero_grad()
        loss = self.loss(fx, px, logqx)
        loss.backward()
        optim.step()
        loss = loss.detach().cpu().item()

        return loss

    def train_step_on_target_batch(self, x, px, fx, optim, minibatch_size=None):
        """Training function on a fixed batch: iterate once over the whole batch

        Parameters:
        -----------
            x: batch of points in target space sampled from some PDF p(x)
            px: values of p(x) for the batch
            fx: values of the target (un-normalized) PDF
            optim: pytorch optimizer class
            minibatch_size: Optional. Size of each minibatch for gradient steps.

        Notes:
        ------
            if minibatch_size is unset (or None), then it is set to the size of the full batch and a single
            gradient step is taken
        """
        n_points = x.shape[0]
        if minibatch_size is None:
            minibatch_size = n_points

        begin = 0
        while begin < n_points:
            end = min(begin + minibatch_size, n_points)
            loss = self.train_step_on_target_minibatch(
                x[begin:end],
                px[begin:end],
                fx[begin:end],
                optim,
            )
            early_stop = self.process_loss(loss)
            if early_stop:
                break
            begin = end

    def train_on_target_batch(self, x, px, fx, optim, n_epochs, minibatch_size=None):
        """Training function on a fixed batch: train for a fixed number of epochs on each batch

        Parameters:
        -----------
            x: batch of points in target space sampled from some PDF p(x)
            px: values of p(x) for the batch
            fx: values of the target (un-normalized) PDF
            optim: pytorch optimizer class
            n_epochs: number of iterations over the full batch
            minibatch_size: Optional. Size of each minibatch for gradient steps

        Notes:
        ------
            if minibatch_size is unset (or None), then it is set to the size of the full batch and a single
            gradient step is taken
        """
        logger.info(f"Training on batch: {x.shape[0]} points")
        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch + 1}/{n_epochs}")
            try:
                self.train_step_on_target_batch(x, px, fx, optim, minibatch_size)
            except InvalidLossError as e:
                logger.error(e)
                self.handle_invalid_loss(e)

    @staticmethod
    def generate_target_batch_from_posterior(n_points, f, target_posterior):
        """Generate a batch of training examples in target space from a specified distribution
        Parameters:
        -----------
            n_points: size of the batch
            f: function to evaluate on the sampled points
            target_posterior: distribution from which to sample points in target_space

        Returns:
        --------
            x,px,fx
                sampled points, sampling distribution PDF values, function values
        """
        with target_posterior(n_points) as xlpx:
            x = xlpx[:, :-1]
            px = torch.exp(- xlpx[:, -1])
        fx = f(x)

        return x, px, fx

    def train_on_target_batches_from_posterior(self,
                                               *,
                                               f,
                                               batch_size,
                                               n_batches=1,
                                               n_epochs_per_batch,
                                               minibatch_size=None,
                                               target_posterior,
                                               optim):
        """Main training function: iterate over a fixed number of batches sampled in target space and
         train for a fixed number of epochs on each batch

        Parameters:
        -----------
            f: un-normalized target PDF
            batch_size: number of points per batch
            n_batches: number of batches to train over
            n_epochs_per_batch: number of iterations over each full batch before sampling a new one
            minibatch_size: Optional. Size of each minibatch for gradient steps
            target_posterior: distribution from which points are sampled in target space
            optim: pytorch optimizer class

        Notes:
        ------
            if minibatch_size is unset (or None), then it is set to the size of the full batch and a single
            gradient step is taken
        """
        logger.info(f"Training on {n_batches} independent batches of size {batch_size}")
        logger.info(f"Performing {n_epochs_per_batch} epochs per batch")
        logger.info(f"Each epoch applies {int(ceil(batch_size / minibatch_size))} gradient steps")

        for batch in range(n_batches):
            logger.info("Generating a new batch of points")
            x, px, fx = self.generate_target_batch_from_posterior(batch_size, f, target_posterior)
            self.train_on_target_batch(x, px, fx, optim, n_epochs_per_batch, minibatch_size)


class BasicStatefulTrainer(BasicTrainer, GenericTrainerAPI):
    def __init__(self, flow, latent_prior, **kwargs):
        super(BasicStatefulTrainer, self).__init__(flow, latent_prior)

        # Setting up the saved configuration accessed through the GenericTrainerAPI
        # Note that the GenericTrainerAPI, operates at the level of a single batch
        # As a result, only relevant keys are part of the config

        # The keys are stored separately so as to be immutable
        self.config_keys = ("n_epochs", "minibatch_size", "optim")

        self.config = {
            "n_epochs": None,
            "minibatch_size": None,
            "optim": None
        }

        for key in kwargs:
            if key in self.config_keys:
                self.config[key] = kwargs[key]

        self.record = TrainingRecord()

    def process_loss(self, loss):
        self.record.log_loss(loss)
        return super(BasicStatefulTrainer, self).process_loss(loss)

    def train_step_on_target_minibatch(self, x, px, fx, optim):
        super(BasicStatefulTrainer, self).train_step_on_target_minibatch(x, px, fx, optim)
        self.record.next_step()

    def train_step_on_target_batch(self, x, px, fx, optim, minibatch_size=None):
        self.record.new_epoch()
        super(BasicStatefulTrainer, self).train_step_on_target_batch(x, px, fx, optim, minibatch_size)

    def train_on_target_batches_from_posterior(self,
                                               *,
                                               f,
                                               batch_size,
                                               n_batches=1,
                                               n_epochs_per_batch,
                                               minibatch_size=None,
                                               target_posterior,
                                               optim):
        """Train over several epochs over several batches with explicit arguments - overriding the current config
        We save the full config of this run here and therefore have a different config format
        """

        config = {
            "f": f,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "n_epochs_per_batch": n_epochs_per_batch,
            "minibatch_size": minibatch_size,
            "target_posterior": target_posterior,
            "optim": optim
        }

        self.record = TrainingRecord(config=config)
        super(BasicStatefulTrainer, self).train_on_target_batches_from_posterior(**config)
        return self.record

    def set_config(self, **kwargs):
        """Set the saved configuration

        This is part of the GenericTrainerAPI, which provides facilities for training
        *on a batch* therefore
        """
        for key in kwargs:
            if key in self.config_keys:
                self.config[key] = kwargs[key]
            else:
                logger.warning(f"Specified config key {key} is not recognized and was ignored")

    def get_config(self):
        return self.config

    def train_on_batch(self, x, px, fx, **kwargs):
        """Train on a batch of points using the saved configuration"""
        self.set_config(**kwargs)
        self.record = TrainingRecord(config=self.config)
        self.train_on_target_batch(x, px, fx, **self.config)
        return self.record

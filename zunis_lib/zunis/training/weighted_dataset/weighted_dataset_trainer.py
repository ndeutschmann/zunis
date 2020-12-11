import logging
import torch
from copy import deepcopy
from collections import OrderedDict
from math import isfinite, ceil
from better_abc import ABC, abstractmethod, abstract_attribute
from .training_record import TrainingRecord
from zunis.utils.logger import set_verbosity as set_verbosity_fct
from zunis.utils.exceptions import AvertedCUDARuntimeError, NoCheckpoint, TrainingInterruption


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
            - x:
                batch of points in target space sampled from a distribution p(x)
            - px:
                the corresponding batch of p(x) values
            - fx:
                the (un-normalized) target weights
            - kwargs:
                trainer-specific options. This overrides the config

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

    @abstractmethod
    def sample_forward(self, n_points):
        """Sample points using the model"""

    def reset(self):
        """Reinitinalize trainer and model. Optional element of the API.
        Raises an error if not implemented
        """
        raise NotImplementedError("reset method not available for this class")


class BasicTrainer(ABC):
    """Basic trainer implementation: sample points in target space from a fixed distribution and train over
    a fixed number of epochs on each batch, over a fixed number of batches.

    This is the implementation of all training facilities for this training mode at a low level: no automation,
    tracking, checkpointing etc is performed. No state or history is conserved.

    Rationale: the training implementation should be independent from the tracking and from the API
    """

    set_verbosity = set_verbosity_fct

    def __init__(self, flow, latent_prior, verbosity=None):
        self.flow = flow
        self.latent_prior = latent_prior
        self.loss = abstract_attribute()
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__ + ":" + hex(id(self)))
        self.set_verbosity(verbosity)

    def sample_forward(self, n_points):
        if self.flow.inverse:
            self.flow.invert()

        xj = self.latent_prior(n_points)
        with torch.no_grad():
            xj = self.flow(xj)
        return xj.detach()

    def process_loss(self, loss):
        if not isfinite(loss):
            raise InvalidLossError(f"loss value {loss} is not valid")
        self.logger.debug(f"Loss: {loss:.3e}")
        return False

    def handle_invalid_loss(self, error):
        raise

    def handle_cuda_error(self, error):
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

        Parameters
        ----------
            x:
                batch of points in target space sampled from some PDF p(x)
            px:
                values of p(x) for the batch
            fx:
                values of the target (un-normalized) PDF
            optim:
                pytorch optimizer object
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

        Parameters
        ----------
            x: torch.Tensor
                batch of points in target space sampled from some PDF p(x)
            px: torch.Tensor
                values of p(x) for the batch
            fx: torch.Tensor
                values of the target (un-normalized) PDF
            optim: torch.optim.Optimize
                pytorch optimizer object
            minibatch_size: None or int
                Optional. Size of each minibatch for gradient steps.

        Notes
        -----
            if minibatch_size is unset (or None), then it is set to the size of the full batch and a single
            gradient step is taken
        """
        n_points = x.shape[0]
        if minibatch_size is None:
            minibatch_size = n_points
        elif isinstance(minibatch_size, float) and 0 < minibatch_size <= 1.:
            minibatch_size = int(minibatch_size * x.shape[0])
        else:
            assert isinstance(minibatch_size,
                              int) and minibatch_size > 0, f"minibatch size must be None, a float in ]0,1] or a " \
                                                           f"positive int. Got {type(minibatch_size)}"

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
                raise TrainingInterruption("Early stopping condition was raised")
            begin = end

    def train_on_target_batch(self, x, px, fx, optim, n_epochs, minibatch_size=None):
        """Training function on a fixed batch: train for a fixed number of epochs on each batch

        Parameters
        ----------
            x:
                batch of points in target space sampled from some PDF p(x)
            px:
                values of p(x) for the batch
            fx:
                values of the target (un-normalized) PDF
            optim:
                pytorch optimizer object
            n_epochs:
                number of iterations over the full batch
            minibatch_size:
                Optional. Size of each minibatch for gradient steps

        Notes
        -----
            if minibatch_size is unset (or None), then it is set to the size of the full batch and a single
            gradient step is taken
        """
        self.logger.info(f"Training on batch: {x.shape[0]} points")
        for epoch in range(n_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{n_epochs}")
            try:
                self.train_step_on_target_batch(x, px, fx, optim, minibatch_size)
            except InvalidLossError as e:
                self.logger.exception("Invalid loss detected - handling started")
                self.handle_invalid_loss(e)
            except AvertedCUDARuntimeError as e:
                self.logger.exception("CUDA error averted - handling started")
                self.handle_cuda_error(e)

    @staticmethod
    def generate_target_batch_from_posterior(n_points, f, target_posterior):
        """Generate a batch of training examples in target space from a specified distribution

        Parameters
        ----------
            n_points:
                size of the batch
            f:
                function to evaluate on the sampled points
            target_posterior:
                distribution from which to sample points in target_space

        Returns
        -------
            tuple of torch.Tensor
                (x,px,fx): sampled points, sampling distribution PDF values, function values
        """
        xlpx = target_posterior(n_points)
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

        Parameters
        ----------
            f:
                un-normalized target PDF
            batch_size:
                number of points per batch
            n_batches:
                number of batches to train over
            n_epochs_per_batch:
                number of iterations over each full batch before sampling a new one
            minibatch_size:
                Optional. Size of each minibatch for gradient steps
            target_posterior: distribution from which points are sampled in target space
            optim:
                pytorch optimizer object

        Notes
        -----
            if minibatch_size is unset (or None), then it is set to the size of the full batch and a single
            gradient step is taken
        """
        self.logger.info(f"Training on {n_batches} independent batches of size {batch_size}")
        self.logger.info(f"Performing {n_epochs_per_batch} epochs per batch")
        self.logger.info(f"Each epoch applies {int(ceil(batch_size / minibatch_size))} gradient steps")

        for batch in range(n_batches):
            self.logger.info("Generating a new batch of points")
            x, px, fx = self.generate_target_batch_from_posterior(batch_size, f, target_posterior)
            self.train_on_target_batch(x, px, fx, optim, n_epochs_per_batch, minibatch_size)


class BasicStatefulTrainer(BasicTrainer, GenericTrainerAPI):
    def __init__(self, flow, latent_prior, checkpoint=True, checkpoint_on_cuda=True, checkpoint_path=None, max_reloads=None,
                 **kwargs):
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

        # Checkpointing logic
        self.checkpoint = checkpoint
        self.checkpoint_on_cuda = checkpoint_on_cuda
        self.checkpoint_data = None
        self.checkpoint_path = checkpoint_path
        self.n_reloads = 0
        if max_reloads is None:
            self.max_reloads = None if not checkpoint else 1
        else:
            self.max_reloads = max_reloads

        if self.checkpoint_path is None:
            self.record = TrainingRecord()
        else:
            self.record = TrainingRecord(checkpoint=checkpoint_path)

    def set_checkpoint(self):
        """Save the current model state as a checkpoint"""
        if not self.checkpoint:
            self.logger.error("This trainer cannot save checkpoints")
            raise AssertionError("This trainer cannot save checkpoints")

        state_dict = deepcopy(self.flow.state_dict())
        if self.checkpoint_on_cuda:
            self.checkpoint_data = state_dict
        else:
            self.checkpoint_data = OrderedDict(
                [(key, value.cpu()) for key, value in state_dict.items()]
            )
        if self.checkpoint_path is not None:
            torch.save(self.checkpoint_data, self.checkpoint_path)

    def restore_checkpoint(self, path=None):
        """Restore from a checkpoint if available"""
        # This function is called when something fails
        # as a result, it should always raise its specific type of error: NoCheckpoint so that the
        # more important failure cause can be reported
        # We use logging.exception to still report the local error stack trace in logs

        # If a path is provided, use it and fail otherwise
        # Use a NoCheckpoint exception to allow whatever error triggered this function to be the main failure cause
        if path is not None:
            try:
                self.flow.load_state_dict(torch.load(path))
                return
            except Exception:
                self.logger.exception(f"Error when loading state dict from file")
                raise NoCheckpoint(f"Could not load checkpoint from {path}")

        # Default functioning mode: no path specified, use initialization settings
        # We have to have set the options to keep checkpoints in memory and have saved at least once
        if self.checkpoint and self.checkpoint_data is not None:
            self.logger.warning("Reloading the latest checkpoint from memory")
            try:
                self.flow.load_state_dict(self.checkpoint_data)
                return
            # In case of error, log the stack trace and try loading from disk if possible
            except Exception:
                self.logger.exception("Could not load checkpoint from memory")
            if self.checkpoint_path is not None:
                self.logger.warning("Trying to load latest checkpoint from disk")
                try:
                    self.flow.load_state_dict(torch.load(self.checkpoint_path))
                    return
                # Again, if we fail, log the stack trace and raise a NoCheckpoint to allow upstream triage
                except Exception:
                    self.logger.exception("Error when loading state dict from file")
                    raise NoCheckpoint(f"Could not load checkpoint from {self.checkpoint_path}")

        raise NoCheckpoint("No checkpoint available")

    def restore_checkpoint_except(self):
        """Try to to restore from a checkpoint as a response to an exception"""
        if "checkpoint" in self.record and self.n_reloads < self.max_reloads:
            self.logger.warning(f"Attempting checkpoint reload {self.n_reloads + 1}/{self.max_reloads}")
            self.restore_checkpoint()
            self.n_reloads += 1
            return True
        else:
            self.logger.error("Cannot reset to a previous checkpoint to sidestep exception")
            return False

    def handle_invalid_loss(self, error):
        """In case of invalid loss, we try to reload the latest checkpoint"""
        if self.restore_checkpoint_except():
            return

        # Otherwise reload the checkpoint and re-raise the error
        try:
            self.logger.warning("An InvalidLossError could not be sidestepped. Trying to reload checkpoint before interrupting the training")
            self.restore_checkpoint()
            self.logger.warning("Could load the latest checkpoint")
            interruptor = TrainingInterruption("Invalid loss error interrupted training")
            self.logger.error(repr(interruptor))
            raise interruptor
        except NoCheckpoint:
            pass

        super(BasicStatefulTrainer, self).handle_invalid_loss(error)

    def handle_cuda_error(self, error):
        """In case of a cuda error, we try to reload the latest checkpoint"""
        if self.restore_checkpoint_except():
            return

        # Otherwise reload the checkpoint and re-raise the error
        try:
            self.logger.warning("An AvertedCUDARuntimeError could not be sidestepped. Trying to reload checkpoint before interrupting the training")
            self.restore_checkpoint()
            self.logger.warning("Could load the latest checkpoint")
            interruptor = TrainingInterruption("CUDA error interrupted training")
            self.logger.error(repr(interruptor))
            raise interruptor
        except NoCheckpoint:
            pass

        super(BasicStatefulTrainer, self).handle_cuda_error(error)

    def process_loss(self, loss):
        """Handle invalid losses by reloading checkpoint and handle logging and checkpointing if valid"""

        # parent class checks that the loss is valid
        output = super(BasicStatefulTrainer, self).process_loss(loss)

        # Handle logging and checkpointing
        self.record.log_loss(loss)
        if self.checkpoint and self.record["loss"] <= self.record["best_loss"]:
            self.set_checkpoint()
        return output

    def train_step_on_target_minibatch(self, x, px, fx, optim):
        loss = super(BasicStatefulTrainer, self).train_step_on_target_minibatch(x, px, fx, optim)
        self.record.next_step()
        return loss

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
        """
        for key in kwargs:
            if key in self.config_keys:
                self.config[key] = kwargs[key]
            else:
                self.logger.error(f"Specified config key {key} is not recognized")
                raise KeyError(f"Config key {key} was not recognized")

    def get_config(self):
        """Return the saved configuration"""
        return self.config

    def train_on_batch(self, x, px, fx, **kwargs):
        """Train on a batch of points using the saved configuration"""
        self.set_config(**kwargs)

        try:
            checkpoint_path = kwargs["checkpoint_path"]
        except KeyError:
            checkpoint_path = self.checkpoint_path

        self.record = TrainingRecord(checkpoint=checkpoint_path)

        optim = self.config["optim"]
        if optim is None:
            error = ValueError("trainer parameter 'optim' must be set before or at training")
            self.logger.error(repr(error))
            raise error

        n_epochs = self.config["n_epochs"]
        self.logger.debug(f"n_epochs in trainer: {n_epochs}")
        if n_epochs is None:
            error = ValueError("trainer parameter 'n_epochs' must be set before or at training")
            self.logger.error(repr(error))
            raise error

        minibatch_size = self.config["minibatch_size"]
        self.train_on_target_batch(x, px, fx, optim=optim, n_epochs=n_epochs, minibatch_size=minibatch_size)
        return self.record

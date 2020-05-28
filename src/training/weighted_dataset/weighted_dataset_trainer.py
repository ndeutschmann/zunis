import logging
import torch
from math import isfinite, ceil
from better_abc import ABC, abstractmethod, abstract_attribute
from .training_state import TrainingState

logger = logging.getLogger(__name__)


class InvalidLossError(ValueError):
    """Value error that indicates something is wrong with a loss value specifically"""
    pass


class GenericTrainer(ABC):
    def __init__(self, flow, latent_prior):
        self.flow = flow
        self.latent_prior = latent_prior
        self.loss = abstract_attribute()

    def sample_forward(self, n_points):
        if self.flow.inverse:
            self.flow.invert()

        points = self.latent_prior(n_points)
        return points

    def process_loss(self, loss):
        if not isfinite(loss):
            raise InvalidLossError(f"loss value {loss} is not valid")
        logger.debug(f"Loss: {loss:.3e}")
        return False

    def handle_invalid_loss(self, error):
        raise

    def compute_loss_no_grad(self, xpx, fx):
        if not self.flow.inverse:
            self.flow.invert()

        with torch.no_grad():
            px = xpx[:, -1]
            xj = torch.cat(xpx[:, :-1], torch.ones(xpx.size[0], 1), dim=1)

            zj = self.flow(xj)
            z = zj[:, :-1]
            logqx = zj[:, -1] + self.latent_prior.log_prob(z)
            loss = self.loss(fx, px, logqx)

        return loss.cpu().item()

    def train_step_on_target_minibatch(self, xpx, fx, optim):
        if not self.flow.inverse:
            self.flow.invert()

        px = xpx[:, -1]
        xj = torch.cat(xpx[:, :-1], torch.ones(xpx.size[0], 1), dim=1)

        zj = self.flow(xj)
        z = zj[:, :-1]
        logqx = zj[:, -1] + self.latent_prior.log_prob(z)
        optim.zero_grad()
        loss = self.loss(fx, px, logqx)
        loss.backward()
        optim.step()
        loss = loss.detach().cpu().item()

        return loss

    def train_step_on_target_batch(self, xpx, fx, optim, minibatch_size=None):
        n_points = xpx.shape[0]
        if minibatch_size is None:
            minibatch_size = n_points

        begin = 0
        while begin < n_points:
            end = min(begin + minibatch_size, n_points)
            loss = self.train_step_on_target_minibatch(
                xpx[begin:end],
                fx[begin:end],
                optim,
            )
            early_stop = self.process_loss(loss)
            if early_stop:
                break

    def train_on_target_batch(self, xpx, fx, optim, n_epochs, minibatch_size=None):
        logger.info(f"Training on batch: {xpx.size[0]} points")
        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch + 1}/{n_epochs}")
            try:
                self.train_step_on_target_batch(xpx, fx, optim, minibatch_size)
            except InvalidLossError as e:
                logger.error(e)
                self.handle_invalid_loss(e)

    @staticmethod
    def generate_target_batch_from_posterior(n_points, f, target_posterior):
        xpx = target_posterior.sample(n_points)
        fx = f(xpx[:, :-1])

        return xpx, fx

    def train_on_target_batches_from_posterior(self,
                                               *,
                                               f,
                                               batch_size,
                                               n_batches=1,
                                               n_epochs_per_batch,
                                               minibatch_size=None,
                                               target_posterior,
                                               optim):
        logger.info(f"Training on {n_batches} independent batches of size {batch_size}")
        logger.info(f"Performing {n_epochs_per_batch} epochs per batch")
        logger.info(f"Each epoch applies {int(ceil(batch_size / minibatch_size))} gradient steps")

        for batch in range(n_batches):
            logger.info("Generating a new batch of points")
            xpx, fx = self.generate_target_batch_from_posterior(batch_size, f, target_posterior)
            self.train_on_target_batch(xpx, fx, optim, n_epochs_per_batch, minibatch_size)


class GenericStatefulTrainer(GenericTrainer):
    def __init__(self, flow, latent_prior):
        super(GenericStatefulTrainer, self).__init__(flow, latent_prior)
        self.state = TrainingState()

    def process_loss(self, loss):
        self.state.log_loss(loss)
        return super(GenericStatefulTrainer, self).process_loss(loss)

    def train_step_on_target_minibatch(self, xpx, fx, optim):
        super(GenericStatefulTrainer, self).train_step_on_target_minibatch(xpx, fx, optim)
        self.state.next_step()

    def train_step_on_target_batch(self, xpx, fx, optim, minibatch_size=None):
        self.state.next_epoch()
        super(GenericStatefulTrainer, self).train_step_on_target_batch(xpx, fx, optim, minibatch_size)

    def train_on_target_batches_from_posterior(self,
                                               *,
                                               f,
                                               batch_size,
                                               n_batches=1,
                                               n_epochs_per_batch,
                                               minibatch_size=None,
                                               target_posterior,
                                               optim):
        self.state = TrainingState()
        super(GenericStatefulTrainer, self).train_on_target_batches_from_posterior(f=f,
                                                                                   batch_size=batch_size,
                                                                                   n_batches=n_batches,
                                                                                   n_epochs_per_batch=n_epochs_per_batch,
                                                                                   minibatch_size=minibatch_size,
                                                                                   target_posterior=target_posterior,
                                                                                   optim=optim)
        return self.state

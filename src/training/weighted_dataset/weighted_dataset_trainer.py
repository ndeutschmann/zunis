import logging
import torch
from math import isfinite
from better_abc import ABC, abstractmethod, abstract_attribute

logger = logging.getLogger(__name__)


class InvalidLossError(ValueError):
    """Value error that indicates something is wrong with a loss value specifically"""
    pass


class GenericTrainer(ABC):
    def __init__(self):
        self.callbacks = None
        self.flow = abstract_attribute()
        self.latent_prior = abstract_attribute()
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

    def train_step_on_target_batch(self, xpx, fx, optim, batch_size=None):
        n_points = xpx.shape[0]
        if batch_size is None:
            batch_size = n_points

        begin = 0
        while begin < n_points:
            end = min(begin + batch_size, n_points)
            loss = self.train_step_on_target_minibatch(
                xpx[begin:end],
                fx[begin:end],
                optim,
            )
            early_stop = self.process_loss(loss)
            if early_stop:
                break

    def train_on_batch(self, xpx, fx, optim, n_epochs, batch_size=None):
        logger.info(f"Training on batch: {xpx.size[0]} points")
        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch + 1}/{n_epochs}")
            try:
                self.train_step_on_target_batch(xpx, fx, optim, batch_size)
            except InvalidLossError as e:
                logger.error(e)
                self.handle_invalid_loss(e)

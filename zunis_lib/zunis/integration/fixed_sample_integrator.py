"""Integrator that does not sample points during the training phase but uses a fixed dataset of points"""
from collections.abc import Sequence, Mapping
import pickle

import numpy as np
import torch

from .base_integrator import BaseIntegrator


class FixedSampleSurveyIntegrator(BaseIntegrator):
    """Integrator that trains its model during the survey phase using a pre-computed sample provided externally"""

    def __init__(self, f, trainer, sample=None, n_iter=None, n_iter_survey=1, n_iter_refine=10,
                 n_points=None, n_points_survey=None, n_points_refine=10000, use_survey=False,
                 verbosity=None, trainer_verbosity=None, **kwargs):
        """

        Parameters
        ----------
        f: callable
            ZuNIS-compatible function
        trainer: BasicTrainer
            trainer object used to perform the survey
        sample: tuple of torch.Tensor
            (x, fx, px): target-space point batch drawn from some PDF p, function value batch, PDF value batch p(x)
        n_iter: int
            number of iterations (used for both survey and  refine unless specified)
        n_iter_survey: int
            number of iterations for survey
        n_iter_refine: int
            number of iterations for refine
        n_points: int
            number of points for both survey and refine unless specified
        n_points_survey: int
            number of points for survey
        n_points_refine: int
            number of points for refine
        use_survey: bool
            whether to use the integral estimations from the survey phase. This makes error estimation formally
            incorrect since samples from the refine depend on the survey training, but these correlation can be negligible
            in some cases.
        verbosity: int
            level of verbosity for the integrator-level logger
        trainer_verbosity: int
            level of verbosity for the trainer-level logger
        kwargs
        """
        super(FixedSampleSurveyIntegrator, self).__init__(f=f,
                                                          trainer=trainer,
                                                          trainer_verbosity=trainer_verbosity,
                                                          n_iter=n_iter,
                                                          n_iter_survey=n_iter_survey,
                                                          n_iter_refine=n_iter_refine,
                                                          n_points=n_points,
                                                          n_points_survey=n_points_survey,
                                                          n_points_refine=n_points_refine,
                                                          use_survey=use_survey,
                                                          verbosity=verbosity,
                                                          **kwargs)

        self.sample = sample

    def sample_survey(self, n_points=None, **kwargs):
        """Sample points from the internally stored sample

        Parameters
        ----------
        n_points: int, None
            size of the batch to select from the sample
        kwargs

        Returns
        -------
            tuple of torch.Tensor
                (x,px,fx): sampled points, sampling distribution PDF values, function values

        """
        assert self.sample is not None, "The training sample must be instantiated before starting the survey"

        if n_points is None:
            n_points = self.n_points_survey

        if n_points is None:
            return self.sample

        x, px, fx = self.sample

        sample_size = x.shape[0]
        assert n_points <= sample_size

        # Sample n_points indices randomly among the sample_size options with equal probability
        indices = torch.multinomial(torch.ones(sample_size), n_points)
        return x[indices], px[indices], fx[indices]

    def set_sample(self, sample):
        """Assign a sample to be trained on

        Parameters
        ----------
        sample: tuple of torch.Tensor
            (x,px,fx): sampled points, sampling distribution PDF values, function values
        """
        assert len(sample) == 3, "A sample is a Sequence of three elements"
        for el in sample:
            assert isinstance(el, torch.Tensor), "All elements of a sample must be tensors"
        assert sample[0].shape[0] == sample[1].shape[0] == sample[2].shape[0], \
            "All elements of a sample must share the same batch size"
        assert len(sample[0].shape) == 2 and len(sample[1].shape) == len(sample[2].shape) == 1, \
            "Sample shapes must be (n_batch, n_dim), (n_batch), (n_batch)"
        assert sample[0].shape[1] == self.model_trainer.flow.d, "Number of dimensions must match the flow"

        self.sample = sample

    def set_sample_pickle(self, pickle_path, device=None):
        """Assign a sample to be trained on from a pickle file
        The pickle file must either contain a tuple (x,px,fx) of point batch, PDF value batch, function batch
        or a mapping with keys "x", "px", "fx". In either case, these batches must be valid inputs for torch.tensor

        Parameters
        ----------
        pickle_path: str
            path to the pickle file.
        device: torch.device, None
            device on which to send the sample. If none is provided, flow parameter device will be used
        """
        with open(pickle_path, "rb") as picklefile:
            pickle_sample = pickle.load(picklefile)

        if device is None:
            device = next(self.model_trainer.flow.parameters()).device

        if isinstance(pickle_sample, Sequence):
            self.set_sample([torch.tensor(el).to(device) for el in pickle_sample])

        elif isinstance(pickle_sample, Mapping):
            self.set_sample([
                torch.tensor(pickle_sample[key]).to(device) for key in ["x", "px", "fx"]
            ])

        else:
            raise TypeError("Pickled sample must be either sequences or mappings")

    def set_sample_csv(self, csv_path, device=None, delimiter=",", dtype=np.float):
        """Assign a sample to be trained on from a csv file
        The file must contain equal length rows with at least four columns, all numerical.
        All columns but the last two are interpreted as point coordinates,
        the next-to-last is the point PDF and the last is the function value.

        Parameters
        ----------
        csv_path: str
            path to the csv file
        device: torch.device
            device to which to send the sample
        delimiter: str
            delimiter of the csv file
        """

        data = np.genfromtxt(csv_path, delimiter=delimiter, dtype=dtype)
        x = torch.tensor(data[:, :-2]).to(device)
        px = torch.tensor(data[:, -2]).to(device)
        fx = torch.tensor(data[:, -1]).to(device)

        self.set_sample((x, px, fx))

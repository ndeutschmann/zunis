"""Utilities for setting up PyTorch"""
import torch
import logging

logger = logging.getLogger(__name__)


def get_device(cuda_ID=0, needs_cuda=False):
    """Choose

    Parameters
    ----------
    cuda_ID: int
        if CUDA is available, use device with this ID
    needs_cuda: bool
        whether to raise an error if CUDA is not available

    Returns
    -------
        torch.device
    """
    if torch.has_cuda:
        device = torch.device(f"cuda:{cuda_ID}")
        logger.warning(f"Using CUDA:{cuda_ID}")
    else:
        if needs_cuda:
            raise RuntimeError("No CUDA device is available")
        device = torch.device("cpu")
        logger.warning(f"Using CPU")
    return device

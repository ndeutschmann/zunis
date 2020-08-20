"""Utilities for setting up PyTorch"""
import torch
import logging

logger = logging.getLogger(__name__)


def get_device(cuda_ID=0):
    """Choose

    Parameters
    ----------
    cuda_ID: int
        if CUDA is available, use device with this ID

    Returns
    -------
        torch.device
    """
    if torch.has_cuda:
        device = torch.device(f"cuda:{cuda_ID}")
        logger.warning(f"Using CUDA:{cuda_ID}")
    else:
        device = torch.device("cpu")
        logger.warning(f"Using CPU")
    return device

"""Tools to build out-of-the-box trainers"""
import torch
from .dkl_training import BasicStatefulDKLTrainer
from zunis.models.generators import create_hypercube_flow
from zunis.models.flows.sampling import FactorizedGaussianSampler


def create_dkl_trainer(d, model_type="realnvp", mask_type="checkerboard", model_params=None,
                       device=torch.device("cpu"), n_epochs=10, optim_class=None, lr=1.e-4, **params):
    """Create a dkl trainer with the a default flow model"""
    if model_params is None:
        model_params = dict()

    model = create_hypercube_flow(d, model_type=model_type, mask_type=mask_type, **model_params).to(device)
    prior = FactorizedGaussianSampler(d=d, device=device)

    if optim_class is None:
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optim = optim_class(model.parameters(),lr=lr)

    return BasicStatefulDKLTrainer(model, prior, optim=optim, n_epochs=n_epochs, **params)

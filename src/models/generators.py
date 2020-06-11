"""Tools to build out-of-the box models"""

import torch
from src.models.flows.sequential import InvertibleSequentialFlow
from src.models.flows.coupling_cells.real_nvp import RealNVP
from src.models.flows.analytic_flows.element_wise import InvertibleAnalyticSigmoid


def n_ary_mask(d, n, offset):
    return [(i + offset) % n == 0 for i in range(d)]


def create_n_ary_hypercube_realnvp(d, n=2, repetitions=2, d_hidden=256, n_hidden=16):
    layers = []
    for rep in range(repetitions):
        for i in range(n):
            mask = n_ary_mask(d, n, i)
            layers.append(
                RealNVP(d=d,
                        mask=mask,
                        d_hidden=d_hidden,
                        n_hidden=n_hidden, ),
            )
    layers.append(InvertibleAnalyticSigmoid(d=d))

    model = InvertibleSequentialFlow(d, layers)
    return model


def create_checkerboard_hypercube_realnvp(d, repetitions=2, d_hidden=256, n_hidden=16):
    return create_n_ary_hypercube_realnvp(d, 2, repetitions, d_hidden, n_hidden)


def create_hypercube_flow(d, model_type="realnvp", mask_type="checkerboard", **params):
    """Provide a high-level generator with default behavior"""

    if model_type == "realnvp":
        if mask_type == "checkerboard":
            return create_checkerboard_hypercube_realnvp(d, **params)
        if mask_type == "n_ary":
            return create_n_ary_hypercube_realnvp(d, **params)
        else:
            raise ValueError(f"Mask type {mask_type} does not exist")

    else:
        raise ValueError(f"Model type {model_type} does not exist")

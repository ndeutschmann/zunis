"""Tools to build out-of-the box models"""

from zunis.models.flows.masking import n_ary_mask
from zunis.models.flows.sequential.invertible_sequential import InvertibleSequentialFlow
from zunis.models.flows.coupling_cells.real_nvp import RealNVP
from zunis.models.flows.analytic_flows.element_wise import InvertibleAnalyticSigmoid


def create_n_ary_hypercube_flow(cell, d, n=2, repetitions=2, **cell_options):
    """Create a sequential normalizing flow on the d-dimensional unit hypercube based on a given
    coupling cell with a n-ary mask. Subsequent cells have masks offset by one at each step until
    all possible masks have been used.

    Parameters
    ----------
    cell:
        coupling cell class
    d: int
        dimension of the unit hypercube
    n:
        period of the n-ary mask
    repetitions:
        how many times a full repetition through all possible masks is repeated
    cell_options:
        options to be passed at the coupling cell instantiation

    Returns
    -------
    InvertibleSequentialFlow
    """
    layers = []
    for rep in range(repetitions):
        for i in range(n):
            mask = n_ary_mask(d, n, i)
            layers.append(
                cell(d=d,
                     mask=mask,
                     **cell_options
                     ),
            )
    layers.append(InvertibleAnalyticSigmoid(d=d))

    model = InvertibleSequentialFlow(d, layers)
    return model


def create_n_ary_hypercube_realnvp(d, n=2, repetitions=2, d_hidden=256, n_hidden=16):
    return create_n_ary_hypercube_flow(RealNVP, d, n=n, repetitions=repetitions, d_hidden=d_hidden, n_hidden=n_hidden)


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

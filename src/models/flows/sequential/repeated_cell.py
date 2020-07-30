from functools import partial

import torch

from .invertible_sequential import InvertibleSequentialFlow
from src.models.flows.coupling_cells.general_coupling import InvertibleCouplingCell
from src.models.flows.analytic_flows.analytic_flow import InvertibleAnalyticFlow
from ..masking import n_ary_mask_strategy

from src.models.flows.coupling_cells.real_nvp import RealNVP
from src.models.flows.coupling_cells.piecewise_coupling.piecewise_linear import PWLinearCoupling
from src.models.flows.analytic_flows.element_wise import InvertibleAnalyticSigmoid


class MaskListRepeatedCellFlow(InvertibleSequentialFlow):
    """Sequential normalizing flow based on repeating a single coupling cell type
    with each a different mask as a sequence of transformations"""

    def __init__(self, d, cell, masks, *, input_cell=None, output_cell=None, cell_params=None, input_cell_params=None,
                 output_cell_params=None):
        """

        Parameters
        ----------
        d: int
            dimensionality of the space
        cell:
            callable that yields an instance of src.models.flows.coupling_cells.general_coupling.InvertibleCouplingCell
            the coupling cell to be repeated.
        masks: list of lists of bool
            list of masks for each flow in the cell. The order is from latent space to target space
        input_flow: , None
            flow model to be used as an input. Typically to deform the input space to be suited to the repeated cell
        output_flow: , None
            flow model to be used as an output. Typically to deform the input space to be suited to the repeated cell
        cell_params: dict
            parameters of the cell
        input_cell_params: dict, None
            parameters of the input cell
        output_cell_params: dict
            parameters of the output cell
        """
        assert issubclass(cell, InvertibleCouplingCell), "The repeated cell must be an InvertibleCouplingCell"

        if cell_params is None:
            cell_params = dict()
        if input_cell_params is None:
            input_cell_params = dict()
        if output_cell_params is None:
            output_cell_params = dict()

        flows = []
        if input_cell is not None:
            assert isinstance(input_cell, InvertibleAnalyticFlow), "The input cell must be an InvertibleAnalyticFlow"
            flows.append(input_cell(**input_cell_params))

        for mask in masks:
            flows.append(cell(mask=mask, **cell_params))

        if output_cell is not None:
            assert isinstance(output_cell, InvertibleAnalyticFlow), "The output cell must be an InvertibleAnalyticFlow"
            flows.append(output_cell(**output_cell_params))

        super(MaskListRepeatedCellFlow, self).__init__(d=d, flows=flows)


class RepeatedCellFlow(MaskListRepeatedCellFlow):
    """Repeated cell flow: a sequential model with a coupling cell model
    This is meant as a higher-level API to generate boilerplate models based on architecture and masking strategy
    """
    cells = {"realnvp": (RealNVP, None, InvertibleAnalyticSigmoid),
             "pwlinear": (PWLinearCoupling, None, None)}

    masking = {
        "checkerboard": partial(n_ary_mask_strategy, n=2, repetitions=2)
    }

    def __init__(self, d, cell="pwlinear", masking="checkerboard", *, input_cell=None, output_cell=None,
                 cell_params=None, input_cell_params=None, output_cell_params=None, masking_options=None):
        """

        Parameters
        ----------
        d
        cell
        masking
        input_cell
        output_cell
        cell_params
        input_cell_params
        output_cell_params
        """
        cell, input_cell, output_cell = self.cells[cell]
        if masking_options is None:
            masking_options = dict()
        masks = self.masking[masking](d, **masking_options)
        super(RepeatedCellFlow, self).__init__(d, cell=self.cells, masks=masks,
                                               input_cell=input_cell, output_cell=output_cell,
                                               cell_params=cell_params, input_cell_params=input_cell_params,
                                               output_cell_params=output_cell_params)

import torch
from .invertible_sequential import InvertibleSequentialFlow
from src.models.flows.coupling_cells.general_coupling import InvertibleCouplingCell
from src.models.generators import n_ary_mask

from src.models.flows.coupling_cells.real_nvp import RealNVP
from src.models.flows.coupling_cells.piecewise_coupling.piecewise_linear import PWLinearCoupling


class RepeatedCellFlow(InvertibleSequentialFlow):
    """Sequential normalizing flow based on repeating a single coupling cell type
    with each a different mask as a sequence of transformations"""

    def __init__(self, d, cell, masks, **cell_params):
        assert issubclass(cell, InvertibleCouplingCell)
        flows = []
        for mask in masks:
            flows.append(cell(mask=mask, **cell_params))

        super(RepeatedCellFlow, self).__init__(d=d, flows=flows)


class NAryMaskRepeatedCellFlow(RepeatedCellFlow):
    """Sequential normalizing flow based on repeating a single coupling cell with a n-ary
    masking strategy:
    masks have every nth entry True (dimension transformed) and every other False (dimension used as a parameter
    for the transform). Each subsequent cell has the mask of the previous one offset by one position to the right.
    We make a number of repetitions of cycling through all such different masks.
    """

    def __init__(self, d, cell, n, repetitions, **cell_params):
        masks = []
        for rep in range(repetitions):
            for i in range(n):
                masks.append(n_ary_mask(d, n, i))
        super(NAryMaskRepeatedCellFlow, self).__init__(d, cell, masks, **cell_params)


class NAryMaskRealNVPFlow(NAryMaskRepeatedCellFlow):
    """Sequential normalizing flow based on repeating a RealNVP coupling cell with a n-ary
    masking strategy:
    masks have every nth entry True (dimension transformed) and every other False (dimension used as a parameter
    for the transform). Each subsequent cell has the mask of the previous one offset by one position to the right.
    We make a number of repetitions of cycling through all such different masks.
    """

    def __init__(self, d, n=2, repetitions=1, **cell_params):
        """

        Parameters
        ----------
        d: int
            dimension
        n: int
            period of the n-ary mask
        repetitions: int
            number of times to cycle through all possible n-ary masks
        cell_params:
            parameters for :py:class:`src.models.flows.coupling_cells.real_nvp.RealNVP`
        """
        super(NAryMaskRealNVPFlow, self).__init__(d, RealNVP, n, repetitions, **cell_params)


class NAryMaskPWLinearFlow(NAryMaskRepeatedCellFlow):
    """Sequential normalizing flow based on repeating a RealNVP coupling cell with a n-ary
    masking strategy:
    masks have every nth entry True (dimension transformed) and every other False (dimension used as a parameter
    for the transform). Each subsequent cell has the mask of the previous one offset by one position to the right.
    We make a number of repetitions of cycling through all such different masks.
    """

    def __init__(self, d, n=2, repetitions=1, **cell_params):
        """

        Parameters
        ----------
        d: int
            dimension
        n: int
            period of the n-ary mask
        repetitions: int
            number of times to cycle through all possible n-ary masks
        cell_params:
            parameters for :py:class:`src.models.flows.coupling_cells.piecewise_coupling.piecewise_linear.PWLinearCoupling`
        """
        super(NAryMaskPWLinearFlow, self).__init__(d, PWLinearCoupling, n, repetitions, **cell_params)

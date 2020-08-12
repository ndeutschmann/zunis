"""Module for the generation of masks for coupling cells"""


def n_ary_mask(d, n, offset):
    """Create a n-ary mask with n entries (a list of bool with each nth entry True)

    Parameters
    ----------
    d: int
        numbers of entries in the mask
    n: int
        period of the mask
    offset: int
        offset of the True entries

    Returns
    -------
    list of bool
        True/False mask with each nth entry True
    """
    return [(i + offset) % n == 0 for i in range(d)]


def n_ary_mask_strategy(d, n=2, repetitions=1):
    """Generate an list of masks using the n-ary periodic strategy: every nth entry is True and each next mask is
    offset by one position with respect to the previous one. We loop for a given number of repetitions through all
    possible masks.

    Parameters
    ----------
    d: int
        dimensionality of the space
    n: int
        periodicity of the mask
    repetitions: int
        number of iterations through all possible masks

    Returns
    -------
    list of list of bool
        list of masks
    """

    masks = []
    for rep in range(repetitions):
        for i in range(n):
            masks.append(n_ary_mask(d, n, i))

    return masks

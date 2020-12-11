"""Module for the generation of masks for coupling cells"""


def get_bin(x, n=0):
    """
    Get the binary representation of x.

    Parameters
    ----------
    x : int
    n : int
        Minimum number of digits. If x needs less digits in binary, the rest
        is filled with zeros.

    Returns
    -------
    list of binary digits
    """

    y = format(x, 'b').zfill(n)

    return [int(i) for i in str(y)]


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


def iflow_strategy(d, repetitions=1):
    """Generate a list of masks using the strategy of Gao et al. arXiv:2001.05486.
    Each dimension is numbered and the number is converted into the binary representation. Then,
    one bit after another, starting with the most significant bit, is converted into a boolean.
    This determines wether the dimension is transformed in this cell or not. In order to be 
    able to test for all correlations, for each mask, the inverse mask is used too. 
    

    The minimal amount of cells will be used to represent all possible correlations.

    Parameters
    ----------
    d: int
        number of dimensions

    Returns
    -------
    list of list of bool
        list of masks

    """

    n = len(get_bin(d - 1, 0))
    masks = [1] * 2 * n * repetitions
    dims = [int(i) for i in range(d)]

    dims_bin = [list(x) for x in zip(*list(map(get_bin, dims, [n] * d)))]
    j = 0
    for k in range(repetitions):
        for i in range(n):
            s = dims_bin[i][:]

            masks[j] = [bool(x) for x in s]

            masks[j + 1] = [not bool(x) for x in s]
            j = j + 2

    return (masks)


def maximal_masking_strategy(d, repetitions=1):
    """Generate a list of masks using the maximally complex option:
    (not really but within the reasonable options)

    Each variable is transformed using all the others as parameters, looping
    over all variables. The size of the model is therefore linear with the
    number of dimensions

    Parameters
    ----------
    d: int
        number of dimensions
    repetitions: int
        number of repetitions

    Returns
    -------

    """

    masks = [[False] * d for _ in range(d * repetitions)]
    for r in range(repetitions):
        for i in range(d):
            masks[i + r * d][i] = True
    return masks
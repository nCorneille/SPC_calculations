import numpy as np
from numba import njit


def get_first_element(conditioned_data):
    """
    :param conditioned_data: some condition on an np.array
    :return: the first element of condition_data which is True, or len(conditioned_data) if every element is False
    """
    if not np.any(conditioned_data):
        return len(conditioned_data)
    else:
        return np.argmax(conditioned_data)


def upper_CL_rule(data, UCL):
    """
    :param data: Set of observations
    :param UCL: Upper control limit
    :return: first index for which data[index] >= UCL
    """
    return get_first_element(data >= UCL)


def lower_CL_rule(data, LCL):
    """
    :param data: Set of observations
    :param LCL: Lower control limit
    :return: first index for which data[index] <= LCL
    """
    return get_first_element(data <= LCL)


def double_sided_CI_rule(data, LCL, UCL):
    """
    :param data: Set of observations
    :param LCL: Lower control limit
    :param UCL: Upper control limit
    :return: first index for which data[index] not in (LCL, UCL)
    """
    return get_first_element(np.logical_or(data <= LCL, data >= UCL))


@njit
def n_points_suffice_condition(conditioned_data: np.array(bool), n: int):
    """
    Based on code by Henry Shackleton (https://www.javaer101.com/en/article/1008280.html)
    Uses njit (numba) for optimization

    :param conditioned_data: Set of conditioned observations
    :param n: number of points
    :return: first index for which conditioned_data[index-6:index] are all True
    """

    out = len(conditioned_data)
    for i in range(out - n + 1):
        found = True
        for j in range(n):
            if not conditioned_data[i + j]:
                found = False
                break
            if found:
                #TODO: what is out exactly?
                out = i + n
                break

    return out

def n_points_above_CL(data: np.array(float), n: int, CL: float):
    """
    :param data: Set of observations
    :param n: number of points
    :return: first index for which data[index-6:index] are all >CL
    """
    return n_points_suffice_condition(data > CL, n)

def n_points_below_CL(data: np.array(float), n: int, CL: float):
    """
    :param data: Set of observations
    :param n: number of points
    :return: first index for which data[index-6:index] are all <CL
    """
    return n_points_suffice_condition(data < CL, n)


#TODO: increasing sequence; n_points_suffice_condition(np.diff...) + 1 ?

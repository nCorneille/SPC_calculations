import numpy as np


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
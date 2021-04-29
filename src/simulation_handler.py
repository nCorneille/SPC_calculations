from matrix_handler import MarkovChainModel
import numpy as np


def get_first_element(conditioned_data):
    '''
    :param conditioned_data: some condition on an np.array
    :return: the first element of condition_data which is True, or len(conditioned_data) if every element is False
    '''
    if not np.any(conditioned_data):
        return len(conditioned_data)
    else:
        return np.argmax(conditioned_data)


def upper_CL_rule(data, LCL, UCL, CL):
    '''
    :param data: Set of observations
    :param LCL: unused, only exists to make SimulationHandler.simulate(..) work
    :param UCL: Upper control limit
    :param CL: unused, only exists to make SimulationHandler.simulate(..) work
    :return: first index for which data[index] >= UCL
    '''
    return get_first_element(data >= UCL)


def lower_CL_rule(data, LCL, UCL, CL):
    '''
    :param data: Set of observations
    :param LCL: Lower control limit
    :param UCL: unused, only exists to make SimulationHandler.simulate(..) work
    :param CL: unused, only exists to make SimulationHandler.simulate(..) work
    :return: first index for which data[index] <= LCL
    '''
    return get_first_element(data <= LCL)


def double_sided_CI_rule(data, LCL, UCL, CL):
    '''
    :param data: Set of observations
    :param LCL: Lower control limit
    :param UCL: Upper control limit
    :param CL: unused, only exists to make SimulationHandler.simulate(..) work
    :return: first index for which data[index] not in (LCL, UCL)
    '''
    return get_first_element(np.logical_or(data <= LCL, data >= UCL))


class SimulationHandler:

    def __init__(self, stopping_rules, LCL: float = -3, UCL: float = 3, CL: float = 0):
        '''
        Class wrapper for handling simulations on control charts
        :param stopping_rules: rules which trigger an OC signal
        :param LCL: Lower control limit
        :param UCL: Upper control limit
        :param CL: Center line
        '''
        self.LCL = LCL
        self.UCL = UCL
        self.CL = CL
        self.stopping_rules = stopping_rules

    def simulate(self, sampling_distribution, max_iterations: int = 1000000):
        '''
        Samples max_iterations points from sampling_distribution and returns
        the minimum index for which an OC signal is given
        :param sampling_distribution: function which returns random samples from a distribution
        :param max_iterations: the maximum number of iterations
        :return: the index which gives an OC signal + 1
        '''
        data = sampling_distribution(max_iterations)
        n = []

        for rule in self.stopping_rules:
            n.append(rule(data=data, LCL=self.LCL, UCL=self.UCL, CL=self.CL))

        return np.min(n) + 1

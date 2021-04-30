import numpy as np
from numba import njit


class SimulationHandler:

    def __init__(self, sampling_distribution, stopping_rules, LCL: float = -3, UCL: float = 3, CL: float = 0):
        """
        Class wrapper for handling simulations on control charts.

        :param sampling_distribution: function which returns random samples from a distribution
        :param stopping_rules: rules which trigger an OC signal. This should be a list of functions which take in an
            np.array of data as argument. For example: sim = SimulationHandler(
            [lambda x: double_sided_CI_rule(x, sim.LCL, sim.UCL)] is allowed.
        :param LCL: Lower control limit
        :param UCL: Upper control limit
        :param CL: Center line
        """
        self.sampling_distribution = sampling_distribution
        self.LCL = LCL
        self.UCL = UCL
        self.CL = CL
        self.stopping_rules = stopping_rules


    def simulate(self, max_iterations: int = 1000000):
        """
        Samples max_iterations points from sampling_distribution and returns
        the minimum index for which an OC signal is given.

        :param max_iterations: the maximum number of iterations
        :return: the index which gives an OC signal + 1
        """
        data: np.array(float) = self.sampling_distribution(max_iterations)
        n = []

        for rule in self.stopping_rules:
            n.append(rule(data))

        return np.min(n) + 1

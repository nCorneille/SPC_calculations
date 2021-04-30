import numpy as np
from numba import njit


class RunsRules:
    @staticmethod
    def get_first_element(conditioned_data: np.array(bool)) -> int:
        """
        :param conditioned_data: some condition on an np.array
        :return: the first element of condition_data which is True, or len(conditioned_data) if every element is False
        """
        if not np.any(conditioned_data):
            return len(conditioned_data)
        else:
            return np.argmax(conditioned_data)

    @staticmethod
    @njit
    def get_first_sequence(conditioned_data: np.array(bool), n: int) -> int:
        """
        Generalizes get_first_element to sequences of consecutive conditions.
        Based on code by Henry Shackleton (https://www.javaer101.com/en/article/1008280.html)
        Uses njit (numba) for optimization; @njit makes the function ~15× faster at 1million data points.

        :param conditioned_data: Set of conditioned observations
        :param n: number of points
        :return: first index for which conditioned_data[index:index+n] are all True
        """

        out = len(conditioned_data)
        for i in range(out - n + 1):
            found = True
            for j in range(n):
                if not conditioned_data[i + j]:
                    found = False
                    break
            if found:
                out = i
                break

        return out

    @staticmethod
    def upper_CL_rule(data, UCL):
        """
        :param data: Set of observations
        :param UCL: Upper control limit
        :return: first index for which data[index] >= UCL
        """
        return RunsRules.get_first_element(data >= UCL)

    @staticmethod
    def lower_CL_rule(data, LCL):
        """
        :param data: Set of observations
        :param LCL: Lower control limit
        :return: first index for which data[index] <= LCL
        """
        return RunsRules.get_first_element(data <= LCL)

    @staticmethod
    def double_sided_CI_rule(data, LCL, UCL):
        """
        :param data: Set of observations
        :param LCL: Lower control limit
        :param UCL: Upper control limit
        :return: first index for which data[index] not in (LCL, UCL)
        """
        return RunsRules.get_first_element(np.logical_or(data <= LCL, data >= UCL))

    @staticmethod
    def n_points_above_CL(data: np.array(float), n: int, CL: float):
        """
        :param data: Set of observations
        :param n: number of points
        :param CL: Center line
        :return: first index for which data[index-6:index] are all >CL
        """
        return RunsRules.get_first_sequence(data > CL, n) + n

    @staticmethod
    def n_points_below_CL(data: np.array(float), n: int, CL: float):
        """
        :param data: Set of observations
        :param n: number of points
        :param CL: Center line
        :return: first index for which data[index-6:index] are all <CL
        """
        return RunsRules.get_first_sequence(data < CL, n) + n

    # TODO: check n_points_increasing and n_points_decreasing for correctness
    @staticmethod
    def n_points_increasing(data: np.array(float), n: int):
        diff = np.diff(data)
        return RunsRules.get_first_sequence(diff > 0, n - 1) + n

    @staticmethod
    def n_points_decreasing(data: np.array(float), n: int):
        diff = np.diff(data)
        return RunsRules.get_first_sequence(diff > 0, n - 1) + n



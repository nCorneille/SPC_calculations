import numpy as np
from scipy import optimize
import csv


def read_model(filename: str, params: dict) -> np.array:
    """
    Code based on function written by Mariia Turchina
    (https://github.com/thatmariia/stochastic-modelling/blob/master/model_python/ModelsSetup.py)

    Reads a matrix from .csv file
    :param filename: string - .csv location
    :param params: dictionary[string] - expresses elements of csv in terms of {mu, sigma, k}
    :return: square probability matrix with variable names
    """
    filename += ".csv"

    shape = (0, 0)
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            shape = (shape[0] + 1, shape[1])

            if len(row) > shape[1]:
                shape = (shape[0], len(row))

    assert shape[0] == shape[1], "The matrix isn't square"

    model_matrix = []

    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            model_row = []
            for cell in row:
                for key in params.keys():
                    cell = cell.replace(key, params[key])
                # model_cell = eval(cell, d)
                model_row.append(cell)
            model_matrix.append(np.array(model_row))

    return np.array(model_matrix)


def evaluate_model(matrix: np.array, params: dict) -> np.matrix:
    """
    Code based on function written by Mariia Turchina
    (https://github.com/thatmariia/stochastic-modelling/blob/master/model_python/ModelsSetup.py)

    Reads a matrix from .csv file
    :param matrix: np.array[np.array[string]] - matrix of variable names
    :param params: map from variable names to values
    :return: square probability matrix evaluated with values at params
    """

    model_matrix = []

    for row in matrix:
        model_row = []
        for cell in row:
            model_cell = eval(cell, params)
            model_row.append(model_cell)
        model_matrix.append(np.array(model_row))

    return np.asmatrix(np.array(model_matrix))


class MarkovChainModel:
    def __init__(self, matrix_path, params, values):
        self.params = params
        self.values = values
        self.matrix = read_model(matrix_path, params)
        self.dim = self.matrix.shape[0]

    def calculate_values(self, k: float) -> dict:
        return dict({"k": k}, **self.values.copy())

    def calculate_ARL(self, k: float) -> float:
        """
        Calculates the ARL for a given k
        :param k: k such that the OC interval is R - [mu - k sigma, mu + k sigma]
        :return: Average Run Length for given k
        """
        identity_minus_matrix_inv = np.linalg.inv(
            np.identity(self.dim) - evaluate_model(self.matrix, self.calculate_values(k)))

        return (np.eye(1, self.dim, 0) @ identity_minus_matrix_inv @ np.ones(self.dim)).item()

    def find_control_limit(self, target_ARL: float, epsilon: float = 0.001) -> float:
        """
        Find a value k for which calculate_ARL is approximately target_ARL
        :param target_ARL: The value we wish for calculate_ARL to have
        :param epsilon: Tolerance of the root-finding
        :return: A floating point value such that |target_ARL - calculate_ARL| < epsilon
        """
        output = optimize.root_scalar(lambda k: self.calculate_ARL(k) - target_ARL,
                                      bracket=[0.2, 5], x0=3, xtol=epsilon)

        return output.root

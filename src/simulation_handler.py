from matrix_handler import MarkovChainModel
import numpy as np


class SimulationHandler:

    def __init__(self, control_limits_rule, runs_rules, LCL: float = -3, UCL: float = 3):
        self.LCL = LCL
        self.UCL = UCL
        self.runs_rules = runs_rules
        self.control_limits_rule = control_limits_rule

    def determine_LCL(self, markov_chain: MarkovChainModel, ARL: float) -> None:
        self.LCL = markov_chain.calculate_ARL(ARL)

    def simulate(self, sampling_distribution, max_iterations: int = 1000000):
        random_variables = sampling_distribution(max_iterations)

        IC_OC_array = self.control_limits_rule(random_variables, self.LCL, self.UCL)
        first_OC = np.argmax(IC_OC_array)

        if first_OC == 0 and not IC_OC_array[0]:
            first_OC = max_iterations - 1

        n = [first_OC]

        if not self.runs_rules:
            return n[0]+1

        for rule in self.runs_rules:
            for i in range(max_iterations - rule[0]):
                if np.all(rule[1](random_variables[i:i + rule[0]])):
                    n.append(i + rule[0])
                    break

        return np.min(n)+1

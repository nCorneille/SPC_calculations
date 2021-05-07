import os
import timeit

import matplotlib.pyplot as plt

from scipy import stats
from scipy import optimize

from src.simulation_handler import *
from src.matrix_handler import MarkovChainModel
from src.runs_rules import *


def main():
    prev_dir = os.path.dirname(__file__) + "/../"

    C2_model = MarkovChainModel(prev_dir + "matrix_C2",
                                {"p0": "phi(2)-phi(-2)", "p1": "phi(k)-phi(2)", "p2": "phi(-2)-phi(-k)"},
                                {"phi": stats.norm.cdf})

    test_model = MarkovChainModel(prev_dir + "matrix_test",
                                  {"p0": "phi(k)-phi(-k)"},
                                  {"phi": stats.norm.cdf})

    p = 0.05
    r = 1
    CCC_model = MarkovChainModel(prev_dir + "matrix_CCC_no-trends",
                                 # TODO: cdf(k) should be cdf(LCL), with LCL = f(k)
                                 {"p(X=m)": "pmf(median)", "p(X>m)": "1-cdf(median)",
                                  "p(l<X<m)": "cdf(median-1)-cdf(k)"},
                                 {"median": 3, "pmf": lambda k: stats.nbinom.pmf(k, r, p),
                                  "cdf": lambda k: stats.nbinom.cdf(k, r, p)})

    print("C2 ARL: {}".format(C2_model.calculate_ARL(3)))
    print("no RR ARL: {}".format(test_model.calculate_ARL(3)))
    print("CCC model no trends: {}".format(CCC_model.calculate_ARL(3)))

    # using properties of test_sim inside the definition of test_sim is allowed because of
    # pass by reference lambda body magic (in Python it is safe to assume objects behave like pointers)
    test_sim = SimulationHandler(lambda x: np.random.normal(0, 1, x),
                                 [lambda x: RunsRules.double_sided_CI_rule(x, test_sim.LCL, test_sim.UCL)],
                                 -3, 3)

    n = []
    for i in range(20000):
        n.append(test_sim.simulate_run_length(10000))

    print(np.mean(n))


def runs_rule_test():
    n = 1000000
    m = 1
    for _ in range(m):
        data: np.array(float) = np.random.normal(0, 1, n)

        n_u = RunsRules.n_points_above_CL(data, 9, 0.5)
        n_l = RunsRules.n_points_below_CL(data, 9, -0.5)

        return min(n_u, n_l)


def simulation_CCC_charts():
    r = 4
    p = 0.005
    p_tilde = 0.008
    changepoint = -1

    median = 0
    while stats.nbinom.cdf(median, r, p) - 1 / 2 < 0:
        median += 1

    sd = np.sqrt((1-p)*r)/p
    #print(median)
    #print(sd)

    def sampling_distr(x,p_): return np.random.negative_binomial(r, p_, x)

    LCL = max(0, median - 2 * sd)
    sim = SimulationHandler(lambda x: sampling_distr(x, p),
                            [lambda x: RunsRules.n_points_above_CL(x, 9, median),
                             lambda x: RunsRules.n_points_below_CL(x, 9, median),
                             lambda x: RunsRules.n_points_increasing(x, 6),
                             lambda x: RunsRules.n_points_decreasing(x, 6),
                             lambda x: RunsRules.lower_CL_rule(x, LCL)])


    def simulate(_LCL):
        out = []
        for i in range(1000):
            out.append(sim.simulate_run_length_variable_rules([lambda x: RunsRules.n_points_above_CL(x, 9, median),
                                                               lambda x: RunsRules.n_points_below_CL(x, 9, median),
                                                               lambda x: RunsRules.n_points_increasing(x, 6),
                                                               lambda x: RunsRules.n_points_decreasing(x, 6),
                                                               lambda x: RunsRules.lower_CL_rule(x, _LCL)],
                                                              changepoint, lambda x: sampling_distr(x, p_tilde)))
        return np.mean(np.array(out))

    print(simulate(LCL))
    output = optimize.root_scalar(lambda k: simulate(median - k * sd) - 230,
                         bracket=[0, 3], x0=1.5, xtol=0.01)

    print(output.root)
    print(simulate(output.root))
    #density = stats.gaussian_kde(out)
    #x = np.linspace(0, 2000, 5000)

    #plt.plot(x, density(x))
    #plt.show()


if __name__ == "__main__":
    #print(timeit.timeit(runs_rule_test, number=1))
    simulation_CCC_charts()

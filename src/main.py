import os

from scipy import stats

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

    # using properties of test_simulation inside the definition of test_simulation is allowed because of
    # pass by reference lambda body magic (in Python it is safe to assume objects behave like pointers)
    test_simulation = SimulationHandler([lambda x: double_sided_CI_rule(x, test_simulation.LCL, test_simulation.UCL)],
                                        -3, 3)

    n = []
    for i in range(20000):
        n.append(test_simulation.simulate(lambda x: np.random.normal(0, 1, x), 10000))

    print(np.mean(n))


if __name__ == "__main__":
    main()

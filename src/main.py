import os
from scipy import stats

from src.matrix_handler import MarkovChainModel


def main():
    prev_dir = os.path.dirname(__file__) + "/../"

    C2_model = MarkovChainModel(prev_dir + "matrix_C2",
                                {"p0": "phi(2)-phi(-2)", "p1": "phi(k)-phi(2)", "p2": "phi(-2)-phi(-k)"},
                                {"phi": stats.norm.cdf})

    test_model = MarkovChainModel(prev_dir + "matrix_test",
                                  {"p0": "phi(k)-phi(-k)"},
                                  {"phi": stats.norm.cdf})

    print("C2 ARL: {}".format(C2_model.calculate_ARL(3)))
    print("no RR ARL: {}".format(test_model.calculate_ARL(3)))


if __name__ == "__main__":
    main()
# a collection of payoff function
import sympy as sp
import numpy as np
from sympy import Interval
import matplotlib.pyplot as plt

from helper_methods import deprecated


@deprecated
class PayoffFunc:

    def __init__(self):
        self.list = ['Inf', 'Sup', 'limSup', 'limInf', 'lower_MP', 'upper_MP']

    @staticmethod
    def Inf(partial_play):
        return min(partial_play)

    @staticmethod
    def Sup(partial_play):
        return max(partial_play)

    @staticmethod
    def LimSup(partial_play):
        value = []
        for n in range(len(partial_play)):
            value.append(max(list(partial_play)[n:]))
        return min(value)

    @staticmethod
    def LimInf(partial_play):
        value = []
        for n in range(len(partial_play)):
            value.append(min(list(partial_play)[n:]))
        return max(value)

    @staticmethod
    def lower_Mp(partial_play):
        value = []

        for i in range(1, len(partial_play)):
            value.append(PayoffFunc.LimInf(partial_play)/i)
        return value[-1]

    @staticmethod
    def higher_MP(partial_play):
        value = []

        for i in range(1, len(partial_play)):
            value.append(PayoffFunc.LimSup(partial_play) / i)
        return value[-1]

    def get_list_of_func(self):
        return self.list

    def get_no_of_func(self):
        return len(self.list)


@deprecated
def main():
    # test payoff functions

    partial_play = set((1, 2, 3, 5))
    np.random.seed(100)
    result = []
    for i in range(10, 100):
        partial_play = np.random.randint(1, 11, i)
        MP_l = PayoffFunc.lower_Mp(partial_play.tolist())
        MP_h = PayoffFunc.higher_MP(partial_play.tolist())

        result.append([MP_h, MP_l])

    plt.plot(result)
    plt.show()
    # print(PayoffFunc.Sup(partial_play))
    # print(PayoffFunc.LimSup(partial_play.tolist()))
    # print(PayoffFunc.LimInf(partial_play.tolist()))


if __name__ == "__main__":
    main()
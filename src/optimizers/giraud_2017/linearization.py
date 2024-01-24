from scipy import optimize as scipy_opt
import matplotlib.pyplot as plt
import numpy as np


class Linearization(object):
    # linearization of f(dP_c) = 1/sqrt(dP_c)
    # see 'https://songhuiming.github.io/pages/2015/09/22/piecewise-linear-function-and-the-explanation/' for explanation
    def __init__(self, DP_C_MIN, DP_C_MAX, L):
        self.x = np.linspace(DP_C_MIN, DP_C_MAX, 100)
        self.y = 1 / np.sqrt(self.x)

        x_init = np.linspace(np.sqrt(DP_C_MIN), np.sqrt(DP_C_MAX), L + 1) ** 2
        y_init = 1 / np.sqrt(x_init)

        self.p_init = np.append(x_init, y_init)
        self.p, e = scipy_opt.curve_fit(
            self.piecewise_linear, self.x, self.y, p0=self.p_init
        )

        forcing_idx = np.array([0, L, L + 1, 2 * L + 1])
        self.p[forcing_idx] = self.p_init[forcing_idx]
        assert np.all(
            np.diff(self.p[: L + 1]) > 0
        ), "x points are not strickly increasing"

    def piecewise_linear(
        self,
        x,
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        y0,
        y1,
        y2,
        y3,
        y4,
        y5,
    ):
        X = [x0, x1, x2, x3, x4, x5]
        Y = [y0, y1, y2, y3, y4, y5]
        condlist = []
        for i in range(len(X) - 1):
            if i == 0:
                condlist.append(x < X[i + 1])
            elif i == len(X) - 2:
                condlist.append(x >= X[i])
            else:
                condlist.append((x >= X[i]) & (x < X[i + 1]))

        K = [(Y[i] - Y[i + 1]) / (X[i] - X[i + 1]) for i in range(len(X) - 1)]
        B = [Y[i] - K[i] * X[i] for i in range(len(X) - 1)]

        def makeFunc(k, b):
            return lambda x: k * x + b

        funclist = [makeFunc(K[i], B[i]) for i in range(len(X) - 1)]
        return np.piecewise(x, condlist, funclist)

    def calculate(self, x):
        return self.piecewise_linear(x, *self.p)

    def validate_plot(self, DP_C_MIN, DP_C_MAX):
        xd = np.linspace(DP_C_MIN, DP_C_MAX, 1000)
        plt.plot(self.x, self.y, "o")
        plt.plot(xd, self.piecewise_linear(xd, *self.p))
        print(self.p)
        plt.show()


if __name__ == "__main__":
    lin = Linearization(100, 100000, 5)
    lin.validate_plot(100, 100000)

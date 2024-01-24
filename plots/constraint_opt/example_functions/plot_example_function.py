import numpy as np
import math
import matplotlib.pyplot as plt


def function(xx, q):
    return xx + pow((math.exp(xx) - q), 2)


def plot_heat_demand_example(start, end, initializations, colors, q):
    x = list(np.linspace(start, end, 10000))
    y = [function(xx, q) for xx in x]

    plt.plot(x, y)
    for index, initialization in enumerate(initializations):
        plt.scatter(
            initialization,
            function(initialization, q),
            color=colors[index],
            label="initialization {}".format(initialization),
        )
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"$y=x+(e^x-{})^2$".format(q))
    plt.savefig("High heat demand function example.png")
    plt.show()


plot_heat_demand_example(
    start=-10,
    end=3.5,
    initializations=[3.5, 3.25, 3, 2, -2, -8],
    colors=["#00FFFF", "r", "g", "y", "k", "c"],
    q=16,
)

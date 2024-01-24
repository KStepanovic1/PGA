import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from .icnn_convex_f_inference import neurons_ext

"""
This file is used to plot mean squared error of PLNN and ICNN when predicting 
specified convex function. On x-axis is the size of the neural network
(number of neurons in each hidden layer).
"""

def calculate_plot_bounds(mean, deviation):
    """
    Calculate upper and lower bounds for the plot.
    """
    up_bound, down_bound = [], []
    for i in range(len(mean)):
        up_bound.append(mean[i] + deviation[i])
        down_bound.append(mean[i] - deviation[i])
    return up_bound, down_bound


if __name__ == "__main__":
    layer_sizes = [[1, 1], [1, 1, 1], [2, 1], [3, 1], [5, 1], [10, 10, 10, 1], [100, 100, 100, 1]]
    result_p: Path = Path(__file__).parents[1] / "2D_convex_function"
    networks = ["icnn", "plnn"]
    mse = {"icnn": [], "plnn": []}
    std = {"icnn": [], "plnn": []}
    for network in networks:
        for layer_size in layer_sizes:
            data = list(np.array(
                pd.read_csv(
                    result_p.joinpath(
                        network + "_mse_" + neurons_ext(layer_size) + ".csv"
                    ),
                )
            ))
            data = [i[1:] for i in data]
            data = np.array(data)
            mse[network].append(np.mean(data))
            std[network].append(np.std(data))
    fig, ax = plt.subplots()
    up_bound_plnn, down_bound_plnn = calculate_plot_bounds(mse["plnn"], std["plnn"])
    up_bound_icnn, down_bound_icnn = calculate_plot_bounds(mse["icnn"], std["icnn"])
    x = list(range(len(layer_sizes)))
    ax.plot(
        x,
        mse["plnn"],
        color="b",
        label="PLNN",
    )
    ax.fill_between(
        x,
        down_bound_plnn,
        up_bound_plnn,
        color="b",
        alpha=0.1,
    )
    ax.plot(
        x,
        mse["icnn"],
        color="r",
        label="ICNN",
    )
    ax.fill_between(
        x,
        down_bound_icnn,
        up_bound_icnn,
        color="r",
        alpha=0.1,
    )
    ax.set_title("MSE of PLNN and ICNN")
    ax.legend()
    ax.set_xlabel("Neural network size")
    ax.set_ylabel("Mean squared error")
    ax.set_xticks(x)
    ax.set_xticklabels(
        (
            "$[1]$",
            "$[1,1]$",
            "$[2]$",
            "$[3]$",
            "$[5]$",
            "$[10,10,10]$",
            "$[100,100,100]$"
        )
    )
    plt.show()

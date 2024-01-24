try:
    import pickle5 as pickle
except:
    import pickle
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd


def plot(heat_continuous, heat_discrete, el_continuous, el_discrete) -> None:
    fig, ax = plt.subplots()
    title = "Heat demand real and discrete data"
    ax.set_xlabel("Heat demand [MWh]")
    ax.set_ylabel("Density")
    ax.set_title(title)
    sns.kdeplot(heat_continuous, linewidth=3, color="g", shade=True)
    y_max = max(ax.get_lines()[0].get_data()[1])
    heat_demand_rescale = [
        i / (max(list(heat_discrete.values())) / y_max)
        for i in list(heat_demand_discrete.values())
    ]
    plt.bar([5, 20, 40], heat_demand_rescale, width=[10, 20, 70])
    ax.set_xticks([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.get_xticklabels()[1].set_color("blue")
    ax.get_xticklabels()[3].set_color("blue")
    ax.get_xticklabels()[5].set_color("blue")
    ax.legend(["Continuous data", "Discrete data"])
    plt.show()

    fig, ax = plt.subplots()
    title = "Electricity price real and discrete data"
    ax.set_xlabel("Electricity_price [e/MWh]")
    ax.set_ylabel("Density")
    ax.set_title(title)
    sns.kdeplot(el_continuous, linewidth=3, color="g", shade=True)
    y_max = max(ax.get_lines()[0].get_data()[1])
    electricity_price_rescale = [
        i / (max(list(el_discrete.values())) / y_max)
        for i in list(electricity_price_discrete.values())
    ]
    plt.bar([20, 40, 60], electricity_price_rescale, width=[25, 30, 80])
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180])
    ax.get_xticklabels()[2].set_color("blue")
    ax.get_xticklabels()[4].set_color("blue")
    ax.get_xticklabels()[6].set_color("blue")
    ax.legend(["Continuous data", "Discrete data"])
    plt.show()


def count(data, disc):
    dict = {}
    for i in disc:
        s = np.count_nonzero(data == str(i)) / len(data)
        dict[str(i)] = s
    return dict


if __name__ == "__main__":
    continuous_data = np.array(pd.read_csv("processed_data.csv", header=None))
    continuous_data = np.delete(continuous_data, 0, 0)
    discrete_data = np.array(pd.read_csv("processed_data_discrete.csv", header=None))
    discrete_data = np.delete(discrete_data, 0, 0)
    heat_demand_continuous = [float(i) for i in continuous_data[:, 3]]
    electricity_price_continuous = [float(i) for i in continuous_data[:, 2]]
    heat_demand_discrete = count(discrete_data[:, 3], [5, 20, 40])
    heat_demand_max = max(list(heat_demand_discrete.values()))
    electricity_price_discrete = count(discrete_data[:, 2], [20, 40, 60])
    el_price_max = max(list(electricity_price_discrete.values()))
    plot(
        heat_demand_continuous,
        heat_demand_discrete,
        electricity_price_continuous,
        electricity_price_discrete,
    )

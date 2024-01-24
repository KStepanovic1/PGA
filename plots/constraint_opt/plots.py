import cv2
import gzip
import os
import numpy as np
import pandas as pd
import seaborn
import re
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pathlib import Path
from util import config
from util.shared import *
from util.config import (
    GridProperties,
    PhysicalProperties,
    TimeParameters,
    CHPPreset1,
    PipePreset1,
)


def reformat_outlet_inlet_temp(out_temp, in_temp, time_delay):
    out_temp_, in_temp_, time_delay_ = [], [], []
    N = len(out_temp)
    for i in range(N):
        in_temp__ = string_to_list(in_temp[i])
        time_delay__ = string_to_list(time_delay[i])
        for j in range(len(in_temp__)):
            out_temp_.append(out_temp[i])
            in_temp_.append(in_temp__[j])
            time_delay_.append(time_delay__[j])
    return out_temp_, in_temp_, time_delay_


def plot_text_and_dependence(ax, start_time, N, defining_indices, x, y):
    """
    Plot time periods beside scatter dots, and color in cyan dots representing dependence
    between supply outlet and supply inlet temperatures
    """
    for i in range(N):
        index = start_time + i
        ax.text(
            x[index],
            y[index],
            str(index),
        )
        if index in defining_indices:
            ax.scatter(
                x[index],
                y[index],
                c="c",
                marker="*",
            )


def string_to_list(x):
    """
    Transforms string of shape [x, y, z,...] into list of integers.
    """
    y = []
    start_index = 1  # index 0 is reserved for [
    end_index = len(x) - 1
    if "," not in x:
        y.append(float(x[start_index:end_index]))
    else:
        for m in re.finditer(",", x):
            y.append(float(x[start_index : m.start(0)]))
            start_index = m.start(0) + 2
        y.append(float(x[start_index:end_index]))
    return y


def plot(start_time, N):
    """
    Plot scatter figures as specified in global variable plots
    """
    paths = {
        "Results": Path(__file__).parents[2] / "results",
        "Plots": Path(__file__).parents[2] / "plots",
    }
    supply_time_delays_l = []
    for case in cases.values():
        name = (
            "data_num_{}_".format(GridProperties["ConsumerNum"])
            + "heat_demand_real_world"
            + "_for_L = {}".format(PipePreset1["Length"])
            + "_time_interval_3600_max_Q_70MW_deltaT_3C"
            + ".csv"
        )
        read_path = os.path.join(paths["Results"] / "constraint_opt", name)
        store_path = os.path.join(paths["Plots"] / "constraint_opt", case)
        with open(
            read_path,
            "rb",
        ) as f:
            data = pd.read_csv(f)
        for i in range(GridProperties["ConsumerNum"]):
            heat_demand = data["Heat demand {}".format(i + 1)]
            underdelivered_heat = data["Underdelivered heat {}".format(i + 1)]
            supply_in_temp = data["Supply in temp {}".format(i + 1)]
            supply_out_temp = data["Supply out temp {}".format(i + 1)]
            supply_mass_flow = data["Supply mass flow {}".format(i + 1)]
            supply_time_delay = data["Supply time delay {}".format(i + 1)]
            supply_in_temps_delay = data["Supply in temps-time delay {}".format(i + 1)]
            return_in_temp = data["Ret in temp {}".format(i + 1)]
            return_out_temp = data["Ret out temp {}".format(i + 1)]

            mean, variance = np.mean(heat_demand), np.var(heat_demand)
            print("Heat demand mean is {}".format(mean))
            print("Heat demand variance is {}".format(variance))
            """
            # plot supply inlet-outlet temperature
            plt.scatter(supply_in_temp, supply_out_temp, marker="*")
            plt.xlabel("Supply inlet temp [C]")
            plt.ylabel("Supply outlet temp [C]")
            plt.title(plots["sup-temp"])
            plt.savefig(os.path.join(store_path, plots["sup-temp"] + ".png"))
            plt.show()

            # plot heat demand-mass flow
            plt.scatter(heat_demand, supply_mass_flow, marker="*")
            plt.xlabel("Heat demand [MW]")
            plt.ylabel("Supply mass flow [kg/s]")
            plt.title(plots["q-m"])
            plt.savefig(os.path.join(store_path, plots["q-m"] + ".png"))
            plt.show()

            # plot supply outlet-return inlet temperature
            plt.scatter(supply_out_temp, return_in_temp, marker="*")
            plt.xlabel("Supply outlet temp [C]")
            plt.ylabel("Return inlet temp [C]")
            plt.title(plots["hes-temp"])
            plt.savefig(os.path.join(store_path, plots["hes-temp"] + ".png"))
            plt.show()

            # plot underdelivered heat demand
            plt.scatter(
                list(range(len(underdelivered_heat))), underdelivered_heat, marker="*"
            )
            plt.xlabel("Time steps [h]")
            plt.ylabel("Underdelivered heat [MWh]")
            plt.title(plots["q-viol"])
            plt.savefig(os.path.join(store_path, plots["q-viol"] + ".png"))
            plt.show()

            # plot supply outlet temperature changes
            supply_out_delta = []
            for i in range(len(supply_out_temp) - 1):
                supply_out_delta.append(supply_out_temp[i + 1] - supply_out_temp[i])
            plt.scatter(
                list(range(len(supply_out_delta))), supply_out_delta, marker="*"
            )
            plt.xlabel("Time steps [h]")
            plt.ylabel("Supply network outlet temperature [C]")
            plt.title(plots["s-temp-out"])
            plt.savefig(os.path.join(store_path, plots["s-temp-out"] + ".png"))
            plt.show()

            # plot return inlet-outlet temperature
            plt.scatter(return_in_temp, return_out_temp, marker="*")
            plt.xlabel("Return inlet temp [C]")
            plt.ylabel("Return outlet temp [C]")
            plt.title(plots["ret-temp"])
            plt.savefig(os.path.join(store_path, plots["ret-temp"] + ".png"))
            plt.show()

            plot outlet temperature at time-step t dependence on the last N inlet temperatures and mass flows
            from previous N time steps t-N, t-N+1,...,t
            The relation we are trying to approximate is also influenced by variables not plotted here:
            heat demand, produced heat, processes in the HES and return pipe.
            supply_in_temp_ = supply_in_temp[start_time : start_time + N]
            supply_mass_flow_ = supply_mass_flow[start_time : start_time + N]
            supply_out_temp_ = supply_out_temp[start_time + N - 1]
            # how many time-steps it takes from inlet temperatures to arrive at the outlet of the pipe?
            supply_time_delay_ = string_to_list(supply_time_delay[start_time + N - 1])
            # inlet temperature from which time-steps define outlet temperatures
            defining_indices = []
            for x in supply_time_delay_:
                defining_indices.append(start_time + N - 1 - x)
            fig, ax = plt.subplots()
            ax.scatter(supply_in_temp_, supply_mass_flow_, c="blue", marker="*")
            plot_text_and_dependence(
                ax,
                start_time,
                N,
                defining_indices,
                x=supply_in_temp_,
                y=supply_mass_flow_,
            )
            ax.scatter(
                supply_out_temp_, min(supply_mass_flow_), c="red", marker="o", s=50
            )
            ax.text(supply_out_temp_, min(supply_mass_flow_), str(start_time + N - 1))
            ax.set_xlabel("Supply inlet temp [C]")
            ax.set_ylabel("Mass flow [kg/s]")
            ax.set_title(plots["sup-temp-conv"])
            ax.legend(["Inlet temp", "Inlet-outlet dependence"])
            fig.savefig(
                os.path.join(
                    store_path,
                    plots["sup-temp-conv"] + ".png",
                )
            )
            plt.show()

            out_temp_r, in_temp_r, time_delay_r = reformat_outlet_inlet_temp(
                out_temp=supply_out_temp,
                in_temp=supply_in_temps_delay,
                time_delay=supply_time_delay,
            )
            

            # plot 3D plot - supply outlet, supply inlet temperature and defining time delays
            fig = plt.figure(figsize=(10, 7))
            ax = plt.axes(projection="3d")
            ax.scatter3D(out_temp_r, in_temp_r, time_delay_r, color="green")
            ax.set_xlabel("Supply outlet temp [C]")
            ax.set_ylabel("Supply inlet temp [C]")
            ax.set_zlabel("Time delay [h]")
            ax.set_title(plots["sup-temp-time-delay"])
            fig.savefig(
                os.path.join(
                    store_path,
                    plots["sup-temp-time-delay"] + ".png",
                )
            )
            plt.show()

            # plot supply inlet-outlet temperatures as the function of time delays
            time_delays_unique = list(set(time_delay_r))
            for time_delay in time_delays_unique:
                title = (
                    "Supply in-out temp for L={len} km and time-delay {delay} h".format(
                        len=PipePreset1["Length"] / 1000, delay=time_delay
                    )
                )
                indices = [i for i, x in enumerate(time_delay_r) if x == time_delay]
                in_temp = [x for i, x in enumerate(in_temp_r) if i in indices]
                out_temp = [x for i, x in enumerate(out_temp_r) if i in indices]
                plt.scatter(in_temp, out_temp, marker="*")
                plt.xlabel("Supply inlet temp [C]")
                plt.ylabel("Supply outlet temp [C]")
                plt.title(title)
                plt.savefig(os.path.join(store_path, title + ".png"))
                plt.show()
            """
            # plot time delays data distribution
            for time_delay in supply_time_delay:
                supply_time_delays_l.extend(string_to_list(time_delay))
            seaborn.kdeplot(supply_time_delays_l)
            plt.show()


if __name__ == "__main__":
    start_time = 1000
    N = 24
    plots = {
        "sup-temp": "Supply inlet-outlet temp for L = {} km".format(
            PipePreset1["Length"] / 1000
        ),
        "q-m": "Heat demand-mass flow L = {} km".format(PipePreset1["Length"] / 1000),
        "hes-temp": "Supply outlet-return inlet temp for L = {} km".format(
            PipePreset1["Length"] / 1000
        ),
        "q-viol": "Violation of the heat demand for L={} km".format(
            PipePreset1["Length"] / 1000
        ),
        "s-temp-out": "Supply outlet temperature changes for L={} km".format(
            PipePreset1["Length"] / 1000
        ),
        "ret-temp": "Return inlet-outlet temp for L = {} km".format(
            PipePreset1["Length"] / 1000
        ),
        "sup-temp-conv": "N={stime} supply inlet-outlet temp for L={len} km".format(
            stime=start_time, len=PipePreset1["Length"] / 1000
        ),
        "sup-temp-time-delay": "Supply outlet temperature-inlet temperature dependence on time delays for L={len} km".format(
            len=PipePreset1["Length"] / 1000
        ),
    }
    plot(start_time=start_time, N=N)

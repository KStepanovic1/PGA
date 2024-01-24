import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from src.util.config import PipePreset1, TimeParameters
from src.optimizers.constraint_opt.dhn_nn.plot import Plot
from src.optimizers.constraint_opt.dhn_nn.param import opt_steps

plt.rcParams["hatch.color"] = "#262626"


@dataclass
class DataPredictions:
    def __init__(self):
        self.data = {
            "Supply_inlet_temp": [],
            "Supply_outlet_temp": [],
            "Mass_flow": [],
            "Delivered_heat": [],
        }


class PlotResult(Plot):
    def __init__(self, result_p, plot_p):
        self.day: int = 24
        self.result_p: Path = result_p
        self.plot_p: Path = plot_p
        self.result_plnn_milp_p: Path = result_p.joinpath("plnn_milp")
        self.result_icnn_gd_p: Path = result_p.joinpath("icnn_gd")
        self.result_plnn_gd_p: Path = result_p.joinpath("plnn_gd")
        self.result_basic_strategy_p: Path = result_p.joinpath("basic_strategy")
        self.parent_p: Path = Path(__file__).parent
        self.colors: dict = {
            "MINLP": "#006600",
            "PLNN-MILP": "#0000cc",
            "ICNN-GD": "#b30000",
            # "ICNN-GD-g": "#b300b3",
            "PLNN-GD": "#cc5200",
            "BS": "#737373",
        }
        self.viol_lab: list = [
            "Supply inlet temperature",
            "Supply outlet temperature",
            "Delivered heat",
            "Mass flow",
        ]
        self.N: int = len(opt_steps) * 24
        self.nlmip_profit = pickle.load(
            open(
                Path(__file__).parents[4]
                / "results/li_2016/li_2016_day_ahead/objective_function_sum_L_12000_.pickle",
                "rb",
            )
        )
        self.rl_profit = pickle.load(
            open(
                Path(__file__).parents[4]
                / "results/rl_full_state/rl_full_state/objective_function_sum_L_12000_.pickle",
                "rb",
            )
        )
        self.nlmip_violations = pickle.load(
            open(
                Path(__file__).parents[4]
                / "results/li_2016/li_2016_day_ahead_sim/percentage_violation_ep_all_L_12000_",
                "rb",
            )
        )
        self.rl_violations = pickle.load(
            open(
                Path(__file__).parents[4]
                / "results/rl_full_state/rl_full_state_sim/percentage_violation_ep_all_L_12000_Q_0_",
                "rb",
            )
        )

    @staticmethod
    def plot_mark(ax, x, y, color, mark):
        """
        Plot mark on existing figure.
        """
        for i in range(len(x)):
            ax.plot(x[i], y[i], color, marker=mark, markersize=5)
        return ax

    def calculate_sum(self):
        """
        Calculate sum of profit and produced heat for all days.
        """
        profit = {
            "PLNN-MILP": [],
            "ICNN-GD": [],
            "ICNN-GD-g": [],
            "PLNN-GD": [],
            "Basic strategy": [],
            "MINLP": [],
            "RL": [],
        }
        produced_heat = {
            "PLNN-MILP": [],
            "ICNN-GD": [],
            "PLNN-GD": [],
            "Basic strategy": [],
        }
        for opt_step in opt_steps:
            plnn_milp = pd.read_csv(
                self.result_plnn_milp_p.joinpath("results " + str(opt_step))
            )[["Profit", "Produced heat"]]
            icnn_gd = pd.read_csv(
                self.result_icnn_gd_p.joinpath("results " + str(opt_step))
            )[["Profit", "Produced heat"]]
            icnn_gd_g = pd.read_csv(
                self.result_icnn_gd_p.joinpath(
                    "results g non-descreasing " + str(opt_step)
                )
            )[["Profit", "Produced heat"]]
            basic_strategy = pd.read_csv(
                self.result_basic_strategy_p.joinpath("results " + str(opt_step))
            )[["Profit", "Produced heat"]]
            plnn_gd = pd.read_csv(
                self.result_plnn_gd_p.joinpath("results " + str(opt_step))
            )[["Profit", "Produced heat"]]
            profit["PLNN-MILP"].append(sum(plnn_milp["Profit"]))
            profit["ICNN-GD"].append(sum(icnn_gd["Profit"]))
            profit["ICNN-GD-g"].append(sum(icnn_gd_g["Profit"]))
            profit["Basic strategy"].append(sum(basic_strategy["Profit"]))
            profit["PLNN-GD"].append(sum(plnn_gd["Profit"]))
            profit["MINLP"].append(self.nlmip_profit[opt_step])
            profit["RL"].append(self.rl_profit[opt_step])
            produced_heat["PLNN-MILP"].append(sum(plnn_milp["Produced heat"]))
            produced_heat["ICNN-GD"].append(sum(icnn_gd["Produced heat"]))
            produced_heat["Basic strategy"].append(sum(basic_strategy["Produced heat"]))
        return profit, produced_heat

    def reformat_violation(self, condition_flag) -> Tuple[list, list]:
        temp = {
            k: round((v / self.N) * 100, 2)
            for k, v in dict(
                sorted(dict(Counter(condition_flag)).items(), key=lambda item: item[0])
            ).items()
        }
        x = list(np.cumsum(list(temp.values())))
        y = list(temp.keys())
        if x[0] > 5:
            x.insert(0, 0)
            y.insert(0, 0)
        return x, y

    def plot_profit(self, opt_step, path):
        """
        One day profit.
        """
        title = "Single-day cost of methods in DNNs+Opt framework"
        day = list(range(1, self.day + 1))
        fig, ax = plt.subplots()
        plnn_milp = pd.read_csv(
            self.result_plnn_milp_p.joinpath("results " + str(opt_step))
        )["Profit"]
        icnn_gd = pd.read_csv(
            self.result_icnn_gd_p.joinpath("results " + str(opt_step))
        )["Profit"]
        icnn_gd_g = pd.read_csv(
            self.result_icnn_gd_p.joinpath("results g non-decreasing " + str(opt_step))
        )["Profit"]
        plnn_gd = pd.read_csv(
            self.result_plnn_gd_p.joinpath("results " + str(opt_step))
        )["Profit"]
        basic_strategy = pd.read_csv(
            self.result_basic_strategy_p.joinpath("results " + str(opt_step))
        )["Profit"]
        ax.plot(day, plnn_milp, color=self.colors["PLNN-MILP"], label="PLNN-MILP")
        PlotResult.plot_mark(
            ax=ax, x=day, y=plnn_milp, color=self.colors["PLNN-MILP"], mark="*"
        )
        ax.plot(
            day,
            icnn_gd,
            color=self.colors["ICNN-GD"],
            label="ICNN-GD≡ICNN-MILP",
        )
        PlotResult.plot_mark(
            ax=ax, x=day, y=icnn_gd, color=self.colors["ICNN-GD"], mark="x"
        )
        # ax.plot(day, icnn_gd_g, color=self.colors["ICNN-GD-g"], label="ICNN-GD$_↗$")
        # PlotResult.plot_mark(
        #    ax=ax, x=day, y=icnn_gd_g, color=self.colors["ICNN-GD-g"], mark="1"
        # )
        ax.plot(day, plnn_gd, color=self.colors["PLNN-GD"], label="PLNN-GD")
        PlotResult.plot_mark(
            ax=ax, x=day, y=plnn_gd, color=self.colors["PLNN-GD"], mark="o"
        )
        ax.plot(day, basic_strategy, color=self.colors["BS"], label="Basic strategy")
        PlotResult.plot_mark(
            ax=ax, x=day, y=basic_strategy, color=self.colors["BS"], mark="s"
        )
        plt.axvspan(6, 11, color="0.85")
        plt.axvspan(12, 17, color="0.85")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Cost [€]")
        ax.set_title(title)
        ax.legend()
        plt.savefig(path.joinpath("Profit"))
        plt.show()
        return (
            sum(plnn_milp) / len(plnn_milp),
            sum(icnn_gd) / len(icnn_gd),
            sum(icnn_gd_g) / len(icnn_gd_g),
            sum(plnn_gd) / len(plnn_gd),
            sum(basic_strategy) / len(basic_strategy),
        )

    def plot_produced_heat(self, opt_step, path):
        """
        One day produced electricity.
        """
        title = "Single-day produced heat of methods in DNNs+Opt framework"
        day = list(range(1, self.day + 1))
        fig, ax = plt.subplots()
        plnn_milp = pd.read_csv(
            self.result_plnn_milp_p.joinpath("results " + str(opt_step))
        )["Produced heat"]
        icnn_gd = pd.read_csv(
            self.result_icnn_gd_p.joinpath("results " + str(opt_step))
        )["Produced heat"]
        icnn_gd_g = pd.read_csv(
            self.result_icnn_gd_p.joinpath("results g non-decreasing " + str(opt_step))
        )["Produced heat"]
        plnn_gd = pd.read_csv(
            self.result_plnn_gd_p.joinpath("results " + str(opt_step))
        )["Produced heat"]
        basic_strategy = pd.read_csv(
            self.result_basic_strategy_p.joinpath("results " + str(opt_step))
        )["Produced heat"]
        ax.plot(day, plnn_milp, color=self.colors["PLNN-MILP"], label="PLNN-MILP")
        PlotResult.plot_mark(
            ax=ax, x=day, y=plnn_milp, color=self.colors["PLNN-MILP"], mark="*"
        )
        ax.plot(day, icnn_gd, color=self.colors["ICNN-GD"], label="ICNN-GD≡ICNN-MILP")
        PlotResult.plot_mark(
            ax=ax, x=day, y=icnn_gd, color=self.colors["ICNN-GD"], mark="x"
        )
        # ax.plot(
        #    day,
        #    icnn_gd_g,
        #    color=self.colors["ICNN-GD-g"],
        #    label="ICNN-GD$_{↗}$",
        # )
        # PlotResult.plot_mark(
        #    ax=ax, x=day, y=icnn_gd_g, color=self.colors["ICNN-GD"], mark="1"
        # )
        ax.plot(day, plnn_gd, color=self.colors["PLNN-GD"], label="PLNN-GD")
        PlotResult.plot_mark(
            ax=ax, x=day, y=plnn_gd, color=self.colors["PLNN-GD"], mark="o"
        )
        ax.plot(day, basic_strategy, color=self.colors["BS"], label="Basic strategy")
        PlotResult.plot_mark(
            ax=ax, x=day, y=basic_strategy, color=self.colors["BS"], mark="s"
        )
        plt.axvspan(6, 11, color="0.85")
        plt.axvspan(12, 17, color="0.85")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Heat [MWh]")
        ax.set_title(title)
        ax.legend()
        plt.savefig(path.joinpath("Produced heat"))
        plt.show()
        return (
            sum(plnn_milp) / len(plnn_milp),
            sum(icnn_gd) / len(icnn_gd),
            sum(icnn_gd_g) / len(icnn_gd_g),
            sum(plnn_gd) / len(plnn_gd),
            sum(basic_strategy) / len(basic_strategy),
        )

    def plot_heat_demand(self, opt_step, path):
        """
        One day heat demand.
        """
        title = "Single-day heat demand"
        day = list(range(1, self.day + 1))
        fig, ax = plt.subplots()
        demand = pd.read_csv(
            self.result_plnn_milp_p.joinpath("results " + str(opt_step))
        )["Demand"]
        ax.plot(day, demand, color="black")
        PlotResult.plot_mark(ax=ax, x=day, y=demand, color="k", mark="*")
        plt.axvspan(6, 11, color="0.85")
        plt.axvspan(12, 17, color="0.85")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Heat [MWh]")
        ax.set_title(title)
        plt.savefig(path.joinpath("Heat demand"))
        # plt.show()

    def plot_electricity_price(self, opt_step, path):
        """
        One day electricity price.
        """
        title = "Single-day electricity price"
        day = list(range(1, self.day + 1))
        fig, ax = plt.subplots()
        price = pd.read_csv(
            self.result_plnn_milp_p.joinpath("results " + str(opt_step))
        )["Price"]
        ax.plot(day, price, color="black")
        PlotResult.plot_mark(ax=ax, x=day, y=price, color="k", mark="*")
        plt.axvspan(6, 11, color="0.85")
        plt.axvspan(12, 17, color="0.85")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Electricity price [€]")
        ax.set_title(title)
        plt.savefig(path.joinpath("Electricity price"))
        # plt.show()

    def plot_single_day(self, opt_step):
        """
        Plot profit, produced heat, heat demand and electricity price of selected single day.
        """
        path = self.parent_p.joinpath("single_day_plot").joinpath(
            "opt_step_" + str(opt_step)
        )
        os.mkdir(path)
        plnn_milp_p, icnn_gd_p, icnn_gd_g_p, plnn_gd_p, bs_p = self.plot_profit(
            opt_step=opt_step, path=path
        )
        plnn_milp_h, icnn_gd_h, icnn_gd_g_h, plnn_gd_h, bs_h = self.plot_produced_heat(
            opt_step=opt_step, path=path
        )
        self.plot_heat_demand(opt_step=opt_step, path=path)
        self.plot_electricity_price(opt_step=opt_step, path=path)
        return (
            plnn_milp_p,
            icnn_gd_p,
            icnn_gd_g_p,
            plnn_gd_p,
            bs_p,
            plnn_milp_h,
            icnn_gd_h,
            icnn_gd_g_h,
            plnn_gd_h,
            bs_h,
        )

    def plot_box_predictions(self):
        """
        Plot predictions as a box plot.
        """
        plnn_short_horizon = pd.read_csv(
            self.result_p.joinpath(
                "plnn_prediction_err_short_horizon_time_delay_52.csv"
            )
        )[["Supply_inlet_temp", "Supply_outlet_temp", "Mass_flow", "Delivered_heat"]]
        icnn_short_horizon = pd.read_csv(
            self.result_p.joinpath(
                "icnn_prediction_err_short_horizon_time_delay_52.csv"
            )
        )[["Supply_inlet_temp", "Supply_outlet_temp", "Mass_flow", "Delivered_heat"]]
        N_short_hor = len(plnn_short_horizon)
        plnn_long_horizon = pd.read_csv(
            self.result_p.joinpath("plnn_prediction_err_long_horizon_time_delay_52.csv")
        )[["Supply_inlet_temp", "Supply_outlet_temp", "Mass_flow", "Delivered_heat"]]
        icnn_long_horizon = pd.read_csv(
            self.result_p.joinpath("icnn_prediction_err_long_horizon_time_delay_52.csv")
        )[["Supply_inlet_temp", "Supply_outlet_temp", "Mass_flow", "Delivered_heat"]]
        N_long_hor = len(plnn_long_horizon)
        print(
            "Supply inlet temperature plnn error one-step "
            + str(sum(plnn_short_horizon["Supply_inlet_temp"]) / N_short_hor)
        )
        print(
            "Supply outlet temperature plnn error one-step "
            + str(sum(plnn_short_horizon["Supply_outlet_temp"]) / N_short_hor)
        )
        print(
            "Mass flow plnn error one-step "
            + str(sum(plnn_short_horizon["Mass_flow"]) / N_short_hor)
        )
        print(
            "Supply inlet temperature icnn error one-step "
            + str(sum(icnn_short_horizon["Supply_inlet_temp"]) / N_short_hor)
        )
        print(
            "Supply outlet temperature icnn error one-step "
            + str(sum(icnn_short_horizon["Supply_outlet_temp"]) / N_short_hor)
        )
        print(
            "Mass flow icnn error one-step "
            + str(sum(icnn_short_horizon["Mass_flow"]) / N_short_hor)
        )

        print(
            "Supply inlet temperature plnn error one-step std "
            + str(np.std(np.array(plnn_short_horizon["Supply_inlet_temp"])))
        )
        print(
            "Supply outlet temperature plnn error one-step std "
            + str(np.std(np.array(plnn_short_horizon["Supply_outlet_temp"])))
        )
        print(
            "Mass flow plnn error one-step std "
            + str(np.std(np.array(plnn_short_horizon["Mass_flow"])))
        )
        print(
            "Supply inlet temperature icnn error one-step std "
            + str(np.std(np.array(icnn_short_horizon["Supply_inlet_temp"])))
        )
        print(
            "Supply outlet temperature icnn error one-step std "
            + str(np.std(np.array(icnn_short_horizon["Supply_outlet_temp"])))
        )
        print(
            "Mass flow icnn error one-step std "
            + str(np.std(np.array(icnn_short_horizon["Mass_flow"])))
        )

        print(
            "Delivered heat plnn error one-step "
            + str(sum(plnn_short_horizon["Delivered_heat"]) / N_short_hor)
        )
        print(
            "Delivered heat plnn error one-step std "
            + str(np.std(np.array(plnn_short_horizon["Delivered_heat"])))
        )
        print(
            "Delivered heat icnn error one-step "
            + str(sum(icnn_short_horizon["Delivered_heat"]) / N_short_hor)
        )
        print(
            "Delivered heat icnn error one-step std "
            + str(np.std(np.array(icnn_short_horizon["Delivered_heat"])))
        )

        print(
            "Supply inlet temperature plnn error multi-step "
            + str(sum(plnn_long_horizon["Supply_inlet_temp"]) / N_long_hor)
        )
        print(
            "Supply outlet temperature plnn error multi-step "
            + str(sum(plnn_long_horizon["Supply_outlet_temp"]) / N_long_hor)
        )
        print(
            "Mass flow plnn error multi-step "
            + str(sum(plnn_long_horizon["Mass_flow"]) / N_long_hor)
        )
        print(
            "Supply inlet temperature icnn error multi-step "
            + str(sum(icnn_long_horizon["Supply_inlet_temp"]) / N_long_hor)
        )
        print(
            "Supply outlet temperature icnn error multi-step "
            + str(sum(icnn_long_horizon["Supply_outlet_temp"]) / N_long_hor)
        )
        print(
            "Mass flow icnn error multi-step "
            + str(sum(icnn_long_horizon["Mass_flow"]) / N_long_hor)
        )

        print(
            "Supply inlet temperature plnn error multi-step std "
            + str(np.std(np.array(plnn_long_horizon["Supply_inlet_temp"])))
        )
        print(
            "Supply outlet temperature plnn error multi-step std "
            + str(np.std(np.array(plnn_long_horizon["Supply_outlet_temp"])))
        )
        print(
            "Mass flow plnn error multi-step std "
            + str(np.std(np.array(plnn_long_horizon["Mass_flow"])))
        )
        print(
            "Supply inlet temperature icnn error multi-step std "
            + str(np.std(np.array(icnn_long_horizon["Supply_inlet_temp"])))
        )
        print(
            "Supply outlet temperature icnn error multi-step std "
            + str(np.std(np.array(icnn_long_horizon["Supply_outlet_temp"])))
        )
        print(
            "Mass flow icnn error multi-step std "
            + str(np.std(np.array(icnn_long_horizon["Mass_flow"])))
        )

        print(
            "Delivered heat plnn error multi-step "
            + str(sum(plnn_long_horizon["Delivered_heat"]) / N_long_hor)
        )
        print(
            "Delivered heat plnn error multi-step std "
            + str(np.std(np.array(plnn_long_horizon["Delivered_heat"])))
        )
        print(
            "Delivered heat icnn error multi-step "
            + str(sum(icnn_long_horizon["Delivered_heat"]) / N_long_hor)
        )
        print(
            "Delivered heat icnn error multi-step std "
            + str(np.std(np.array(icnn_long_horizon["Delivered_heat"])))
        )

    def plot_box(self):
        """
        Plot profit and produced heat of all days as box-plot.
        """
        profit, produced_heat = self.calculate_sum()
        labels = ["PLNN-MILP", "ICNN-GD", "BS"]
        # plot profit
        fig, ax = plt.subplots()
        ax.boxplot(
            [profit["PLNN-MILP"], profit["ICNN-GD"], profit["Basic strategy"]],
            labels=labels,
            notch=False,
            boxprops=dict(color="b"),
            capprops=dict(color="b"),
            whiskerprops=dict(color="b"),
            medianprops=dict(color="g"),
        )
        ax.set_ylabel("Cost [€]")
        ax.set_title("Cost")
        plt.grid()
        plt.show()
        # plot produced heat
        fig, ax = plt.subplots()
        ax.boxplot(
            [
                produced_heat["PLNN-MILP"],
                produced_heat["ICNN-GD"],
                produced_heat["Basic strategy"],
            ],
            labels=labels,
            notch=False,
            boxprops=dict(color="b"),
            capprops=dict(color="b"),
            whiskerprops=dict(color="b"),
            medianprops=dict(color="g"),
        )
        ax.set_ylabel("Heat [MWh]")
        ax.set_title("Produced heat")
        plt.grid()
        plt.show()

    def plot_bar(self):
        """
        Plot cumulative cost during the 30 days period.
        """
        path = self.parent_p.joinpath("plot_bar")
        profit, produced_heat = self.calculate_sum()
        print(
            "Total produced heat of PLNN-MILP is "
            + str(sum(produced_heat["PLNN-MILP"]))
        )
        print("Total produced heat of ICNN-GD is " + str(sum(produced_heat["ICNN-GD"])))
        profit_k = ["MINLP", "PLNN-MILP", "ICNN-GD≡\nICNN-MILP", "PLNN-GD", "BS"]
        profit_v = [
            sum(profit["MINLP"]),
            sum(profit["PLNN-MILP"]),
            sum(profit["ICNN-GD-g"]),
            sum(profit["PLNN-GD"]),
            sum(profit["Basic strategy"]),
        ]
        patterns = ["/", "\\", "|", "-", "+"]
        fig, ax = plt.subplots()
        x = range(len(profit_k))
        ax.bar(x, profit_v, color=list(self.colors.values()), hatch=patterns)
        ax.set_ylabel("Cost [€]")
        ax.set_title("Thirty days cost")
        ax.set_xticks(x)
        _ = ax.set_xticklabels(profit_k)
        ax.set_yticklabels(
            [
                "0",
                "25,000",
                "50,000",
                "75,000",
                "100,000",
                "125,000",
                "150,000",
                "175,000",
                "200,000",
            ]
        )
        plt.savefig(path.joinpath("Thirty days cost"))
        plt.show()

    def get_violations(self) -> Tuple[dict, dict]:
        violations_plnn_milp = {
            "Supply inlet temperature": [],
            "Supply outlet temperature": [],
            "Delivered heat": [],
            "Mass flow": [],
        }
        violations_icnn_gd = {
            "Supply inlet temperature": [],
            "Supply outlet temperature": [],
            "Delivered heat": [],
            "Mass flow": [],
        }
        for opt_step in opt_steps:
            # PLNN-MILP
            plnn_milp = pd.read_csv(
                self.result_plnn_milp_p.joinpath(
                    "violations percentage " + str(opt_step)
                )
            )[
                [
                    "Supply inlet temperature",
                    "Supply outlet temperature",
                    "Delivered heat",
                    "Mass flow",
                ]
            ]
            violations_plnn_milp["Supply inlet temperature"].extend(
                list(plnn_milp["Supply inlet temperature"])
            )
            violations_plnn_milp["Supply outlet temperature"].extend(
                list(plnn_milp["Supply outlet temperature"])
            )
            violations_plnn_milp["Delivered heat"].extend(
                list(plnn_milp["Delivered heat"])
            )
            violations_plnn_milp["Mass flow"].extend(list(plnn_milp["Mass flow"]))
            # ICNN-GD
            icnn_gd = pd.read_csv(
                self.result_icnn_gd_p.joinpath("violations percentage " + str(opt_step))
            )[
                [
                    "Supply inlet temperature",
                    "Supply outlet temperature",
                    "Delivered heat",
                    "Mass flow",
                ]
            ]

            violations_icnn_gd["Supply inlet temperature"].extend(
                list(icnn_gd["Supply inlet temperature"])
            )
            violations_icnn_gd["Supply outlet temperature"].extend(
                list(icnn_gd["Supply outlet temperature"])
            )
            violations_icnn_gd["Delivered heat"].extend(list(icnn_gd["Delivered heat"]))
            violations_icnn_gd["Mass flow"].extend(list(icnn_gd["Mass flow"]))
        return violations_plnn_milp, violations_icnn_gd

    def plot_violations(self):
        """
        Plot quantile plots: On x-axis is the percentage of hours and on y-axis is the percentage of violations.
        """
        path = self.parent_p.joinpath("plot_feasibility")
        plnn_milp, icnn_gd = self.get_violations()
        for violation in self.viol_lab:
            title = violation + " violation"
            fig, ax = plt.subplots()
            x1, y1 = self.reformat_violation(plnn_milp[violation])
            x2, y2 = self.reformat_violation(icnn_gd[violation])
            x3, y3 = self.reformat_violation(self.nlmip_violations[violation])
            ax.plot(x1, y1, self.colors["PLNN-MILP"], label="PLNN-MILP")
            PlotResult.plot_mark(ax, x1, y1, self.colors["PLNN-MILP"], "*")
            ax.plot(x2, y2, self.colors["ICNN-GD"], label="ICNN-GD≡ICNN-MILP")
            PlotResult.plot_mark(ax, x2, y2, self.colors["ICNN-GD"], "x")
            ax.plot(x3, y3, self.colors["MINLP"], label="MINLP")
            PlotResult.plot_mark(ax, x3, y3, self.colors["MINLP"], "s")
            ax.set_xlabel("Percentage of hours [%]")
            ax.set_ylabel("Violation percentage [%]")
            ax.legend()
            ax.set_title(title)
            plt.savefig(path.joinpath(title))
            plt.show()

    @staticmethod
    def create_neuron_string(neuron) -> str:
        """
        Create string of the neuron depending on the type of neurons.
        For example, if the neuron is int (5) string is 5.
        If the neuron is list ([5,5]) string is 5_5.
        """
        neuron_ = ""
        if type(neuron) is int:
            neuron_ = str(neuron)
        elif type(neuron) is list:
            for nn in neuron:
                neuron_ += str(nn)
                neuron_ += "_"
            neuron_ = neuron_[:-1]
        else:
            print("Incorrect type of neuron!")
            exit(1)
        return neuron_

    @staticmethod
    def calculate_plot_bounds(mean, deviation):
        """
        Calculate upper and lower bounds for the plot.
        """
        up_bound, down_bound = [], []
        for i in range(len(mean)):
            up_bound.append(mean[i] + deviation[i])
            down_bound.append(mean[i] - deviation[i])
        return up_bound, down_bound

    def plot_predictions(self):
        """
        Plot predictions of four variables: supply inlet temperature, supply outlet temperature, mass flow (state variables)
        and delivered heat (output variable) as the function of neural network size.
        """
        time_delay = 10
        neurons = [
            [1],
            [1, 1],
            [3],
            [5],
            [5, 3],
            [10],
            [10, 10],
            [50, 50],
            [100, 100, 100],
        ]
        columns = [
            "Supply_inlet_temp",
            "Supply_outlet_temp",
            "Mass_flow",
            "Delivered_heat",
        ]
        y_name = {
            "Supply_inlet_temp": "Root mean squared error [C]",
            "Supply_outlet_temp": "Root mean squared error [C]",
            "Mass_flow": "Root mean squared error [kg/s]",
            "Delivered_heat": "Root mean squared error [MWh]",
        }
        title_name = {
            "Supply_inlet_temp": "Supply inlet temperature",
            "Supply_outlet_temp": "Supply outlet temperature",
            "Mass_flow": "Mass flow",
            "Delivered_heat": "Delivered heat",
        }
        mirror_error = {
            "Supply_inlet_temp_one_step": 4.73,
            "Supply_outlet_temp_one_step": 4.50,
            "Mass_flow_one_step": 38.61,
            "Delivered_heat_one_step": 2.72,
            "Mass_flow_multi_step": 100.02,
            "Delivered_heat_multi_step": 9.29,
        }
        plnn_one_step_means = DataPredictions()
        plnn_one_step_std = DataPredictions()
        plnn_multi_step_means = DataPredictions()
        plnn_multi_step_std = DataPredictions()
        icnn_one_step_means = DataPredictions()
        icnn_one_step_std = DataPredictions()
        icnn_multi_step_means = DataPredictions()
        icnn_multi_step_std = DataPredictions()
        monotonic_icnn_one_step_means = DataPredictions()
        monotonic_icnn_one_step_std = DataPredictions()
        monotonic_icnn_multi_step_means = DataPredictions()
        monotonic_icnn_multi_step_std = DataPredictions()
        for neuron in neurons:
            neuron_ = PlotResult.create_neuron_string(neuron)
            plnn_one_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/plnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            plnn_multi_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/plnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_multi_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            icnn_one_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/icnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            icnn_multi_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/icnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_multi_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            monotonic_icnn_one_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/monotonic_icnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            monotonic_icnn_multi_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/monotonic_icnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_multi_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            for column in columns:
                plnn_one_step_means.data[column].append(
                    sum(plnn_one_step[column]) / len(plnn_one_step[column])
                )
                plnn_one_step_std.data[column].append(
                    np.std(np.array(plnn_one_step[column]))
                )
                plnn_multi_step_means.data[column].append(
                    sum(plnn_multi_step[column]) / len(plnn_multi_step[column])
                )
                plnn_multi_step_std.data[column].append(
                    np.std(np.array(plnn_multi_step[column]))
                )
                icnn_one_step_means.data[column].append(
                    sum(icnn_one_step[column]) / len(icnn_one_step[column])
                )
                icnn_one_step_std.data[column].append(
                    np.std(np.array(icnn_one_step[column]))
                )
                icnn_multi_step_means.data[column].append(
                    sum(icnn_multi_step[column]) / len(icnn_multi_step[column])
                )
                icnn_multi_step_std.data[column].append(
                    np.std(np.array(icnn_multi_step[column]))
                )
                monotonic_icnn_one_step_means.data[column].append(
                    sum(monotonic_icnn_one_step[column])
                    / len(monotonic_icnn_one_step[column])
                )
                monotonic_icnn_one_step_std.data[column].append(
                    np.std(np.array(monotonic_icnn_one_step[column]))
                )
                monotonic_icnn_multi_step_means.data[column].append(
                    sum(monotonic_icnn_multi_step[column])
                    / len(monotonic_icnn_multi_step[column])
                )
                monotonic_icnn_multi_step_std.data[column].append(
                    np.std(np.array(monotonic_icnn_multi_step[column]))
                )
        for column in columns:
            title = title_name[column] + " one step prediction"
            fig, ax = plt.subplots()
            up_bound_plnn, down_bound_plnn = PlotResult.calculate_plot_bounds(
                plnn_one_step_means.data[column], plnn_one_step_std.data[column]
            )
            up_bound_icnn, down_bound_icnn = PlotResult.calculate_plot_bounds(
                icnn_one_step_means.data[column], icnn_one_step_std.data[column]
            )
            (
                up_bound_monotonic_icnn,
                down_bound_monotonic_icnn,
            ) = PlotResult.calculate_plot_bounds(
                monotonic_icnn_one_step_means.data[column],
                monotonic_icnn_one_step_std.data[column],
            )
            x = list(range(len(neurons)))
            ax.plot(x, plnn_one_step_means.data[column], color="b", label="PLNN")
            ax.fill_between(x, down_bound_plnn, up_bound_plnn, color="b", alpha=0.1)
            ax.plot(x, icnn_one_step_means.data[column], color="r", label="ICNN")
            ax.fill_between(x, down_bound_icnn, up_bound_icnn, color="r", alpha=0.1)
            ax.plot(
                x,
                monotonic_icnn_one_step_means.data[column],
                color="g",
                label="Monotonic ICNN",
            )
            ax.fill_between(
                x,
                down_bound_monotonic_icnn,
                up_bound_monotonic_icnn,
                color="g",
                alpha=0.1,
            )
            plt.plot(
                x,
                [mirror_error[column + "_one_step"]] * len(x),
                color="k",
                linestyle="--",
                label="Mirror",
            )
            ax.set_title(title)
            ax.legend()
            ax.set_xlabel("Neural network size")
            ax.set_ylabel(y_name[column])
            ax.set_xticks(x)
            ax.set_xticklabels(
                (
                    "$[1]$",
                    "$[1,1]$",
                    "$[3]$",
                    "$[5]$",
                    "$[5,3]$",
                    "$[10]$",
                    "$[10,10]$",
                    "$[50,50]$",
                    "$[100,100,100]$",
                )
            )
            plt.xticks(fontsize=7)
            plt.savefig(self.plot_p.joinpath("predictions").joinpath(title))
            plt.show()

            title = title_name[column] + " six steps prediction"
            fig, ax = plt.subplots()
            up_bound_plnn, down_bound_plnn = PlotResult.calculate_plot_bounds(
                plnn_multi_step_means.data[column], plnn_multi_step_std.data[column]
            )
            up_bound_icnn, down_bound_icnn = PlotResult.calculate_plot_bounds(
                icnn_multi_step_means.data[column], icnn_multi_step_std.data[column]
            )
            (
                up_bound_monotonic_icnn,
                down_bound_monotonic_icnn,
            ) = PlotResult.calculate_plot_bounds(
                monotonic_icnn_multi_step_means.data[column],
                monotonic_icnn_multi_step_std.data[column],
            )
            x = list(range(len(neurons)))
            ax.plot(x, plnn_multi_step_means.data[column], color="b", label="PLNN")
            ax.fill_between(x, down_bound_plnn, up_bound_plnn, color="b", alpha=0.1)
            ax.plot(x, icnn_multi_step_means.data[column], color="r", label="ICNN")
            ax.fill_between(x, down_bound_icnn, up_bound_icnn, color="r", alpha=0.1)
            ax.plot(
                x,
                monotonic_icnn_multi_step_means.data[column],
                color="g",
                label="Monotonic ICNN",
            )
            ax.fill_between(
                x,
                down_bound_monotonic_icnn,
                up_bound_monotonic_icnn,
                color="g",
                alpha=0.1,
            )
            if column != "Supply_inlet_temp" and column != "Supply_outlet_temp":
                plt.axhline(
                    mirror_error[column + "_multi_step"],
                    color="k",
                    linestyle="--",
                    label="Mirror",
                )
            ax.set_title(title)
            ax.set_xlabel("Neural network size")
            ax.set_ylabel(y_name[column])
            ax.legend()
            ax.set_xticks(x)
            ax.set_xticklabels(
                (
                    "$[1]$",
                    "$[1,1]$",
                    "$[3]$",
                    "$[5]$",
                    "$[5,3]$",
                    "$[10]$",
                    "$[10,10]$",
                    "$[50,50]$",
                    "$[100,100,100]$",
                )
            )
            plt.xticks(fontsize=7)
            plt.savefig(self.plot_p.joinpath("predictions").joinpath(title))
            plt.show()

    def plot_predictions_with_q_sequence(self):
        """
        Plot predictions of four variables: supply inlet temperature, supply outlet temperature, mass flow (state variables)
        and delivered heat (output variable) where predictions are made without q sequence and with q sequence being part of the
        function g input. The role of these plots is to inspect whether adding sequence of previous heat demands increases
        prediction accuracy of the neural network.
        """
        time_delay = 10
        neurons = [
            [1],
            [1, 1],
            [3],
            [5],
            [10],
            [10, 10],
            [50, 50],
        ]
        columns = [
            "Supply_inlet_temp",
            "Supply_outlet_temp",
            "Mass_flow",
            "Delivered_heat",
        ]
        y_name = {
            "Supply_inlet_temp": "Root mean squared error [C]",
            "Supply_outlet_temp": "Root mean squared error [C]",
            "Mass_flow": "Root mean squared error [kg/s]",
            "Delivered_heat": "Root mean squared error [MWh]",
        }
        title_name = {
            "Supply_inlet_temp": "Supply inlet temperature with Q sequence",
            "Supply_outlet_temp": "Supply outlet temperature with Q sequence",
            "Mass_flow": "Mass flow with Q sequence",
            "Delivered_heat": "Delivered heat with Q sequence",
        }
        # without q sequence
        plnn_one_step_means = DataPredictions()
        plnn_one_step_std = DataPredictions()
        plnn_multi_step_means = DataPredictions()
        plnn_multi_step_std = DataPredictions()
        icnn_one_step_means = DataPredictions()
        icnn_one_step_std = DataPredictions()
        icnn_multi_step_means = DataPredictions()
        icnn_multi_step_std = DataPredictions()
        monotonic_icnn_one_step_means = DataPredictions()
        monotonic_icnn_one_step_std = DataPredictions()
        monotonic_icnn_multi_step_means = DataPredictions()
        monotonic_icnn_multi_step_std = DataPredictions()
        # with q sequence
        plnn_one_step_means_with_q = DataPredictions()
        plnn_one_step_std_with_q = DataPredictions()
        plnn_multi_step_means_with_q = DataPredictions()
        plnn_multi_step_std_with_q = DataPredictions()
        icnn_one_step_means_with_q = DataPredictions()
        icnn_one_step_std_with_q = DataPredictions()
        icnn_multi_step_means_with_q = DataPredictions()
        icnn_multi_step_std_with_q = DataPredictions()
        monotonic_icnn_one_step_means_with_q = DataPredictions()
        monotonic_icnn_one_step_std_with_q = DataPredictions()
        monotonic_icnn_multi_step_means_with_q = DataPredictions()
        monotonic_icnn_multi_step_std_with_q = DataPredictions()
        for neuron in neurons:
            neuron_ = PlotResult.create_neuron_string(neuron)
            plnn_one_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/plnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            plnn_multi_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/plnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_multi_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            icnn_one_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/icnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            icnn_multi_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/icnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_multi_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            monotonic_icnn_one_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/monotonic_icnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            monotonic_icnn_multi_step = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/monotonic_icnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_multi_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]

            plnn_one_step_with_q = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/plnn_prediction_with_q_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            plnn_multi_step_with_q = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/plnn_prediction_with_q_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_multi_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            icnn_one_step_with_q = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/icnn_prediction_with_q_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            icnn_multi_step_with_q = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/icnn_prediction_with_q_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_multi_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            monotonic_icnn_one_step_with_q = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/monotonic_icnn_prediction_with_q_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            monotonic_icnn_multi_step_with_q = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/monotonic_icnn_prediction_with_q_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_multi_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )[columns]
            for column in columns:
                plnn_one_step_means.data[column].append(
                    sum(plnn_one_step[column]) / len(plnn_one_step[column])
                )
                plnn_one_step_std.data[column].append(
                    np.std(np.array(plnn_one_step[column]))
                )
                plnn_multi_step_means.data[column].append(
                    sum(plnn_multi_step[column]) / len(plnn_multi_step[column])
                )
                plnn_multi_step_std.data[column].append(
                    np.std(np.array(plnn_multi_step[column]))
                )
                icnn_one_step_means.data[column].append(
                    sum(icnn_one_step[column]) / len(icnn_one_step[column])
                )
                icnn_one_step_std.data[column].append(
                    np.std(np.array(icnn_one_step[column]))
                )
                icnn_multi_step_means.data[column].append(
                    sum(icnn_multi_step[column]) / len(icnn_multi_step[column])
                )
                icnn_multi_step_std.data[column].append(
                    np.std(np.array(icnn_multi_step[column]))
                )
                monotonic_icnn_one_step_means.data[column].append(
                    sum(monotonic_icnn_one_step[column])
                    / len(monotonic_icnn_one_step[column])
                )
                monotonic_icnn_one_step_std.data[column].append(
                    np.std(np.array(monotonic_icnn_one_step[column]))
                )
                monotonic_icnn_multi_step_means.data[column].append(
                    sum(monotonic_icnn_multi_step[column])
                    / len(monotonic_icnn_multi_step[column])
                )
                monotonic_icnn_multi_step_std.data[column].append(
                    np.std(np.array(monotonic_icnn_multi_step[column]))
                )

                plnn_one_step_means_with_q.data[column].append(
                    sum(plnn_one_step_with_q[column])
                    / len(plnn_one_step_with_q[column])
                )
                plnn_one_step_std_with_q.data[column].append(
                    np.std(np.array(plnn_one_step_with_q[column]))
                )
                plnn_multi_step_means_with_q.data[column].append(
                    sum(plnn_multi_step_with_q[column])
                    / len(plnn_multi_step_with_q[column])
                )
                plnn_multi_step_std_with_q.data[column].append(
                    np.std(np.array(plnn_multi_step_with_q[column]))
                )
                icnn_one_step_means_with_q.data[column].append(
                    sum(icnn_one_step_with_q[column])
                    / len(icnn_one_step_with_q[column])
                )
                icnn_one_step_std_with_q.data[column].append(
                    np.std(np.array(icnn_one_step_with_q[column]))
                )
                icnn_multi_step_means_with_q.data[column].append(
                    sum(icnn_multi_step_with_q[column])
                    / len(icnn_multi_step_with_q[column])
                )
                icnn_multi_step_std_with_q.data[column].append(
                    np.std(np.array(icnn_multi_step_with_q[column]))
                )
                monotonic_icnn_one_step_means_with_q.data[column].append(
                    sum(monotonic_icnn_one_step_with_q[column])
                    / len(monotonic_icnn_one_step_with_q[column])
                )
                monotonic_icnn_one_step_std_with_q.data[column].append(
                    np.std(np.array(monotonic_icnn_one_step_with_q[column]))
                )
                monotonic_icnn_multi_step_means_with_q.data[column].append(
                    sum(monotonic_icnn_multi_step_with_q[column])
                    / len(monotonic_icnn_multi_step_with_q[column])
                )
                monotonic_icnn_multi_step_std_with_q.data[column].append(
                    np.std(np.array(monotonic_icnn_multi_step_with_q[column]))
                )
        for column in columns:
            title = title_name[column] + " one step prediction"
            fig, ax = plt.subplots()
            up_bound_plnn, down_bound_plnn = PlotResult.calculate_plot_bounds(
                plnn_one_step_means.data[column], plnn_one_step_std.data[column]
            )
            up_bound_icnn, down_bound_icnn = PlotResult.calculate_plot_bounds(
                icnn_one_step_means.data[column], icnn_one_step_std.data[column]
            )
            (
                up_bound_monotonic_icnn,
                down_bound_monotonic_icnn,
            ) = PlotResult.calculate_plot_bounds(
                monotonic_icnn_one_step_means.data[column],
                monotonic_icnn_one_step_std.data[column],
            )
            (
                up_bound_plnn_with_q,
                down_bound_plnn_with_q,
            ) = PlotResult.calculate_plot_bounds(
                plnn_one_step_means_with_q.data[column],
                plnn_one_step_std_with_q.data[column],
            )
            (
                up_bound_icnn_with_q,
                down_bound_icnn_with_q,
            ) = PlotResult.calculate_plot_bounds(
                icnn_one_step_means_with_q.data[column],
                icnn_one_step_std_with_q.data[column],
            )
            (
                up_bound_monotonic_icnn_with_q,
                down_bound_monotonic_icnn_with_q,
            ) = PlotResult.calculate_plot_bounds(
                monotonic_icnn_one_step_means_with_q.data[column],
                monotonic_icnn_one_step_std_with_q.data[column],
            )
            # without q sequence
            x = list(range(len(neurons)))
            ax.plot(x, plnn_one_step_means.data[column], color="b", label="PLNN")
            ax.fill_between(x, down_bound_plnn, up_bound_plnn, color="b", alpha=0.1)
            ax.plot(x, icnn_one_step_means.data[column], color="r", label="ICNN")
            ax.fill_between(x, down_bound_icnn, up_bound_icnn, color="r", alpha=0.1)
            ax.plot(
                x,
                monotonic_icnn_one_step_means.data[column],
                color="g",
                label="Monotonic ICNN",
            )
            ax.fill_between(
                x,
                down_bound_monotonic_icnn,
                up_bound_monotonic_icnn,
                color="g",
                alpha=0.1,
            )
            # with q sequence
            ax.plot(
                x,
                plnn_one_step_means_with_q.data[column],
                color="c",
                label="PLNN with Q seq",
            )
            ax.fill_between(
                x, down_bound_plnn_with_q, up_bound_plnn_with_q, color="c", alpha=0.1
            )
            ax.plot(
                x,
                icnn_one_step_means_with_q.data[column],
                color="m",
                label="ICNN with Q seq",
            )
            ax.fill_between(
                x, down_bound_icnn_with_q, up_bound_icnn_with_q, color="m", alpha=0.1
            )
            ax.plot(
                x,
                monotonic_icnn_one_step_means_with_q.data[column],
                color="y",
                label="Monotonic ICNN with Q seq",
            )
            ax.fill_between(
                x,
                down_bound_monotonic_icnn_with_q,
                up_bound_monotonic_icnn_with_q,
                color="y",
                alpha=0.1,
            )
            ax.set_title(title)
            ax.legend()
            ax.set_xlabel("Neural network size")
            ax.set_ylabel(y_name[column])
            ax.set_xticks(x)
            ax.set_xticklabels(
                (
                    "$[1]$",
                    "$[1,1]$",
                    "$[3]$",
                    "$[5]$",
                    "$[10]$",
                    "$[10,10]$",
                    "$[50,50]$",
                )
            )
            plt.xticks(fontsize=7)
            plt.savefig(self.plot_p.joinpath("predictions").joinpath(title))
            plt.show()

            title = title_name[column] + " six steps prediction"
            fig, ax = plt.subplots()
            # without q sequence
            up_bound_plnn, down_bound_plnn = PlotResult.calculate_plot_bounds(
                plnn_multi_step_means.data[column], plnn_multi_step_std.data[column]
            )
            up_bound_icnn, down_bound_icnn = PlotResult.calculate_plot_bounds(
                icnn_multi_step_means.data[column], icnn_multi_step_std.data[column]
            )
            (
                up_bound_monotonic_icnn,
                down_bound_monotonic_icnn,
            ) = PlotResult.calculate_plot_bounds(
                monotonic_icnn_multi_step_means.data[column],
                monotonic_icnn_multi_step_std.data[column],
            )
            # with q sequence
            (
                up_bound_plnn_with_q,
                down_bound_plnn_with_q,
            ) = PlotResult.calculate_plot_bounds(
                plnn_multi_step_means_with_q.data[column],
                plnn_multi_step_std_with_q.data[column],
            )
            (
                up_bound_icnn_with_q,
                down_bound_icnn_with_q,
            ) = PlotResult.calculate_plot_bounds(
                icnn_multi_step_means_with_q.data[column],
                icnn_multi_step_std_with_q.data[column],
            )
            (
                up_bound_monotonic_icnn_with_q,
                down_bound_monotonic_icnn_with_q,
            ) = PlotResult.calculate_plot_bounds(
                monotonic_icnn_multi_step_means_with_q.data[column],
                monotonic_icnn_multi_step_std_with_q.data[column],
            )
            x = list(range(len(neurons)))
            ax.plot(x, plnn_multi_step_means.data[column], color="b", label="PLNN")
            ax.fill_between(x, down_bound_plnn, up_bound_plnn, color="b", alpha=0.1)
            ax.plot(x, icnn_multi_step_means.data[column], color="r", label="ICNN")
            ax.fill_between(x, down_bound_icnn, up_bound_icnn, color="r", alpha=0.1)
            ax.plot(
                x,
                monotonic_icnn_multi_step_means.data[column],
                color="g",
                label="Monotonic ICNN",
            )
            ax.fill_between(
                x,
                down_bound_monotonic_icnn,
                up_bound_monotonic_icnn,
                color="g",
                alpha=0.1,
            )
            ax.plot(
                x,
                plnn_multi_step_means_with_q.data[column],
                color="c",
                label="PLNN with Q seq",
            )
            ax.fill_between(
                x, down_bound_plnn_with_q, up_bound_plnn_with_q, color="c", alpha=0.1
            )
            ax.plot(
                x,
                icnn_multi_step_means_with_q.data[column],
                color="m",
                label="ICNN with Q seq",
            )
            ax.fill_between(
                x, down_bound_icnn_with_q, up_bound_icnn_with_q, color="m", alpha=0.1
            )
            ax.plot(
                x,
                monotonic_icnn_multi_step_means_with_q.data[column],
                color="y",
                label="Monotonic ICNN with Q seq",
            )
            ax.fill_between(
                x,
                down_bound_monotonic_icnn_with_q,
                up_bound_monotonic_icnn_with_q,
                color="y",
                alpha=0.1,
            )
            ax.set_title(title)
            ax.set_xlabel("Neural network size")
            ax.set_ylabel(y_name[column])
            ax.legend()
            ax.set_xticks(x)
            ax.set_xticklabels(
                (
                    "$[1]$",
                    "$[1,1]$",
                    "$[3]$",
                    "$[5]$",
                    "$[10]$",
                    "$[10,10]$",
                    "$[50,50]$",
                )
            )
            plt.xticks(fontsize=7)
            plt.savefig(self.plot_p.joinpath("predictions").joinpath(title))
            plt.show()

    def plot_delivered_heat_with_state_predictions_and_without(self, step_type):
        """
        Delivered heat prediction with real state as input and delivered heat prediction with state predictions as input.
        The role of these plots is to inspect the influence of state prediction propagation on the prediction accuracy of delivered heat.
        """
        time_delay = 10
        neurons = [
            [1],
            [1, 1],
            [3],
            [5],
            [5, 3],
            [10],
            [10, 10],
            [50, 50],
            [100, 100, 100],
        ]
        (
            plnn_state_pred_mean,
            plnn_state_pred_std,
            icnn_state_pred_mean,
            icnn_state_pred_std,
            monotonic_icnn_state_pred_mean,
            monotonic_icnn_state_pred_std,
        ) = ([], [], [], [], [], [])
        (
            plnn_real_mean,
            plnn_real_std,
            icnn_real_mean,
            icnn_real_std,
            monotonic_icnn_real_mean,
            monotonic_icnn_real_std,
        ) = ([], [], [], [], [], [])
        for neuron in neurons:
            neuron_ = PlotResult.create_neuron_string(neuron)
            plnn_state_pred = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/plnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_"
                    + step_type
                    + "_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )["Delivered_heat"]
            icnn_state_pred = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/icnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_"
                    + step_type
                    + "_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )["Delivered_heat"]
            monotonic_icnn_state_pred = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/monotonic_icnn_prediction_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_"
                    + step_type
                    + "_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )["Delivered_heat"]
            plnn_real_feature = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/plnn_prediction_real_world_feature_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )["0"]
            icnn_real_feature = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/icnn_prediction_real_world_feature_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )["0"]
            monotonic_icnn_real_feature = pd.read_csv(
                self.result_p.joinpath(
                    "predictions/monotonic_icnn_prediction_real_world_feature_L_"
                    + str(PipePreset1["Length"])
                    + "_s_err_one_step_time_delay_"
                    + str(time_delay)
                    + "_neurons_"
                    + neuron_
                    + ".csv"
                )
            )["0"]
            plnn_state_pred_mean.append(sum(plnn_state_pred) / len(plnn_state_pred))
            plnn_state_pred_std.append(np.std(np.array(plnn_state_pred)))
            icnn_state_pred_mean.append(sum(icnn_state_pred) / len(icnn_state_pred))
            icnn_state_pred_std.append(np.std(np.array(icnn_state_pred)))
            monotonic_icnn_state_pred_mean.append(
                sum(monotonic_icnn_state_pred) / len(monotonic_icnn_state_pred)
            )
            monotonic_icnn_state_pred_std.append(
                np.array(np.std(monotonic_icnn_state_pred))
            )
            plnn_real_mean.append(sum(plnn_real_feature) / len(plnn_real_feature))
            plnn_real_std.append(np.std(np.array(plnn_real_feature)))
            icnn_real_mean.append(sum(icnn_real_feature) / len(icnn_real_feature))
            icnn_real_std.append(np.std(np.array(icnn_real_std)))
            monotonic_icnn_real_mean.append(
                sum(monotonic_icnn_real_feature) / len(monotonic_icnn_real_feature)
            )
            monotonic_icnn_real_std.append(
                np.array(np.std(monotonic_icnn_real_feature))
            )
        title = (
            "Delivered heat with and without state prediction "
            + step_type
            + " prediction"
        )
        fig, ax = plt.subplots()
        up_bound_plnn_pred, down_bound_plnn_pred = PlotResult.calculate_plot_bounds(
            plnn_state_pred_mean, plnn_state_pred_std
        )
        up_bound_icnn_pred, down_bound_icnn_pred = PlotResult.calculate_plot_bounds(
            icnn_state_pred_mean, icnn_state_pred_std
        )
        (
            up_bound_monotonic_icnn_pred,
            down_bound_monotonic_icnn_pred,
        ) = PlotResult.calculate_plot_bounds(
            monotonic_icnn_state_pred_mean, monotonic_icnn_state_pred_std
        )
        up_bound_plnn_real, down_bound_plnn_real = PlotResult.calculate_plot_bounds(
            plnn_real_mean, plnn_real_std
        )
        up_bound_icnn_real, down_bound_icnn_real = PlotResult.calculate_plot_bounds(
            icnn_real_mean, icnn_real_std
        )
        (
            up_bound_monotonic_icnn_real,
            down_bound_monotonic_icnn_real,
        ) = PlotResult.calculate_plot_bounds(
            monotonic_icnn_real_mean, monotonic_icnn_real_std
        )
        x = list(range(len(neurons)))
        ax.plot(x, plnn_state_pred_mean, color="b", label="PLNN+state pred")
        ax.fill_between(
            x, down_bound_plnn_pred, up_bound_plnn_pred, color="b", alpha=0.1
        )
        ax.plot(x, icnn_state_pred_mean, color="r", label="ICNN+state pred")
        ax.fill_between(
            x, down_bound_icnn_pred, up_bound_icnn_pred, color="r", alpha=0.1
        )
        ax.plot(
            x,
            monotonic_icnn_state_pred_mean,
            color="g",
            label="Monotonic ICNN+state pred",
        )
        ax.fill_between(
            x,
            down_bound_monotonic_icnn_pred,
            up_bound_monotonic_icnn_pred,
            color="g",
            alpha=0.1,
        )

        ax.plot(x, plnn_real_mean, color="c", label="PLNN+state real")
        ax.fill_between(
            x, down_bound_plnn_real, up_bound_plnn_real, color="c", alpha=0.1
        )
        ax.plot(x, icnn_real_mean, color="m", label="ICNN+state real")
        ax.fill_between(
            x, down_bound_icnn_real, up_bound_icnn_real, color="m", alpha=0.1
        )
        ax.plot(
            x,
            monotonic_icnn_real_mean,
            color="y",
            label="Monotonic ICNN+state real",
        )
        ax.fill_between(
            x,
            down_bound_monotonic_icnn_real,
            up_bound_monotonic_icnn_real,
            color="y",
            alpha=0.1,
        )

        ax.set_title("Delivered heat")
        ax.legend()
        ax.set_xlabel("Neural network size")
        ax.set_ylabel("Root mean squared error [MWh]")
        ax.set_xticks(x)
        ax.set_xticklabels(
            (
                "$[1]$",
                "$[1,1]$",
                "$[3]$",
                "$[5]$",
                "$[5,3]$",
                "$[10]$",
                "$[10,10]$",
                "$[50,50]$",
                "$[100,100,100]$",
            )
        )
        plt.xticks(fontsize=7)
        plt.savefig(self.plot_p.joinpath("predictions").joinpath(title))
        plt.show()

    @staticmethod
    def get_last_element(x):
        """
        Iterates through the list backwards, and gets the first element that is not nan.
        """
        y = []
        for i in range(len(x)):
            for j in range(1, len(x[i]) + 1):
                if not math.isnan(x[i][-j]):
                    y.append(x[i][-j])
                    break
        return y

    @staticmethod
    def summarize_validation_loss(
        val_loss_summary, val_loss_no_early_stop, val_loss_early_stop, fun_type
    ):
        """
        Calculate mean, upper and lower bounds of validation losses without and with early stopping.
        """
        val_loss = {}
        val_loss["early_stop"] = PlotResult.get_last_element(x=val_loss_early_stop)
        val_loss["no_early_stop"] = PlotResult.get_last_element(
            x=val_loss_no_early_stop
        )
        for regulator in ["early_stop", "no_early_stop"]:
            mean = sum(val_loss[regulator]) / len(val_loss[regulator])
            std_dev = np.std(np.array(val_loss[regulator]))
            val_loss_summary[fun_type][regulator]["mean"].append(mean)
            val_loss_summary[fun_type][regulator]["up_bound"].append(mean + std_dev)
            val_loss_summary[fun_type][regulator]["low_bound"].append(mean - std_dev)
        return val_loss_summary

    def plot_training_validation_loss(self, early_stop, nn_type, num_iter):
        """
        Plot training and validation loss as the function of number of epochs.
        Plot validation losses without and with early stopping as the function of neural network size.
        """
        # these are validation losses for the creation of the second plot, where number of parameters is on the x-axis
        # and validation loss from trainings without and with early stopping on y-axis.
        val_loss_summary = {
            "_state_": {
                "early_stop": {"mean": [], "up_bound": [], "low_bound": []},
                "no_early_stop": {"mean": [], "up_bound": [], "low_bound": []},
            },
            "_output_": {
                "early_stop": {"mean": [], "up_bound": [], "low_bound": []},
                "no_early_stop": {"mean": [], "up_bound": [], "low_bound": []},
            },
        }
        if early_stop:
            ext = "_early_stop"
        else:
            ext = ""
        neurons = [
            # [1],
            # [1,1],
            # [3],
            # [5],
            # [5,3],
            # [10],
            [10, 10],
            [50, 50],
            [100, 100, 100],
        ]
        x = list(range(len(neurons)))
        fun_types = ["_state_", "_output_"]
        # y_max = {"_state_function_": 0.154, "_output_function_": 0.029}
        for neuron in neurons:
            for fun_type in fun_types:
                # names of training loss, validation loss without early stopping and validation loss with early stopping
                # corresponding to number of neurons and function type
                train_loss_: str = (
                    "train_loss_"
                    + nn_type
                    + fun_type
                    + "neurons_"
                    + PlotResult.create_neuron_string(neuron)
                    + ext
                    + ".csv"
                )
                val_loss_: str = (
                    "val_loss_"
                    + nn_type
                    + fun_type
                    + "neurons_"
                    + PlotResult.create_neuron_string(neuron)
                    + ext
                    + ".csv"
                )
                val_loss_early_stop_: str = (
                    "val_loss_"
                    + nn_type
                    + fun_type
                    + "neurons_"
                    + PlotResult.create_neuron_string(neuron)
                    + "_early_stop"
                    + ".csv"
                )
                # reading training loss, validation loss (without early stopping) and validation loss with early stopping
                # corresponding to their pre-determined names
                train_loss = list(
                    np.array(
                        pd.read_csv(
                            self.result_p.joinpath("train_val_loss").joinpath(
                                train_loss_
                            )
                        ).iloc[0:num_iter, 1:]
                    )
                )
                val_loss = list(
                    np.array(
                        pd.read_csv(
                            self.result_p.joinpath("train_val_loss").joinpath(val_loss_)
                        ).iloc[0:num_iter, 1:]
                    )
                )
                val_loss_early_stop = list(
                    np.array(
                        pd.read_csv(
                            self.result_p.joinpath("train_val_loss").joinpath(
                                val_loss_early_stop_
                            )
                        ).iloc[0:num_iter, 1:]
                    )
                )
                early_stop_epoch: int = len(val_loss_early_stop[1])
                # calculate mean, upper and lower bounds for validation losses without and with early stopping
                val_loss_summary = PlotResult.summarize_validation_loss(
                    val_loss_summary,
                    val_loss,
                    val_loss_early_stop,
                    fun_type,
                )
                # plot training and validation loss without early stopping with the early stopping hyperparameter as vertical line
                # only for the first iteration
                title = nn_type + fun_type + "train_val_loss"
                name = (
                    title + "_neurons_" + PlotResult.create_neuron_string(neuron) + ext
                )
                fig, ax = plt.subplots()
                ax.plot(train_loss[1], color="r", label="Training loss")
                ax.plot(val_loss[1], color="b", label="Validation loss")
                plt.axvline(
                    x=early_stop_epoch, color="k", label="Early stop", linestyle="--"
                )
                # ax.set_ylim(ymin=0)
                # ax.set_ylim(ymax=0.5 * max(list(train_loss)))
                ax.set_ylim(ymax=0.0007)
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Mean squared error")
                ax.set_title(title)
                ax.legend()
                plt.savefig(
                    self.plot_p.joinpath("training_validation_loss").joinpath(
                        name + "_zoom_in_3.png"
                    )
                )
                plt.show()
        for fun_type in fun_types:
            fig, ax = plt.subplots()
            title = nn_type + fun_type + "validation_loss"
            ax.plot(
                x,
                val_loss_summary[fun_type]["no_early_stop"]["mean"],
                color="r",
                label="No early stop",
            )
            ax.fill_between(
                x,
                val_loss_summary[fun_type]["no_early_stop"]["up_bound"],
                val_loss_summary[fun_type]["no_early_stop"]["low_bound"],
                color="r",
                alpha=0.1,
            )
            plt.scatter(
                x,
                val_loss_summary[fun_type]["no_early_stop"]["mean"],
                color="r",
                marker="*",
            )
            ax.plot(
                x,
                val_loss_summary[fun_type]["early_stop"]["mean"],
                color="b",
                label="Early stop",
            )
            ax.fill_between(
                x,
                val_loss_summary[fun_type]["early_stop"]["up_bound"],
                val_loss_summary[fun_type]["early_stop"]["low_bound"],
                color="b",
                alpha=0.1,
            )
            plt.scatter(
                x,
                val_loss_summary[fun_type]["early_stop"]["mean"],
                color="b",
                marker="*",
            )
            ax.legend()
            ax.set_xlabel("Neural network size")
            ax.set_ylabel("Validation loss -- Mean squared error")
            ax.set_xticks(x)
            ax.set_xticklabels(
                (
                    # "$[1]$",
                    # "$[1,1]$",
                    # "$[3]$",
                    # "$[5]$",
                    # "$[5,3]$",
                    # "$[10]$",
                    "$[10,10]$",
                    "$[50,50]$",
                    "$[100,100,100]$",
                )
            )
            ax.set_title(title)
            plt.xticks(fontsize=7)
            plt.savefig(
                self.plot_p.joinpath("training_validation_loss").joinpath(
                    title + ".png"
                )
            )
            plt.show()

    def inspect_relaxation_of_monotonicity_restrictions(self, time_delay):
        """
        The role of this function is to inspect how relaxation of monotonicity restrictions on state variables
        in input convex neural network influences prediction accuracy on delivered heat.
        """
        path: Path = self.result_p.joinpath("predictions").joinpath(
            "relax_monotonic_icnn"
        )
        dict = {
            "plnn_": {
                "one_step": {"mean": [], "std": []},
                "multi_step": {"mean": [], "std": []},
            },
            "icnn": {
                "one_step": {"mean": [], "std": []},
                "multi_step": {"mean": [], "std": []},
            },
            "": {
                "one_step": {"mean": [], "std": []},
                "multi_step": {"mean": [], "std": []},
            },
            "relax_all_": {
                "one_step": {"mean": [], "std": []},
                "multi_step": {"mean": [], "std": []},
            },
            "relax_tau_in_": {
                "one_step": {"mean": [], "std": []},
                "multi_step": {"mean": [], "std": []},
            },
            "relax_tau_out_": {
                "one_step": {"mean": [], "std": []},
                "multi_step": {"mean": [], "std": []},
            },
            "relax_m_": {
                "one_step": {"mean": [], "std": []},
                "multi_step": {"mean": [], "std": []},
            },
        }
        labels = {
            "plnn_": "PLNN",
            "icnn": "ICNN",
            "": "Monotonic ICNN",
            "relax_all_": r"ICNN Nonmon over $\tau^{s,in}, \tau^{s,out}, m$",
            "relax_tau_in_": r"ICNN Nonmon over $\tau^{s,in}$",
            "relax_tau_out_": r"ICNN Nonmon over $\tau^{s,out}$",
            "relax_m_": r"ICNN Nonmon over m",
        }
        colors = {
            "plnn_": "k",
            "icnn": "b",
            "": "r",
            "relax_all_": "m",
            "relax_tau_in_": "g",
            "relax_tau_out_": "y",
            "relax_m_": "c",
        }
        neurons = [[3], [5], [5, 3], [10], [10, 10], [50, 50]]
        for neuron in neurons:
            neuron_ = PlotResult.create_neuron_string(neuron)
            for type in dict.keys():
                for step in ["one_step", "multi_step"]:
                    if type == "plnn_":
                        nn_name = (
                            "plnn_prediction_L_"
                            + str(PipePreset1["Length"])
                            + "_s_err_"
                            + step
                            + "_time_delay_"
                            + str(time_delay)
                            + "_neurons_"
                            + neuron_
                            + ".csv"
                        )
                    elif type == "icnn":
                        nn_name = (
                            "icnn_prediction_L_"
                            + str(PipePreset1["Length"])
                            + "_s_err_"
                            + step
                            + "_time_delay_"
                            + str(time_delay)
                            + "_neurons_"
                            + neuron_
                            + ".csv"
                        )
                    else:
                        nn_name = (
                            "monotonic_icnn_prediction_"
                            + type
                            + "L_"
                            + str(PipePreset1["Length"])
                            + "_s_err_"
                            + step
                            + "_time_delay_"
                            + str(time_delay)
                            + "_neurons_"
                            + neuron_
                            + ".csv"
                        )
                    delivered_heat = pd.read_csv(path.joinpath(nn_name))[
                        "Delivered_heat"
                    ]
                    dict[type][step]["mean"].append(
                        sum(delivered_heat) / len(delivered_heat)
                    )
                    dict[type][step]["std"].append(np.std(delivered_heat))
        x = list(range(len(neurons)))
        for step in ["one_step", "multi_step"]:
            fig, ax = plt.subplots()
            title = "Analysis -- relaxing monotonic constraints " + step
            ax.set_title(title)
            for type in dict.keys():
                up_bound, down_bound = PlotResult.calculate_plot_bounds(
                    dict[type][step]["mean"], dict[type][step]["std"]
                )
                ax.plot(
                    x, dict[type][step]["mean"], label=labels[type], color=colors[type]
                )
                ax.fill_between(x, up_bound, down_bound, color=colors[type], alpha=0.1)
                ax.legend(loc="upper left")
                ax.set_xlabel("Neural network size")
                ax.set_ylabel("Root mean squared error [MWh]")
                ax.set_xticks(x)
                ax.set_xticklabels(
                    (
                        # "$[1]$",
                        # "$[1,1]$",
                        "$[3]$",
                        "$[5]$",
                        "$[5,3]$",
                        "$[10]$",
                        "$[10,10]$",
                        "$[50,50]$",
                        # "$[100,100,100]$"
                    )
                )
                plt.xticks(fontsize=7)
            plt.savefig(self.plot_p.joinpath(title))
            plt.show()

    def modify_tau_in(self, neurons, time_delay):
        """
        How does prediction accuracy of delivered heat change if we omit supply inlet temperature
        as feature from PLNN, ICNN and Monotonic ICNN?
        How does prediction accuracy of delivered heat change if we reinforce monotonicity restriction
        over supply inlet temperature?
        """
        path_plot = plot_p.joinpath("modify_tau_in")
        folders = ["without_tau_in", "predictions", "predictions/relax_monotonic_icnn"]
        nn_types = ["plnn", "icnn", "monotonic_icnn"]
        steps = ["one_step", "multi_step"]
        titles = {
            "one_step": r"One step prediction -- $\tau^{s,in}$ modifications",
            "multi_step": r"Multi step prediction -- $\tau^{s,in}$ modifications",
        }
        styles = {
            "without_tau_in": {
                "plnn": {
                    "label": r"PLNN -$\tau^{s,in}$",
                    "linestyle": "dashed",
                    "ext": "",
                },
                "icnn": {
                    "label": r"ICNN -$\tau^{s,in}$",
                    "linestyle": "dashed",
                    "ext": "",
                },
                "monotonic_icnn": {
                    "label": r"Mon ICNN -$\tau^{s,in}$",
                    "linestyle": "dashed",
                    "ext": "",
                },
            },
            "predictions": {
                "plnn": {
                    "label": r"PLNN $\tau^{s,in} \nearrow \swarrow$",
                    "linestyle": "solid",
                    "ext": "",
                },
                "icnn": {
                    "label": r"ICNN $\tau^{s,in}  \nearrow \swarrow$",
                    "linestyle": "solid",
                    "ext": "",
                },
                "monotonic_icnn": {
                    "label": r"Mon ICNN $\tau^{s,in}  \nearrow$",
                    "linestyle": "dotted",
                    "ext": "",
                },
            },
            "predictions/relax_monotonic_icnn": {
                "plnn": {
                    "label": r"PLNN/ICNN $\tau^{s,in} \nearrow $",
                    "linestyle": "dotted",
                    "ext": "_relax_tau_out_m",
                },
                "icnn": {
                    "label": r"PLNN/ICNN $\tau^{s,in}  \nearrow $",
                    "linestyle": "dotted",
                    "ext": "_relax_tau_out_m",
                },
                "monotonic_icnn": {
                    "label": r"Mon ICNN $\tau^{s,in}  \nearrow \swarrow$",
                    "linestyle": "solid",
                    "ext": "_relax_tau_in",
                },
            },
        }
        colors = {"plnn": "b", "icnn": "r", "monotonic_icnn": "g"}
        for step in steps:
            dict = {
                "without_tau_in": {
                    "plnn": {"mean": [], "std": []},
                    "icnn": {"mean": [], "std": []},
                    "monotonic_icnn": {"mean": [], "std": []},
                },
                "predictions": {
                    "plnn": {"mean": [], "std": []},
                    "icnn": {"mean": [], "std": []},
                    "monotonic_icnn": {"mean": [], "std": []},
                },
                "predictions/relax_monotonic_icnn": {
                    "plnn": {"mean": [], "std": []},
                    "icnn": {"mean": [], "std": []},
                    "monotonic_icnn": {"mean": [], "std": []},
                },
            }
            for folder in folders:
                path = self.result_p.joinpath(folder)
                for nn_type in nn_types:
                    for neuron in neurons:
                        neuron_ = PlotResult.create_neuron_string(neuron)
                        if folder == "predictions/relax_monotonic_icnn" and (
                            nn_type == "plnn" or nn_type == "icnn"
                        ):
                            nn_type_ = "monotonic_icnn"
                        else:
                            nn_type_ = nn_type
                        nn_name = (
                            nn_type_
                            + "_prediction"
                            + styles[folder][nn_type]["ext"]
                            + "_L_"
                            + str(PipePreset1["Length"])
                            + "_s_err_"
                            + step
                            + "_time_delay_"
                            + str(time_delay)
                            + "_neurons_"
                            + neuron_
                            + ".csv"
                        )
                        delivered_heat = pd.read_csv(path.joinpath(nn_name))[
                            "Delivered_heat"
                        ]
                        dict[folder][nn_type]["mean"].append(
                            sum(delivered_heat) / len(delivered_heat)
                        )
                        dict[folder][nn_type]["std"].append(np.std(delivered_heat))
            fig, ax = plt.subplots()
            x = list(range(len(neurons)))
            title = titles[step]
            for folder in folders:
                for nn_type in nn_types:
                    if folder == "predictions/relax_monotonic_icnn" and (
                        nn_type == "plnn"
                    ):
                        continue
                    up_bound, down_bound = PlotResult.calculate_plot_bounds(
                        dict[folder][nn_type]["mean"], dict[folder][nn_type]["std"]
                    )
                    ax.plot(
                        x,
                        dict[folder][nn_type]["mean"],
                        label=styles[folder][nn_type]["label"],
                        color=colors[nn_type],
                        linestyle=styles[folder][nn_type]["linestyle"],
                    )
                    ax.fill_between(
                        x, up_bound, down_bound, color=colors[nn_type], alpha=0.1
                    )
                    ax.legend(loc="upper left")
                    ax.set_xlabel("Neural network size")
                    ax.set_ylabel("Root mean squared error [MWh]")
                    ax.set_xticks(x)
                    ax.set_title(title)
                    ax.set_xticklabels(
                        (
                            "$[1]$",
                            "$[1,1]$",
                            "$[3]$",
                            "$[5]$",
                            "$[5,3]$",
                            "$[10]$",
                            "$[10,10]$",
                            "$[50,50]$",
                        )
                    )
                    plt.xticks(fontsize=7)
            plt.savefig(path_plot.joinpath(step))
            plt.show()

    def zoom_in_tau_in(self, neurons, time_delay):
        """
        Zoom in predictions with best choices for supply inlet temperature monotonicity restrictions
        in order to show decreasing trend of functions with increasing number of neurons.
        This trend can not be seen when number of neurons starts from [1].
        """
        path_plot = plot_p.joinpath("predictions")
        path = self.result_p.joinpath("predictions/relax_monotonic_icnn")
        nn_types = [
            "plnn_prediction",
            "monotonic_icnn_prediction_relax_tau_in",
            "icnn_prediction",
        ]
        steps = ["one_step", "multi_step"]
        titles = {
            "one_step": r"One step prediction with $\tau^{s,in}$ modifications",
            "multi_step": r"Multi step prediction with $\tau^{s,in}$ modifications",
        }
        styles = {
            "plnn_prediction": {
                "label": r"PLNN $\tau^{s,in} \nearrow \swarrow$",
                "color": "b",
                "linestyle": "solid",
            },
            "monotonic_icnn_prediction_relax_tau_in": {
                "label": r"Mon ICNN $\tau^{s,in} \nearrow \swarrow$",
                "color": "g",
                "linestyle": "solid",
            },
            "icnn_prediction": {
                "label": r"ICNN $\tau^{s,in} \nearrow \swarrow$",
                "color": "r",
                "linestyle": "solid",
            },
        }
        for step in steps:
            dict = {
                "plnn_prediction": {"mean": [], "std": []},
                "monotonic_icnn_prediction_relax_tau_in": {"mean": [], "std": []},
                "icnn_prediction": {"mean": [], "std": []},
            }
            for nn_type in nn_types:
                for neuron in neurons:
                    neuron_ = PlotResult.create_neuron_string(neuron)
                    nn_name = (
                        nn_type
                        + "_L_"
                        + str(PipePreset1["Length"])
                        + "_s_err_"
                        + step
                        + "_time_delay_"
                        + str(time_delay)
                        + "_neurons_"
                        + neuron_
                        + ".csv"
                    )
                    delivered_heat = pd.read_csv(path.joinpath(nn_name))[
                        "Delivered_heat"
                    ]
                    dict[nn_type]["mean"].append(
                        sum(delivered_heat) / len(delivered_heat)
                    )
                    dict[nn_type]["std"].append(np.std(delivered_heat))
            fig, ax = plt.subplots()
            x = list(range(len(neurons)))
            title = titles[step]
            for nn_type in nn_types:
                up_bound, down_bound = PlotResult.calculate_plot_bounds(
                    dict[nn_type]["mean"], dict[nn_type]["std"]
                )
                ax.plot(
                    x,
                    dict[nn_type]["mean"],
                    label=styles[nn_type]["label"],
                    color=styles[nn_type]["color"],
                    linestyle=styles[nn_type]["linestyle"],
                )
                ax.fill_between(
                    x, up_bound, down_bound, color=styles[nn_type]["color"], alpha=0.1
                )
                ax.legend(loc="upper left")
                ax.set_xlabel("Neural network size")
                ax.set_ylabel("Root mean squared error [MWh]")
                ax.set_xticks(x)
                ax.set_title(title)
                ax.set_xticklabels(
                    (
                        "$[3]$",
                        "$[5]$",
                        "$[5,3]$",
                        "$[10]$",
                        "$[10,10]$",
                        "$[50,50]$",
                    )
                )
                plt.xticks(fontsize=7)
            plt.savefig(path_plot.joinpath("Delivered heat " + step + " zoom_in"))
            plt.show()

    def inspect_monotonic_heat_restriction(self, neurons, time_delay):
        """
        Plot prediction accuracy on delivered heat for monotonic ICNN and monotonic ICNN
        where monotonicity restriction over supply inlet temperature is relaxed for two cases:
        the first one is when produced heat variables in function f are not restricted and
        the second one is when produced heat variables in function f are restricted to be non-decreasing.
        """
        path_plot = plot_p.joinpath("monotonic_heat")
        path = self.result_p.joinpath("monotonic_heat")
        steps = ["one_step", "multi_step"]
        titles = {
            "one_step": r"One step prediction with monotonic heat restriction",
            "multi_step": r"Multi step prediction with monotonic heat restriction",
        }
        styles = {
            "monotonic_icnn_prediction": {
                "label": r"Mon ICNN $h \nearrow \swarrow$",
                "color": "r",
                "linestyle": "solid",
            },
            "monotonic_icnn_prediction_heat": {
                "label": r"Mon ICNN $h \nearrow$",
                "color": "r",
                "linestyle": "dashed",
            },
            "monotonic_icnn_prediction_relax_tau_in": {
                "label": r"Mon ICNN $\tau^{s,in} \nearrow \swarrow, h \nearrow \swarrow$",
                "color": "b",
                "linestyle": "solid",
            },
            "monotonic_icnn_prediction_heat_relax_tau_in": {
                "label": r"Mon ICNN $\tau^{s,in} \nearrow \swarrow, h \nearrow$",
                "color": "b",
                "linestyle": "dashed",
            },
        }
        for step in steps:
            dict = {
                "monotonic_icnn_prediction": {"mean": [], "std": []},
                "monotonic_icnn_prediction_heat": {"mean": [], "std": []},
                "monotonic_icnn_prediction_relax_tau_in": {"mean": [], "std": []},
                "monotonic_icnn_prediction_heat_relax_tau_in": {"mean": [], "std": []},
            }
            for nn_type in styles.keys():
                for neuron in neurons:
                    neuron_ = PlotResult.create_neuron_string(neuron)
                    nn_name = (
                        nn_type
                        + "_L_"
                        + str(PipePreset1["Length"])
                        + "_s_err_"
                        + step
                        + "_time_delay_"
                        + str(time_delay)
                        + "_neurons_"
                        + neuron_
                        + ".csv"
                    )
                    delivered_heat = pd.read_csv(path.joinpath(nn_name))[
                        "Delivered_heat"
                    ]
                    dict[nn_type]["mean"].append(
                        sum(delivered_heat) / len(delivered_heat)
                    )
                    dict[nn_type]["std"].append(np.std(delivered_heat))
            fig, ax = plt.subplots()
            x = list(range(len(neurons)))
            title = titles[step]
            for nn_type in styles.keys():
                up_bound, down_bound = PlotResult.calculate_plot_bounds(
                    dict[nn_type]["mean"], dict[nn_type]["std"]
                )
                ax.plot(
                    x,
                    dict[nn_type]["mean"],
                    label=styles[nn_type]["label"],
                    color=styles[nn_type]["color"],
                    linestyle=styles[nn_type]["linestyle"],
                )
                ax.fill_between(
                    x, up_bound, down_bound, color=styles[nn_type]["color"], alpha=0.1
                )
                ax.legend(loc="upper left")
                ax.set_xlabel("Neural network size")
                ax.set_ylabel("Root mean squared error [MWh]")
                ax.set_xticks(x)
                ax.set_title(title)
                ax.set_xticklabels(
                    (
                        "$[1]$",
                        "$[1,1]$",
                        "$[3]$",
                        "$[5]$",
                        "$[5,3]$",
                        "$[10]$",
                        "$[10,10]$",
                        "$[50,50]$",
                    )
                )
                plt.xticks(fontsize=7)
            plt.savefig(path_plot.joinpath(titles[step] + ".png"))
            plt.show()

    def sum(self, path, optimizer, coeff_day, column):
        """
        Calculate sum of specific column for different days.
        """
        sum = 0
        for coeff in coeff_day:
            sum += np.sum(
                np.array(
                    pd.read_csv(path.joinpath(optimizer + "_" + str(coeff) + ".csv"))[
                        column
                    ]
                )
            )
        return sum

    def average(self, path, optimizer, coeff_day, column):
        """
        Calculate average of specific column for different days.
        """
        sum = 0
        for coeff in coeff_day:
            sum += np.sum(
                np.array(
                    pd.read_csv(path.joinpath(optimizer + "_" + str(coeff) + ".csv"))[
                        column
                    ]
                )
            )
        return sum / (len(coeff_day) * TimeParameters["PlanningHorizon"])

    def standard_deviation(self, path, optimizer, coeff_day, column):
        """
        Calculate standard deviation of specific column for all days.
        """
        temp = []
        for coeff in coeff_day:
            temp.extend(
                list(
                    pd.read_csv(path.joinpath(optimizer + "_" + str(coeff) + ".csv"))[
                        column
                    ]
                )
            )
        temp = np.array(temp)
        standard_deviation = np.std(temp)
        return standard_deviation

    def optimization(self, neurons, plot_p):
        """
        Plot results of optimization step: operation cost, percentage of violations, runtime and optimality gap of different algorithms.
        """
        x = list(range(len(neurons)))
        optimizers = ["abdollahi", "li"]
        metrics = ["profit", "violations", "runtime", "optimality_gap"]
        violations = [
            "Supply_inlet_violation",
            "Supply_outlet_violation",
            "Delivered_heat_violation",
            "Mass_flow_violation",
        ]
        paths = {
            "abdollahi": Path(__file__).parents[4] / "results/abdollahi_2015",
            "li": Path(__file__).parents[4] / "results/li_2016",
            "plot": plot_p.joinpath("optimization"),
        }
        colors = {"abdollahi": "y", "li": "b"}
        labels = {"abdollahi": "LP", "li": "MINLP"}
        titles = {
            "profit": "Operation cost",
            "violations": {
                "Supply_inlet_violation": "Percentage violation of supply inlet temperature",
                "Supply_outlet_violation": "Percentage violation of supply outlet temperature",
                "Delivered_heat_violation": "Percentage violation of delivered heat",
                "Mass_flow_violation": "Percentage violation of mass flow",
            },
            "runtime": "Runtime",
            "optimality_gap": "Optimality gap",
        }
        y_axis = {
            "profit": "Operation cost [e]",
            "violations": "Violation [%]",
            "runtime": "Runtime [s]",
            "optimality_gap": "Optimality gap [%]",
        }
        results = {
            "abdollahi": {
                "profit": [],
                "violations": {
                    "Supply_inlet_violation": [],
                    "Supply_outlet_violation": [],
                    "Delivered_heat_violation": [],
                    "Mass_flow_violation": [],
                },
                "runtime": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
                "optimality_gap": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
            },
            "li": {
                "profit": [],
                "violations": {
                    "Supply_inlet_violation": [],
                    "Supply_outlet_violation": [],
                    "Delivered_heat_violation": [],
                    "Mass_flow_violation": [],
                },
                "runtime": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
                "optimality_gap": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
            },
        }
        for optimizer in optimizers:
            results[optimizer]["profit"].append(
                self.sum(
                    path=paths[optimizer],
                    optimizer=optimizer,
                    coeff_day=opt_steps,
                    column="Profit",
                )
            )
            for violation_type in results[optimizer]["violations"].keys():
                results[optimizer]["violations"][violation_type].append(
                    self.average(
                        path=paths[optimizer],
                        optimizer=optimizer,
                        coeff_day=opt_steps,
                        column=violation_type,
                    )
                )
            # mean value
            results[optimizer]["runtime"]["mean"].append(
                self.average(
                    path=paths[optimizer],
                    optimizer=optimizer,
                    coeff_day=opt_steps,
                    column="Runtime",
                )
            )

            results[optimizer]["optimality_gap"]["mean"].append(
                self.average(
                    path=paths[optimizer],
                    optimizer=optimizer,
                    coeff_day=opt_steps,
                    column="Optimality_gap",
                )
            )
            # standard deviation
            results[optimizer]["runtime"]["std"].append(
                self.standard_deviation(
                    path=paths[optimizer],
                    optimizer=optimizer,
                    coeff_day=opt_steps,
                    column="Runtime",
                )
            )

            results[optimizer]["optimality_gap"]["std"].append(
                self.standard_deviation(
                    path=paths[optimizer],
                    optimizer=optimizer,
                    coeff_day=opt_steps,
                    column="Optimality_gap",
                )
            )
            # upper bound
            results[optimizer]["runtime"]["upper_bound"] = self.calculate_plot_bounds(
                results[optimizer]["runtime"]["mean"],
                results[optimizer]["runtime"]["std"],
            )[0]
            results[optimizer]["optimality_gap"][
                "upper_bound"
            ] = self.calculate_plot_bounds(
                results[optimizer]["optimality_gap"]["mean"],
                results[optimizer]["optimality_gap"]["std"],
            )[
                0
            ]
            results[optimizer]["runtime"]["lower_bound"] = self.calculate_plot_bounds(
                results[optimizer]["runtime"]["mean"],
                results[optimizer]["runtime"]["std"],
            )[1]
            results[optimizer]["optimality_gap"][
                "lower_bound"
            ] = self.calculate_plot_bounds(
                results[optimizer]["optimality_gap"]["mean"],
                results[optimizer]["optimality_gap"]["std"],
            )[
                1
            ]
        for metric in metrics:
            if metric == "violations":
                for violation in violations:
                    fig, ax = plt.subplots()
                    for optimizer in optimizers:
                        plt.axhline(
                            results[optimizer]["violations"][violation][0],
                            label=labels[optimizer],
                            color=colors[optimizer],
                        )
                    ax.set_title(titles[metric][violation])
                    ax.set_xticks(x)
                    ax.set_ylabel(y_axis[metric])
                    ax.set_xlabel("Neural network size")
                    ax.set_xticklabels(
                        (
                            "$[1]$",
                            "$[1,1]$",
                            "$[3]$",
                            "$[5]$",
                            "$[5,3]$",
                            "$[10]$",
                            "$[10,10]$",
                            "$[50,50]$",
                        )
                    )
                    plt.legend()
                    plt.savefig(
                        paths["plot"].joinpath(titles[metric][violation] + ".png")
                    )
                    plt.show()
            elif metric == "runtime" or metric == "optimality_gap":
                fig, ax = plt.subplots()
                for optimizer in optimizers:
                    plt.plot(
                        x,
                        results[optimizer][metric]["mean"] * len(x),
                        label=labels[optimizer],
                        color=colors[optimizer],
                    )
                    plt.fill_between(
                        x,
                        results[optimizer][metric]["lower_bound"] * len(x),
                        results[optimizer][metric]["upper_bound"] * len(x),
                        color=colors[optimizer],
                        alpha=0.1,
                    )
                ax.set_title(titles[metric])
                ax.set_xticks(x)
                ax.set_ylabel(y_axis[metric])
                ax.set_xlabel("Neural network size")
                ax.set_xticklabels(
                    (
                        "$[1]$",
                        "$[1,1]$",
                        "$[3]$",
                        "$[5]$",
                        "$[5,3]$",
                        "$[10]$",
                        "$[10,10]$",
                        "$[50,50]$",
                    )
                )
                plt.legend()
                plt.savefig(paths["plot"].joinpath(titles[metric] + ".png"))
                plt.show()
            else:
                fig, ax = plt.subplots()
                for optimizer in optimizers:
                    plt.axhline(
                        results[optimizer][metric][0],
                        label=labels[optimizer],
                        color=colors[optimizer],
                    )
                ax.set_title(titles[metric])
                ax.set_xticks(x)
                ax.set_ylabel(y_axis[metric])
                ax.set_xlabel("Neural network size")
                ax.set_xticklabels(
                    (
                        "$[1]$",
                        "$[1,1]$",
                        "$[3]$",
                        "$[5]$",
                        "$[5,3]$",
                        "$[10]$",
                        "$[10,10]$",
                        "$[50,50]$",
                    )
                )
                plt.legend()
                plt.savefig(paths["plot"].joinpath(titles[metric] + ".png"))
                plt.show()


if __name__ == "__main__":
    result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
    plot_p: Path = Path(__file__).parents[4] / "plots/constraint_opt"
    plot = PlotResult(result_p=result_p, plot_p=plot_p)

    # plot cost, produced heat, heat demand and electricity price for the single day
    """
    opt_step = [70]
    mean_profit = {
        "PLNN-MILP": [],
        "ICNN-GD": [],
        "ICNN-GD-g": [],
        "PLNN-GD": [],
        "BS": [],
    }
    mean_heat = {
        "PLNN-MILP": [],
        "ICNN-GD": [],
        "ICNN-GD-g": [],
        "PLNN-GD": [],
        "BS": [],
    }
    for opt_step in opt_step:
        (
            plnn_milp_p,
            icnn_gd_p,
            icnn_gd_g_p,
            plnn_gd_p,
            bs_p,
            plnn_milp_h,
            icnn_gd_h,
            icnn_gd_g_h,
            plnn_gd_h,
            bs_h,
        ) = plot.plot_single_day(opt_step=opt_step)
        mean_profit["PLNN-MILP"].append(plnn_milp_p)
        mean_profit["ICNN-GD"].append(icnn_gd_p)
        mean_profit["ICNN-GD-g"].append(icnn_gd_g_p)
        mean_profit["PLNN-GD"].append(plnn_gd_p)
        mean_profit["BS"].append(bs_p)
        mean_heat["PLNN-MILP"].append(plnn_milp_h)
        mean_heat["ICNN-GD"].append(icnn_gd_h)
        mean_heat["ICNN-GD-g"].append(icnn_gd_g_h)
        mean_heat["PLNN-GD"].append(plnn_gd_h)
        mean_heat["BS"].append(bs_h)
    mean_profit = pd.DataFrame(mean_profit)
    mean_heat = pd.DataFrame(mean_heat)
    mean_profit.to_csv(result_p.joinpath("mean_profit.csv"))
    mean_heat.to_csv(result_p.joinpath("mean_heat.csv"))
    """
    # plot cost and produced heat of all days as box plot
    # plot.plot_box()
    # plot.plot_violations()
    # plot.plot_bar()
    # plot.plot_predictions()
    # plot.plot_delivered_heat_with_state_predictions_and_without(step_type="multi")
    # plot.plot_training_validation_loss(
    #    early_stop=False, nn_type="plnn", num_iter=3
    # )
    # plot.inspect_monotonic_heat_restriction(
    #    neurons=[[1], [1, 1], [3], [5], [5, 3], [10], [10, 10], [50, 50]], time_delay=10
    # )
    plot.optimization(
        neurons=[[1], [1, 1], [3], [5], [5, 3], [10], [10, 10], [50, 50]], plot_p=plot_p
    )

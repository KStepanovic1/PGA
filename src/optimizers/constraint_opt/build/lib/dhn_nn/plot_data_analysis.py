import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pingouin as pg
import pickle

from collections import Counter
from pathlib import Path
from tensorflow.keras.models import load_model
from scipy.stats import pearsonr, spearmanr, kendalltau

from util.config import GridProperties, PipePreset1, ProducerPreset1
from util.config import Dataset as ConfigDataset
from src.optimizers.constraint_opt.dhn_nn.plot import Plot
from src.optimizers.constraint_opt.dhn_nn.tensor_constraint import ParNonNeg
from src.optimizers.constraint_opt.dhn_nn.functions import plugs_to_list, string_to_list


class PlotDataAnalysis(Plot):
    """
    Plot various analysis of the data.
    """

    def __init__(self, result_p, columns, time_delay):
        super().__init__()
        self.parent_p: Path = Path(__file__).parent
        self.save_path: Path = Path(__file__).parents[1] / "dhn_nn/y_h_q_plot"
        self.result_p: Path = result_p
        self.time_delay = time_delay
        self.cut_index_initial_state = 15  # initially pipe is full of hot water, and that influences first steps outcome.
        with open(self.parent_p.joinpath("state_dict_plnn.pkl"), "rb") as f:
            self.state_dict = pickle.load(f)
        with open(self.parent_p.joinpath("output_dict_plnn.pkl"), "rb") as f:
            self.output_dict = pickle.load(f)
        self.y_max = self.output_dict["Delivered heat 1 max"]
        self.y_min = self.output_dict["Delivered heat 1 min"]
        self.data, self.x_s = self.read_data(columns=columns)
        self.N: int = len(self.data)
        self.x, self.y = self.create_data()
        self.h = self.x["h_t"]
        self.y = self.y["y"]
        self.q = self.x["q_t"]
        self.grid_input = self.x["grid input"]
        self.x_norm = self.normalize()

    def read_data(self, columns):
        """
        Reading data from the main file, corresponding to requested columns.
        """
        with open(
            self.result_p.joinpath(
                "data_num_{}_heat_demand_real_world_for_L={}_time_interval_3600_max_Q_70MW_deltaT_{}C.csv".format(
                    GridProperties["ConsumerNum"],
                    PipePreset1["Length"],
                    ProducerPreset1["Generators"][0]["MaxRampRateTemp"],
                )
            ),
            "rb",
        ) as f:
            data = pd.read_csv(f)[columns]
        data = data.iloc[self.cut_index_initial_state :]
        with open(self.result_p.joinpath("x_s.csv"), "rb") as f:
            x_s = pd.read_csv(f)["q_{}".format(1 + 1)]
        return data, x_s

    def create_data(self):
        """
        Create dataframe in a suitable form for further analysis.
        """
        columns_x = [
            "tau_in_t_1",
            "tau_out_t_1",
            "m_t_1",
            "q_t",
            "h_t",
            "h_t_1",
            "h_t_t",
            "grid input",
        ]
        columns_y = ["y"]
        x, y = [], []
        for i in range(self.N):
            if (
                i < self.N + self.cut_index_initial_state - 1
                and i > self.cut_index_initial_state
            ):
                temp = []
                temp.append(self.data["Supply in temp 1"][i])
                temp.append(self.data["Supply out temp 1"][i])
                temp.append(self.data["Supply mass flow 1"][i])
                temp.append(self.data["Heat demand 1"][i + 1])
                temp.append(self.data["Produced heat"][i + 1])
                temp.append(self.data["Produced heat"][i])
                temp.append(self.data["Produced heat"][i + 1])
                temp.append(self.data["Grid input"][i + 1])
                x.append(temp)
                y.append(
                    [
                        self.data["Delivered heat 1"][i + 1],
                    ]
                )
        x = pd.DataFrame(x, columns=columns_x)
        y = pd.DataFrame(y, columns=columns_y)
        return x, y

    def normalize(self):
        """
        Normalize the dataset.
        """
        x_norm = pd.DataFrame()
        x_norm["tau_in_t_1"] = (
            self.x["tau_in_t_1"] - self.state_dict["Supply in temp 1 min"]
        ) / (
            self.state_dict["Supply in temp 1 max"]
            - self.state_dict["Supply in temp 1 min"]
        )
        x_norm["tau_out_t_1"] = (
            self.x["tau_out_t_1"] - self.state_dict["Supply out temp 1 min"]
        ) / (
            self.state_dict["Supply out temp 1 max"]
            - self.state_dict["Supply out temp 1 min"]
        )
        x_norm["m_t_1"] = (
            self.x["m_t_1"] - self.state_dict["Supply mass flow 1 min"]
        ) / (
            self.state_dict["Supply mass flow 1 max"]
            - self.state_dict["Supply mass flow 1 min"]
        )
        x_norm["q_t"] = (self.x["q_t"] - self.state_dict["Heat demand 1 min"]) / (
            self.state_dict["Heat demand 1 max"] - self.state_dict["Heat demand 1 min"]
        )
        x_norm["h_t"] = (self.x["h_t"] - self.state_dict["Produced heat min"]) / (
            self.state_dict["Produced heat max"] - self.state_dict["Produced heat min"]
        )
        x_norm["h_t_1"] = (self.x["h_t_1"] - self.state_dict["Produced heat min"]) / (
            self.state_dict["Produced heat max"] - self.state_dict["Produced heat min"]
        )
        x_norm["h_t_t"] = (self.x["h_t_t"] - self.state_dict["Produced heat min"]) / (
            self.state_dict["Produced heat max"] - self.state_dict["Produced heat min"]
        )
        return x_norm

    def get_plnn(self, model_s_p, model_out_p):
        """
        Get prediction by PLNN.
        """
        model_state = load_model(self.parent_p.joinpath(model_s_p), compile=False)
        model_out = load_model(self.parent_p.joinpath(model_out_p), compile=False)
        model_state_pred = model_state.predict(
            self.x_norm[["tau_in_t_1", "tau_out_t_1", "m_t_1", "q_t", "h_t"]]
        )
        model_out_feature: np.array = np.array(self.x_norm[["h_t_1", "h_t_t"]])

        y_ = model_out.predict(
            np.concatenate((model_state_pred, model_out_feature), axis=1),
        )
        y_ = self.y_min + y_ * (self.y_max - self.y_min)
        return y_

    def get_icnn(self, model_s_p, model_out_p):
        """
        Get prediction by ICNN.
        """
        model_state = load_model(self.parent_p.joinpath(model_s_p), compile=False)
        model_out = load_model(
            self.parent_p.joinpath(model_out_p),
            compile=False,
            custom_objects={"ParNonNeg": ParNonNeg},
        )
        model_state_pred = model_state.predict(
            self.x_norm[["tau_in_t_1", "tau_out_t_1", "m_t_1", "q_t", "h_t"]]
        )
        model_out_feature: np.array = np.array(self.x_norm[["h_t_1", "h_t_t"]])

        y_ = model_out.predict(
            np.concatenate((model_state_pred, model_out_feature), axis=1),
        )
        y_ = self.y_min + y_ * (self.y_max - self.y_min)
        return y_

    def plot_h_y(self, y_plnn, y_icnn):
        """
        Plot produced heat (x-axis) versus delivered heat (y-axis).
        """
        title = "Produced heat-Delivered heat"
        plt.plot(self.h, self.y, "b.", label="Real")
        plt.plot(self.h, y_plnn, "r.", label="PLNN")
        plt.plot(self.h, y_icnn, "g.", label="ICNN")
        plt.title(title)
        plt.xlabel("Produced heat [MWh]")
        plt.ylabel("Delivered heat [MWh]")
        plt.legend()
        plt.grid()
        plt.savefig(self.save_path.joinpath(title))
        plt.show()

    def plot_q_y(self):
        """
        Plot heat demand (x-axis) versus delivered heat (y-axis).
        """
        title = "Heat demand-Delivered heat"
        plt.plot(self.q, self.y, ".")
        plt.title(title)
        plt.xlabel("Heat demand [MWh]")
        plt.ylabel("Delivered heat [MWh]")
        plt.grid()
        plt.savefig(self.save_path.joinpath(title))
        plt.show()

    def plot_grid_input_produced_heat(self):
        """
        Plot input to the grid (in MWh, x-axis) versus produced heat (y-axis).
        """
        title = "Grid input-Produced heat"
        plt.plot(self.grid_input, self.h, ".")
        plt.title(title)
        plt.xlabel("Grid input [MWh]")
        plt.ylabel("Produced heat [MWh]")
        plt.grid()
        plt.savefig(self.save_path.joinpath(title))
        plt.show()

    def plot_outlet_plugs_histogram(self):
        """
        Plot histogram of output plugs delays.
        Output plug delay (x-axis): plug[t][time-step it is inserted] - plug[t+1][time-step it is inserted].
        Number of time-steps (y-axis).
        """
        with open(self.result_p.joinpath("supply_pipe_plugs.pickle"), "rb") as f:
            supply_plugs = pickle.load(f)
        N = len(supply_plugs)
        diff = []
        for i in range(N - 1):
            diff_ = int(
                plugs_to_list(supply_plugs[i + 1])[-1][-1]
                - plugs_to_list(supply_plugs[i])[-1][-1]
            )
            diff.append(diff_)
        unique_diff = list(set(diff))
        plt.hist(diff, bins=unique_diff)
        plt.xlabel("Plug delay [h]")
        plt.ylabel("Number of time-steps")
        plt.title("Histogram of output plug delays")
        plt.xticks(range(unique_diff[-1]))
        plt.show()
        diff_count = Counter(diff)
        print(
            "Plugs with 0 and 1 delay make %.2f data"
            % (100 * (diff_count[0] + diff_count[1]) / N)
        )

    def data_inspect(self):
        """
        Data was carefully and randomly generated so that it has as less violations as possible.
        This function plots violations of four metrics -- underdelivered heat, supply inlet temperature,
        supply outlet temperature and mass flow.
        It also calculates percentage of the data with each violation.
        """
        underdelivered_heat = self.data["Underdelivered heat 1"].tolist()
        plt.plot(underdelivered_heat)
        plt.xlabel("Time_steps [h]")
        plt.ylabel("Underdelivered heat [MWh]")
        plt.title("Underdelivered heat")
        plt.show()

        max_supply_in_temp = self.data["Max supply in temp"].tolist()
        plt.plot(max_supply_in_temp)
        plt.xlabel("Time_steps [h]")
        plt.ylabel("Violation of maximum supply inlet temperature [C]")
        plt.show()

        supply_out_temp = self.data["Supply out temp 1"].tolist()
        min_supply_out_temp = [
            s_out_temp - 70 if s_out_temp < 70 else 0 for s_out_temp in supply_out_temp
        ]
        plt.plot(min_supply_out_temp)
        plt.xlabel("Time_steps [h]")
        plt.ylabel("Violation of minimum supply outlet temperature [C]")
        plt.show()

        mass_flow = self.data["Supply mass flow 1"].tolist()
        max_mass_flow = [m - 810 if m > 810 else 0 for m in mass_flow]
        plt.plot(max_mass_flow)
        plt.xlabel("Time-steps [h]")
        plt.ylabel("Violation of maximum mass flow [kg/s]")
        plt.show()

        violation_count = {
            "Underdelivered heat": 0,
            "Max supply inlet temp": 0,
            "Min supply outlet temp": 0,
            "Max mass flow": 0,
        }

        for i in range(self.N):
            if underdelivered_heat[i] < -0.1:
                violation_count["Underdelivered heat"] += 1
            if max_supply_in_temp[i] > 0:
                violation_count["Max supply inlet temp"] += 1
            if min_supply_out_temp[i] < 0:
                violation_count["Min supply outlet temp"] += 1
            if max_mass_flow[i] > 0:
                violation_count["Max mass flow"] += 1
        print(
            "Data that violates underdelivered heat makes %.2f percent of the data"
            % (100 * (violation_count["Underdelivered heat"]) / self.N)
        )
        print(
            "Data that violates maximum supply inlet temperature makes %.2f percent of the data"
            % (100 * (violation_count["Max supply inlet temp"]) / self.N)
        )
        print(
            "Data that violates minimum supply outlet temperature makes %.2f percent of the data"
            % (100 * (violation_count["Min supply outlet temp"]) / self.N)
        )
        print(
            "Data that violates maximum mass flow makes %.2f percent of the data"
            % (100 * (violation_count["Max mass flow"]) / self.N)
        )

    def plot_heat_demand(self):
        """
        Plots real-world and normalized heat demand.
        """
        plt.plot(self.data["Heat demand 1"])
        plt.title("Regular heat demand")
        plt.show()
        plt.plot(self.x_s)
        plt.title("Normalized heat demand")
        plt.show()

    def plot_mass_flow_delta(self):
        """
        Plot histogram of mass flow differences.
        """
        delta_m = []
        mass_flow = self.data["Supply mass flow 1"].tolist()
        for i in range(1, self.N):
            delta_m.append(round(mass_flow[i] - mass_flow[i - 1], 2))
        print("Minimum delta mass flow is {}".format(min(delta_m)))
        print("Maximum delta mass flow is {}".format(max(delta_m)))
        plt.hist(delta_m)
        plt.xlabel("Delta mass flow")
        plt.ylabel("")
        plt.title("Histogram of delta mass flow for L={}".format(PipePreset1["Length"]))
        plt.show()

    def is_there_mass_flow_cap(self):
        """
        Does the mass flow always take the maximum value of 805 when the heat demand
        is underdelivered?
        """
        dict = {"Index": [], "Mass flow": [], "Underdelivered heat": []}
        underdelivered_heat = self.data["Underdelivered heat 1"].tolist()
        mass_flow = self.data["Supply mass flow 1"].tolist()
        for i in range(self.N):
            if underdelivered_heat[i] < 0:
                dict["Index"].append(
                    i + 15
                )  # 15 has to be add in order for this to correspond to the data (but I do not know why)
                dict["Mass flow"].append(mass_flow[i])
                dict["Underdelivered heat"].append(underdelivered_heat[i])
        df = pd.DataFrame.from_dict(dict, orient="columns")
        df.to_csv(self.result_p.joinpath("is_there_cap_on_mass_flow.csv"))

    def eighty_percent(self, time_delays):
        """
        Calculate the time delay containing eighty percent or more of the data.
        """
        N = 0.8 * len(time_delays)
        counter = Counter(time_delays)
        s = 0
        for key in counter.keys():
            s += counter[key]
            if s > N:
                break
        return key

    def plot_time_delays_histogram(self):
        """
        Plot the histogram of time delays -- how long does it take for the water chunk
        to arrive from the inlet to the outlet of the pipe.
        """
        time_delays_ = self.data["Supply time delay 1"]
        time_delays = []
        for time_delay in time_delays_:
            time_delay = string_to_list(time_delay)
            for j in range(len(time_delay)):
                time_delays.append(time_delay[j])
        unique_time_delays = list(set(time_delays))
        eighty_percent_time_delay = self.eighty_percent(time_delays)
        plt.hist(time_delays, bins=unique_time_delays)
        plt.xticks(unique_time_delays, unique_time_delays)
        plt.axvline(
            x=eighty_percent_time_delay, color="red", label="80 percent of the data"
        )
        plt.xlabel("Time delays")
        plt.ylabel("Number of time delays")
        plt.title("Histogram of time delays for L={}".format(PipePreset1["Length"]))
        plt.legend()
        plt.show()

    def does_underdelivered_heat_means_supply_outlet_temp_violation(self):
        """
        The role of this function is to inspect whether violation of the supply outlet temperature (<70C)
        drives underdelivered heat demand.
        The conclusion is: When supply outlet temperature is <70, heat demand will be underdelivered.
        When heat demand is underdelivered, supply outlet temperature can be both less/greater than 70C.
        Therefore, backtracking supply inlet temperature(s) than underdelivers the heat will also
        address the problem of violating supply outlet temperatures.
        """
        underdelivered_heat = self.data["Underdelivered heat 1"].tolist()
        supply_out_temp = self.data["Supply out temp 1"].tolist()
        dict = {"Underdelivered heat": [], "Supply outlet temperature": []}
        for i in range(self.N):
            if underdelivered_heat[i] <= -0.1:
                dict["Underdelivered heat"].append(underdelivered_heat[i])
                dict["Supply outlet temperature"].append(supply_out_temp[i])
        df = pd.DataFrame.from_dict(dict, orient="columns")
        df.to_csv(
            self.result_p.joinpath(
                "relation_underdelivered_heat_supply_outlet_temperature_violation.csv"
            )
        )

    def plot_heat_demand_per_season(self):
        """
        Plot distribution of heat demand depending on the season: winter, spring, autumn, summer.
        """
        path = Path(__file__).parents[4] / "data"
        data = pd.read_csv(
            path / ConfigDataset["FileName"],
            usecols=["Season", "Time_day", "Price", "Heat_demand"],
        )
        season = data["Season"].tolist()
        heat_demand = data["Heat_demand"].tolist()
        heat_demand_per_season = {
            "winter": [],
            "spring": [],
            "summer": [],
            "autumn": [],
        }
        for index, value in enumerate(season):
            if value == 0:
                heat_demand_per_season["winter"].append(heat_demand[index])
            elif value == 1:
                heat_demand_per_season["spring"].append(heat_demand[index])
            elif value == 2:
                heat_demand_per_season["summer"].append(heat_demand[index])
            elif value == 3:
                heat_demand_per_season["autumn"].append(heat_demand[index])
        plt.hist(heat_demand_per_season["winter"])
        plt.xlabel("Heat demand [MW]")
        plt.title("Heat demand during winter")
        plt.show()
        plt.hist(heat_demand_per_season["spring"])
        plt.xlabel("Heat demand [MW]")
        plt.title("Heat demand during spring")
        plt.show()
        plt.hist(heat_demand_per_season["summer"])
        plt.xlabel("Heat demand [MW]")
        plt.title("Heat demand during summer")
        plt.show()
        plt.hist(heat_demand_per_season["autumn"])
        plt.xlabel("Heat demand [MW]")
        plt.title("Heat demand during autumn")
        plt.show()
        print(
            "Number of hours during winter {}".format(
                len(heat_demand_per_season["winter"])
            )
        )
        print(
            "Number of hours during spring {}".format(
                len(heat_demand_per_season["spring"])
            )
        )
        print(
            "Number of hours during summer {}".format(
                len(heat_demand_per_season["summer"])
            )
        )
        print(
            "Number of hours during autumn {}".format(
                len(heat_demand_per_season["autumn"])
            )
        )

    def correlation(self):
        """
        Calculate pearsons, spearmans correlation and partial correlation between different variables.
        """
        dict = {
            "Supply in temp 1": [],
            "Supply out temp 1": [],
            "Supply mass flow 1": [],
            "Ret out temp 1": [],
            "Produced heat": [],
            "Delivered heat 1": [],
        }
        pairs = [
            ["Supply in temp 1", "Produced heat"],
            ["Supply in temp 1", "Delivered heat 1"],
            ["Supply in temp 1", "Supply mass flow 1"],
            ["Supply in temp 1", "Supply out temp 1"],
            ["Produced heat", "Delivered heat 1"],
            ["Supply mass flow 1", "Produced heat"],
            ["Supply mass flow 1", "Delivered heat 1"],
        ]
        pairs_par_corr = [["Supply in temp 1", "Produced heat"], ["Supply in temp 1", "Delivered heat 1"], [ "Supply out temp 1", "Delivered heat 1"]]
        for key in dict.keys():
            for i in range(
                self.cut_index_initial_state + 1,
                self.N + self.cut_index_initial_state - 1,
            ):
                dict[key].append(self.data[key][i])
        for pair in pairs:
            r, p = spearmanr(self.data[pair[0]], self.data[pair[1]])
            print(
                "Correlation coefficient between "
                + pair[0]
                + " and "
                + pair[1]
                + " is "
                + str(r)
                + " and confidence interval is "
                + str(p)
            )
        for pair in pairs_par_corr:
           stats = pg.partial_corr(
            self.data,
            x=pair[0],
            y=pair[1],
            covar="Supply mass flow 1",
            method="spearman",
           )
           print(stats)


if __name__ == "__main__":
    time_delay: int = 10
    plot = PlotDataAnalysis(
        result_p=Path(__file__).parents[4] / "results/constraint_opt",
        columns=[
            "Supply in temp 1",
            "Supply out temp 1",
            "Supply mass flow 1",
            "Supply time delay 1",
            "Grid input",
            "Produced heat",
            "Delivered heat 1",
            "Heat demand 1",
            "Max supply in temp",
            "Underdelivered heat 1",
            "Ret out temp 1",
        ],
        time_delay=time_delay,
    )
    """
    y_plnn = plot.get_plnn(
        model_s_p="model_state_plnn.h5", model_out_p="model_output_plnn.h5"
    )
    y_icnn = plot.get_icnn(
        model_s_p="model_state_icnn.h5", model_out_p="model_output_icnn.h5"
    )
    plot.normalize()
    plot.plot_h_y(y_plnn, y_icnn)
    plot.plot_q_y()
    plot.plot_grid_input_produced_heat()
    plot.plot_plugs_histogram()
    plot.plot_time_delays_histogram()
    plot.plot_mass_flow_delta()
    plot.data_inspect()
    """
    plot.plot_time_delays_histogram()

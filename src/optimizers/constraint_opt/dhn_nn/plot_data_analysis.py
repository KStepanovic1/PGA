from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.plot import Plot


class PlotDataAnalysis(Plot):
    """
    Analyse the data from the dataset.
    """

    def __init__(self, result_p, plot_p):
        super().__init__(result_p, plot_p)
        self.data = pd.read_csv(
            self.result_p.joinpath(
                "data_num_1_heat_demand_real_world_for_L=4000_time_interval_3600_max_Q_70MW_deltaT_5C.csv"
            )
        )

    def plot_mass_flow(self):
        """
        Plot mass flow of entire dataset.
        """
        mass_flow = self.data.loc[:, "Supply mass flow 1"]
        plt.plot(mass_flow)
        plt.xlabel("Time-step [h]")
        plt.ylabel("Mass flow [kg/s]")
        plt.title("Maximal mass flow is {} kg/s".format(max(list(mass_flow))))
        plt.savefig(self.plot_p.joinpath("Mass flow.png"))
        plt.show()

    def plot_supply_inlet_temperature(self):
        """
        Plot supply inlet of entire dataset.
        """
        tau_in_ = []
        time = []
        tau_in = self.data.loc[:, "Supply in temp 1"]
        for i in range(len(tau_in)):
            if i % 10 == 0:
                tau_in_.append(tau_in[i])
                time.append(i)
        plt.plot(time, tau_in_)
        plt.xlabel("Time-step [h]")
        plt.ylabel("Supply inlet temperature [C]")
        plt.title("Minimal supply inlet temperature is {} C".format(min(list(tau_in))))
        plt.savefig(self.plot_p.joinpath("Supply inlet temperature.png"))
        plt.show()

    def plot_heat_demand(self):
        """
        Plot heat demand of entire dataset.
        """
        heat_demand = self.data[["Heat demand 1"]]
        fig, ax = plt.subplots(figsize=(18, 6))
        plt.plot(heat_demand)
        plt.xlabel("Time-step [h]")
        plt.ylabel("Heat demand [MWh]")
        plt.axvline(10545, label="Testing dataset", color="red")
        plt.axvline(1681, label="Year 1", color="green")
        plt.axvline(4560, label="Year 2", color="green")
        plt.axvline(7444, label="Year 3", color="green")
        plt.axvline(10325, label="Year 4", color="green")
        plt.axvline(len(heat_demand), label="Year 3", color="green")
        plt.title("Heat demand")
        plt.legend()
        plt.savefig(self.plot_p.joinpath("Heat demand.png"))
        plt.show()

    def calculate_mean_and_median_heat_demand(self):
        heat_demand = self.data[["Heat demand 1"]]
        heat_demand = np.array(heat_demand)
        mean = np.mean(heat_demand)
        median = np.median(heat_demand)
        print("Mean is {}".format(mean))
        print("Median is {}".format(median))

    def plot_period_heat_demand(self, start, end):
        """
        Plot heat demand during specified time-period [start, end].
        """
        heat_demand = self.data.loc[start : end - 1, "Heat demand 1"]
        fig, ax = plt.subplots(figsize=(18, 6))
        plt.plot(heat_demand)
        plt.xlabel("Time-step [h]")
        plt.ylabel("Heat demand [MWh]")
        plt.title("Heat demand")
        plt.legend()
        plt.savefig(
            self.plot_p.joinpath(
                "Period of heat demand from {} to {}.png".format(start, end)
            )
        )
        plt.show()

    def plot_weekly_heat_demand(self):
        """
        Plot weekly heat demand of the testing dataset.
        Under the assumption that the first data point corresponds to the midnight (00:00 - 01:00),
        the first week starts in the first midnight data point after the start of the testing dataset.
        """
        day = 24
        week = day * 7
        start = 10324 # corresponds to the hour 16.11.2019 00:00 - 16.11.2019 01:00
        N_week = int((len(self.data) - start) / week)
        for i in range(N_week):
            weekly_heat_demand = list(self.data.loc[
                start + i * week : start + (i + 1) * week, "Heat demand 1"
            ])
            print(weekly_heat_demand)
            fig, ax = plt.subplots(figsize=(18, 6))
            plt.plot(weekly_heat_demand)
            plt.xlabel("Time-step [h]")
            plt.ylabel("Heat demand [MWh]")
            plt.title("Weekly heat demand variation: Week {}".format(i + 1))
            plt.xticks(list(range(1, week + 1, 5)), [str(i) for i in range(1, week + 1,5)])
            plt.savefig(
                self.plot_p.joinpath("Heat demand during week {}.png".format(i + 1))
            )
            plt.show()


if __name__ == "__main__":
    plot = PlotDataAnalysis(
        result_p=Path(__file__).parents[4] / "results/constraint_opt",
        plot_p=Path(__file__).parents[4] / "plots/constraint_opt/data_analysis",
    )
    # plot.plot_supply_inlet_temperature()
    # plot.plot_heat_demand()
    # plot.plot_mass_flow()
    # plot.calculate_mean_and_median_heat_demand()
    # plot.plot_period_heat_demand(start=1900, end=1972)
    plot.plot_weekly_heat_demand()

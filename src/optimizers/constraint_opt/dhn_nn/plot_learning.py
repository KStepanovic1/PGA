from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *
from src.optimizers.constraint_opt.dhn_nn.plot import Plot


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


class PlotLearning(Plot):
    def __init__(self, result_p, plot_p):
        super(PlotLearning, self).__init__(result_p, plot_p)
        self.result_p: Path = result_p
        self.plot_p: Path = plot_p
        self.result_plnn_milp_p: Path = result_p.joinpath("plnn_milp")
        self.result_icnn_gd_p: Path = result_p.joinpath("icnn_gd")
        self.result_plnn_gd_p: Path = result_p.joinpath("plnn_gd")
        self.parent_p: Path = Path(__file__).parent

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
            "Supply_inlet_temp": "Root mean squared error [$^{\circ}$C]",
            "Supply_outlet_temp": "Root mean squared error [$^{\circ}$C]",
            "Mass_flow": "Root mean squared error [kg/s]",
            "Delivered_heat": "Root mean squared error [MW]",
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
            neuron_ = Plot.create_neuron_string(neuron)
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
            up_bound_plnn, down_bound_plnn = calculate_plot_bounds(
                plnn_one_step_means.data[column], plnn_one_step_std.data[column]
            )
            up_bound_icnn, down_bound_icnn = calculate_plot_bounds(
                icnn_one_step_means.data[column], icnn_one_step_std.data[column]
            )
            (
                up_bound_monotonic_icnn,
                down_bound_monotonic_icnn,
            ) = calculate_plot_bounds(
                monotonic_icnn_one_step_means.data[column],
                monotonic_icnn_one_step_std.data[column],
            )
            x = list(range(len(neurons)))
            #ax.plot(x, plnn_one_step_means.data[column], color="b", label="PLNN")
            #ax.fill_between(x, down_bound_plnn, up_bound_plnn, color="b", alpha=0.1)
            #ax.plot(x, icnn_one_step_means.data[column], color="r", label="ICNN")
            #ax.fill_between(x, down_bound_icnn, up_bound_icnn, color="r", alpha=0.1)
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
            # ax.set_title(title)
            # ax.legend()
            ax.set_xlabel("Neural network size", fontsize=14)
            ax.set_ylabel(y_name[column], fontsize=14)
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
            plt.savefig(self.plot_p.joinpath("predictions").joinpath(title+".pdf"))
            plt.show()
            title = title_name[column] + " six steps prediction"
            fig, ax = plt.subplots()
            up_bound_plnn, down_bound_plnn = Plot.calculate_plot_bounds(
                plnn_multi_step_means.data[column], plnn_multi_step_std.data[column]
            )
            up_bound_icnn, down_bound_icnn = Plot.calculate_plot_bounds(
                icnn_multi_step_means.data[column], icnn_multi_step_std.data[column]
            )
            (
                up_bound_monotonic_icnn,
                down_bound_monotonic_icnn,
            ) = Plot.calculate_plot_bounds(
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
            neuron_ = Plot.create_neuron_string(neuron)
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
            up_bound_plnn, down_bound_plnn = Plot.calculate_plot_bounds(
                plnn_one_step_means.data[column], plnn_one_step_std.data[column]
            )
            up_bound_icnn, down_bound_icnn = Plot.calculate_plot_bounds(
                icnn_one_step_means.data[column], icnn_one_step_std.data[column]
            )
            (
                up_bound_monotonic_icnn,
                down_bound_monotonic_icnn,
            ) = Plot.calculate_plot_bounds(
                monotonic_icnn_one_step_means.data[column],
                monotonic_icnn_one_step_std.data[column],
            )
            (
                up_bound_plnn_with_q,
                down_bound_plnn_with_q,
            ) = Plot.calculate_plot_bounds(
                plnn_one_step_means_with_q.data[column],
                plnn_one_step_std_with_q.data[column],
            )
            (
                up_bound_icnn_with_q,
                down_bound_icnn_with_q,
            ) = Plot.calculate_plot_bounds(
                icnn_one_step_means_with_q.data[column],
                icnn_one_step_std_with_q.data[column],
            )
            (
                up_bound_monotonic_icnn_with_q,
                down_bound_monotonic_icnn_with_q,
            ) = Plot.calculate_plot_bounds(
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
            up_bound_plnn, down_bound_plnn = Plot.calculate_plot_bounds(
                plnn_multi_step_means.data[column], plnn_multi_step_std.data[column]
            )
            up_bound_icnn, down_bound_icnn = Plot.calculate_plot_bounds(
                icnn_multi_step_means.data[column], icnn_multi_step_std.data[column]
            )
            (
                up_bound_monotonic_icnn,
                down_bound_monotonic_icnn,
            ) = Plot.calculate_plot_bounds(
                monotonic_icnn_multi_step_means.data[column],
                monotonic_icnn_multi_step_std.data[column],
            )
            # with q sequence
            (
                up_bound_plnn_with_q,
                down_bound_plnn_with_q,
            ) = Plot.calculate_plot_bounds(
                plnn_multi_step_means_with_q.data[column],
                plnn_multi_step_std_with_q.data[column],
            )
            (
                up_bound_icnn_with_q,
                down_bound_icnn_with_q,
            ) = Plot.calculate_plot_bounds(
                icnn_multi_step_means_with_q.data[column],
                icnn_multi_step_std_with_q.data[column],
            )
            (
                up_bound_monotonic_icnn_with_q,
                down_bound_monotonic_icnn_with_q,
            ) = Plot.calculate_plot_bounds(
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
            neuron_ = Plot.create_neuron_string(neuron)
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
        up_bound_plnn_pred, down_bound_plnn_pred = Plot.calculate_plot_bounds(
            plnn_state_pred_mean, plnn_state_pred_std
        )
        up_bound_icnn_pred, down_bound_icnn_pred = Plot.calculate_plot_bounds(
            icnn_state_pred_mean, icnn_state_pred_std
        )
        (
            up_bound_monotonic_icnn_pred,
            down_bound_monotonic_icnn_pred,
        ) = Plot.calculate_plot_bounds(
            monotonic_icnn_state_pred_mean, monotonic_icnn_state_pred_std
        )
        up_bound_plnn_real, down_bound_plnn_real = Plot.calculate_plot_bounds(
            plnn_real_mean, plnn_real_std
        )
        up_bound_icnn_real, down_bound_icnn_real = Plot.calculate_plot_bounds(
            icnn_real_mean, icnn_real_std
        )
        (
            up_bound_monotonic_icnn_real,
            down_bound_monotonic_icnn_real,
        ) = Plot.calculate_plot_bounds(
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
            [1],
            [1,1],
            [3],
            [5],
            [5,3],
            [10],
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
                    + Plot.create_neuron_string(neuron)
                    + ext
                    + ".csv"
                )
                val_loss_: str = (
                    "val_loss_"
                    + nn_type
                    + fun_type
                    + "neurons_"
                    + Plot.create_neuron_string(neuron)
                    + ext
                    + ".csv"
                )
                val_loss_early_stop_: str = (
                    "val_loss_"
                    + nn_type
                    + fun_type
                    + "neurons_"
                    + Plot.create_neuron_string(neuron)
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
                val_loss_summary = Plot.summarize_validation_loss(
                    val_loss_summary,
                    val_loss,
                    val_loss_early_stop,
                    fun_type,
                )
                # plot training and validation loss without early stopping with the early stopping hyperparameter as vertical line
                # only for the first iteration
                title = nn_type + fun_type + "train_val_loss"
                name = title + "_neurons_" + Plot.create_neuron_string(neuron) + ext
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
                #ax.set_title(title)
                ax.legend()
                #plt.savefig(
                 #   self.plot_p.joinpath("training_validation_loss").joinpath(
                 #       name + "_zoom_in_3.png"
                 #   )
                #)
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
            #ax.set_title(title)
            plt.xticks(fontsize=7)
            plt.savefig(
                self.plot_p.joinpath("training_validation_loss").joinpath(
                    title + ".pdf"
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
            neuron_ = Plot.create_neuron_string(neuron)
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
                up_bound, down_bound = Plot.calculate_plot_bounds(
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
                        neuron_ = Plot.create_neuron_string(neuron)
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
                    up_bound, down_bound = Plot.calculate_plot_bounds(
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
                    neuron_ = Plot.create_neuron_string(neuron)
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
                up_bound, down_bound = Plot.calculate_plot_bounds(
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
                    neuron_ = Plot.create_neuron_string(neuron)
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
                up_bound, down_bound = Plot.calculate_plot_bounds(
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

    def plot_rmse_error_as_the_function_of_heat_demand(self, keys, dataset):
        """
        Plot mean of root mean squared errors of prediction for PLNN and ICNN networks as the function
        of heat demand span.
        """
        titles = {
            "tau_in": "Supply inlet temperature",
            "tau_out": "Supply outlet temperature",
            "m": "Mass flow",
            "y": "Delivered heat",
            "total": "Supply inlet temperature+Supply outlet temperature+Mass flow+Delivered heat",
        }
        (
            dict_mean_plnn,
            dict_std_plnn,
        ) = self.get_rmse_error_as_the_function_of_heat_demand(
            folder="plnn", dataset=dataset, ad="plnn", keys=keys
        )
        (
            dict_mean_icnn,
            dict_std_icnn,
        ) = self.get_rmse_error_as_the_function_of_heat_demand(
            folder="relax_monotonic_icnn",
            dataset=dataset,
            ad="monotonic_icnn",
            keys=keys,
        )
        (
            dict_mean_monotonic_icnn,
            dict_std_monotonic_icnn,
        ) = self.get_rmse_error_as_the_function_of_heat_demand(
            folder="monotonic_icnn",
            dataset=dataset,
            ad="monotonic_icnn",
            keys=keys,
        )
        for key in keys:
            fig, ax = plt.subplots(figsize=(18, 6))
            plt.bar(
                list(range(len(dict_mean_plnn[key].values()))),
                dict_mean_plnn[key].values(),
                yerr=dict_std_plnn[key].values(),
                width=0.2,
                color="blue",
                label="PLNN",
                capsize=10,
            )
            plt.bar(
                [x + 0.2 for x in list(range(len(dict_mean_icnn[key].values())))],
                dict_mean_icnn[key].values(),
                yerr=dict_std_icnn[key].values(),
                width=0.2,
                color="red",
                label="ICNN",
                capsize=10,
            )
            """
            plt.bar(
                [x + 0.4 for x in list(range(len(dict_mean_monotonic_icnn[key].values())))],
                dict_mean_monotonic_icnn[key].values(),
                yerr=dict_std_monotonic_icnn[key].values(),
                width=0.2,
                color="green",
                label="Monotonic ICNN",
                capsize=10,
            )
            """
            plt.xticks(
                list(range(len(dict_mean_plnn[key].values()))),
                [
                    "0-5[MW]",
                    "5-10[MW]",
                    "10-15[MW]",
                    "15-20[MW]",
                    "20-25[MW]",
                    "25-30[MW]",
                    "30-35[MW]",
                    "35-40[MW]",
                    "40-45[MW]",
                    "45-50[MW]",
                    "50-55[MW]",
                    "55-60[MW]",
                    "60-65[MW]",
                    "65-70[MW]",
                ],
            )
            plt.ylabel("Root mean squared error")
            plt.title(titles[key])
            plt.legend()
            plt.savefig(
                self.plot_p.joinpath("predictions").joinpath(
                    titles[key]
                    + "plnn icnn prediction over the "
                    + dataset
                    + " as the function of heat demand "
                    + ".png"
                )
            )
            plt.show()

    def get_rmse_error_as_the_function_of_heat_demand(self, folder, dataset, ad, keys):
        """
        Get mean of RMSE errors for supply inlet temperature, supply outlet temperature, mass flow and delivered heat
        as the function of span of mean heat demand.
        """
        dict_, dict_mean, dict_std = {}, {}, {}
        heat_demand = self.data["Heat demand 1"]
        # heat demand interval
        interval = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        # read predictions
        files = os.listdir(
            self.result_p.joinpath("evaluate_rolling_horizon_predictions")
            .joinpath(folder)
            .joinpath(dataset)
        )
        predictions = []
        for file in files:
            with open(
                self.result_p.joinpath("evaluate_rolling_horizon_predictions")
                .joinpath(folder)
                .joinpath(dataset)
                .joinpath(file),
                "rb",
            ) as file:
                predictions_ = pickle.load(file)[keys]
            predictions.append(predictions_)
        predictions = pd.concat(predictions, ignore_index=True)
        for key in keys:
            dict_[key] = {}
            dict_mean[key] = {}
            dict_std[key] = {}
            for i in interval[:-1]:
                dict_[key][str(i)] = []
                dict_mean[key][str(i)] = 0
                dict_std[key][str(i)] = 0
        for i in range(1, predictions.shape[0] - TimeParameters["PlanningHorizon"]):
            heat_demand_mean = np.mean(
                heat_demand[i : i + TimeParameters["PlanningHorizon"]]
            )
            for j in range(len(interval) - 1):
                if (
                    heat_demand_mean >= interval[j]
                    and heat_demand_mean < interval[j + 1]
                ):
                    for key in keys:
                        dict_[key][str(interval[j])].append(
                            np.mean(
                                predictions[key][
                                    i - 1 : i - 1 + TimeParameters["PlanningHorizon"]
                                ]
                            )
                        )
                    break
        # calculate the mean of the root mean squared error
        for key in keys:
            fig, ax = plt.subplots(figsize=(18, 6))
            for i in interval[:-1]:
                if len(dict_[key][str(i)]) > 0:
                    dict_mean[key][str(i)] = sum(dict_[key][str(i)]) / len(
                        dict_[key][str(i)]
                    )
                    dict_std[key][str(i)] = np.std(np.array(dict_[key][str(i)]))
                else:
                    dict_mean[key][str(i)] = np.nan
                    dict_std[key][str(i)] = np.nan
        return dict_mean, dict_std


if __name__ == "__main__":
    result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
    plot_p: Path = Path(__file__).parents[4] / "plots/constraint_opt"
    plot = PlotLearning(result_p=result_p, plot_p=plot_p)

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
    plot.plot_training_validation_loss(
        early_stop=False, nn_type="monotonic_icnn", num_iter=3
    )
    # plot.inspect_monotonic_heat_restriction(
    #    neurons=[[1], [1, 1], [3], [5], [5, 3], [10], [10, 10], [50, 50]], time_delay=10
    # )
    #plot.plot_rmse_error_as_the_function_of_heat_demand(
    #    dataset="training_dataset", keys=["tau_in", "tau_out", "m", "y", "total"]
    #)
    # plot.plot_predictions()

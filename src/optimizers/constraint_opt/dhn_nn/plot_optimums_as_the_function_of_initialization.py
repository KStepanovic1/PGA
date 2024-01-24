from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *
from src.optimizers.constraint_opt.dhn_nn.plot import Plot


class PlotOptimums(Plot):
    """
    Plot operation costs and operation costs+violations as the function of initialization.
    """

    def __init__(self, result_p, plot_p):
        super().__init__(result_p, plot_p)
        self.N_models = 3  # number of NN models
        self.result_p = self.result_p.joinpath("relax_monotonic_icnn_gd")

    def plot_operation_cost(self, opt_step, initializations, neuron):
        """
        Plot operation cost and operation cost+penalty as the function of ICNN-GD optimization initialization
        for different ICNN models.
        """
        # create dictionary for colors
        colors = {
            "0.125": "#FF6103",
            "0.25": "k",
            "0.375": "m",
            "0.5": "c",
            "0.625": "y",
            "0.75": "g",
            "0.875": "b",
            "0.9375": "#7FFF00",
            "1": "r",
        }
        # create dictionary for saving results
        dict = {}
        for initialization in initializations:
            dict[str(initialization)] = {}
            dict[str(initialization)]["Profit"] = []
            dict[str(initialization)]["Profit plus delivered heat penalty"] = []
        for initialization in initializations:
            result_p_initialization = self.result_p.joinpath(
                "initialization_{}".format(initialization)
            ).joinpath("MPC_episode_length_72_hours")
            for model_num in range(self.N_models):
                optimizer = (
                    "{}_icnn_gd_init_neurons_".format(model_num)
                    + Plot.create_neuron_string(neuron)
                    + "_opt_step"
                )
                # sum operation cost for all days for one DNN model
                operation_cost = Plot.sum(
                    path=result_p_initialization,
                    optimizer=optimizer,
                    coeff_day=opt_step,
                    column="Profit",
                    null_solve=False,
                )
                dict[str(initialization)]["Profit"].append(operation_cost)
                # sum of violations percentage for all days
                delivered_heat_violation = Plot.sum(
                    path=result_p_initialization,
                    optimizer=optimizer,
                    coeff_day=opt_step,
                    column="Delivered_heat_violation",
                    null_solve=False,
                )
                dict[str(initialization)]["Profit plus delivered heat penalty"].append(
                    operation_cost + pow(delivered_heat_violation, 2)
                )

        fig, ax = plt.subplots(figsize=(18, 6))
        counter = 0
        x_ticks = ["Model 0", "Model 1", "Model 2"]
        x = [xx + 10 * xx for xx in range(self.N_models)]
        for model in range(self.N_models):
            delta_xlabel = 0
            for initialization in initializations:
                if counter == 0:
                    plt.bar(
                        counter + delta_xlabel,
                        dict[str(initialization)]["Profit"][model],
                        color=colors[str(initialization)],
                        label="initialization={}MW".format(
                            re_normalize_variable(
                                var=initialization,
                                min=self.produced_heat_min,
                                max=self.produced_heat_max,
                            )
                        ),
                        width=0.2,
                    )
                    delta_xlabel += 0.5
                else:
                    plt.bar(
                        counter + delta_xlabel,
                        dict[str(initialization)]["Profit"][model],
                        color=colors[str(initialization)],
                        width=0.2,
                    )
                    delta_xlabel += 0.5
            counter += 10
        plt.xlabel("Models of the relaxed input convex neural network")
        plt.ylabel(
            r"$J = \sum_{\tau=t}^T a_1 \cdot h_\tau+a_2 \cdot p_\tau -c_\tau \cdot p_\tau$ [e]"
        )
        plt.legend()
        plt.title(
            r"$J = \sum_{\tau=t}^T a_1 \cdot h_\tau+a_2 \cdot p_\tau -c_\tau \cdot p_\tau$"
        )
        plt.xticks(x, x_ticks)
        plt.savefig(
            self.plot_p.joinpath(
                "Relax ICNN Operation cost as the function of initialization opt step={}.png".format(
                    opt_step[0]
                )
            )
        )
        plt.show()

        fig, ax = plt.subplots(figsize=(18, 6))
        counter = 0
        x_ticks = ["Model 0", "Model 1", "Model 2"]
        x = [xx + 10 * xx for xx in range(self.N_models)]
        for model in range(self.N_models):
            delta_xlabel = 0
            for initialization in initializations:
                if counter == 0:
                    plt.bar(
                        counter + delta_xlabel,
                        dict[str(initialization)]["Profit plus delivered heat penalty"][
                            model
                        ],
                        color=colors[str(initialization)],
                        label="initialization={}".format(
                            re_normalize_variable(
                                var=initialization,
                                min=self.produced_heat_min,
                                max=self.produced_heat_max,
                            )
                        ),
                        width=0.2,
                    )
                    delta_xlabel += 0.5
                else:
                    plt.bar(
                        counter + delta_xlabel,
                        dict[str(initialization)]["Profit plus delivered heat penalty"][
                            model
                        ],
                        color=colors[str(initialization)],
                        width=0.2,
                    )
                    delta_xlabel += 0.5
            counter += 10
        plt.xlabel("Models of the relaxed input convex neural network")
        plt.ylabel(
            r"$J = \sum_{\tau=t}^T a_1 \cdot h_\tau+a_2 \cdot p_\tau -c_\tau \cdot p_\tau + C \cdot (y_\tau-q_\tau)^2$ [e]"
        )
        plt.legend()
        plt.title(
            r"$J = \sum_{\tau=t}^T a_1 \cdot h_\tau+a_2 \cdot p_\tau -c_\tau \cdot p_\tau + C \cdot (y_\tau-q_\tau)^2$"
        )
        plt.xticks(x, x_ticks)
        plt.ylim(-10000, 10000)
        plt.savefig(
            self.plot_p.joinpath(
                "Relax ICNN Operation cost+delivered heat penalty as the function of initialization opt step={}.png".format(
                    opt_step[0]
                )
            )
        )
        plt.show()


if __name__ == "__main__":
    result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
    plot_p: Path = (
        Path(__file__).parents[4]
        / "plots/constraint_opt/optimums_as_the_function_of_initialization"
    )
    plot = PlotOptimums(result_p=result_p, plot_p=plot_p)
    plot.plot_operation_cost(
        opt_step=[1024],
        initializations=[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375, 1],
        neuron=[50, 50],
    )

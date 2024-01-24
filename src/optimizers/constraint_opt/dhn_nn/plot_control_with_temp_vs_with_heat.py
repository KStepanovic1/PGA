from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *
from src.optimizers.constraint_opt.dhn_nn.plot import Plot


class PlotControlWithTempVsControlWithHeat(Plot):
    """
    Analyse the difference in terms of operation cost and constraints violations
    between control with temperature and control with heat of PLNN-MILP planner.
    """

    def __init__(self, result_p, plot_p):
        super().__init__(result_p, plot_p)
        self.result_p = self.result_p.joinpath("plnn_milp/MPC_episode_length_72_hours")
        self.plot_p = self.plot_p.joinpath("plnn_milp_control_with_temp_vs_heat")
        self.N_models = 3

    def plot_operation_cost(self, neurons):
        """
        Plot operation cost of PLNN-MILP when the simulation is carried out
        with supply inlet temperature as decision variable and heat as decision variable.
        """
        x = list(range(len(neurons)))
        optimizers = ["control with temperature", "control with heat"]
        paths = {
            "control with heat": self.result_p.joinpath("control with heat"),
            "control with temperature": self.result_p.joinpath(
                "control with temperature"
            ),
        }
        results = {
            "control with heat": {
                "mean": [],
                "std": [],
                "upper_bound": [],
                "lower_bound": [],
            },
            "control with temperature": {
                "mean": [],
                "std": [],
                "upper_bound": [],
                "lower_bound": [],
            },
        }
        for optimizer in optimizers:
            for neuron in neurons:
                operation_cost = []
                for model in range(self.N_models):
                    optimizer_ = (
                        "{}_".format(model)
                        + "plnn_milp_init_neurons_"
                        + Plot.create_neuron_string(neuron)
                        + "_opt_step"
                    )
                    operation_cost.append(
                        Plot.sum(
                            path=paths[optimizer],
                            optimizer=optimizer_,
                            coeff_day=opt_steps["math_opt"],
                            column="Profit",
                        )
                    )
                mean = sum(operation_cost) / len(operation_cost)
                results[optimizer]["mean"].append(mean)
                std = np.std(np.array(operation_cost))
                results[optimizer]["std"].append(std)
            results[optimizer]["upper_bound"] = calculate_plot_bounds(
                results[optimizer]["mean"], results[optimizer]["std"]
            )[0]
            results[optimizer]["lower_bound"] = calculate_plot_bounds(
                results[optimizer]["mean"], results[optimizer]["std"]
            )[1]
        fig, ax = plt.subplots()
        plt.plot(
            x,
            results["control with temperature"]["mean"],
            label="Simulation with temperature",
            color="b",
        )
        plt.fill_between(
            x,
            results["control with temperature"]["lower_bound"],
            results["control with temperature"]["upper_bound"],
            color="b",
            alpha=0.1,
        )
        plt.plot(
            x,
            results["control with heat"]["mean"],
            label="Simulation with heat",
            color="r",
        )
        plt.fill_between(
            x,
            results["control with heat"]["lower_bound"],
            results["control with heat"]["upper_bound"],
            color="r",
            alpha=0.1,
        )
        ax.set_title("Operational cost")
        ax.set_xticks(x)
        ax.set_ylabel("Operational cost [e]")
        ax.set_xlabel("Neural network size")
        ax.set_xticklabels(("$[1]$", "$[1,1]$", "$[10]$"))
        plt.legend()
        plt.grid()
        plt.savefig(
            self.plot_p.joinpath(
                "Operational cost from time-step {} png".format(
                    opt_steps["math_opt"][0]
                )
            )
        )
        plt.show()

    def plot_violations(self, neurons):
        """
        Plot violations of PLNN-MILP when the simulation is carried out
        with supply inlet temperature as decision variable and heat as decision variable.
        """
        x = list(range(len(neurons)))
        optimizers = ["control with temperature", "control with heat"]
        violations = [
            "Supply_inlet_violation",
            "Supply_outlet_violation",
            "Delivered_heat_violation",
            "Mass_flow_violation",
        ]
        paths = {
            "control with heat": self.result_p.joinpath("control with heat"),
            "control with temperature": self.result_p.joinpath(
                "control with temperature"
            ),
        }
        colors = {"control with temperature": "b", "control with heat": "r"}
        labels = {
            "control with temperature": "Simulation with temperature",
            "control with heat": "Simulation with heat",
        }
        titles = {
            "Supply_inlet_violation": "Supply inlet temperature violation",
            "Supply_outlet_violation": "Supply outlet temperature violation",
            "Delivered_heat_violation": "Delivered heat violation",
            "Mass_flow_violation": "Mass flow violation",
        }
        results = {
            "control with temperature": {
                "Supply_inlet_violation": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
                "Supply_outlet_violation": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
                "Delivered_heat_violation": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
                "Mass_flow_violation": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
            },
            "control with heat": {
                "Supply_inlet_violation": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
                "Supply_outlet_violation": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
                "Delivered_heat_violation": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
                "Mass_flow_violation": {
                    "mean": [],
                    "std": [],
                    "upper_bound": [],
                    "lower_bound": [],
                },
            },
        }
        for optimizer in optimizers:
            for neuron in neurons:
                for violation_type in violations:
                    average_violation = []
                    for model in range(self.N_models):
                        optimizer_ = (
                            "{}_".format(model)
                            + "plnn_milp_init_neurons_"
                            + Plot.create_neuron_string(neuron)
                            + "_opt_step"
                        )
                        average_violation.append(
                            self.average(
                                path=paths[optimizer],
                                optimizer=optimizer_,
                                coeff_day=opt_steps["math_opt"],
                                column=violation_type,
                                null_solve=False,
                            )
                        )
                    mean = round(sum(average_violation) / len(average_violation), 1)
                    std = round(np.std(np.array(average_violation)), 1)
                    results[optimizer][violation_type]["mean"].append(mean)
                    results[optimizer][violation_type]["std"].append(std)
            for violation_type in violations:
                results[optimizer][violation_type][
                    "upper_bound"
                ] = calculate_plot_bounds(
                    results[optimizer][violation_type]["mean"],
                    results[optimizer][violation_type]["std"],
                )[
                    0
                ]
                results[optimizer][violation_type][
                    "lower_bound"
                ] = calculate_plot_bounds(
                    results[optimizer][violation_type]["mean"],
                    results[optimizer][violation_type]["std"],
                )[
                    1
                ]
        for violation in violations:
            fig, ax = plt.subplots()
            for optimizer in optimizers:
                plt.plot(
                    x,
                    results[optimizer][violation]["mean"],
                    label=labels[optimizer],
                    color=colors[optimizer],
                )
                plt.fill_between(
                    x,
                    results[optimizer][violation]["lower_bound"],
                    results[optimizer][violation]["upper_bound"],
                    color=colors[optimizer],
                    alpha=0.1,
                )
            ax.set_title(titles[violation])
            ax.set_xticks(x)
            ax.set_ylabel("Violation percentage [%]")
            ax.set_xlabel("Neural network size")
            ax.set_xticklabels(("$[1]$", "$[1,1]$", "$[10]$"))
            plt.legend()
            plt.savefig(
                self.plot_p.joinpath(
                    titles[violation]
                    + " from time-step {}.png".format(opt_steps["math_opt"][0])
                )
            )
            plt.show()

    def plot_optimization_operation_cost_plus_violations(
        self, neurons, supply_inlet_violation_penalty, delivered_heat_penalty
    ):
        """
        Add penalty*violations to operation cost, and plot results.
        """
        x = list(range(len(neurons)))
        optimizers = ["control with temperature", "control with heat"]
        violations = {
            "Supply_inlet_violation": 0,
            "Supply_outlet_violation": 0,
            "Mass_flow_violation": 0,
            "Delivered_heat_violation": 0,
        }
        paths = {
            "control with heat": self.result_p.joinpath("control with heat"),
            "control with temperature": self.result_p.joinpath(
                "control with temperature"
            ),
        }
        results = {
            "control with heat": {
                "mean": [],
                "std": [],
                "upper_bound": [],
                "lower_bound": [],
            },
            "control with temperature": {
                "mean": [],
                "std": [],
                "upper_bound": [],
                "lower_bound": [],
            },
        }
        for optimizer in optimizers:
            for neuron in neurons:
                cost_plus_violation = []
                for model in range(self.N_models):
                    for violation in violations.keys():
                        violations[violation] = 0
                    optimizer_ = (
                        "{}_".format(model)
                        + "plnn_milp_init_neurons_"
                        + Plot.create_neuron_string(neuron)
                        + "_opt_step"
                    )
                    operation_cost = Plot.sum(
                        path=paths[optimizer],
                        optimizer=optimizer_,
                        coeff_day=opt_steps["math_opt"],
                        column="Profit",
                    )
                    for violation in violations.keys():
                        violations[violation] = Plot.sum(
                            path=paths[optimizer],
                            optimizer=optimizer_,
                            coeff_day=opt_steps["math_opt"],
                            column=violation,
                        )
                    cost_plus_violation_ = Plot.sum_cost_and_violations(
                        operation_cost=operation_cost,
                        supply_inlet_violation=violations["Supply_inlet_violation"],
                        supply_outlet_violation=violations["Supply_outlet_violation"],
                        mass_flow_violation=violations["Mass_flow_violation"],
                        delivered_heat_violation=violations["Delivered_heat_violation"],
                        supply_inlet_violation_penalty=supply_inlet_violation_penalty,
                        delivered_heat_penalty=delivered_heat_penalty,
                    )
                    cost_plus_violation.append(cost_plus_violation_)
                mean = sum(cost_plus_violation) / len(cost_plus_violation)
                results[optimizer]["mean"].append(mean)
                std = np.std(np.array(cost_plus_violation))
                results[optimizer]["std"].append(std)
            results[optimizer]["upper_bound"] = calculate_plot_bounds(
                results[optimizer]["mean"], results[optimizer]["std"]
            )[0]
            results[optimizer]["lower_bound"] = calculate_plot_bounds(
                results[optimizer]["mean"], results[optimizer]["std"]
            )[1]
        fig, ax = plt.subplots()
        plt.plot(
            x,
            results["control with temperature"]["mean"],
            label="Simulation with temperature",
            color="b",
        )
        plt.fill_between(
            x,
            results["control with temperature"]["lower_bound"],
            results["control with temperature"]["upper_bound"],
            color="b",
            alpha=0.1,
        )
        plt.plot(
            x,
            results["control with heat"]["mean"],
            label="Simulation with heat",
            color="r",
        )
        plt.fill_between(
            x,
            results["control with heat"]["lower_bound"],
            results["control with heat"]["upper_bound"],
            color="r",
            alpha=0.1,
        )
        ax.set_title("Operational cost+penalty")
        ax.set_xticks(x)
        ax.set_ylabel("Operational cost+penalty [e]")
        ax.set_xlabel("Neural network size")
        ax.set_xticklabels(("$[1]$", "$[1,1]$", "$[10]$"))
        plt.legend()
        plt.grid()
        plt.savefig(
            self.plot_p.joinpath(
                "Operational cost plus constraint violation from time-step {} penalty {}.png".format(
                    opt_steps["math_opt"][0], delivered_heat_penalty
                )
            )
        )
        plt.show()


if __name__ == "__main__":
    result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
    plot_p: Path = Path(__file__).parents[4] / "plots/constraint_opt"
    plot = PlotControlWithTempVsControlWithHeat(result_p=result_p, plot_p=plot_p)
    # plot.plot_operation_cost(neurons=[[1], [1, 1], [10]])
    # plot.plot_violations(neurons=[[1], [1, 1], [10]])
    plot.plot_optimization_operation_cost_plus_violations(
        neurons=[[1], [1, 1], [10]],
        supply_inlet_violation_penalty=10,
        delivered_heat_penalty=10,
    )

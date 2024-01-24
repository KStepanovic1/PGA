from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer
from src.optimizers.constraint_opt.dhn_nn.plot import Plot
import scipy

if GridProperties["ConsumerNum"] == 1:
    from src.simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from src.simulator.cases.parallel_consumers import build_grid
curr_time = datetime.datetime.now().strftime("%Y-%m-%d")


class PlotOptimization(Plot):
    """
    Plot results of the optimization phase.
    """

    def __init__(self, result_p, plot_p):
        super().__init__(result_p, plot_p)
        self.N_models = 3

    def plot_optimization_operation_cost(self, neurons, plot_p):
        """
        Plot sum of operational costs for each optimizers
        """
        x = list(range(len(neurons)))
        optimizers = [
            "abdollahi",
            "plnn_milp",
            # "icnn_gd",
            # "monotonic_icnn_gd",
            # "plnn_gd",
        ]
        optimizer_type = {
            "plnn_milp": "plnn_milp",
            "icnn_gd": "icnn_gd",
            "monotonic_icnn_gd": "icnn_gd",
            "plnn_gd": "plnn_gd",
        }
        paths = {
            "abdollahi": Path(__file__).parents[4]
            / "results/abdollahi_2015/MPC_episode_length_72_hours",
            "li": Path(__file__).parents[4]
            / "results/li_2016/MPC_episode_length_72_hours",
            "plnn_milp": Path(__file__).parents[4]
            / "results/constraint_opt/plnn_milp/MPC_episode_length_72_hours/control with heat",
            "icnn_gd": Path(__file__).parents[4]
            / "results/constraint_opt/relax_monotonic_icnn_gd/initialization_1/MPC_episode_length_72_hours",
            "monotonic_icnn_gd": Path(__file__).parents[4]
            / "results/constraint_opt/monotonic_icnn_gd/initialization_1/MPC_episode_length_72_hours",
            "plnn_gd": Path(__file__).parents[4]
            / "results/constraint_opt/plnn_gd/initialization_1/MPC_episode_length_72_hours",
            "plot": plot_p.joinpath("optimization"),
        }
        results = {
            "abdollahi": [],
            "li": [],
            "giraud": [],
            "plnn_milp": {"mean": [], "std": [], "upper_bound": [], "lower_bound": []},
            "icnn_gd": {"mean": [], "std": [], "upper_bound": [], "lower_bound": []},
            "monotonic_icnn_gd": {
                "mean": [],
                "std": [],
                "upper_bound": [],
                "lower_bound": [],
            },
            "plnn_gd": {
                "mean": [],
                "std": [],
                "upper_bound": [],
                "lower_bound": [],
            },
        }
        results["giraud"] = calculate_giraud_operation_cost()
        for optimizer in optimizers:
            if (
                optimizer == "plnn_milp"
                or optimizer == "icnn_gd"
                or optimizer == "monotonic_icnn_gd"
                or optimizer == "plnn_gd"
            ):
                for neuron in neurons:
                    profit = []
                    for model in range(self.N_models):
                        optimizer_ = (
                            "{}_".format(model)
                            + optimizer_type[optimizer]
                            + "_neurons_"
                            + Plot.create_neuron_string(neuron)
                            + "_opt_step"
                        )
                        profit.append(
                            Plot.sum(
                                path=paths[optimizer],
                                optimizer=optimizer_,
                                coeff_day=opt_steps["math_opt"],
                                column="Profit",
                            )
                        )
                    all_none = all(element is None for element in profit)
                    if all_none:
                        results[optimizer]["mean"].append(None)
                        results[optimizer]["std"].append(None)
                    else:
                        mean = sum(profit) / len(profit)
                        results[optimizer]["mean"].append(mean)
                        std = np.std(np.array(profit))
                        results[optimizer]["std"].append(std)
                results[optimizer]["upper_bound"] = calculate_plot_bounds(
                    results[optimizer]["mean"], results[optimizer]["std"]
                )[0]
                results[optimizer]["lower_bound"] = calculate_plot_bounds(
                    results[optimizer]["mean"], results[optimizer]["std"]
                )[1]
            else:
                results[optimizer].append(
                    Plot.sum(
                        path=paths[optimizer],
                        optimizer=optimizer,
                        coeff_day=opt_steps["math_opt"],
                        column="Profit",
                    )
                )
        fig, ax = plt.subplots()
        plt.axhline(
            results["abdollahi"],
            label="LP",
            color="y",
        )
        # plt.axhline(results["giraud"], label="Giraud", color="c")
        """
        plt.axhline(
            results["li"],
            label="MINLP",
            color="g",
        )
        """
        plt.plot(
            x,
            results["plnn_milp"]["mean"],
            label="PLNN-MILP",
            color="b",
        )
        plt.fill_between(
            x,
            results["plnn_milp"]["lower_bound"],
            results["plnn_milp"]["upper_bound"],
            color="b",
            alpha=0.1,
        )
        """
        plt.plot(
            x,
            results["icnn_gd"]["mean"],
            label="ICNN-GD",
            color="r",
        )
        plt.fill_between(
            x,
            results["icnn_gd"]["lower_bound"],
            results["icnn_gd"]["upper_bound"],
            color="r",
            alpha=0.1,
        )
        plt.plot(
            x,
            results["monotonic_icnn_gd"]["mean"],
            label="Monotonic ICNN-GD",
            color="#00FF00",
        )
        plt.fill_between(
            x,
            results["monotonic_icnn_gd"]["lower_bound"],
            results["monotonic_icnn_gd"]["upper_bound"],
            color="#00FF00",
            alpha=0.1,
        )
        plt.plot(
            x,
            results["plnn_gd"]["mean"],
            label="PLNN-GD",
            color="#FF9033",
        )
        plt.fill_between(
            x,
            results["plnn_gd"]["lower_bound"],
            results["plnn_gd"]["upper_bound"],
            color="#FF9033",
            alpha=0.1,
        )
        """
        ax.set_title("Operational cost")
        ax.set_xticks(x)
        ax.set_ylabel("Operational cost [e]")
        ax.set_xlabel("Neural network size")
        ax.set_xticklabels(("$[1]$", "$[1,1]$", "$[3]$", "$[5]$"))
        plt.legend()
        plt.grid()
        plt.ylim(results["giraud"] - 3000, results["abdollahi"][0] + 3000)
        plt.savefig(
            paths["plot"].joinpath(
                "Operational cost from time-step {}".format(opt_steps["math_opt"][0])
                + curr_time
                + ".png"
            )
        )
        plt.show()

    def plot_optimization_violations(self, neurons, plot_p):
        """
        Plot average of percent of violations.
        Average is calculated as the sum of violations accross of all days divided by the number of days*planning horizon.
        Percent of violations is calculated with respect to the maximum possible violation for each type of violation.
        """
        x = list(range(len(neurons)))
        optimizers = ["abdollahi", "plnn_milp", "icnn_gd"]

        violations = [
            "Supply_inlet_violation",
            "Supply_outlet_violation",
            "Delivered_heat_violation",
            "Mass_flow_violation",
        ]
        paths = {
            "abdollahi": Path(__file__).parents[4]
            / "results/abdollahi_2015/MPC_episode_length_72_hours",
            "li": Path(__file__).parents[4]
            / "results/li_2016/MPC_episode_length_72_hours",
            "plnn_milp": Path(__file__).parents[4]
            / "results/constraint_opt/plnn_milp/MPC_episode_length_72_hours/control with heat",
            "icnn_gd": Path(__file__).parents[4]
            / "results/constraint_opt/relax_monotonic_icnn_gd/initialization_1/MPC_episode_length_72_hours",
            "plot": plot_p.joinpath("optimization/Unit_violation"),
        }
        colors = {"abdollahi": "y", "li": "g", "plnn_milp": "b", "icnn_gd": "r"}
        labels = {
            "abdollahi": "LP",
            "li": "MINLP",
            "plnn_milp": "PLNN-MILP",
            "icnn_gd": "ICNN-GD",
        }
        titles = {
            "Supply_inlet_violation": "Supply inlet temperature violation",
            "Supply_outlet_violation": "Supply outlet temperature violation",
            "Delivered_heat_violation": "Delivered heat violation",
            "Mass_flow_violation": "Mass flow violation",
        }
        results = {
            "abdollahi": {
                "Supply_inlet_violation": [],
                "Supply_outlet_violation": [],
                "Delivered_heat_violation": [],
                "Mass_flow_violation": [],
            },
            "li": {
                "Supply_inlet_violation": [],
                "Supply_outlet_violation": [],
                "Delivered_heat_violation": [],
                "Mass_flow_violation": [],
            },
            "plnn_milp": {
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
            "icnn_gd": {
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
            if optimizer == "plnn_milp" or optimizer == "icnn_gd":
                for neuron in neurons:
                    for violation_type in violations:
                        average_violation = []
                        for model in range(self.N_models):
                            optimizer_ = (
                                "{}_".format(model)
                                + optimizer
                                + "_init_neurons_"
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
                        all_none = all(element is None for element in average_violation)
                        if all_none:
                            results[optimizer][violation_type]["mean"].append(None)
                            results[optimizer][violation_type]["std"].append(None)
                        else:
                            mean = round(
                                sum(average_violation) / len(average_violation), 1
                            )
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
            else:
                for violation_type in violations:
                    results[optimizer][violation_type].append(
                        round(
                            self.average(
                                path=paths[optimizer],
                                optimizer=optimizer,
                                coeff_day=opt_steps["math_opt"],
                                column=violation_type,
                                null_solve=False,
                            ),
                            0,
                        )
                    )
        for violation in violations:
            fig, ax = plt.subplots()
            for optimizer in optimizers:
                if optimizer == "abdollahi" or optimizer == "li":
                    plt.axhline(
                        results[optimizer][violation][0],
                        label=labels[optimizer],
                        color=colors[optimizer],
                    )
                elif optimizer == "plnn_milp" or optimizer == "icnn_gd":
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
            ax.set_xticklabels(("$[1]$", "$[1,1]$", "$[10]$", "$[10,10]$", "$[50,50]$"))
            plt.legend()
            plt.savefig(
                paths["plot"].joinpath(
                    titles[violation]
                    + " from time-step {}.png".format(opt_steps["math_opt"][0])
                )
            )
            plt.show()

    def plot_optimization_operation_cost_plus_violations(
        self, neurons, plot_p, supply_inlet_violation_penalty, delivered_heat_penalty
    ):
        """
        Add penalty*violations to operation cost, and plot results.
        """
        x = list(range(len(neurons)))
        optimizers = [
            "abdollahi",
            "plnn_milp",
            # "icnn_gd",
            # "monotonic_icnn_gd",
            # "plnn_gd",
        ]
        violations = {
            "Supply_inlet_violation_percent": 0,
            "Supply_outlet_violation_percent": 0,
            "Mass_flow_violation_percent": 0,
            "Delivered_heat_violation_percent": 0,
        }
        optimizer_type = {
            "plnn_milp": "plnn_milp",
            "icnn_gd": "icnn_gd",
            "monotonic_icnn_gd": "icnn_gd",
            "plnn_gd": "plnn_gd",
        }
        paths = {
            "abdollahi": Path(__file__).parents[4]
            / "results/abdollahi_2015/MPC_episode_length_72_hours",
            "li": Path(__file__).parents[4]
            / "results/li_2016/MPC_episode_length_72_hours",
            "plnn_milp": Path(__file__).parents[4]
            / "results/constraint_opt/plnn_milp/MPC_episode_length_72_hours/control with heat",
            "icnn_gd": Path(__file__).parents[4]
            / "results/constraint_opt/relax_monotonic_icnn_gd/initialization_1/MPC_episode_length_72_hours",
            "monotonic_icnn_gd": Path(__file__).parents[4]
            / "results/constraint_opt/monotonic_icnn_gd/initialization_1/MPC_episode_length_72_hours",
            "plnn_gd": Path(__file__).parents[4]
            / "results/constraint_opt/plnn_gd/initialization_1/MPC_episode_length_72_hours",
            "plot": plot_p.joinpath("optimization/Percent_violation"),
        }
        results = {
            "abdollahi": [],
            "li": [],
            "plnn_milp": {"mean": [], "std": [], "upper_bound": [], "lower_bound": []},
            "icnn_gd": {"mean": [], "std": [], "upper_bound": [], "lower_bound": []},
            "monotonic_icnn_gd": {
                "mean": [],
                "std": [],
                "upper_bound": [],
                "lower_bound": [],
            },
            "plnn_gd": {
                "mean": [],
                "std": [],
                "upper_bound": [],
                "lower_bound": [],
            },
        }
        results["giraud"] = calculate_giraud_operation_cost()
        for optimizer in optimizers:
            if (
                optimizer == "plnn_milp"
                or optimizer == "icnn_gd"
                or optimizer == "monotonic_icnn_gd"
                or optimizer == "plnn_gd"
            ):
                for neuron in neurons:
                    cost_plus_violation = []
                    for model in range(self.N_models):
                        for violation in violations.keys():
                            violations[violation] = 0
                        optimizer_ = (
                            "{}_".format(model)
                            + optimizer_type[optimizer]
                            + "_neurons_"
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
                            supply_inlet_violation=violations[
                                "Supply_inlet_violation_percent"
                            ],
                            supply_outlet_violation=violations[
                                "Supply_outlet_violation_percent"
                            ],
                            mass_flow_violation=violations[
                                "Mass_flow_violation_percent"
                            ],
                            delivered_heat_violation=violations[
                                "Delivered_heat_violation_percent"
                            ],
                            supply_inlet_violation_penalty=supply_inlet_violation_penalty,
                            delivered_heat_penalty=delivered_heat_penalty,
                        )
                        cost_plus_violation.append(cost_plus_violation_)
                    all_none = all(element is None for element in cost_plus_violation)
                    if all_none:
                        results[optimizer]["mean"].append(None)
                        results[optimizer]["std"].append(None)
                    else:
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
            else:
                for violation in violations.keys():
                    violations[violation] = 0
                operation_cost = Plot.sum(
                    path=paths[optimizer],
                    optimizer=optimizer,
                    coeff_day=opt_steps["math_opt"],
                    column="Profit",
                )
                for violation in violations.keys():
                    violations[violation] = Plot.sum(
                        path=paths[optimizer],
                        optimizer=optimizer,
                        coeff_day=opt_steps["math_opt"],
                        column=violation,
                    )
                cost_plus_violation_ = Plot.sum_cost_and_violations(
                    operation_cost=operation_cost,
                    supply_inlet_violation=violations["Supply_inlet_violation_percent"],
                    supply_outlet_violation=violations[
                        "Supply_outlet_violation_percent"
                    ],
                    mass_flow_violation=violations["Mass_flow_violation_percent"],
                    delivered_heat_violation=violations[
                        "Delivered_heat_violation_percent"
                    ],
                    supply_inlet_violation_penalty=supply_inlet_violation_penalty,
                    delivered_heat_penalty=delivered_heat_penalty,
                )
                results[optimizer].append(cost_plus_violation_)
        fig, ax = plt.subplots()
        plt.axhline(
            results["abdollahi"],
            label="LP",
            color="y",
        )
        # plt.axhline(results["giraud"], label="Giraud", color="c")
        """
        plt.axhline(
            results["li"],
            label="MINLP",
            color="g",
        )
        """
        plt.plot(
            x,
            results["plnn_milp"]["mean"],
            label="PLNN-MILP",
            color="b",
        )
        plt.fill_between(
            x,
            results["plnn_milp"]["lower_bound"],
            results["plnn_milp"]["upper_bound"],
            color="b",
            alpha=0.1,
        )
        """
        plt.plot(
            x,
            results["icnn_gd"]["mean"],
            label="ICNN-GD",
            color="r",
        )
        plt.fill_between(
            x,
            results["icnn_gd"]["lower_bound"],
            results["icnn_gd"]["upper_bound"],
            color="r",
            alpha=0.1,
        )
        plt.plot(
            x,
            results["monotonic_icnn_gd"]["mean"],
            label="Monotonic ICNN-GD",
            color="#00FF00",
        )
        plt.fill_between(
            x,
            results["monotonic_icnn_gd"]["lower_bound"],
            results["monotonic_icnn_gd"]["upper_bound"],
            color="#00FF00",
            alpha=0.1,
        )
        plt.plot(
            x,
            results["plnn_gd"]["mean"],
            label="PLNN-GD",
            color="#FF9033",
        )
        plt.fill_between(
            x,
            results["plnn_gd"]["lower_bound"],
            results["plnn_gd"]["upper_bound"],
            color="#FF9033",
            alpha=0.1,
        )
        """
        ax.set_title("Operational cost+penalty")
        ax.set_xticks(x)
        ax.set_ylabel("Operational cost+penalty [e]")
        ax.set_xlabel("Neural network size")
        ax.set_xticklabels(("$[1]$", "$[1,1]$", "$[3]$", "$[5]$"))
        plt.legend()
        plt.grid()
        plt.ylim(results["giraud"] - 8000, results["abdollahi"][0] + 8000)
        plt.savefig(
            paths["plot"].joinpath(
                "Operational cost plus constraint violation from time-step {} penalty {}".format(
                    opt_steps["math_opt"][0], delivered_heat_penalty
                )
                + curr_time
                + ".png"
            )
        )
        plt.show()

    def plot_optimization_runtime(self, neurons, plot_p):
        """
        Plot sum of runtimes of solving each day for each optimizers
        """
        x = list(range(len(neurons)))
        optimizers = [
            "abdollahi",
            # "li",
            "plnn_milp",
            # "icnn_gd",
            # "monotonic_icnn_gd",
            # "plnn_gd",
        ]
        optimizer_type = {
            "plnn_milp": "plnn_milp",
            "icnn_gd": "icnn_gd",
            "monotonic_icnn_gd": "icnn_gd",
            "plnn_gd": "plnn_gd",
        }
        paths = {
            "abdollahi": Path(__file__).parents[4]
            / "results/abdollahi_2015/MPC_episode_length_72_hours",
            "li": Path(__file__).parents[4]
            / "results/li_2016/MPC_episode_length_72_hours",
            "plnn_milp": Path(__file__).parents[4]
            / "results/constraint_opt/plnn_milp/MPC_episode_length_72_hours/control with heat",
            "icnn_gd": Path(__file__).parents[4]
            / "results/constraint_opt/relax_monotonic_icnn_gd/initialization_1/MPC_episode_length_72_hours",
            "plot": plot_p.joinpath("optimization"),
            "monotonic_icnn_gd": Path(__file__).parents[4]
            / "results/constraint_opt/monotonic_icnn_gd/initialization_1/MPC_episode_length_72_hours",
            "plnn_gd": Path(__file__).parents[4]
            / "results/constraint_opt/plnn_gd/initialization_1/MPC_episode_length_72_hours",
        }
        colors = {"abdollahi": "y", "li": "g", "plnn_milp": "b"}
        labels = {"abdollahi": "LP", "li": "MINLP", "plnn_milp": "PLNN-MILP"}
        results = {
            "abdollahi": [],
            "li": [],
            "plnn_milp": {"mean": [], "std": [], "upper_bound": [], "lower_bound": []},
            "icnn_gd": {"mean": [], "std": [], "upper_bound": [], "lower_bound": []},
            "monotonic_icnn_gd": {
                "mean": [],
                "std": [],
                "upper_bound": [],
                "lower_bound": [],
            },
            "plnn_gd": {"mean": [], "std": [], "upper_bound": [], "lower_bound": []},
        }
        for optimizer in optimizers:
            if (
                optimizer == "plnn_milp"
                or optimizer == "icnn_gd"
                or optimizer == "monotonic_icnn_gd"
                or optimizer == "plnn_gd"
            ):
                for neuron in neurons:
                    runtime = []
                    for model in range(self.N_models):
                        optimizer_ = (
                            "{}_".format(model)
                            + optimizer_type[optimizer]
                            + "_neurons_"
                            + Plot.create_neuron_string(neuron)
                            + "_opt_step"
                        )
                        runtime.append(
                            self.sum(
                                path=paths[optimizer],
                                optimizer=optimizer_,
                                coeff_day=opt_steps["math_opt"],
                                column="Runtime",
                            )
                        )
                    all_none = all(element is None for element in runtime)
                    if all_none:
                        results[optimizer]["mean"].append(None)
                        results[optimizer]["std"].append(None)
                    else:
                        mean = sum(runtime) / len(runtime)
                        results[optimizer]["mean"].append(mean)
                        std = np.std(np.array(runtime))
                        results[optimizer]["std"].append(std)
                results[optimizer]["upper_bound"] = calculate_plot_bounds(
                    results[optimizer]["mean"], results[optimizer]["std"]
                )[0]
                results[optimizer]["lower_bound"] = calculate_plot_bounds(
                    results[optimizer]["mean"], results[optimizer]["std"]
                )[1]
            else:
                results[optimizer].append(
                    self.sum(
                        path=paths[optimizer],
                        optimizer=optimizer,
                        coeff_day=opt_steps["math_opt"],
                        column="Runtime",
                    )
                )
        fig, ax = plt.subplots()
        plt.axhline(
            results["abdollahi"],
            label="LP",
            color="y",
        )
        """
        plt.axhline(
            results["li"],
            label="MINLP",
            color="g",
        )
        """
        plt.plot(
            x,
            results["plnn_milp"]["mean"],
            label="PLNN-MILP",
            color="b",
        )
        plt.fill_between(
            x,
            results["plnn_milp"]["lower_bound"],
            results["plnn_milp"]["upper_bound"],
            color="b",
            alpha=0.1,
        )
        """
        plt.plot(
            x,
            results["icnn_gd"]["mean"],
            label="ICNN-GD",
            color="r",
        )
        plt.fill_between(
            x,
            results["icnn_gd"]["lower_bound"],
            results["icnn_gd"]["upper_bound"],
            color="r",
            alpha=0.1,
        )
        plt.plot(
            x,
            results["monotonic_icnn_gd"]["mean"],
            label="Monotonic ICNN-GD",
            color="#00FF00",
        )
        plt.fill_between(
            x,
            results["monotonic_icnn_gd"]["lower_bound"],
            results["monotonic_icnn_gd"]["upper_bound"],
            color="#00FF00",
            alpha=0.1,
        )
        plt.plot(
            x,
            results["plnn_gd"]["mean"],
            label="PLNN-GD",
            color="#FF9033",
        )
        plt.fill_between(
            x,
            results["plnn_gd"]["lower_bound"],
            results["plnn_gd"]["upper_bound"],
            color="#FF9033",
            alpha=0.1,
        )
        """
        ax.set_title("Runtime for solving the model")
        ax.set_xticks(x)
        ax.set_ylabel("Runtime [s]")
        ax.set_xlabel("Neural network size")
        ax.set_xticklabels(("$[1]$", "$[1,1]$", "$[3]$", "$[5]$"))
        plt.legend()
        plt.savefig(
            paths["plot"].joinpath(
                "Runtime from time-step {}".format(opt_steps["math_opt"][0])
                + curr_time
                + ".png"
            )
        )
        plt.show()

    def plot_optimality_gap(self, neurons, plot_p):
        """
        Plot optimality gap with 90% confidence intervals of mathematical solver.
        """
        x = list(range(len(neurons)))
        optimizers = ["abdollahi", "plnn_milp"]
        paths = {
            "abdollahi": Path(__file__).parents[4]
            / "results/abdollahi_2015/MPC_episode_length_72_hours",
            "li": Path(__file__).parents[4]
            / "results/li_2016/MPC_episode_length_72_hours",
            "plnn_milp": Path(__file__).parents[4]
            / "results/constraint_opt/plnn_milp/MPC_episode_length_72_hours/control with heat",
            "plot": plot_p.joinpath("optimization"),
        }
        colors = {"abdollahi": "y", "li": "g", "plnn_milp": "b"}
        labels = {"abdollahi": "LP", "li": "MINLP", "plnn_milp": "PLNN-MILP"}
        results = {
            "abdollahi": [],
            "li": [],
            "plnn_milp": {"mean": [], "std": [], "upper_bound": [], "lower_bound": []},
        }
        for optimizer in optimizers:
            files = os.listdir(paths[optimizer])
            if optimizer == "plnn_milp":
                for neuron in neurons:
                    optimality_gap = []
                    for model in range(self.N_models):
                        optimizer_ = (
                            "{}_plnn_milp_init_neurons_".format(model)
                            + Plot.create_neuron_string(neuron)
                            + "_opt_step"
                        )
                        optimality_gap.append(
                            self.average(
                                path=paths[optimizer],
                                optimizer=optimizer_,
                                coeff_day=opt_steps["math_opt"],
                                column="Optimality_gap",
                                null_solve=False,
                            )
                        )
                    all_none = all(element is None for element in optimality_gap)
                    if all_none:
                        results[optimizer]["mean"].append(None)
                        results[optimizer]["std"].append(None)
                    else:
                        mean = sum(optimality_gap) / len(optimality_gap)
                        results[optimizer]["mean"].append(mean)
                        std = np.std(np.array(optimality_gap))
                        results[optimizer]["std"].append(std)
                results[optimizer][
                    "upper_bound"
                ] = calculate_ninety_percent_confidence_interval(
                    results[optimizer]["mean"], results[optimizer]["std"]
                )[
                    0
                ]
                results[optimizer][
                    "lower_bound"
                ] = calculate_ninety_percent_confidence_interval(
                    results[optimizer]["mean"], results[optimizer]["std"]
                )[
                    1
                ]
            else:
                results[optimizer].append(
                    round(
                        self.average(
                            path=paths[optimizer],
                            optimizer=optimizer,
                            coeff_day=opt_steps["math_opt"],
                            column="Optimality_gap",
                            null_solve=False,
                        ),
                        0,
                    )
                )
        fig, ax = plt.subplots()
        plt.axhline(
            results["abdollahi"],
            label="LP",
            color="y",
        )
        """
        plt.axhline(
            results["li"],
            label="MINLP",
            color="g",
        )
        """
        plt.plot(
            x,
            results["plnn_milp"]["mean"],
            label="PLNN-MILP",
            color="b",
        )
        plt.fill_between(
            x,
            results["plnn_milp"]["lower_bound"],
            results["plnn_milp"]["upper_bound"],
            color="b",
            alpha=0.1,
        )
        ax.set_title("Optimality gap")
        ax.set_xticks(x)
        ax.set_ylabel("Optimality gap [%]")
        ax.set_xlabel("Neural network size")
        ax.set_xticklabels(("$[1]$", "$[1,1]$", "$[10]$", "$[10,10]$", "$[50,50]$"))
        plt.legend()
        plt.savefig(
            paths["plot"].joinpath(
                "Optimality gap from time-step {}.png".format(opt_steps["math_opt"][0])
            )
        )
        plt.show()

    def get_gradient_iteration_number_per_nn_size(self, result_p, plot_p):
        """
        Get mean and standard deviation of number of gradient descent iterations for different neural network sizes.
        Get mean and standard deviation of iteration time length for different neural network sizes.
        """
        result_p_iteration = result_p.joinpath("relax_monotonic_icnn_gd").joinpath(
            "heat_gradients"
        )
        result_p_time = result_p.joinpath("relax_monotonic_icnn_gd")
        opt_step: int = 12
        MPC_step: int = 5
        date: str = "2023-05-30"
        model_num: int = 0
        neural_network_sizes = [[1], [1, 1], [10], [10, 10]]
        dict_iteration, dict_time = {}, {}
        for neuron in neural_network_sizes:
            dict_iteration[str(neuron)] = []
            dict_time[str(neuron)] = []
            file_name_time = (
                "{}_icnn_gd_init_neurons_".format(model_num)
                + Plot.create_neuron_string(neuron)
                + "_opt_step_{}_".format(opt_step)
                + date
                + ".csv"
            )
            data_time = pd.read_csv(result_p_time.joinpath(file_name_time))["Runtime"]
            for i in range(MPC_step):
                file_name_iteration = (
                    "{}_heat_gradients_neurons_".format(model_num)
                    + Plot.create_neuron_string(neuron)
                    + "_opt_step_{}_MPC_step_{}_".format(opt_step, i)
                    + date
                    + ".csv"
                )
                data_iteration = pd.read_csv(
                    result_p_iteration.joinpath(file_name_iteration)
                )
                dict_iteration[str(neuron)].append(len(data_iteration))
                dict_time[str(neuron)].append(data_time.iloc[i] / len(data_iteration))
        for neuron in neural_network_sizes:
            mean_iteration_number = sum(dict_iteration[str(neuron)]) / len(
                dict_iteration[str(neuron)]
            )
            std_iteration_number = np.std(np.array(dict_iteration[str(neuron)]))
            mean_iteration_time = sum(dict_time[str(neuron)]) / len(
                dict_time[str(neuron)]
            )
            std_iteration_time = np.std(np.array(dict_time[str(neuron)]))
            print(
                "Mean iteration number for "
                + str(neuron)
                + " is {}".format(mean_iteration_number)
            )
            print(
                "Standard deviation iteration number for "
                + str(neuron)
                + " is {}".format(std_iteration_number)
            )
            print(
                "Mean iteration time for "
                + str(neuron)
                + " is {}".format(mean_iteration_time)
            )
            print(
                "Standard deviation iteration time for "
                + str(neuron)
                + " is {}".format(std_iteration_time)
            )

    def plot_gdco_ipdd_algorithm(self, C):
        """
        Plot objective value, approximate objective value and infeasibility as the function of computational time.
        """
        result_p: Path = self.result_p.joinpath(
            "monotonic_icnn_gd/initialization_1/MPC_episode_length_1_hours"
        )
        plot_p: Path = self.plot_p.joinpath(
            "const_gradient_descent_optimization"
        ).joinpath("Neural network")
        J = {"gdco": [], "ipdd": [], "milp": [2042.18]}
        hat_J = {"gdco": [], "ipdd": []}
        constraint_violations = {"gdco": {}, "ipdd": {}, "milp": {}}
        for i in range(TimeParameters["PlanningHorizon"]):
            constraint_violations["gdco"]["constraint {}".format(i)] = []
            constraint_violations["ipdd"]["constraint {}".format(i)] = []
            constraint_violations["milp"]["constraint {}".format(i)] = [0]
        constraint_violations_max = {"gdco": [], "ipdd": []}
        computational_time = {"gdco": {}, "ipdd": {}, "milp": {}}
        J["gdco"] = list(
            pd.read_csv(result_p.joinpath("J_C={}_gdco.csv".format(C)))["0"]
        )
        J["ipdd"] = list(
            pd.read_csv(
                result_p.joinpath("J_C={}_eta=1.1_warm_start_up_ipdd.csv".format(C))
            )["0"]
        )
        J["milp"] = list(
            pd.read_csv(self.result_p.joinpath("monotonic_icnn_plnn/J.csv"))["0"]
        )
        hat_J["gdco"] = list(
            pd.read_csv(result_p.joinpath("hat_J_C={}_gdco.csv".format(C)))["0"]
        )
        hat_J["ipdd"] = list(
            pd.read_csv(
                result_p.joinpath("hat_J_C={}_eta=1.1_warm_start_up_ipdd.csv".format(C))
            )["0"]
        )
        for i in range(TimeParameters["PlanningHorizon"]):
            constraint_violations["gdco"]["constraint {}".format(i)] = list(
                pd.read_csv(
                    result_p.joinpath("Constraint_violations_C={}_gdco.csv".format(C))
                )["constraint {}".format(i)]
            )
            constraint_violations["ipdd"]["constraint {}".format(i)] = list(
                pd.read_csv(
                    result_p.joinpath(
                        "Constraint_violations_C={}_eta=1.1_warm_start_up_ipdd.csv".format(
                            C
                        )
                    )
                )["constraint {}".format(i)]
            )
        constraint_violations["milp"] = list(
            pd.read_csv(
                self.result_p.joinpath("monotonic_icnn_plnn/Constraint_violations.csv")
            )["0"]
        )
        x = list(
            pd.read_csv(result_p.joinpath("Execution_time_C={}_gdco.csv".format(C)))[
                "0"
            ]
        )
        y = list(
            pd.read_csv(
                result_p.joinpath(
                    "Execution_time_C={}_eta=1.1_warm_start_up_ipdd.csv".format(C)
                )
            )["0"]
        )
        N = min(len(x), len(y))
        print(scipy.stats.pearsonr(x[:N],y[:N]))
        computational_time["milp"] = list(
            pd.read_csv(
                self.result_p.joinpath("monotonic_icnn_plnn/Execution_time.csv")
            )["0"]
        )
        computational_time["gdco"] = [sum(x[: i + 1]) for i in range(len(x))]
        computational_time["ipdd"] = [sum(y[: i + 1]) for i in range(len(y))]
        for i in range(len(J["gdco"])):
            # Extract the ith element from each list and find the absolute max
            max_value = float("-inf")
            max_constraint_index = None
            for constraint_number in constraint_violations["gdco"]:
                if abs(constraint_violations["gdco"][constraint_number][i]) > max_value:
                    max_value = abs(constraint_violations["gdco"][constraint_number][i])
                    max_constraint_index = constraint_number
            constraint_violations_max["gdco"].append(
                constraint_violations["gdco"][max_constraint_index][i]
            )
        for i in range(len(J["ipdd"])):
            # Extract the ith element from each list and find the absolute max
            max_value = float("-inf")
            max_constraint_index = None
            for constraint_number in constraint_violations["ipdd"]:
                if abs(constraint_violations["ipdd"][constraint_number][i]) > max_value:
                    max_value = abs(constraint_violations["ipdd"][constraint_number][i])
                    max_constraint_index = constraint_number
            constraint_violations_max["ipdd"].append(
                constraint_violations["ipdd"][max_constraint_index][i]
            )
        plt.plot(
            computational_time["gdco"][0],
            J["gdco"][0],
            color="#B6C800",
            marker="P",
            markersize=12,
            linestyle="dashed",
            label="PM",
        )
        plt.axhline(
            y=J["gdco"][0],
            xmin=computational_time["gdco"][0] / 1000,
            xmax=1,
            color="#B6C800",
            linestyle="dashed",
        )
        plt.plot(
            computational_time["milp"],
            J["milp"],
            color="#16502E",
            marker="v",
            markersize=10,
            linestyle="dotted",
            label="MILP",
        )
        plt.plot(
            computational_time["gdco"],
            J["gdco"],
            color="#3374FF",
            marker="o",
            markersize=7,
            label="PGA",
            linestyle="solid",
        )
        plt.plot(
            computational_time["ipdd"],
            J["ipdd"],
            color="#E31D1D",
            marker="*",
            markersize=7,
            label="IPDD",
            linestyle="dashdot",
        )
        # plt.axhline(y=2642, color="#33754E", linestyle="--", label=r"MILP$_{T_{lim}=1000s}$")
        plt.xlim([0, 1000])
        plt.ylim([J["gdco"][0] - 10, J["milp"][-1] + 25])
        plt.xlabel("Computational time [s]", fontsize=14)
        plt.ylabel("Objective value [€]", fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='lower right')
        # plt.tight_layout()
        plt.savefig(plot_p.joinpath("Objective_value_C={}.pdf".format(C)))
        plt.show()
        # approximate objective function
        # plt.figure(figsize=(20, 5))
        plt.plot(
            computational_time["gdco"],
            hat_J["gdco"],
            color="#3374FF",
            marker="o",
            markersize=6,
            label="PGDA",
        )
        plt.plot(
            computational_time["ipdd"],
            hat_J["ipdd"],
            color="#E31D1D",
            marker="*",
            markersize=7,
            label="IPDD",
        )
        plt.axhline(
            y=J["milp"][0],
            color="#33754E",
            linestyle="--",
            label=r"MILP$_{T_{lim}=3600s}$",
        )
        plt.xlabel("Computational time [s]", fontsize=11)
        plt.ylabel("Approximate Objective Value", fontsize=11)
        plt.xlim([0, 1000])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_p.joinpath("Approximate_objective_value_C={}.png".format(C)))
        plt.show()
        # constraint violations
        # plt.figure(figsize=(20, 5))
        plt.plot(
            computational_time["gdco"][0],
            constraint_violations_max["gdco"][0],
            color="#B6C800",
            marker="P",
            markersize=12,
            linestyle="dashed",
            label="PM",
        )
        plt.axhline(
            y=constraint_violations_max["gdco"][0],
            xmin=computational_time["gdco"][0] / 1000,
            xmax=1,
            color="#B6C800",
            linestyle="dashed",
        )
        plt.plot(
            computational_time["milp"],
            constraint_violations["milp"],
            color="#16502E",
            marker="v",
            markersize=10,
            linestyle="dotted",
            label="MILP",
        )
        plt.plot(
            computational_time["gdco"],
            constraint_violations_max["gdco"],
            color="#3374FF",
            marker="o",
            markersize=7,
            label="PGA",
            linestyle="solid",
        )
        plt.plot(
            computational_time["ipdd"],
            constraint_violations_max["ipdd"],
            color="#E31D1D",
            marker="*",
            markersize=7,
            label="IPDD",
            linestyle="dashdot",
        )
        plt.xlabel("Computational time [s]", fontsize=14)
        plt.ylabel("Constraint value [MW]", fontsize=14)
        plt.xlim([0, 1000])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='lower right')
        # plt.tight_layout()
        plt.savefig(plot_p.joinpath("Constraint_values_C={}.pdf".format(C)))
        plt.show()

    def plot_gdco_ipdd_algorithm_li(self, C):
        """
        Plot objective value, approximate objective value and infeasibility as the function of computational time.
        """
        result_p: Path = self.result_p.joinpath("li_2016_gd")
        plot_p: Path = self.plot_p.joinpath(
            "const_gradient_descent_optimization"
        ).joinpath("DHS")
        J = {"pga": [], "ipdd": [], "nlp": []}
        constraint_violations_max = {"pga": [], "ipdd": []}
        computational_time = {"pga": {}, "ipdd": {}, "nlp": {}}
        J["pga"] = list(
            pd.read_csv(result_p.joinpath("J_C={}_gdco.csv".format(C)))["0"]
        )
        J["ipdd"] = list(
            pd.read_csv(result_p.joinpath("J_C={}_ipdd.csv".format(C)))["0"]
        )
        J["nlp"] = list(pd.read_csv(result_p.joinpath("J_nlp.csv"))["0"])
        constraint_violations_max["pga"] = list(
            pd.read_csv(result_p.joinpath("Max_constraint_violation_pga.csv"))["0"]
        )
        constraint_violations_max["nlp"] = list(
            pd.read_csv(result_p.joinpath("Constraint_value_nlp.csv"))["0"]
        )
        constraint_violations_max["ipdd"] = list(
            pd.read_csv(result_p.joinpath("Max_constraint_violation_ipdd.csv"))["0"]
        )
        x = list(
            pd.read_csv(result_p.joinpath("Execution_time_pga.csv".format(C)))["0"]
        )
        y = list(
            pd.read_csv(result_p.joinpath("Execution_time_ipdd.csv".format(C)))["0"]
        )
        N = min(len(x), len(y))
        print(scipy.stats.pearsonr(x[:N], y[:N]))
        z = list(
            pd.read_csv(result_p.joinpath("Execution_time_nlp.csv"))["0"]
        )
        computational_time["pga"] = [sum(x[: i + 1]) for i in range(len(x))]
        computational_time["ipdd"] = [sum(y[: i + 1]) for i in range(len(y))]
        computational_time["nlp"] = [sum(z[: i + 1]) for i in range(len(z))]
        plt.plot(
            computational_time["pga"][0],
            J["pga"][0],
            color="#B6C800",
            marker="P",
            markersize=12,
            linestyle="dashed",
            label="PM",
        )
        plt.axhline(
            y=J["pga"][0],
            xmin=computational_time["pga"][0] / 500,
            xmax=1,
            color="#B6C800",
            linestyle="dashed",
        )
        plt.plot(
            computational_time["nlp"],
            J["nlp"],
            color="#16502E",
            marker="v",
            markersize=10,
            linestyle="dotted",
            label="NLP",
        )
        plt.plot(
            computational_time["pga"],
            J["pga"],
            color="#3374FF",
            marker="o",
            markersize=7,
            label="PGA",
            linestyle="solid",
        )
        plt.plot(
            computational_time["ipdd"],
            J["ipdd"],
            color="#E31D1D",
            marker="*",
            markersize=7,
            label="IPDD",
            linestyle="dashdot",
        )
        plt.xlim([0, 500])
        plt.ylim([J["pga"][0] - 50, J["pga"][1] + 50])
        # plt.ylim([J["gdco"][0] - 10, J["milp"][-1] + 25])
        plt.xlabel("Computational time [s]", fontsize=14)
        plt.ylabel("Objective value [€]", fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='lower right')
        # plt.tight_layout()
        plt.savefig(plot_p.joinpath("Objective_value_li_C={}.pdf".format(C)))
        plt.show()
        # approximate objective function
        # plt.figure(figsize=(20, 5))
        plt.plot(
            computational_time["pga"][0],
            constraint_violations_max["pga"][0],
            color="#B6C800",
            marker="P",
            markersize=12,
            linestyle="dashed",
            label="PM",
        )
        plt.axhline(
            y=constraint_violations_max["pga"][0],
            xmin=computational_time["pga"][0] / 500,
            xmax=1,
            color="#B6C800",
            linestyle="dashed",
        )
        plt.plot(
            computational_time["nlp"],
            constraint_violations_max["nlp"],
            color="#16502E",
            marker="v",
            markersize=10,
            linestyle="dotted",
            label="MILP",
        )
        plt.plot(
            computational_time["pga"],
            constraint_violations_max["pga"],
            color="#3374FF",
            marker="o",
            markersize=7,
            label="PGA",
            linestyle="solid",
        )
        plt.plot(
            computational_time["ipdd"],
            constraint_violations_max["ipdd"],
            color="#E31D1D",
            marker="*",
            markersize=7,
            label="IPDD",
            linestyle="dashdot",
        )
        plt.xlabel("Computational time [s]", fontsize=14)
        plt.ylabel("Constraint value [MW]", fontsize=14)
        plt.xlim([0, 500])
        plt.ylim([-6.7, 5])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='lower right')
        # plt.tight_layout()
        plt.savefig(plot_p.joinpath("Constraint_values_li={}.pdf".format(C)))
        plt.show()

    def mean_std_computational_time(self):
        """
        Print mean and standard deviation of computational time per outer iteration.
        """
        result_p: Path = self.result_p.joinpath(
            "const_gradient_descent_optimization"
        )
        computation_time_gdco = list(
            pd.read_csv(result_p.joinpath("Execution_time_gdco.csv"))["0"]
        )
        computation_time_ipdd = list(
            pd.read_csv(
                result_p.joinpath("Execution_time_ipdd.csv")
            )["0"]
        )
        mean_gdco = statistics.mean(computation_time_gdco)
        std_gdco = statistics.stdev(computation_time_gdco)
        mean_ipdd = statistics.mean(computation_time_ipdd)
        std_ipdd = statistics.stdev(computation_time_ipdd)
        print("GDCO mean is {} and std is {}".format(mean_gdco, std_gdco))
        print("IPDD mean is {} and std is {}".format(mean_ipdd, std_ipdd))


if __name__ == "__main__":
    result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
    plot_p: Path = Path(__file__).parents[4] / "plots/constraint_opt"
    plot = PlotOptimization(result_p=result_p, plot_p=plot_p)
    """
    plot.plot_optimization_operation_cost(
        neurons=[[1], [1, 1], [10], [10, 10], [50, 50]], plot_p=plot_p
    )
    plot.plot_optimization_violations(
        neurons=[[1], [1, 1], [10], [10, 10], [50, 50]], plot_p=plot_p
    )
    plot.plot_optimality_gap(
        neurons=[[1], [1, 1], [10], [10, 10], [50, 50]], plot_p=plot_p
    )
    plot.plot_optimization_operation_cost_plus_violations(
        neurons=[[1], [1, 1], [10], [10, 10], [50, 50]],
        plot_p=plot_p,
        supply_inlet_violation_penalty=100,
        delivered_heat_penalty=100,
    )
    plot.plot_optimization_runtime(
        neurons=[[1], [1, 1], [10], [10, 10], [50, 50]], plot_p=plot_p
    )
    """
    plot.plot_gdco_ipdd_algorithm_li(C=132)

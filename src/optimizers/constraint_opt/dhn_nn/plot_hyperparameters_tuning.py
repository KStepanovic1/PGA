from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *
from src.optimizers.constraint_opt.dhn_nn.plot import Plot
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer

if GridProperties["ConsumerNum"] == 1:
    from src.simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from src.simulator.cases.parallel_consumers import build_grid


class PlotHyperparametersTuning(Plot):
    def __init__(self, result_p, plot_p):
        super().__init__(result_p, plot_p)
        self.N_models = 3
        self.violation_y_label = {
            "Supply_inlet_violation": "Violation of maximum supply inlet temperature [C]",
            "Supply_outlet_violation": "Violation of minimum supply inlet temperature [C]",
            "Mass_flow_violation": "Violation of maximum mass flow [kg/s]",
            "Delivered_heat_violation": "Underdelivered heat [MWh]",
        }

    def evaluate_learning_rate(
        self, experiment_folder, experiment_sub_folder, learning_rates, neurons
    ):
        """
        Evaluate two learning rates, 0.01 and 0.001, where gradient descent updates were carried for 5000 iterations,
        based on their resulting operation cost and violations.
        """
        result_p = self.result_p.joinpath(experiment_folder).joinpath(
            experiment_sub_folder
        )
        variables = [
            "Profit",
            "Supply_inlet_violation",
            "Supply_outlet_violation",
            "Mass_flow_violation",
            "Delivered_heat_violation",
        ]
        # create dictionary of colors and marks
        annotation = {"color": {}, "mark": {}}
        color_list = ["r", "b", "g"]
        mark_list = ["o", "*", "v"]
        for i, learning_rate in enumerate(learning_rates):
            annotation["color"][str(learning_rate)] = color_list[i]
            annotation["mark"][str(learning_rate)] = mark_list[i]
        # create dictionary for saving results
        dict = {}
        for learning_rate in learning_rates:
            dict[str(learning_rate)] = {}
            for neuron in neurons:
                dict[str(learning_rate)][str(neuron)] = {}
                for var in variables:
                    dict[str(learning_rate)][str(neuron)][var] = []
        for learning_rate in learning_rates:
            result_p_learning_rate = result_p.joinpath(
                "learning_rate_{}_N_10000".format(learning_rate)
            )
            for neuron in neurons:
                for model_num in range(self.N_models):
                    optimizer = (
                        "{}_icnn_gd_init_neurons_".format(model_num)
                        + Plot.create_neuron_string(neuron)
                        + "_opt_step"
                    )
                    # sum operation cost for all days for one DNN model
                    dict[str(learning_rate)][str(neuron)]["Profit"].append(
                        Plot.sum(
                            path=result_p_learning_rate,
                            optimizer=optimizer,
                            coeff_day=opt_steps["math_opt"],
                            column="Profit",
                            null_solve=False,
                        )
                    )
                    for violation in [
                        "Supply_inlet_violation",
                        "Supply_outlet_violation",
                        "Mass_flow_violation",
                        "Delivered_heat_violation",
                    ]:
                        # mean of violations percentage per day (sum of violations percentage for all days/num_days*planning_horizon)
                        dict[str(learning_rate)][str(neuron)][violation].append(
                            Plot.average_actual_violation(
                                path=result_p_learning_rate,
                                optimizer=optimizer,
                                coeff_day=opt_steps["math_opt"],
                                column=violation,
                                null_solve=False,
                            )
                        )
        # plot operation cost
        fig, ax = plt.subplots()
        counter = 0
        x_ticks = ["$[1]$", "$[10]$", "$[50,50]$"]
        x = list(range(len(x_ticks)))
        x = [xx + 0.005 * len(neurons) for xx in x]
        for neuron in neurons:
            delta_xlabel = 0
            for learning_rate in learning_rates:
                if counter == 0:
                    ax.errorbar(
                        counter + delta_xlabel,
                        np.mean(
                            np.array(dict[str(learning_rate)][str(neuron)]["Profit"])
                        ),
                        np.std(
                            np.array(dict[str(learning_rate)][str(neuron)]["Profit"])
                        ),
                        fmt=annotation["mark"][str(learning_rate)],
                        capsize=4,
                        color=annotation["color"][str(learning_rate)],
                        label="alpha={}".format(learning_rate),
                    )
                    delta_xlabel += 0.05
                else:
                    ax.errorbar(
                        counter + delta_xlabel,
                        np.mean(
                            np.array(dict[str(learning_rate)][str(neuron)]["Profit"])
                        ),
                        np.std(
                            np.array(dict[str(learning_rate)][str(neuron)]["Profit"])
                        ),
                        fmt=annotation["mark"][str(learning_rate)],
                        capsize=4,
                        color=annotation["color"][str(learning_rate)],
                    )
                    delta_xlabel += 0.05
            counter += 1
        plt.xlabel("Neural network size")
        plt.ylabel("Operation cost [e]")
        plt.legend()
        plt.title("Operation cost")
        plt.xticks(x, x_ticks)
        plt.savefig(
            self.plot_p.joinpath("Operation cost as the function of learning rate")
        )
        plt.show()
        # plot violations
        for violation in [
            "Supply_inlet_violation",
            "Supply_outlet_violation",
            "Mass_flow_violation",
            "Delivered_heat_violation",
        ]:
            fig, ax = plt.subplots()
            counter = 0
            x_ticks = ["$[1]$", "$[10]$", "$[50,50]$"]
            x = list(range(len(x_ticks)))
            x = [xx + 0.005 * len(neurons) for xx in x]
            for neuron in neurons:
                delta_xlabel = 0
                for learning_rate in learning_rates:
                    if counter == 0:
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(
                                np.array(
                                    dict[str(learning_rate)][str(neuron)][violation]
                                )
                            ),
                            np.std(
                                np.array(
                                    dict[str(learning_rate)][str(neuron)][violation]
                                )
                            ),
                            fmt=annotation["mark"][str(learning_rate)],
                            capsize=4,
                            color=annotation["color"][str(learning_rate)],
                            label="alpha={}".format(learning_rate),
                        )
                        delta_xlabel += 0.05
                    else:
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(
                                np.array(
                                    dict[str(learning_rate)][str(neuron)][violation]
                                )
                            ),
                            np.std(
                                np.array(
                                    dict[str(learning_rate)][str(neuron)][violation]
                                )
                            ),
                            fmt=annotation["mark"][str(learning_rate)],
                            capsize=4,
                            color=annotation["color"][str(learning_rate)],
                        )
                        delta_xlabel += 0.05
                counter += 1
            plt.xlabel("Neural network size")
            plt.ylabel(self.violation_y_label[violation])
            plt.legend()
            plt.title(violation)
            plt.xticks(x, x_ticks)
            plt.savefig(
                self.plot_p.joinpath(
                    violation + " as the function of learning rate.png"
                )
            )
            plt.show()

    def plot_gradient_and_decision_variable_updates_per_iterations(
        self, experiment_folder, experiment_sub_folder, learning_rate
    ):
        """
        For specific model number, neural network size, and MPC steps, plot progress of
        gradient descent updates and produced heats as decision variables through gradient descent iterations.
        """
        result_p = (
            self.result_p.joinpath(experiment_folder)
            .joinpath(experiment_sub_folder)
            .joinpath("learning_rate_{}_N_10000".format(learning_rate))
            .joinpath("heat_gradients")
        )
        model_num = 0
        neurons = "50_50"
        opt_step = 12
        MPC_steps = [0, 1, 2, 3]
        date = "2023-06-21"
        columns = ["h_" + str(i) for i in range(TimeParameters["PlanningHorizon"])]
        columns_grad = ["grad_" + column for column in columns]
        # columns_var = ["new_" + column for column in columns]
        columns_var = ["new_h_0"]
        for mpc_step in MPC_steps:
            name = (
                "{}_heat_gradients_neurons_".format(model_num)
                + neurons
                + "_opt_step_{}_MPC_step_{}_".format(opt_step, mpc_step)
                + date
                + ".csv"
            )
            data_grad = pd.read_csv(result_p.joinpath(name))[columns_grad]
            data_var = pd.read_csv(result_p.joinpath(name))[columns_var]
            # plot gradients
            fig, ax = plt.subplots()
            for index, column in enumerate(columns_grad):
                data = [abs(x) for x in list(data_grad[column])]
                ax.plot(data, label="grad_{}".format(index))
            ax.axhline(y=0.001, label="Stop criteria")
            ax.set_title("Gradient updates through iterations")
            ax.set_xlabel("Number of gradient iterations")
            ax.set_ylabel("Absolute gradient update")
            plt.legend()
            plt.savefig(
                self.plot_p.joinpath(
                    "Gradient descent updates mpc step {}.png".format(mpc_step)
                )
            )
            plt.show()
            # plot zoomed in gradients after 2000 iterations (if gradient descent did not have premature convergence)
            if len(data) > 8000:
                fig, ax = plt.subplots()
                for index, column in enumerate(columns_grad):
                    data = [abs(x) for x in list(data_grad[column])]
                    ax.plot(data[2000:], label="grad_{}".format(index))
                ax.axhline(y=0.001, label="Stop criteria")
                ax.set_title("Gradient updates through iterations")
                ax.set_xlabel("Number of gradient iterations")
                ax.set_ylabel("Absolute gradient update")
                plt.legend()
                plt.savefig(
                    plot_p.joinpath("optimization").joinpath(
                        "Zoomed in gradient descent updates mpc step {}.png".format(
                            mpc_step
                        )
                    )
                )
                plt.show()
            # plot decision variables updates
            fig, ax = plt.subplots()
            for index, column in enumerate(columns_var):
                data = [
                    (self.produced_heat_max - self.produced_heat_min) * x
                    + self.produced_heat_min
                    for x in list(data_var[column])
                ]
                ax.plot(data, label="h_{}".format(index))
            ax.set_title("Decision variables updates through iterations")
            ax.set_xlabel("Number of gradient iterations")
            ax.set_ylabel("Decision variable")
            plt.legend()
            plt.savefig(
                self.plot_p.joinpath(
                    "Decision variables updates mpc step {}.png".format(mpc_step)
                )
            )
            plt.show()
            # plot zoomed in decision variables updates after 2000 iterations (if gradient descent did not have premature convergence)
            if len(data) > 8000:
                fig, ax = plt.subplots()
                for index, column in enumerate(columns_var):
                    data = [
                        (self.produced_heat_max - self.produced_heat_min) * x
                        + self.produced_heat_min
                        for x in list(data_var[column])
                    ]
                    ax.plot(data[2000:], label="h_{}".format(index))
                ax.set_title("Decision variables updates through iterations")
                ax.set_xlabel("Number of gradient iterations")
                ax.set_ylabel("Decision variable")
                plt.legend()
                plt.savefig(
                    self.plot_p.joinpath(
                        "Zoomed in decision variables updates mpc step {}.png".format(
                            mpc_step
                        )
                    )
                )
                plt.show()

    def rescale_heat(self, heat):
        """
        Rescale normalized heat into MWh heat
        """

        rescaled_heat = (
            self.produced_heat_max - self.produced_heat_min
        ) * heat + self.produced_heat_min
        if rescaled_heat < 0:
            rescaled_heat = 0
        return rescaled_heat

    def get_biggest_difference(
        self, produced_heat, delta, patience
    ) -> Tuple[float, int]:
        """
        Get the biggest difference of produced heat between the stop iteration and the end iteration.
        """
        N: int = len(
            produced_heat
        )  # length can be different depending on at which iteration gradient converged (max 10000 iterations)
        max_diff = 0  # minimal difference
        iter = N - 1  # maximal iteration number
        if N <= patience:
            return max_diff, iter
        for i in range(N - patience):
            x = produced_heat[i]
            consecutive_steps = 0
            max_diff = 0
            for j in range(1, patience):
                x_new = produced_heat[i + j]
                if abs(x - x_new) <= delta:
                    consecutive_steps += 1
                else:
                    consecutive_steps = 0
                    break
            if consecutive_steps >= patience - 1:
                for j in range(i, N):
                    x = max(
                        max_diff,
                        abs(
                            self.rescale_heat(produced_heat[j])
                            - self.rescale_heat(produced_heat[-1])
                        ),
                    )
                    if x > max_diff:
                        max_diff = x
                iter = i + patience
                break
        return max_diff, iter

    def simulator_evaluation(self, produced_heat, produced_electricity, opt_step):
        """
        Evaluate feasibility through simulation based on produced heat and electricity.
        """
        demand, price, plugs = get_demand_price_plugs(opt_step)
        if ProducerPreset1["ControlWithTemp"]:
            warnings.warn(
                "As we are determining stop criteria for actions optimized via gradient descent, it is possible to control simulator only with the heat"
            )
        # build simulator
        simulator = build_grid(
            consumer_demands=[demand], electricity_prices=[price], config=config
        )
        # get object's ids
        (
            object_ids,
            producer_id,
            consumer_ids,
            sup_edge_ids,
            ret_edge_ids,
        ) = Optimizer.get_object_ids(simulator)
        history_mass_flow = {
            sup_edge_ids: plugs[0][0][0] / TimeParameters["TimeInterval"],
            ret_edge_ids: plugs[0][0][0] / TimeParameters["TimeInterval"],
        }
        (
            supply_inlet_violation,
            supply_outlet_violation,
            mass_flow_violation,
            delivered_heat_violation,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = run_simulator(
            simulator=simulator,
            object_ids=object_ids,
            producer_id=producer_id,
            consumer_ids=consumer_ids,
            sup_edge_ids=sup_edge_ids,
            produced_heat=produced_heat,
            supply_inlet_temperature=[90] * TimeParameters["PlanningHorizon"],
            produced_electricity=produced_electricity,
            demand=demand,
            price=price,
            plugs=plugs,
            history_mass_flow=history_mass_flow,
        )
        return (
            abs(sum(supply_inlet_violation) / len(supply_inlet_violation)),
            abs(sum(supply_outlet_violation) / len(supply_outlet_violation)),
            abs(sum(mass_flow_violation) / len(mass_flow_violation)),
            abs(sum(delivered_heat_violation) / len(delivered_heat_violation)),
        )

    def get_stop_criteria_per_model(
        self,
        experiments_folder,
        experiments_sub_folder,
        learning_rate,
        delta: int,
        patience: int,
        mpc_steps: int,
        neuron: list,
        nn_models: int,
        opt_step: int,
        date: str,
    ):
        """
        Depending on the type of the model,
        store maximal produced heat difference, stop iteration, operation cost at the stop iteration and operation cost at the last iteration
        for all MPC steps.
        """
        dict_diff: dict = (
            {}
        )  # the largest difference in produced heat from h[iter] to the last produced heat
        dict_h_max_iter: dict = {}  # produced heat at the last iteration
        dict_p_max_iter: dict = {}  # produced electricity at the last iteration
        dict_h_iter: dict = {}  # produced heat at the stop iteration
        dict_p_iter: dict = {}  # produced electricity at the stop iteration
        dict_iter: dict = {}  # stopping iteration
        dict_max_iter: dict = {}  # maximal number of iterations
        dict_obj_fun_max_iter: dict = {}  # operation cost value at the stop iteration
        dict_obj_fun: dict = (
            {}
        )  # operation cost value corresponding to the last iteration
        for model_num in range(nn_models):
            dict_diff[str(model_num)] = []
            dict_h_max_iter[str(model_num)] = []
            dict_h_iter[str(model_num)] = []
            dict_p_max_iter[str(model_num)] = []
            dict_p_iter[str(model_num)] = []
            dict_iter[str(model_num)] = []
            dict_max_iter[str(model_num)] = []
            dict_obj_fun_max_iter[str(model_num)] = []
            dict_obj_fun[str(model_num)] = []
        path = (
            self.result_p.joinpath(experiments_folder)
            .joinpath(experiments_sub_folder)
            .joinpath("learning_rate_{}_N_10000".format(learning_rate))
            .joinpath("heat_gradients")
        )
        with open(
            self.result_p.joinpath("electricity_price.csv"),
            "rb",
        ) as f:
            electricity_prices = list(pd.read_csv(f)["Electricity price"])
        delta = delta / (self.produced_heat_max - self.produced_heat_min)
        for model_num in range(nn_models):
            for mpc_step in range(mpc_steps):
                produced_heat = list(
                    pd.read_csv(
                        path.joinpath(
                            "{}_heat_gradients_neurons_".format(model_num)
                            + Plot.create_neuron_string(neuron)
                            + "_opt_step_{}_MPC_step_{}_".format(opt_step, mpc_step)
                            + date
                            + ".csv"
                        )
                    )["new_h_0"]
                )
                electricity_price = float(
                    electricity_prices[
                        opt_step + mpc_step - time_delay[str(PipePreset1["Length"])] - 1
                    ]
                )
                diff, iteration = self.get_biggest_difference(
                    produced_heat=produced_heat, delta=delta, patience=patience
                )
                dict_diff[str(model_num)].append(diff)
                dict_h_iter[str(model_num)].append(
                    self.rescale_heat(produced_heat[iteration])
                )
                dict_h_max_iter[str(model_num)].append(
                    self.rescale_heat(produced_heat[-1])
                )
                dict_iter[str(model_num)].append(iteration)
                dict_max_iter[str(model_num)].append(len(produced_heat))
                optimal_electricity_max_iter = get_optimal_produced_electricity(
                    produced_heat=[self.rescale_heat(produced_heat[-1])],
                    electricity_price=[electricity_price],
                )
                optimal_electricity = get_optimal_produced_electricity(
                    produced_heat=[self.rescale_heat(produced_heat[iteration])],
                    electricity_price=[electricity_price],
                )
                dict_p_iter[str(model_num)].extend(optimal_electricity)
                dict_p_max_iter[str(model_num)].extend(optimal_electricity_max_iter)
                dict_obj_fun_max_iter[str(model_num)].extend(
                    calculate_operation_cost(
                        produced_heat=[self.rescale_heat(produced_heat[-1])],
                        produced_electricity=optimal_electricity_max_iter,
                        electricity_price=[electricity_price],
                    )
                )
                dict_obj_fun[str(model_num)].extend(
                    calculate_operation_cost(
                        produced_heat=[self.rescale_heat(produced_heat[iteration])],
                        produced_electricity=optimal_electricity,
                        electricity_price=[electricity_price],
                    )
                )
        return (
            dict_diff,
            dict_h_iter,
            dict_p_iter,
            dict_h_max_iter,
            dict_p_max_iter,
            dict_iter,
            dict_max_iter,
            dict_obj_fun,
            dict_obj_fun_max_iter,
        )

    def get_stop_criteria_per_delta_and_patience(
        self,
        experiments_folder,
        experiments_sub_folder,
        learning_rate,
        deltas: list,
        patiences: list,
        mpc_steps: int,
        nn_sizes: list,
        nn_models: int,
        opt_step: int,
        date: str,
    ):
        """
        Depending on the parameters delta and patience,
        get maximal difference of produced heat, number of iterations, objective function, supply inlet temperature violation, supply outlet temperature violation, max flow violation and delivered heat violation.

        """
        dict_heat_diff: dict = Plot.form_3D_dictionary(
            x=deltas, y=patiences, z=nn_sizes
        )
        dict_supply_in_iter: dict = Plot.form_3D_dictionary(
            x=deltas, y=patiences, z=nn_sizes
        )
        dict_supply_in_max_iter: dict = Plot.form_3D_dictionary(
            x=deltas, y=patiences, z=nn_sizes
        )
        dict_supply_out_iter: dict = Plot.form_3D_dictionary(
            x=deltas, y=patiences, z=nn_sizes
        )
        dict_supply_out_max_iter: dict = Plot.form_3D_dictionary(
            x=deltas, y=patiences, z=nn_sizes
        )
        dict_m_iter: dict = Plot.form_3D_dictionary(x=deltas, y=patiences, z=nn_sizes)
        dict_m_max_iter: dict = Plot.form_3D_dictionary(
            x=deltas, y=patiences, z=nn_sizes
        )
        dict_q_iter: dict = Plot.form_3D_dictionary(x=deltas, y=patiences, z=nn_sizes)
        dict_q_max_iter: dict = Plot.form_3D_dictionary(
            x=deltas, y=patiences, z=nn_sizes
        )
        dict_iter: dict = Plot.form_3D_dictionary(x=deltas, y=patiences, z=nn_sizes)
        dict_max_iter: dict = Plot.form_3D_dictionary(x=deltas, y=patiences, z=nn_sizes)
        dict_obj_fun: dict = Plot.form_3D_dictionary(x=deltas, y=patiences, z=nn_sizes)
        dict_obj_fun_max_iter: dict = Plot.form_3D_dictionary(
            x=deltas, y=patiences, z=nn_sizes
        )
        for delta in deltas:
            for patience in patiences:
                for neuron in nn_sizes:
                    (
                        dict_heat_diff_,
                        dict_h_iter_,
                        dict_p_iter_,
                        dict_h_max_iter_,
                        dict_p_max_iter_,
                        dict_iter_,
                        dict_max_iter_,
                        dict_obj_fun_,
                        dict_obj_fun_max_iter_,
                    ) = self.get_stop_criteria_per_model(
                        experiments_folder=experiments_folder,
                        experiments_sub_folder=experiments_sub_folder,
                        learning_rate=learning_rate,
                        delta=delta,
                        patience=patience,
                        mpc_steps=mpc_steps,
                        neuron=neuron,
                        nn_models=nn_models,
                        opt_step=opt_step,
                        date=date,
                    )
                    for nn_model in range(nn_models):
                        dict_heat_diff[str(delta)][str(patience)][str(neuron)].extend(
                            dict_heat_diff_[str(nn_model)]
                        )
                        # mean of violation per neural network model
                        (
                            supply_in_iter,
                            supply_out_iter,
                            m_iter,
                            q_iter,
                        ) = self.simulator_evaluation(
                            produced_heat=dict_h_iter_[str(nn_model)],
                            produced_electricity=dict_p_iter_[str(nn_model)],
                            opt_step=opt_step,
                        )
                        dict_supply_in_iter[str(delta)][str(patience)][
                            str(neuron)
                        ].append(supply_in_iter)
                        dict_supply_out_iter[str(delta)][str(patience)][
                            str(neuron)
                        ].append(supply_out_iter)
                        dict_m_iter[str(delta)][str(patience)][str(neuron)].append(
                            m_iter
                        )
                        dict_q_iter[str(delta)][str(patience)][str(neuron)].append(
                            q_iter
                        )
                        (
                            supply_in_max_iter,
                            supply_out_max_iter,
                            m_max_iter,
                            q_max_iter,
                        ) = self.simulator_evaluation(
                            produced_heat=dict_h_max_iter_[str(nn_model)],
                            produced_electricity=dict_p_max_iter_[str(nn_model)],
                            opt_step=opt_step,
                        )
                        dict_supply_in_max_iter[str(delta)][str(patience)][
                            str(neuron)
                        ].append(supply_in_max_iter)
                        dict_supply_out_max_iter[str(delta)][str(patience)][
                            str(neuron)
                        ].append(supply_out_max_iter)
                        dict_m_max_iter[str(delta)][str(patience)][str(neuron)].append(
                            m_max_iter
                        )
                        dict_q_max_iter[str(delta)][str(patience)][str(neuron)].append(
                            q_max_iter
                        )
                        dict_iter[str(delta)][str(patience)][str(neuron)].extend(
                            dict_iter_[str(nn_model)]
                        )
                        dict_max_iter[str(delta)][str(patience)][str(neuron)].extend(
                            dict_max_iter_[str(nn_model)]
                        )
                        dict_obj_fun[str(delta)][str(patience)][str(neuron)].append(
                            sum(dict_obj_fun_[str(nn_model)])
                        )
                        dict_obj_fun_max_iter[str(delta)][str(patience)][
                            str(neuron)
                        ].append(sum(dict_obj_fun_max_iter_[str(nn_model)]))
        return (
            dict_heat_diff,
            dict_supply_in_iter,
            dict_supply_out_iter,
            dict_m_iter,
            dict_q_iter,
            dict_supply_in_max_iter,
            dict_supply_out_max_iter,
            dict_m_max_iter,
            dict_q_max_iter,
            dict_iter,
            dict_max_iter,
            dict_obj_fun,
            dict_obj_fun_max_iter,
        )

    def plot_stop_criteria(
        self,
        experiments_folder,
        experiments_sub_folder,
        learning_rate,
        deltas: list,
        patiences: list,
        mpc_steps: int,
        nn_sizes: list,
        nn_models: int,
        opt_step: int,
        date: str,
    ):
        """
        Finally, plot maximal difference of produced heat, number of iterations, objective function,
        supply inlet temperature violation, supply outlet temperature violation, mass flow violation and delivered heat violation as the function of stop parameters delta and patience.
        """
        color_neurons = {"[1]": "r", "[10]": "b", "[50, 50]": "g"}
        violations_ylabel = [
            "Supply inlet temperature violation [C]",
            "Supply outlet temperature violation [C]",
            "Mass flow violation [kg/s]",
            "Delivered heat violation [MWh]",
        ]
        violations_title = [
            "Supply inlet temperature violation as the function of stop parameters",
            "Supply outlet temperature violation as the function of stop parameters",
            "Mass flow violation as the function of stop parameters",
            "Delivered heat violation as the function of stop parameters",
        ]
        (
            dict_heat_diff,
            dict_supply_in_iter,
            dict_supply_out_iter,
            dict_m_iter,
            dict_q_iter,
            dict_supply_in_max_iter,
            dict_supply_out_max_iter,
            dict_m_max_iter,
            dict_q_max_iter,
            dict_iter,
            dict_max_iter,
            dict_obj_fun,
            dict_obj_fun_max_iter,
        ) = self.get_stop_criteria_per_delta_and_patience(
            experiments_folder=experiments_folder,
            experiments_sub_folder=experiments_sub_folder,
            learning_rate=learning_rate,
            deltas=deltas,
            patiences=patiences,
            mpc_steps=mpc_steps,
            nn_sizes=nn_sizes,
            nn_models=nn_models,
            opt_step=opt_step,
            date=date,
        )
        x_ticks = []
        for delta in deltas:
            for patience in patiences:
                x_ticks.append(str(delta) + "," + str(patience))
        # plot maximal difference between produced heat in the last iteration and heat from the stop iteration up to the last iteration
        # as the function of delta and patience parameters
        x = list(range(len(x_ticks)))
        x = [xx + 0.1 * len(nn_sizes) / 2 for xx in x]
        fig, ax = plt.subplots(figsize=(18, 6))
        counter = 0
        for delta in deltas:
            for patience in patiences:
                delta_xlabel = 0
                for neuron in nn_sizes:
                    np_array = np.array(
                        dict_heat_diff[str(delta)][str(patience)][str(neuron)]
                    )
                    if counter == 0:
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(np_array),
                            np.std(np_array),
                            fmt="o",
                            capsize=4,
                            color=color_neurons[str(neuron)],
                            label=str(neuron),
                        )
                    else:
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(np_array),
                            np.std(np_array),
                            fmt="o",
                            capsize=4,
                            color=color_neurons[str(neuron)],
                        )
                    delta_xlabel += 0.1
                counter += 1
        plt.xlabel("Delta [MWh], patience")
        plt.ylabel("Absolute difference in produced heat [MWh]")
        plt.legend()
        plt.title("Decision variable difference as the function of stop parameters")
        plt.xticks(x, x_ticks)
        plt.show()
        # plot maximal number of iterations and stop iteration
        x = list(range(len(x_ticks)))
        x = [xx + 0.1 * len(nn_sizes) for xx in x]
        fig, ax = plt.subplots(figsize=(18, 6))
        counter = 0
        for delta in deltas:
            for patience in patiences:
                delta_xlabel = 0
                for neuron in nn_sizes:
                    np_array_iter = np.array(
                        dict_iter[str(delta)][str(patience)][str(neuron)]
                    )
                    np_array_max_iter = np.array(
                        dict_max_iter[str(delta)][str(patience)][str(neuron)]
                    )
                    if counter == 0:
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(np_array_iter),
                            np.std(np_array_iter),
                            fmt="o",
                            capsize=4,
                            color=color_neurons[str(neuron)],
                            label=str(neuron) + " stop iter",
                        )
                        delta_xlabel += 0.05
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(np_array_max_iter),
                            np.std(np_array_max_iter),
                            fmt="x",
                            linestyle="dashed",
                            capsize=4,
                            color=color_neurons[str(neuron)],
                            label=str(neuron) + " max iter",
                        )
                    else:
                        delta_xlabel += 0.1
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(np_array_iter),
                            np.std(np_array_iter),
                            fmt="o",
                            capsize=4,
                            color=color_neurons[str(neuron)],
                        )
                        delta_xlabel += 0.05
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(np_array_max_iter),
                            np.std(np_array_max_iter),
                            fmt="x",
                            capsize=4,
                            color=color_neurons[str(neuron)],
                            linestyle="dashed",
                        )
                counter += 1
        plt.xlabel("Delta [MWh], patience")
        plt.ylabel("Number of iterations")
        plt.legend()
        plt.title("Number of iterations as the function of stop parameters")
        plt.xticks(x, x_ticks)
        plt.savefig(
            self.plot_p.joinpath(
                "Number of iterations as the function of stop parameters.png"
            )
        )
        plt.show()
        # plot operation cost at the stop iteration and the maximal iteration
        x = list(range(len(x_ticks)))
        x = [xx + 0.1 * len(nn_sizes) for xx in x]
        fig, ax = plt.subplots(figsize=(18, 6))
        counter = 0
        for delta in deltas:
            for patience in patiences:
                delta_xlabel = 0
                for neuron in nn_sizes:
                    np_array_iter = np.array(
                        dict_obj_fun[str(delta)][str(patience)][str(neuron)]
                    )
                    np_array_max_iter = np.array(
                        dict_obj_fun_max_iter[str(delta)][str(patience)][str(neuron)]
                    )
                    if counter == 0:
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(np_array_iter),
                            np.std(np_array_iter),
                            fmt="o",
                            capsize=4,
                            color=color_neurons[str(neuron)],
                            label=str(neuron) + " stop iter",
                        )
                        delta_xlabel += 0.05
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(np_array_max_iter),
                            np.std(np_array_max_iter),
                            fmt="x",
                            linestyle="dashed",
                            capsize=4,
                            color=color_neurons[str(neuron)],
                            label=str(neuron) + " max iter",
                        )
                    else:
                        delta_xlabel += 0.1
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(np_array_iter),
                            np.std(np_array_iter),
                            fmt="o",
                            capsize=4,
                            color=color_neurons[str(neuron)],
                        )
                        delta_xlabel += 0.05
                        ax.errorbar(
                            counter + delta_xlabel,
                            np.mean(np_array_max_iter),
                            np.std(np_array_max_iter),
                            fmt="x",
                            capsize=4,
                            color=color_neurons[str(neuron)],
                            linestyle="dashed",
                        )
                counter += 1
        plt.xlabel("Delta [MWh], patience")
        plt.ylabel("Operation cost [e]")
        plt.legend()
        plt.title("Operation cost as the function of stop parameters")
        plt.xticks(x, x_ticks)
        plt.savefig(
            self.plot_p.joinpath(
                "Operation cost as the function of stop parameters.png"
            )
        )
        plt.show()
        # plot four violations at the stop iteration and the maximal iteration
        violations_dict_iter: list = [
            dict_supply_in_iter,
            dict_supply_out_iter,
            dict_m_iter,
            dict_q_iter,
        ]
        violations_dict_max_iter: list = [
            dict_supply_in_max_iter,
            dict_supply_out_max_iter,
            dict_m_max_iter,
            dict_q_max_iter,
        ]
        for i in range(len(violations_dict_iter)):
            x = list(range(len(x_ticks)))
            x = [xx + 0.1 * len(nn_sizes) for xx in x]
            fig, ax = plt.subplots(figsize=(18, 6))
            counter = 0
            for delta in deltas:
                for patience in patiences:
                    delta_xlabel = 0
                    for neuron in nn_sizes:
                        np_array_iter = np.array(
                            violations_dict_iter[i][str(delta)][str(patience)][
                                str(neuron)
                            ]
                        )
                        np_array_max_iter = np.array(
                            violations_dict_max_iter[i][str(delta)][str(patience)][
                                str(neuron)
                            ]
                        )
                        if counter == 0:
                            ax.errorbar(
                                counter + delta_xlabel,
                                np.mean(np_array_iter),
                                np.std(np_array_iter),
                                fmt="o",
                                capsize=4,
                                color=color_neurons[str(neuron)],
                                label=str(neuron) + " stop iter",
                            )
                            delta_xlabel += 0.05
                            ax.errorbar(
                                counter + delta_xlabel,
                                np.mean(np_array_max_iter),
                                np.std(np_array_max_iter),
                                fmt="x",
                                linestyle="dashed",
                                capsize=4,
                                color=color_neurons[str(neuron)],
                                label=str(neuron) + " max iter",
                            )
                        else:
                            delta_xlabel += 0.1
                            ax.errorbar(
                                counter + delta_xlabel,
                                np.mean(np_array_iter),
                                np.std(np_array_iter),
                                fmt="o",
                                capsize=4,
                                color=color_neurons[str(neuron)],
                            )
                            delta_xlabel += 0.05
                            ax.errorbar(
                                counter + delta_xlabel,
                                np.mean(np_array_max_iter),
                                np.std(np_array_max_iter),
                                fmt="x",
                                capsize=4,
                                color=color_neurons[str(neuron)],
                                linestyle="dashed",
                            )
                    counter += 1
            plt.xlabel("Delta [MWh], patience")
            plt.ylabel(violations_ylabel[i])
            plt.legend()
            plt.title(violations_title[i])
            plt.xticks(x, x_ticks)
            plt.savefig(self.plot_p.joinpath(violations_title[i] + ".png"))
            plt.show()


if __name__ == "__main__":
    experiment_folder = experiments_optimization["relax_monotonic_icnn_gd"]["folder"]
    experiment_sub_folder = experiments_optimization["relax_monotonic_icnn_gd"][
        "sub-folder"
    ]
    result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
    plot_p: Path = (
        Path(__file__).parents[4]
        / "plots/constraint_opt/icnn_gd_hyperparameters_tuning"
    )
    plot = PlotHyperparametersTuning(result_p=result_p, plot_p=plot_p)
    """
    plot.evaluate_learning_rate(
        experiment_folder=experiment_folder,
        experiment_sub_folder=experiment_sub_folder,
        learning_rates=[0.01, 0.001],
        neurons=[[1], [10], [50, 50]],
    )
    plot.plot_gradient_and_decision_variable_updates_per_iterations(
        experiment_folder=experiment_folder, experiment_sub_folder=experiment_sub_folder, learning_rate=0.001
    )
    """
    plot.plot_stop_criteria(
        experiments_folder=experiment_folder,
        experiments_sub_folder=experiment_sub_folder,
        learning_rate=0.01,
        deltas=[1, 0.5, 0.25],
        patiences=[10, 100, 1000, 2000],
        mpc_steps=24,
        nn_sizes=[[1], [10], [50, 50]],
        nn_models=3,
        opt_step=12,
        date="2023-06-20",
    )

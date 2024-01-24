"""
Tests whether specified optimization days are out-of-distribution by propagating state through
learned deep neural network models and simulator.
"""
from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer


if GridProperties["ConsumerNum"] == 1:
    from src.simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from src.simulator.cases.parallel_consumers import build_grid


class OutOfDistribution(Optimizer):
    def __init__(
        self,
        experiment_learn_type,
        experiment_opt_type,
        x_s,
        electricity_price,
        supply_pipe_plugs,
        return_pipe_plugs,
        T,
        N_w,
        N_w_q,
        ad,
    ):
        super().__init__(
            experiment_learn_type=experiment_learn_type,
            experiment_opt_type=experiment_opt_type,
            x_s=x_s,
            electricity_price=electricity_price,
            supply_pipe_plugs=supply_pipe_plugs,
            return_pipe_plugs=return_pipe_plugs,
            T=T,
            N_w=N_w,
            N_w_q=N_w_q,
            ad=ad,
        )

    def find_result_file(self, start, result_folder, result_sub_folder):
        """
        Finds result file based on the start of its name and collection of files.
        This implementation throws warning if multiple files with the same name are found.
        This can be the case if the same files have different time stamps, in which case the warning is thrown.
        """
        result_p: Path = self.result_p.joinpath(result_folder).joinpath(
            result_sub_folder
        )
        files = os.listdir(result_p)
        file = csv_file_finder(files=files, start=start, null_solve=False)
        file = pd.read_csv(result_p.joinpath(file))
        h = file["Produced_heat_optimizer"].values.tolist()
        # supply_inlet_temperature = file["T_supply_optimized"].values.tolist()
        supply_inlet_temperature = [90] * len(h)
        return h, supply_inlet_temperature

    def optimize(
        self,
    ):
        """
        Overriding the abstract method optimize
        """
        print(
            "Class not for the optimization, but for the out-of-distribution analysis"
        )

    def state_transition_function(
        self, model, tensor_constraint, tau_in, tau_out, m, heat_demand, h_init, h
    ):
        """
        Based on the specified, already learned state transition function and its input, predict the next state.
        """
        h = normalize_variable(
            var=h, min=self.produced_heat_min, max=self.produced_heat_max
        )
        model = self.load_nn_model(model_p=model, tensor_constraint=tensor_constraint)
        input = [tau_in, tau_out, m]
        input.extend(heat_demand)
        input.extend(h_init)
        input.append(h)
        input = np.array(input).reshape(1, -1)
        pred = model.predict(input)
        pred = pred[0]
        tau_in_pred = pred[0]
        tau_out_pred = pred[1]
        m_pred = pred[2]
        return tau_in_pred, tau_out_pred, m_pred

    def system_output_function(
        self, model, tensor_constraint, tau_in, tau_out, m, h_init, h
    ):
        """
        Based on the specified, already learned system output function and its input, predict the delivered heat.
        """
        h = normalize_variable(
            var=h, min=self.produced_heat_min, max=self.produced_heat_max
        )
        model = self.load_nn_model(model_p=model, tensor_constraint=tensor_constraint)
        input = [tau_in, tau_out, m]
        input.extend(h_init)
        input.append(h)
        input = np.array(input).reshape(1, -1)
        pred = model.predict(input)
        return pred[0][0]

    def propagate(self, h, T_supply, opt_step, tensor_constraint):
        """
        Propagate solutions through the simulator and through learned deep neural network state trabsition and system output functions.
        """
        print(h)
        data = {
            "Supply in temp sim": [],
            "Supply out temp sim": [],
            "Mass flow sim": [],
            "Delivered heat sim": [],
            "Supply in temp opt": [],
            "Supply out temp opt": [],
            "Mass flow opt": [],
            "Delivered heat opt": [],
        }
        # opt step is shifted for the time delay needed for dnn-opt planners
        opt_step = opt_step - time_delay[str(PipePreset1["Length"])] - 1
        tau_in_init: float = self.get_tau_in(opt_step=opt_step)
        tau_out_init: float = self.get_tau_out(opt_step=opt_step)
        m_init: float = self.get_m(opt_step=opt_step)
        h_init: list = self.get_h(opt_step=opt_step)
        # normalized heat demand
        q: list = self.get_demand_(opt_step=opt_step)
        # plugs
        plugs = self.get_plugs(opt_step=opt_step)
        # regular heat demand
        heat_demand: np.array = self.get_demand(opt_step=opt_step)
        # electricity price
        price: list = self.get_price(opt_step=opt_step)
        # build simulator
        simulator = build_grid(
            consumer_demands=[heat_demand],
            electricity_prices=[price],
            config=config,
        )
        # get object's ids
        (
            object_ids,
            producer_id,
            consumer_ids,
            sup_edge_ids,
            ret_edge_ids,
        ) = Optimizer.get_object_ids(simulator)
        for i in range(len(h) - TimeParameters["PlanningHorizon"] + 1):
            produced_heat = h[i : i + TimeParameters["PlanningHorizon"]]
            supply_inlet_temperature = T_supply[
                i : i + TimeParameters["PlanningHorizon"]
            ]
            produced_electricity = [5] * len(produced_heat)
            (
                tau_in_sim,
                tau_out_sim,
                m_sim,
                delivered_heat_sim,
                plugs,
            ) = self.propagate_through_simulator(
                simulator=simulator,
                object_ids=object_ids,
                producer_id=producer_id,
                consumer_ids=consumer_ids,
                sup_edge_ids=sup_edge_ids,
                ret_edge_ids=ret_edge_ids,
                produced_heat=produced_heat,
                supply_inlet_temperature=supply_inlet_temperature,
                produced_electricity=produced_electricity,
                heat_demand=heat_demand,
                price=price,
                plugs=plugs,
                m_init=m_init,
            )
            data["Supply in temp sim"].append(tau_in_sim)
            data["Supply out temp sim"].append(tau_out_sim)
            data["Mass flow sim"].append(m_sim)
            data["Delivered heat sim"].append(delivered_heat_sim)
            tau_in_nn, tau_out_nn, m_nn = self.state_transition_function(
                model=model_state,
                tensor_constraint=tensor_constraint,
                tau_in=tau_in_init,
                tau_out=tau_out_init,
                m=m_init,
                heat_demand=q[0 : 0 + TimeParameters["ActionHorizon"] + 1],
                h_init=h_init,
                h=produced_heat[0],
            )
            data["Supply in temp opt"].append(
                re_normalize_variable(
                    var=tau_in_nn,
                    min=self.state_dict["Supply in temp 1 min"],
                    max=self.state_dict["Supply in temp 1 max"],
                )
            )
            data["Supply out temp opt"].append(
                re_normalize_variable(
                    var=tau_out_nn,
                    min=self.state_dict["Supply out temp 1 min"],
                    max=self.state_dict["Supply out temp 1 max"],
                )
            )
            data["Mass flow opt"].append(
                re_normalize_variable(
                    var=m_nn,
                    min=self.state_dict["Supply mass flow 1 min"],
                    max=self.state_dict["Supply mass flow 1 max"],
                )
            )
            delivered_heat_nn = self.system_output_function(
                model=model_output,
                tensor_constraint=tensor_constraint,
                tau_in=tau_in_nn,
                tau_out=tau_out_nn,
                m=m_nn,
                h_init=h_init,
                h=produced_heat[0],
            )

            data["Delivered heat opt"].append(
                re_normalize_variable(
                    var=delivered_heat_nn,
                    min=self.output_dict["Delivered heat 1 min"],
                    max=self.output_dict["Delivered heat 1 max"],
                )
            )

            # get initial values for the next MPC iteration
            tau_in_init = max(
                normalize_variable(
                    var=tau_in_sim,
                    min=self.state_dict["Supply in temp 1 min"],
                    max=self.state_dict["Supply in temp 1 max"],
                ),
                0.01,
            )
            tau_out_init = max(
                normalize_variable(
                    var=tau_out_sim,
                    min=self.state_dict["Supply out temp 1 min"],
                    max=self.state_dict["Supply out temp 1 max"],
                ),
                0.01,
            )
            m_init = max(
                normalize_variable(
                    var=m_sim,
                    min=self.state_dict["Supply mass flow 1 min"],
                    max=self.state_dict["Supply mass flow 1 max"],
                ),
                0.01,
            )
            # update of previous actions
            h_init = h_init[TimeParameters["ActionHorizon"] :]
            h_init.append(
                max(
                    normalize_variable(
                        var=produced_heat[0],
                        min=self.state_dict["Produced heat min"],
                        max=self.state_dict["Produced heat max"],
                    ),
                    0.01,
                )
            )
            q = self.get_demand_(
                opt_step=opt_step + i + TimeParameters["ActionHorizon"]
            )  # time-step t-1,...,t+T (normalized)
            # update heat demand and electricity price
            heat_demand = self.get_demand(
                opt_step=opt_step + i + TimeParameters["ActionHorizon"]
            )
            price = self.get_price(
                opt_step=opt_step + i + TimeParameters["ActionHorizon"]
            )
        data = pd.DataFrame(data)
        return data

    def propagate_through_simulator(
        self,
        simulator,
        object_ids,
        producer_id,
        consumer_ids,
        sup_edge_ids,
        ret_edge_ids,
        produced_heat,
        supply_inlet_temperature,
        produced_electricity,
        heat_demand,
        price,
        plugs,
        m_init,
    ):
        """
        Propagate produced heat or supply inlet temperature (depending on the parameter ControlWithTemp) through the simulator.
        As the output get supply inlet temperature, supply outlet temperature, mass flow and delivered heat at the time-step 1.
        """
        # create historical mass flow
        # m_init is normalized mass flow
        if ProducerPreset1["ControlWithTemp"]:
            historical_mass_flow = None
        else:
            historical_mass_flow = {
                sup_edge_ids: re_normalize_variable(
                    var=m_init,
                    min=self.state_dict["Supply mass flow 1 min"],
                    max=self.state_dict["Supply mass flow 1 max"],
                ),
                ret_edge_ids: re_normalize_variable(
                    var=m_init,
                    min=self.state_dict["Supply mass flow 1 min"],
                    max=self.state_dict["Supply mass flow 1 max"],
                ),
            }
        (
            supply_inlet_violation,
            supply_outlet_violation,
            mass_flow_violation,
            delivered_heat_violation,
            produced_heat_sim,
            supply_inlet_temperature_sim,
            tau_out_sim,
            mass_flow_sim,
            return_outlet_temperature_sim,
            return_inlet_temperature_sim,
            plugs,
        ) = run_simulator(
            simulator,
            object_ids,
            producer_id,
            consumer_ids,
            sup_edge_ids,
            produced_heat,
            supply_inlet_temperature,
            produced_electricity,
            heat_demand,
            price,
            plugs,
            historical_mass_flow,
        )
        print(supply_inlet_violation)
        print(supply_outlet_violation)
        print(mass_flow_violation)
        print(delivered_heat_violation)
        delivered_heat_sim = (
            (
                PhysicalProperties["HeatCapacity"]
                / PhysicalProperties["EnergyUnitConversion"]
            )
            * mass_flow_sim[0]
            * (tau_out_sim[0] - return_inlet_temperature_sim[0])
        )
        return (
            supply_inlet_temperature_sim[0],
            tau_out_sim[0],
            mass_flow_sim[0],
            delivered_heat_sim,
            plugs,
        )

    def plot_out_of_distribution_analysis(
        self, data_path, save_plot_path, save_plot_name
    ):
        """
        Plot supply inlet temperature, supply outlet temperature, mass flow and delivered heat
        as the result of propagation through simulator and through learned DNN models.
        """
        data = pd.read_csv(data_path)
        # supply inlet temperature
        fig, ax = plt.subplots()
        plt.plot(data["Supply in temp sim"], color="r", label="Through simulator")
        plt.plot(
            data["Supply in temp opt"], color="b", label="Through learned DNN model"
        )
        plt.title("Supply inlet temperature")
        plt.xlabel("Time steps")
        plt.ylabel("Temperature [C]")
        plt.legend()
        plt.savefig(
            save_plot_path.joinpath("Supply inlet temperature " + save_plot_name)
        )
        plt.show()
        # supply outlet temperature
        fig, ax = plt.subplots()
        plt.plot(data["Supply out temp sim"], color="r", label="Through simulator")
        plt.plot(
            data["Supply out temp opt"], color="b", label="Through learned DNN model"
        )
        plt.title("Supply outlet temperature")
        plt.xlabel("Time steps")
        plt.ylabel("Temperature [C]")
        plt.legend()
        plt.savefig(
            save_plot_path.joinpath("Supply outlet temperature " + save_plot_name)
        )
        plt.show()
        # mass flow
        fig, ax = plt.subplots()
        plt.plot(data["Mass flow sim"], color="r", label="Through simulator")
        plt.plot(data["Mass flow opt"], color="b", label="Through learned DNN model")
        plt.title("Mass flow")
        plt.xlabel("Time steps")
        plt.ylabel("Mass flow [kg/s]")
        plt.legend()
        plt.savefig(save_plot_path.joinpath("Mass flow " + save_plot_name))
        plt.show()
        # delivered heat
        fig, ax = plt.subplots()
        plt.plot(data["Delivered heat sim"], color="r", label="Through simulator")
        plt.plot(
            data["Delivered heat opt"], color="b", label="Through learned DNN model"
        )
        plt.title("Delivered heat")
        plt.xlabel("Time steps")
        plt.ylabel("Delivered heat [MWh]")
        plt.legend()
        plt.savefig(save_plot_path.joinpath("Delivered heat " + save_plot_name))
        plt.show()


if __name__ == "__main__":
    # type of neural networks
    experiment_learn_type = experiments_learning["relax_monotonic_icnn"]
    # type of planner
    experiment_opt_type = experiments_optimization["relax_monotonic_icnn_gd"]
    # tensor constraint
    tensor_constraint = experiment_opt_type["tensor_constraint"]
    if (
        experiment_opt_type["optimizer_type"] == "icnn_gd"
        and ProducerPreset1["ControlWithTemp"]
    ):
        warnings.warn(
            "When having GD planner, it is not possible to control the system with temperature."
        )
        sys.exit(1)
    nn_model = 0
    opt_step = 10000
    neurons = [50, 50]
    # name of the learned state transition model
    model_state = (
        "{}_model_state".format(nn_model)
        + experiment_opt_type["model_ext"]
        + "_s_time_delay_{}_".format(time_delay[str(PipePreset1["Length"])])
        + neurons_ext(neurons)
        + "_"
        + experiment_opt_type["nn_type"]
        + ".h5"
    )
    # name of the learned system output model
    model_output = (
        "{}_model_output".format(nn_model)
        + experiment_opt_type["model_ext"]
        + "_s_time_delay_{}_".format(time_delay[str(PipePreset1["Length"])])
        + neurons_ext(neurons)
        + "_"
        + experiment_opt_type["nn_type"]
        + ".h5"
    )
    # name of the result file
    result_name_start: str = (
        "{}_".format(nn_model)
        + experiment_opt_type["optimizer_type"]
        + "_init_"
        + neurons_ext(neurons)
        + "_opt_step_{}_".format(opt_step)
    )
    out_of_distribution_test = OutOfDistribution(
        experiment_learn_type=experiment_learn_type,
        experiment_opt_type=experiment_opt_type,
        x_s="x_s.csv",
        electricity_price="electricity_price.csv",
        supply_pipe_plugs="supply_pipe_plugs.pickle",
        return_pipe_plugs="return_pipe_plugs.pickle",
        T=TimeParameters["PlanningHorizon"],
        N_w=time_delay[str(PipePreset1["Length"])],
        N_w_q=time_delay_q[str(PipePreset1["Length"])],
        ad=experiment_learn_type["ad"],
    )
    # find the produced heat files
    h, supply_inlet_temperature = out_of_distribution_test.find_result_file(
        start=result_name_start,
        result_folder=experiment_opt_type["folder"],
        result_sub_folder=experiment_opt_type["sub-folder"],
    )
    data = out_of_distribution_test.propagate(
        h=h,
        T_supply=supply_inlet_temperature,
        opt_step=opt_step,
        tensor_constraint=tensor_constraint,
    )
    data_path = (
        (Path(__file__).parents[4] / "results/constraint_opt")
        .joinpath("out_of_distribution_test")
        .joinpath(experiment_opt_type["folder"])
        .joinpath(experiment_opt_type["sub-folder"])
        .joinpath(
            "{}_".format(nn_model)
            + experiment_opt_type["optimizer_type"]
            + "_"
            + neurons_ext(neurons)
            + "_opt_step_{}.csv".format(opt_step)
        )
    )
    data.to_csv(data_path)
    out_of_distribution_test.plot_out_of_distribution_analysis(
        data_path=data_path,
        save_plot_path=(Path(__file__).parents[4] / "plots/constraint_opt")
        .joinpath("out_of_distribution_test")
        .joinpath(experiment_opt_type["folder"])
        .joinpath(experiment_opt_type["sub-folder"]),
        save_plot_name="{}_".format(nn_model)
        + experiment_opt_type["optimizer_type"]
        + "_"
        + neurons_ext(neurons)
        + "_opt_step_{}_initialization_{}.png".format(opt_step, 0.875),
    )

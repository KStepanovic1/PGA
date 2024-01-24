from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer
from src.optimizers.constraint_opt.dhn_nn.gradient_descent import GradientDescent

if GridProperties["ConsumerNum"] == 1:
    from src.simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from src.simulator.cases.parallel_consumers import build_grid


def propagate(
    state_model,
    h_init,
    heat_demand_nor_,
    control,
    model_s,
    model_y,
    N_w,
    N_w_q,
):
    """
    Propagate state through learned DNN models, and access convexity in each time-step.
    """
    # indication whether f(g(g...))-q<0
    is_it_convex = True
    # state model
    state_model = tf.Variable(
        np.array(state_model).reshape(1, -1),
        dtype=tf.float32,
    )
    # transform normalized heat demand into tf.Variable
    heat_demand_nor = tf.Variable(
        np.array(heat_demand_nor_).reshape(1, -1), dtype=tf.float32
    )
    # transform initial produced heats+starting optimization point into tf.Variable
    actions = h_init + control
    actions = tf.Variable(np.array(actions).reshape(1, -1), dtype=tf.float32)
    for i in range(TimeParameters["PlanningHorizon"]):
        # dynamic state transition, s_t -> s_{t+1}
        state_model = model_s(
            tf.concat(
                [
                    state_model,
                    heat_demand_nor[:, i : N_w_q + 1 + i],
                    actions[:, i : N_w + 1 + i],
                ],
                axis=1,
            )
        )
        # state_model = tf.math.add(state_model, delta_state_model)
        # system output
        y = (
            model_y(
                tf.concat(
                    [
                        state_model,
                        actions[
                            :,
                            i : N_w + 1 + i,
                        ],
                    ],
                    axis=1,
                )
            )
            - heat_demand_nor_[i + N_w_q]
        )
        if y < 0:
            is_it_convex = False
            break
    return is_it_convex


if __name__ == "__main__":
    N_init = 12
    MPC = 60
    opt_step = 1024
    N_model: int = 3
    controls = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375, 1]
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
    layer_size = [50, 50]
    experiment_learn_type = experiments_learning["monotonic_icnn"]
    experiment_opt_type = experiments_optimization["monotonic_icnn_gd"]
    tensor_constraint: bool = experiment_opt_type[
        "tensor_constraint"
    ]  # does the neural network model used in optimization have overwritten weight constraint class?
    gd_optimizer = GradientDescent(
        experiment_learn_type=experiment_learn_type,
        experiment_opt_type=experiment_opt_type,
        x_s="x_s.csv",
        electricity_price="electricity_price.csv",
        supply_pipe_plugs="supply_pipe_plugs.pickle",
        return_pipe_plugs="return_pipe_plugs.pickle",
        T=TimeParameters["PlanningHorizon"],
        N_w=time_delay[str(PipePreset1["Length"])],
        N_w_q=time_delay_q[str(PipePreset1["Length"])],
    )
    results = {}
    for control in controls:
        results[str(control)] = {}
        for k in range(N_model):
            results[str(control)][str(k)] = []
    for control_ in controls:
        result_p: Path = (
            (Path(__file__).parents[4] / "results/constraint_opt")
            .joinpath(experiment_opt_type["folder"])
            .joinpath(experiment_opt_type["sub-folder"])
            .joinpath("initialization_{}/MPC_episode_length_72_hours".format(control_))
        )
        files = os.listdir(result_p)
        for k in range(N_model):
            file_start = (
                "{}_icnn_gd_init_".format(k)
                + neurons_ext(layer_size)
                + "_opt_step_{}_".format(opt_step)
            )
            file = csv_file_finder(
                files=files,
                start=file_start,
                null_solve=False,
            )
            # produced heat by our optimizer
            produced_heat = pd.read_csv(result_p.joinpath(file))["Q_optimized"].tolist()
            # initial values of internal variables
            tau_in_init: float = round(
                gd_optimizer.get_tau_in(
                    opt_step=opt_step - time_delay[str(PipePreset1["Length"])] - 1
                ),
                round_dig,
            )  # time-step t-1 (normalized)
            tau_out_init: float = round(
                gd_optimizer.get_tau_out(
                    opt_step=opt_step - time_delay[str(PipePreset1["Length"])] - 1
                ),
                round_dig,
            )  # time-step t-1 (normalized)
            m_init: float = round(
                gd_optimizer.get_m(
                    opt_step=opt_step - time_delay[str(PipePreset1["Length"])] - 1
                ),
                round_dig,
            )  # time-step t-1 (normalized)
            h_init: list = gd_optimizer.get_h(
                opt_step=opt_step - time_delay[str(PipePreset1["Length"])] - 1
            )  # time-step t-N_w,...t-1 (normalized)
            heat_demand_nor: list = gd_optimizer.get_demand_(
                opt_step=opt_step - time_delay[str(PipePreset1["Length"])] - 1
            )  # time-step t-1,...,t+T (normalized)
            plugs: list = gd_optimizer.get_plugs(
                opt_step=opt_step - time_delay[str(PipePreset1["Length"])] - 1
            )  # time-step t-1
            # initial values of external variables
            heat_demand: np.array = gd_optimizer.get_demand(
                opt_step=opt_step - time_delay[str(PipePreset1["Length"])] - 1
            )  # time-steps t,...,t+T (not normalized)
            price: list = gd_optimizer.get_price(
                opt_step=opt_step - time_delay[str(PipePreset1["Length"])] - 1
            )  # time-steps t,...,t+T
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
            state_dict = gd_optimizer.get_state_dict()
            for i in range(N_init + MPC - TimeParameters["PlanningHorizon"]):
                history_mass_flow = {
                    sup_edge_ids: re_normalize_variable(
                        var=m_init,
                        min=state_dict["Supply mass flow 1 min"],
                        max=state_dict["Supply mass flow 1 max"],
                    ),
                    ret_edge_ids: re_normalize_variable(
                        var=m_init,
                        min=state_dict["Supply mass flow 1 min"],
                        max=state_dict["Supply mass flow 1 max"],
                    ),
                }
                if ProducerPreset1["ControlWithTemp"]:
                    warnings.warn(
                        "For actions optimized via gradient descent, it is possible to control simulator only with the heat"
                    )
                    break
                model_s: str = (
                    "{}_model_state".format(k)
                    + experiment_learn_type["model_ext"]
                    + "_s_time_delay_{}_".format(time_delay[str(PipePreset1["Length"])])
                    + neurons_ext(layer_size)
                    + "_"
                    + experiment_opt_type["nn_type"]
                    + ".h5",
                )
                model_s = gd_optimizer.load_nn_model(
                    model_p=model_s, tensor_constraint=tensor_constraint
                )
                model_out: str = (
                    "{}_model_output".format(k)
                    + experiment_learn_type["model_ext"]
                    + "_s_time_delay_{}_".format(time_delay[str(PipePreset1["Length"])])
                    + neurons_ext(layer_size)
                    + "_"
                    + experiment_opt_type["nn_type"]
                    + ".h5"
                )
                model_y = gd_optimizer.load_nn_model(
                    model_p=model_out, tensor_constraint=tensor_constraint
                )
                # control = gd_optimizer.get_initial_control(opt_step_index, i)
                control = [control_] * TimeParameters["PlanningHorizon"]
                is_it_convex = propagate(
                    state_model=[tau_in_init, tau_out_init, m_init],
                    h_init=h_init,
                    heat_demand_nor_=heat_demand_nor,
                    control=control,
                    model_s=model_s,
                    model_y=model_y,
                    N_w=time_delay[str(PipePreset1["Length"])],
                    N_w_q=time_delay_q[str(PipePreset1["Length"])],
                )
                if not is_it_convex:
                    results[str(control_)][str(k)].append(i)
                (
                    supply_inlet_violation,
                    supply_outlet_violation,
                    mass_flow_violation,
                    delivered_heat_violation,
                    produced_heat_sim,
                    tau_in_sim,
                    tau_out,
                    m,
                    ret_tau_out_sim,
                    ret_tau_in_sim,
                    plugs,
                ) = run_simulator(
                    simulator=simulator,
                    object_ids=object_ids,
                    producer_id=producer_id,
                    consumer_ids=consumer_ids,
                    sup_edge_ids=sup_edge_ids,
                    produced_heat=produced_heat[
                        i : i + TimeParameters["PlanningHorizon"]
                    ],
                    supply_inlet_temperature=[90] * TimeParameters["PlanningHorizon"],
                    produced_electricity=[5] * TimeParameters["PlanningHorizon"],
                    demand=heat_demand,
                    price=price,
                    plugs=plugs,
                    history_mass_flow=history_mass_flow,
                )
                # get initial values for the next MPC iteration
                tau_in_init = max(
                    normalize_variable(
                        var=tau_in_sim[0],
                        min=state_dict["Supply in temp 1 min"],
                        max=state_dict["Supply in temp 1 max"],
                    ),
                    0.01,
                )
                tau_out_init = max(
                    normalize_variable(
                        var=tau_out[0],
                        min=state_dict["Supply out temp 1 min"],
                        max=state_dict["Supply out temp 1 max"],
                    ),
                    0.01,
                )
                m_init = max(
                    normalize_variable(
                        var=m[0],
                        min=state_dict["Supply mass flow 1 min"],
                        max=state_dict["Supply mass flow 1 max"],
                    ),
                    0.01,
                )
                # update of previous actions
                h_init = h_init[TimeParameters["ActionHorizon"] :]
                h_init.append(
                    max(
                        normalize_variable(
                            var=produced_heat_sim[0],
                            min=state_dict["Produced heat min"],
                            max=state_dict["Produced heat max"],
                        ),
                        0.01,
                    )
                )
                heat_demand_nor = gd_optimizer.get_demand_(
                    opt_step=opt_step
                    - time_delay[str(PipePreset1["Length"])]
                    - 1
                    + i
                    + TimeParameters["ActionHorizon"]
                )  # time-step t-1,...,t+T (normalized)
                # update heat demand and electricity price
                heat_demand = gd_optimizer.get_demand(
                    opt_step=opt_step
                    - time_delay[str(PipePreset1["Length"])]
                    - 1
                    + i
                    + TimeParameters["ActionHorizon"]
                )
                price = gd_optimizer.get_price(
                    opt_step=opt_step
                    - time_delay[str(PipePreset1["Length"])]
                    - 1
                    + i
                    + TimeParameters["ActionHorizon"]
                )
    fig, ax = plt.subplots()
    counter = 0
    x_ticks = ["Model 0", "Model 1", "Model 2"]
    x = [xx + 10 * xx for xx in range(N_model)]
    for model in range(N_model):
        delta_xlabel = 0
        for initialization in controls:
            if counter == 0:
                plt.scatter(
                    [counter + delta_xlabel]
                    * len(results[str(initialization)][str(model)]),
                    results[str(initialization)][str(model)],
                    color=colors[str(initialization)],
                    label="initialization={} MWh".format(
                        re_normalize_variable(
                            var=initialization,
                            min=state_dict["Produced heat min"],
                            max=state_dict["Produced heat max"],
                        )
                    ),
                    marker="*",
                )
                delta_xlabel += 0.5
            else:
                plt.scatter(
                    [counter + delta_xlabel]
                    * len(results[str(initialization)][str(model)]),
                    results[str(initialization)][str(model)],
                    color=colors[str(initialization)],
                    marker="*",
                )
                delta_xlabel += 0.5
        counter += 10
    plt.xlabel("Model number of the relaxed input convex neural network")
    plt.ylabel("Time steps [h]")
    plt.legend()
    plt.title("Time step at which convexity requirement has been violated")
    plt.xticks(x, x_ticks)
    plt.savefig(
        (
            Path(__file__).parents[4]
            / "plots/constraint_opt/optimums_as_the_function_of_initialization"
        ).joinpath(
            "Monotonic ICNN Time step at which convexity requirement has been violated {}.png".format(
                opt_step
            )
        )
    )
    plt.show()

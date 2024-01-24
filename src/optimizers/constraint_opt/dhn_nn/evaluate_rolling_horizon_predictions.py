"""
The role of this file is to calculate the root mean squared errors of four variables: supply inlet temperature, supply outlet temperature, mass flow, and delivered heat
when rolling out planning horizon, at the time-step that equal length of the planning horizon.
"""
from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *
from src.optimizers.constraint_opt.dhn_nn.tensor_constraint import ParNonNeg

if TimeParameters["PlanningHorizon"] != 12:
    warnings.warn(
        "For accessing the rmse in rolling horizon, planning horizon should be 12."
    )
    exit(1)


def load_nn_model(model_p, tensor_constraint, experiment_learn_type):
    """
    Load neural network model, depending on whether the model has overwritten class for constraining weights
    """
    if type(model_p) == tuple:
        model_p = str(model_p[0])
    model_nn_p: Path = (Path(__file__).parents[4] / "models/constraint_opt").joinpath(
        experiment_learn_type["folder"]
    )
    if tensor_constraint:
        model = load_model(
            model_nn_p.joinpath(model_p),
            compile=True,
            custom_objects={"ParNonNeg": ParNonNeg},
        )
    else:
        model = load_model(model_nn_p.joinpath(model_p), compile=True)
    return model


def get_state_model_input(state_model, q, h):
    """
    state_model: list
    q:list
    h: list
    """
    state_model = copy.deepcopy(state_model)
    state_model.extend(q)
    state_model.extend(h)
    state_model = np.array(state_model).reshape(1, -1)
    return state_model


def get_system_output_input(state_model, h):
    """
    state_model: list
    h: list
    """
    system_output = copy.deepcopy(state_model)
    system_output.extend(h)
    system_output = np.array(system_output).reshape(1, -1)
    return system_output


if __name__ == "__main__":
    layer_size = [50, 50]  # size of the DNN model
    N_model = 0  # order of the learned DNN model
    experiment_learn_type = experiments_learning["predictions"]
    experiment_opt_type = experiments_optimization["plnn_milp"]
    tensor_constraint: bool = experiment_opt_type[
        "tensor_constraint"
    ]  # does the neural network model used in optimization have overwritten weight constraint class?
    # the role of optimizer is to retrieve dat from the data matrix
    N_w = time_delay[str(PipePreset1["Length"])]
    N_w_q = time_delay_q[str(PipePreset1["Length"])]
    # produced heat columns
    h_column = ["h_{}".format(i) for i in range(1, N_w + 1 + 1)]
    # heat demand columns
    q_column = ["q_{}".format(i) for i in range(1, N_w_q + 1 + 1)]
    # input to the state model
    x_s: Path = (Path(__file__).parents[4] / "results/constraint_opt").joinpath(
        "x_s.csv"
    )
    x_s = pd.read_csv(x_s)
    start_iters = [11534]
    # output of the state model
    x_y: Path = (Path(__file__).parents[4] / "results/constraint_opt").joinpath(
        "x_y.csv"
    )
    x_y = pd.read_csv(x_y)
    # output of the system output model
    y_y = (Path(__file__).parents[4] / "results/constraint_opt").joinpath("y_y.csv")
    y_y = pd.read_csv(y_y)
    # load state prediction model
    model_s: str = (
        "{}_model_state".format(N_model)
        + experiment_learn_type["model_ext"]
        + "_s_time_delay_{}_".format(time_delay[str(PipePreset1["Length"])])
        + neurons_ext(layer_size)
        + "_"
        + experiment_opt_type["nn_type"]
        + ".h5",
    )
    model_s = load_nn_model(
        model_p=model_s,
        tensor_constraint=tensor_constraint,
        experiment_learn_type=experiment_learn_type,
    )
    # load system output model
    model_out: str = (
        "{}_model_output".format(N_model)
        + experiment_learn_type["model_ext"]
        + "_s_time_delay_{}_".format(time_delay[str(PipePreset1["Length"])])
        + neurons_ext(layer_size)
        + "_"
        + experiment_opt_type["nn_type"]
        + ".h5"
    )
    model_y = load_nn_model(
        model_p=model_out,
        tensor_constraint=tensor_constraint,
        experiment_learn_type=experiment_learn_type,
    )
    for start_iter in start_iters:
        rmse = {"tau_in": [], "tau_out": [], "m": [], "y": [], "total": []}
        for i in range(
            int(len(x_s) * 0.8), len(x_s) - TimeParameters["PlanningHorizon"] - 1
        ):
            print("Iteration {}".format(i))
            # initial values of internal variables
            tau_in_init: float = x_s["in_t_{}".format(N_w)][i]
            tau_out_init: float = x_s["out_t_{}".format(N_w)][i]
            m_init: float = x_s["m_{}".format(N_w)][i]
            state_model = [tau_in_init, tau_out_init, m_init]
            for j in range(TimeParameters["PlanningHorizon"]):
                h: list = x_s.loc[i + j, h_column].tolist()
                q: list = x_s.loc[i + j, q_column].tolist()
                state_model_input = get_state_model_input(state_model, q, h)
                state_model = model_s.predict(state_model_input, verbose=0)
                state_model = list(state_model[0])
                system_output_input = get_system_output_input(state_model, h)
                system_output = model_y.predict(system_output_input, verbose=0)
                system_output = system_output[0][0]
                if j == TimeParameters["PlanningHorizon"] - 1:
                    delivered_heat_rmse = round(
                        np.sqrt(pow((system_output - y_y["tilde_q11"][i + j]), 2)), 2
                    )
                    rmse["y"].append(delivered_heat_rmse)
                    supply_inlet_temperature_rmse = round(
                        np.sqrt(
                            pow(
                                (system_output - x_y["in_t_{}".format(N_w + 1)][i + j]),
                                2,
                            )
                        ),
                        2,
                    )
                    rmse["tau_in"].append(supply_inlet_temperature_rmse)
                    supply_outlet_temperature_rmse = round(
                        np.sqrt(
                            pow(
                                (
                                    system_output
                                    - x_y["out_t_{}".format(N_w + 1)][i + j]
                                ),
                                2,
                            )
                        ),
                        2,
                    )
                    rmse["tau_out"].append(supply_outlet_temperature_rmse)
                    mass_flow_rmse = round(
                        np.sqrt(
                            pow((system_output - x_y["m_{}".format(N_w + 1)][i + j]), 2)
                        ),
                        2,
                    )
                    rmse["m"].append(mass_flow_rmse)
                    rmse["total"].append(
                        delivered_heat_rmse
                        + supply_inlet_temperature_rmse
                        + supply_outlet_temperature_rmse
                        + mass_flow_rmse
                    )
        df = pd.DataFrame(rmse)
        df.to_pickle(
            (Path(__file__).parents[4] / "results/constraint_opt")
            .joinpath("evaluate_rolling_horizon_predictions")
            .joinpath(
                "train_rmse_from_i_{}_to_i_{}_".format(int(len(x_s) * 0.8), len(x_s))
                + experiment_opt_type["nn_type"]
                + ".pkl"
            )
        )

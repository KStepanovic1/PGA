import copy
import numpy as np
import pandas as pd
import time
from gurobipy import *
from pathlib import Path

from src.optimizers.constraint_opt.dhn_nn.functions import (
    run_simulator,
    get_optimal_produced_electricity,
    normalize_variable,
    neurons_ext,
    percent_tau_in,
    percent_tau_out,
    percent_m,
    percent_y,
)
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer
from src.optimizers.constraint_opt.dhn_nn.milp import MILP
from src.optimizers.constraint_opt.dhn_nn.param import opt_steps, time_delay, time_delay_q, round_dig
from src.optimizers.constraint_opt.dhn_nn.config_experiments import experiments_learning, experiments_optimization
from src.util import config
from src.util.config import (
    ProducerPreset1,
    ConsumerPreset1,
    TimeParameters,
    GridProperties,
    PhysicalProperties,
    PipePreset1,
)

if GridProperties["ConsumerNum"] == 1:
    from src.simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from src.simulator.cases.parallel_consumers import build_grid

MPC: int = 12  # number of times we repeat MPC
N_init: int = 12  # to mitigate the effect of initial actions, we start calculating the profit after N_init time-steps

class PLNNMILP(MILP):
    """
    Optimizing multi-step mathematical model based on piecewise linear neural network as mixed-integer linear program.
    x_s: normalized state space dataset (before shuffling)
    supply_pipe_plugs: plugs in the supply pipeline (taken from the row i-1)
    return pipe plugs: plugs in the return pipeline (taken from the row i-1)
    T: planning horizon
    N_w: length of previous actions time window concerning heat.
    N_w_q: length of previous actions time window concerning heat demands.
    """

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
            ad="plnn",
        )

    @staticmethod
    def nn_model(time_step, model, theta, layer_size, input, output, ad):
        """
        Following framework by Fischetti and Jo (2018), transform piecewise linear
        neural network into mixed-integer linear program.
        """
        a, z, s = [], [], []
        for i, nn in enumerate(layer_size[:-1]):
            z.append(
                model.addVars(
                    nn,
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    ub=+GRB.INFINITY,
                    name="z_time_%s_layer_%s" % (time_step, i),
                )
            )
            s.append(
                model.addVars(
                    nn,
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    ub=+GRB.INFINITY,
                    name="s_time_%s_layer_%s" % (time_step, i),
                )
            )
            a.append(
                model.addVars(
                    nn, vtype=GRB.BINARY, name="a_time_%s_layer_%s" % (time_step, i)
                )
            )
        for i in range(len(layer_size)):
            if i == 0:
                model.addConstrs(
                    (
                        z[i][j] - s[i][j]
                        == sum(
                            input[k] * theta["wz " + str(i)][k][j]
                            for k in range(len(input))
                        )
                        + theta["b " + str(i)][j]
                        for j in range(layer_size[i])
                    ),
                    name=ad + "_layer_" + str(i),
                )
                for j in range(layer_size[i]):
                    model.addGenConstrIndicator(
                        a[i][j],
                        True,
                        z[i][j],
                        GRB.LESS_EQUAL,
                        0.0,
                        name=ad + "indicator_constraint_z_" + str(i),
                    )
                    model.addGenConstrIndicator(
                        a[i][j],
                        False,
                        s[i][j],
                        GRB.LESS_EQUAL,
                        0.0,
                        name=ad + "indicator_constraint_s_" + str(i),
                    )
            elif i < len(layer_size) - 1:
                model.addConstrs(
                    (
                        z[i][j] - s[i][j]
                        == sum(
                            z[i - 1][k] * theta["wz " + str(i)][k][j]
                            for k in range(layer_size[i - 1])
                        )
                        + theta["b " + str(i)][j]
                        for j in range(layer_size[i])
                    ),
                    name=ad + "_layer_" + str(i),
                )
                for j in range(layer_size[i]):
                    model.addGenConstrIndicator(
                        a[i][j],
                        True,
                        z[i][j],
                        GRB.LESS_EQUAL,
                        0.0,
                        name=ad + "indicator_constraint_z_" + str(i),
                    )
                    model.addGenConstrIndicator(
                        a[i][j],
                        False,
                        s[i][j],
                        GRB.LESS_EQUAL,
                        0.0,
                        name=ad + "indicator_constraint_s_" + str(i),
                    )
            else:
                model.addConstrs(
                    (
                        output[j]
                        == sum(
                            z[i - 1][k] * theta["wz " + str(i)][k][j]
                            for k in range(layer_size[i - 1])
                        )
                        + theta["b " + str(i)][j]
                        for j in range(layer_size[i])
                    ),
                    name=ad + "_output_layer",
                )
        return model

    @staticmethod
    def extend_input(start, end, variable_base, variable_ext):
        """
        Extend the input, by previous actions time window or previous actions as decision variables.
        """
        for j in range(start, end):
            variable_base.append(variable_ext[j])
        return variable_base

    def shift_previous_actions(self, i, variable_base, h_init, h):
        """
        If length of MPC time horizon is less than length of previous actions time window then in extending the input
        both heat production initial values and heat production as decision variables participate. However, if the
        length of MPC planning horizon is greater than the length of previous actions time window, then when time-step (i)
        becomes >= N_w only heat produced as decision variable extend the input.
        """
        if i < self.N_w:
            input = PLNNMILP.extend_input(
                start=i, end=self.N_w, variable_base=variable_base, variable_ext=h_init
            )
            input = PLNNMILP.extend_input(
                start=0, end=i, variable_base=input, variable_ext=h
            )
        else:
            input = PLNNMILP.extend_input(
                start=i - self.N_w,
                end=i,
                variable_base=variable_base,
                variable_ext=h,
            )
        return input

    def optimize(
        self,
        opt_step,
        tau_in_init,
        tau_out_init,
        m_init,
        h_init,
        q,
        price,
        model_s,
        model_out,
        layer_size_s,
        layer_size_y,
    ):
        """
        Create the mathematical model incorporating system and output models of neural networks.
        """
        start = time.time()  # start measuring execution time
        theta_s = Optimizer.extract_weights(
            model=self.model_nn_p.joinpath(model_s), tensor_constraint=False
        )
        theta_y = Optimizer.extract_weights(
            model=self.model_nn_p.joinpath(model_out), tensor_constraint=False
        )
        model = Model("plnn_milp")
        model.reset()
        model.Params.FeasibilityTol = 0.01  # primal feasibility tolerance
        model.Params.IntFeasTol = 0.01  # integer feasibility tolerance
        model.Params.OptimalityTol = 0.01  # dual feasibility tolerance
        model.setParam("TimeLimit", 10 * 60)

        # heat demand buffer
        epsilon = 0

        # decision variables (not normalized)
        h = model.addVars(self.T, lb=self.H_min, ub=self.H_max, name="h")
        p = model.addVars(self.T, lb=self.P_min, ub=self.P_max, name="p")
        # alpha = model.addVars(self.T, self.num_extreme_points, lb=0, ub=1, name="alpha")
        # produced heat (normalized)
        h_nor = model.addVars(self.T, lb=0, ub=1, name="h_nor")
        # output variable (delivered heat)
        y = model.addVars(self.T, name="y")

        # state variables, s_t,...,s_{t+T} = [tau^{s,in}_t, tau^{s,out}_t, m_t,...,tau^{s,in}_{t+T}, tau^{s,out}_{t+T}, m_{t+T}]
        tau_in = model.addVars(
            self.T,
            lb=0,
            ub=self.T_in_max_nor,
            name="tau_in",
        )
        tau_out = model.addVars(
            self.T,
            lb=self.T_out_min_nor,
            ub=1,
            name="tau_out",
        )
        m = model.addVars(self.T, lb=0, ub=self.m_max_nor, name="m")

        # normalized produced heat
        model.addConstrs(
            (
                h_nor[i]
                == (h[i] - self.state_dict["Produced heat min"])
                / (
                    self.state_dict["Produced heat max"]
                    - self.state_dict["Produced heat min"]
                )
                for i in range(self.T)
            ),
            name="produced_heat_normalize",
        )

        # delivered heat should meet heat demand
        # heat demand at index 0 corresponds to time-step t-1
        model.addConstrs(
            (y[i] <= q[i + self.N_w_q] + epsilon for i in range(self.T)),
            name="delivered_heat_L",
        )

        model.addConstrs(
            (y[i] >= q[i + self.N_w_q] - epsilon for i in range(self.T)),
            name="delivered_heat_G",
        )

        # state model generated by the neural network
        for i in range(self.T):
            if i == 0:
                input = [tau_in_init, tau_out_init, m_init]
                input.extend([q[i], q[i + 1]])
                input.extend(h_init)
                input.append(h_nor[i])
                model = PLNNMILP.nn_model(
                    time_step=i,
                    model=model,
                    theta=theta_s,
                    layer_size=layer_size_s,
                    input=input,
                    output=[tau_in[i], tau_out[i], m[i]],
                    ad="state_time_step_%s" % (i),
                )
            else:
                input = [tau_in[i - 1], tau_out[i - 1], m[i - 1]]
                input.extend([q[i], q[i + 1]])
                input = self.shift_previous_actions(
                    i=i, variable_base=input, h_init=h_init, h=h_nor
                )
                input.append(h_nor[i])
                model = PLNNMILP.nn_model(
                    time_step=i,
                    model=model,
                    theta=theta_s,
                    layer_size=layer_size_s,
                    input=input,
                    output=[tau_in[i], tau_out[i], m[i]],
                    ad="state_time_step_%s" % (i),
                )

        # output model generated by the neural network
        for i in range(self.T):
            if i == 0:
                input = [tau_in[i], tau_out[i], m[i]]
                input.extend(h_init)
                input.append(h_nor[i])
                model = PLNNMILP.nn_model(
                    time_step=i,
                    model=model,
                    theta=theta_y,
                    layer_size=layer_size_y,
                    input=input,
                    output=[y[i]],
                    ad="output_time_step_%s" % (i),
                )
            else:
                input = [tau_in[i], tau_out[i], m[i]]
                input = self.shift_previous_actions(
                    i=i, variable_base=input, h_init=h_init, h=h_nor
                )
                input.append(h_nor[i])
                model = PLNNMILP.nn_model(
                    time_step=i,
                    model=model,
                    theta=theta_y,
                    layer_size=layer_size_y,
                    input=input,
                    output=[y[i]],
                    ad="output_time_step_%s" % (i),
                )
        # sum of alpha
        """
        for i in range(self.T):
            model.addConstr(
                (quicksum(alpha[i, j] for j in range(self.num_extreme_points)) == 1),
                "alpha_sum_(%s)" % (i),
            )

        # heat definition
        for i in range(self.T):
            model.addConstr(
                h[i]
                == quicksum(
                    alpha[i, j]
                    * ProducerPreset1["Generators"][0]["OperationRegion"][j][0]
                    for j in range(self.num_extreme_points)
                ),
                name="heat_definition_(%s)" % (i),
            )

        # electricity definition
        for i in range(self.T):
            model.addConstr(
                p[i]
                == quicksum(
                    alpha[i, j]
                    * ProducerPreset1["Generators"][0]["OperationRegion"][j][1]
                    for j in range(self.num_extreme_points)
                ),
                name="electricity_definition_(%s)" % (i),
            )
        """
        # objective function
        model.setObjective(
            quicksum(
                ProducerPreset1["Generators"][0]["FuelCost"][0] * h[i]
                # + ProducerPreset1["Generators"][0]["FuelCost"][1] * p[i]
                # - price[i] * p[i]
                for i in range(self.T)
            ),
            GRB.MINIMIZE,
        )
        model.write(str(self.model_p.joinpath("plnn_milp_{}.lp".format(opt_step))))
        model.optimize()
        status = model.status
        if status == GRB.INFEASIBLE:
            print("Model is infeasible")
            model.computeIIS()
            model.write(
                str(
                    self.model_p.joinpath(
                        "model_infeasible_constraints_{}.ilp".format(opt_step)
                    )
                )
            )
            model.feasRelaxS(1, True, False, True)
            model.optimize()
        h, p, tau_in = self.extract_variables(model=model, h=h, p=p, tau_in=tau_in)
        gap = 100 * model.MIPGap  # convert relative gap into percentage
        end = time.time()
        return h, p, tau_in, gap, end - start


if __name__ == "__main__":
    N_model: int = 3  # number of neural network models
    layer_sizes = [[1, 1]]
    opt_step: list = opt_steps  # initial steps
    experiment_learn_type = experiments_learning["predictions"]
    experiment_opt_type = experiments_optimization["plnn_milp"]
    result_p: Path = (Path(__file__).parents[4] / "results/constraint_opt").joinpath(
        experiment_opt_type["folder"]
    )
    for layer_size in layer_sizes:
        layer_size_s = copy.deepcopy(layer_size)
        layer_size_s.append(3)
        layer_size_y = copy.deepcopy(layer_size)
        layer_size_y.append(1)
        for k in range(N_model):
            # opt_step means that the initial values for tau_in, tau_out, and m will be taken for time-step N_w+opt_step+1
            # +1 is because querying the data was done as data[N_w:]. Instead of this it could have been done as data[(N_w-1):]
            # heat demand and price for the optimization will start from the time-step (N_w+opt_step+1)+1
            for opt_step_index, opt_step in enumerate(opt_steps["dnn_opt"]):
                results = {
                    "Q_optimized": [],
                    "T_supply_optimized": [],
                    "Produced_electricity": [],
                    "Profit": [],
                    "Heat_demand": [],
                    "Electricity_price": [],
                    "Supply_inlet_violation": [],
                    "Supply_outlet_violation": [],
                    "Mass_flow_violation": [],
                    "Delivered_heat_violation": [],
                    "Runtime": [],
                    "Optimality_gap": [],
                }
                milp_optimizer = PLNNMILP(
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
                # initial values of internal variables
                tau_in_init = round(
                    milp_optimizer.get_tau_in(opt_step=opt_step), round_dig
                )  # time-step t-1 (normalized)
                tau_out_init = round(
                    milp_optimizer.get_tau_out(opt_step=opt_step), round_dig
                )  # time-step t-1 (normalized)
                m_init = round(
                    milp_optimizer.get_m(opt_step=opt_step), round_dig
                )  # time-step t-1 (normalized)
                h_init = milp_optimizer.get_h(
                    opt_step=opt_step
                )  # time-step t-N_w,...t-1 (normalized)
                q = milp_optimizer.get_demand_(
                    opt_step=opt_step
                )  # time-step t-1,...,t+T (normalized)
                plugs = milp_optimizer.get_plugs(opt_step=opt_step)  # time-step t-1
                # initial values of external variables
                demand = milp_optimizer.get_demand(
                    opt_step=opt_step
                )  # time-steps t,...,t+T
                price = milp_optimizer.get_price(
                    opt_step=opt_step
                )  # time-steps t,...,t+T
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
                state_dict = milp_optimizer.get_state_dict()
                for i in range(N_init + MPC):
                    # optimize the model for the solution
                    h, p, tau_in, gap, exec_time = milp_optimizer.optimize(
                        opt_step=opt_step + i,
                        tau_in_init=tau_in_init,
                        tau_out_init=tau_out_init,
                        m_init=m_init,
                        h_init=h_init,
                        q=q,
                        price=price,
                        model_s="{}_model_state_s_time_delay_{}_".format(
                            k, time_delay[str(PipePreset1["Length"])]
                        )
                        + neurons_ext(layer_size)
                        + "_"
                        + experiment_opt_type["nn_type"]
                        + ".h5",
                        model_out="{}_model_output_s_time_delay_{}_".format(
                            k, time_delay[str(PipePreset1["Length"])]
                        )
                        + neurons_ext(layer_size)
                        + "_"
                        + experiment_opt_type["nn_type"]
                        + ".h5",
                        layer_size_s=layer_size_s,
                        layer_size_y=layer_size_y,
                    )
                    # run through the simulator for feasibility verification
                    (
                        supply_inlet_violation,
                        supply_outlet_violation,
                        mass_flow_violation,
                        delivered_heat_violation,
                        produced_heat_sim,
                        tau_out,
                        m,
                        plugs,
                    ) = run_simulator(
                        simulator=simulator,
                        object_ids=object_ids,
                        producer_id=producer_id,
                        consumer_ids=consumer_ids,
                        sup_edge_ids=sup_edge_ids,
                        produced_heat=h,
                        supply_inlet_temperature=tau_in,
                        produced_electricity=p,
                        demand=demand,
                        price=price,
                        plugs=plugs,
                    )
                    p = get_optimal_produced_electricity(
                        produced_heat=produced_heat_sim, electricity_price=price
                    )
                    profit = (
                        ProducerPreset1["Generators"][0]["FuelCost"][0]
                        * produced_heat_sim[0]
                        + ProducerPreset1["Generators"][0]["FuelCost"][1] * p[0]
                        - price[0] * p[0]
                    )
                    # join solutions to the list in order to save the run
                    results["Profit"].append(profit)
                    results["Q_optimized"].append(produced_heat_sim[0])
                    results["T_supply_optimized"].append(tau_in[0])
                    results["Produced_electricity"].append(p[0])
                    results["Heat_demand"].append(demand[0])
                    results["Electricity_price"].append(price[0])
                    results["Optimality_gap"].append(gap)
                    results["Runtime"].append(exec_time)
                    results["Supply_inlet_violation"].append(
                        percent_tau_in(supply_inlet_violation[0])
                    )
                    results["Supply_outlet_violation"].append(
                        percent_tau_out(supply_outlet_violation[0])
                    )
                    results["Mass_flow_violation"].append(
                        percent_m(mass_flow_violation[0])
                    )
                    results["Delivered_heat_violation"].append(
                        percent_y(delivered_heat_violation[0])
                    )
                    # get initial values for the next MPC iteration
                    tau_in_init = max(
                        normalize_variable(
                            var=tau_in[0],
                            min=state_dict["Supply in temp 1 min"],
                            max=state_dict["Supply in temp 1 max"],
                        ),
                        0.001,
                    )
                    tau_out_init = max(
                        normalize_variable(
                            var=tau_out[0],
                            min=state_dict["Supply out temp 1 min"],
                            max=state_dict["Supply out temp 1 max"],
                        ),
                        0.001,
                    )
                    m_init = max(
                        normalize_variable(
                            var=m[0],
                            min=state_dict["Supply mass flow 1 min"],
                            max=state_dict["Supply mass flow 1 max"],
                        ),
                        0.001,
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
                            0,
                        )
                    )
                    # h_init = milp_optimizer.norm_h(h[0])
                    # update heat demand and electricity price
                    demand = milp_optimizer.get_demand(
                        opt_step=opt_step + i + TimeParameters["ActionHorizon"]
                    )
                    price = milp_optimizer.get_price(
                        opt_step=opt_step + i + TimeParameters["ActionHorizon"]
                    )
                    results_df = pd.DataFrame(results)
                    # save results with initial actions
                    results_df.to_csv(
                        result_p.joinpath(
                            "{}_plnn_milp_init_".format(k)
                            + neurons_ext(layer_size)
                            + "_opt_step_{}".format(
                                opt_steps["math_opt"][opt_step_index]
                            )
                            + ".csv"
                        )
                    )
                # save results without initial actions
                results_df = results_df.iloc[N_init:]
                results_df.to_csv(
                    result_p.joinpath(
                        "{}_plnn_milp_".format(k)
                        + neurons_ext(layer_size)
                        + "_opt_step_{}".format(opt_steps["math_opt"][opt_step_index])
                        + ".csv"
                    )
                )

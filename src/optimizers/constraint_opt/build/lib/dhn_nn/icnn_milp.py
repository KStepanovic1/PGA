import time
import numpy as np
import pandas as pd

from gurobipy import *
from pathlib import Path

from .optimizer import Optimizer
from .milp import MILP
from util import config
from util.config import CHPPreset1, TimeParameters, GridProperties, PhysicalProperties

if GridProperties["ConsumerNum"] == 1:
    from ....simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from ....simulator.cases.parallel_consumers import build_grid


class ICNNMILP(MILP):
    """
    Optimizing multi-step mathematical model based on input convex neural network as mixed-integer linear program.
    """

    def __init__(
        self,
        result_p,
        parent_p,
        x_s,
        electricity_price,
        supply_pipe_plugs,
        return_pipe_plugs,
        T,
    ):
        super().__init__(
            result_p,
            parent_p,
            x_s,
            electricity_price,
            supply_pipe_plugs,
            return_pipe_plugs,
            "icnn",
        )
        self.T = T

    @staticmethod
    def nn_model(model, theta, layer_size, input, output, ad):
        """
        Following framework by Fischetti and Jo (2018), transform piecewise linear
        neural network into mixed-integer linear program.
        """
        a, z, s = [], [], []
        for i, nn in enumerate(layer_size[:-1]):
            z.append(
                model.addVars(
                    nn, vtype=GRB.CONTINUOUS, lb=0, ub=+GRB.INFINITY, name="z_" + str(i)
                )
            )
            s.append(
                model.addVars(
                    nn, vtype=GRB.CONTINUOUS, lb=0, ub=+GRB.INFINITY, name="s_" + str(i)
                )
            )
            a.append(model.addVars(nn, vtype=GRB.BINARY, name="a_" + str(i)))
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
                    name=ad + "I_layer_" + str(i),
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
                        + sum(
                            input[k] * theta["wx " + str(i)][k][j]
                            for k in range(len(input))
                        )
                        + theta["b " + str(i)][j]
                        for j in range(layer_size[i])
                    ),
                    name=ad + "I_layer_" + str(i),
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
                        + sum(
                            input[k] * theta["wx " + str(i)][k][j]
                            for k in range(len(input))
                        )
                        + theta["b " + str(i)][j]
                        for j in range(layer_size[i])
                    ),
                    name=ad + "output_layer",
                )
        return model

    def optimize(
        self,
        opt_step,
        tau_in_init,
        tau_out_init,
        m_init,
        h_init,
        model_s,
        model_out,
        layer_size_s,
        layer_size_y,
    ):
        """
        Create the mathematical model incorporating system and output models of neural network.
        """
        start = time.time()
        theta_s = Optimizer.extract_weights(
            model=self.parent_p.joinpath(model_s), state=True
        )
        theta_y = Optimizer.extract_weights(
            model=self.parent_p.joinpath(model_out), state=False
        )
        model = Model("multi_step_icnn_milp")
        # model.Params.FeasibilityTol = 0.01
        # model.Params.OptimalityTol = 0.01
        model.setParam("TimeLimit", 5 * 30)
        model.reset()

        # control variables, u_t,...,u_{t+T} = [h_t,...,h_{t+T}, p_t,...,p_{t+T}]
        h = model.addVars(self.T, lb=self.H_min, ub=self.H_max, name="h")
        p = model.addVars(self.T, lb=self.P_min, ub=self.P_max, name="p")
        alpha = model.addVars(self.T, self.num_extreme_points, lb=0, ub=1, name="alpha")

        # control variables (normalized)
        h_nor = model.addVars(self.T, lb=0, ub=1, name="h_nor")

        # state variables from the initial time-step, s_{t-1} = [tau^{s,in}_{t-1}, tau^{s,out}_{t-1}, m_{t-1}] (normalized)
        tau_in_ = model.addVar(name="tau_in_")
        tau_out_ = model.addVar(name="tau_out_")
        m_ = model.addVar(name="m_")
        h_ = model.addVar(name="h_")

        # state variables, s_t,...,s_{t+T} = [tau^{s,in}_t, tau^{s,out}_t, m_t,...,tau^{s,in}_{t+T}, tau^{s,out}_{t+T}, m_{t+T}]
        tau_in = model.addVars(self.T, ub=self.T_in_max, name="tau_in")
        tau_out = model.addVars(self.T, lb=self.T_out_min, name="tau_out")
        m = model.addVars(self.T, ub=self.m_max, name="m")

        # state variables (normalized)
        tau_in_nor = model.addVars(self.T, lb=0, ub=1, name="tau_in_nor")
        tau_out_nor = model.addVars(self.T, lb=0, ub=1, name="tau_out_nor")
        m_nor = model.addVars(self.T, lb=0, ub=1, name="m_nor")

        # output variable (delivered heat)
        y = model.addVars(self.T, name="y")

        # output variable (normalized)
        y_nor = model.addVars(self.T, lb=0, ub=1, name="y_nor")

        # heat demand (that should be met)
        q_ = model.addVars(self.T, name="q_nor")

        # initialize variables from time-step t-1
        model.addConstr(tau_in_ == tau_in_init, name="tau_in_init")
        model.addConstr(tau_out_ == tau_out_init, name="tau_out_init")
        model.addConstr(m_ == m_init, name="m_init")
        model.addConstr(h_ == h_init, name="h_init")

        # initialize heat demand
        model.addConstrs(
            (q_[i] == self.x_s["q_2"][opt_step + i] for i in range(self.T)),
            name="q_init",
        )

        # inverse transformed normalized variables
        model.addConstrs(
            (
                tau_in_nor[i]
                == (tau_in[i] - self.state_dict["Supply in temp 1 min"])
                / (
                    self.state_dict["Supply in temp 1 max"]
                    - self.state_dict["Supply in temp 1 min"]
                )
                for i in range(self.T)
            ),
            name="tau_in_normalize",
        )
        model.addConstrs(
            (
                tau_out_nor[i]
                == (tau_out[i] - self.state_dict["Supply out temp 1 min"])
                / (
                    self.state_dict["Supply out temp 1 max"]
                    - self.state_dict["Supply out temp 1 min"]
                )
                for i in range(self.T)
            ),
            name="tau_out_normalize",
        )
        model.addConstrs(
            (
                m_nor[i]
                == (m[i] - self.state_dict["Supply mass flow 1 min"])
                / (
                    self.state_dict["Supply mass flow 1 max"]
                    - self.state_dict["Supply mass flow 1 min"]
                )
                for i in range(self.T)
            ),
            name="m_normalize",
        )
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
        model.addConstrs(
            (
                y_nor[i]
                == (y[i] - self.output_dict["Delivered heat 1 min"])
                / (
                    self.output_dict["Delivered heat 1 max"]
                    - self.output_dict["Delivered heat 1 min"]
                )
                for i in range(self.T)
            ),
            name="delivered_heat_normalize",
        )

        # delivered heat should meet heat demand
        model.addConstrs(
            (y_nor[i] == q_[i] for i in range(self.T)),
            name="delivered_heat_equal_heat_demand",
        )

        # state model generated by the neural network
        for i in range(self.T):
            if i == 0:
                model = ICNNMILP.nn_model(
                    model=model,
                    theta=theta_s,
                    layer_size=layer_size_s,
                    input=[tau_in_, tau_out_, m_, q_[i], h_nor[i]],
                    output=[tau_in_nor[i], tau_out_nor[i], m_nor[i]],
                    ad="state_(%s)" % (i),
                )
            else:
                model = ICNNMILP.nn_model(
                    model=model,
                    theta=theta_s,
                    layer_size=layer_size_s,
                    input=[
                        tau_in_nor[i - 1],
                        tau_out_nor[i - 1],
                        m_nor[i - 1],
                        q_[i],
                        h_nor[i],
                    ],
                    output=[tau_in_nor[i], tau_out_nor[i], m_nor[i]],
                    ad="state_(%s)" % (i),
                )

        # output model generated by the neural network
        for i in range(self.T):
            if i == 0:
                model = ICNNMILP.nn_model(
                    model=model,
                    theta=theta_y,
                    layer_size=layer_size_y,
                    input=[tau_in_nor[i], tau_out_nor[i], m_nor[i], h_, h_nor[i]],
                    output=[y_nor[i]],
                    ad="output_(%s)" % (i),
                )
            else:
                model = ICNNMILP.nn_model(
                    model=model,
                    theta=theta_y,
                    layer_size=layer_size_y,
                    input=[
                        tau_in_nor[i],
                        tau_out_nor[i],
                        m_nor[i],
                        h_nor[i - 1],
                        h_nor[i],
                    ],
                    output=[y_nor[i]],
                    ad="output_(%s)" % (i),
                )

        # sum of alpha
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
                    alpha[i, j] * CHPPreset1["OperationRegion"][j][0]
                    for j in range(self.num_extreme_points)
                ),
                name="heat_definition_(%s)" % (i),
            )

        # electricity definition
        for i in range(self.T):
            model.addConstr(
                p[i]
                == quicksum(
                    alpha[i, j] * CHPPreset1["OperationRegion"][j][1]
                    for j in range(self.num_extreme_points)
                ),
                name="electricity_definition_(%s)" % (i),
            )

        # objective function
        model.setObjective(
            quicksum(
                CHPPreset1["FuelCost"][0] * h[i]
                + CHPPreset1["FuelCost"][1] * p[i]
                - self.electricity_price[i + opt_step - 1] * p[i]
                for i in range(self.T)
            ),
            GRB.MINIMIZE,
        )
        model.write("math_model_multi_step_dhn.lp")
        model.optimize()
        status = model.status
        if status == GRB.INFEASIBLE:
            print("Model is infeasible")
            model.computeIIS()
            model.write("model_infeasible_constraints.ilp")
            model.feasRelaxS(1, True, False, True)
            model.optimize()
        h_nor = model.getAttr("X", h_nor)
        h = model.getAttr("X", h)
        p = model.getAttr("X", p)
        obj = model.getObjective()
        obj = obj.getValue()
        h = np.array(h.values())
        p = np.array(p.values())
        profit = (
            CHPPreset1["FuelCost"][0] * h[0]
            + CHPPreset1["FuelCost"][1] * p[0]
            - self.electricity_price[opt_step - 1] * p[0]
        )
        gap = 100 * model.MIPGap  # convert relative gap into percentage
        end = time.time()
        return profit, h, p, gap, end - start


if __name__ == "__main__":
    T: int = 5  # planning horizon
    N: int = 24  # number of times we repeat MPC
    opt_step: list = [
        70,
        80,
        90,
        100,
        144,
        180,
        208,
        212,
        234,
        272,
        287,
    ]  # initial steps
    result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
    result_p_icnn_milp: Path = result_p.joinpath("icnn_milp")
    for opt_step in opt_step:
        results = {
            "Profit": [],
            "Produced heat": [],
            "Produced electricity": [],
            "Gaps": [],
            "Execution time": [],
            "Demand": [],
            "Price": [],
        }
        violations = {
            "Supply inlet temperature": [],
            "Supply outlet temperature": [],
            "Delivered heat": [],
            "Mass flow": [],
        }
        icnn_optimizer = ICNNMILP(
            result_p=Path(__file__).parents[4] / "results/constraint_opt",
            parent_p=Path(__file__).parent,
            x_s="x_s.csv",
            electricity_price="electricity_price.csv",
            supply_pipe_plugs="supply_pipe_plugs.pickle",
            return_pipe_plugs="return_pipe_plugs.pickle",
            T=T,
        )
        # initial values of internal variables
        tau_in_init = icnn_optimizer.get_tau_in(opt_step=opt_step)  # time-step t-1
        tau_out_init = icnn_optimizer.get_tau_out(opt_step=opt_step)  # time-step t-1
        m_init = icnn_optimizer.get_m(opt_step=opt_step)  # time-step t-1
        h_init = icnn_optimizer.get_h(opt_step=opt_step)  # time-step t-1
        plugs = icnn_optimizer.get_plugs(opt_step=opt_step)  # time-step t-1
        # initial values of external variables
        demand = icnn_optimizer.get_demand(
            opt_step=opt_step, T=T
        )  # time-steps t,...,t+T-1
        price = icnn_optimizer.get_price(
            opt_step=opt_step, T=T
        )  # time-steps t,...,t+T-1
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
        for i in range(N):
            simulator.reset([demand], [price], plugs)
            obj, h, p, gap, exec_time = icnn_optimizer.optimize(
                opt_step=opt_step + i,
                tau_in_init=tau_in_init,
                tau_out_init=tau_out_init,
                m_init=m_init,
                h_init=h_init,
                model_s="model_state_icnn.h5",
                model_out="model_output_icnn.h5",
                layer_size_s=[25, 25, 3],
                layer_size_y=[10, 10, 1],
            )
            simulator.run(heat=[h], electricity=[p])
            # join solutions to the list in order to save the run
            results["Profit"].append(obj)
            results["Produced heat"].append(h[0])
            results["Produced electricity"].append(p[0])
            results["Demand"].append(demand[0])
            results["Price"].append(price[0])
            results["Gaps"].append(gap)
            results["Execution time"].append(exec_time)
            grid_status = simulator.get_object_status(
                object_ids=object_ids,
                get_temp=True,
                get_ms=True,
                get_pressure=False,
                get_violation=True,
            )
            # get initial values for the next MPC iteration
            tau_in_init = icnn_optimizer.norm_tau_in(
                grid_status[sup_edge_ids]["Temp"][0][0]
            )
            tau_out_init = icnn_optimizer.norm_tau_out(
                grid_status[sup_edge_ids]["Temp"][1][0]
            )
            m_init = icnn_optimizer.norm_m(grid_status[sup_edge_ids]["Mass flow"][0][0])
            # h_init = icnn_optimizer.norm_h(
            #    PhysicalProperties["HeatCapacity"]
            #    * 10 ** (-6)
            #    * m_init
            #    * (tau_in_init - grid_status[ret_edge_ids]["Temp"][1][0])
            # )
            h_init = icnn_optimizer.norm_h(h[0])
            plugs = simulator.get_pipe_states(time_step=1)
            demand = icnn_optimizer.get_demand(opt_step=opt_step + i + 1, T=T)
            price = icnn_optimizer.get_price(opt_step=opt_step + i + 1, T=T)
            # verify feasibility
            # violation = simulator.get_condition_violation_one_step()
            violations["Supply inlet temperature"].append(
                grid_status[producer_id]["Violation"]["supply temp"][0]
            )
            violations["Supply outlet temperature"].append(
                grid_status[consumer_ids]["Violation"]["supply temp"][0]
            )
            violations["Delivered heat"].append(
                grid_status[consumer_ids]["Violation"]["heat delivered"][0]
            )
            violations["Mass flow"].append(
                grid_status[sup_edge_ids]["Violation"]["flow speed"][0]
            )
            results_df = pd.DataFrame(results)
            violations_df = pd.DataFrame(violations)
            results_df.to_csv(
                result_p_icnn_milp.joinpath("results {}".format(opt_step))
            )
            violations_df.to_csv(
                result_p_icnn_milp.joinpath("violations {}".format(opt_step))
            )

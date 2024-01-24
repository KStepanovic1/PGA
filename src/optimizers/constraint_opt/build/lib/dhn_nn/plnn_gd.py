import time

import tensorflow as tf
import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow.keras.models import load_model

from util import config
from util.config import PhysicalProperties, GridProperties, CHPPreset1
from .optimizer import Optimizer
from .tensor_constraint import ParNonNeg
from ..setting import opt_steps

if GridProperties["ConsumerNum"] == 1:
    from ....simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from ....simulator.cases.parallel_consumers import build_grid


class GradientDescent(Optimizer):
    """
    Optimizing multi-step mathematical model based on neural network via stochastic gradient descent.
    """

    def __init__(
        self,
        result_p,
        parent_p,
        x_s,
        electricity_price,
        supply_pipe_plugs,
        return_pipe_plugs,
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

    def get_starting_point(self, tau_in, tau_out, m, h, heat_demand, T):
        """
        Get starting point of the optimization, including future starting points.
        """
        starting_point = [
            tau_in,  # \tau^{s,in}_{t-1}
            tau_out,  # \tau^{s,out}_{t-1}
            m,  # m_{t-1}
        ]
        starting_point.extend(heat_demand)  # q_t,...,q_{t+T}
        starting_point.append(h)  # h_{t-1} (value has to be known)
        starting_point.extend([1] * T)
        return starting_point

    def get_heat_demand(self, opt_step, T):
        """
        Get heat demand that should be satisfied.
        """
        return list(self.x_s["q_2"][opt_step : opt_step + T])  # q_t,...,q_{t+T}

    def optimize_CHP_operation_region(self, h_nor, c, T):
        """
        Based on the calculated heat and convexity of the problem, get electricity production
        as the point on the edge of CHP operation region.
        """
        h_nor: np.array = np.array(h_nor)
        c: np.array = np.array(c)
        h, p = [], []
        for i in range(T):
            h.append(
                (
                    self.state_dict["Produced heat max"]
                    - self.state_dict["Produced heat min"]
                )
                * h_nor[i]
                + self.state_dict["Produced heat min"]
            )
            delta_c = CHPPreset1["FuelCost"][1] - c[i]
            p.append(Optimizer.calculate_p(h=h[i], delta_c=delta_c))
        return h, p

    def optimize(
        self,
        model_s_p,
        model_out_p,
        layer_size_s,
        layer_size_y,
        opt_step,
        T,
        learning_rate,
        iteration_number,
        starting_point,
        heat_demand,
        delta_grad,
    ):
        """
        Optimize the model via gradient descent.
        """
        start = time.time()  # start measuring execution time
        model_s = load_model(self.parent_p.joinpath(model_s_p))
        model_y = load_model(
            self.parent_p.joinpath(model_out_p),
            compile=False,
            custom_objects={"ParNonNeg": ParNonNeg},
        )
        x = np.array(starting_point).reshape(1, -1)
        x = tf.Variable(x, dtype="float32")
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        index_start = 3 + T + 2 - 1  # index of the first element of produced heat
        end_index = len(starting_point)  # index of the last element of produced heat
        for i in range(iteration_number):
            gradient = self.calculate_gradient(
                x=x, model_s=model_s, model_y=model_y, q=heat_demand, T=T
            )
            # zip calculated gradient and previous value of x
            zipped = zip([gradient], [x])
            # update value of input variable according to the calculated gradient
            opt.apply_gradients(zipped)
            heat_gradients = gradient[0][index_start:end_index]
            if tf.math.reduce_all(tf.less_equal(tf.abs(heat_gradients), delta_grad)):
                break
        h, p = self.optimize_CHP_operation_region(
            h_nor=x[0][index_start:end_index],
            c=self.electricity_price[opt_step - 1 : opt_step - 1 + T],
            T=T,
        )
        obj = sum(
            CHPPreset1["FuelCost"][0] * h[i]
            + CHPPreset1["FuelCost"][1] * p[i]
            - self.electricity_price[opt_step - 1 + i] * p[i]
            for i in range(T)
        )
        h = np.array(h)
        p = np.array(p)
        profit = (
            CHPPreset1["FuelCost"][0] * h[0]
            + CHPPreset1["FuelCost"][1] * p[0]
            - self.electricity_price[opt_step - 1] * p[0]
        )
        end = time.time()
        return profit, h, p, end - start

    def calculate_gradient(self, x, model_s, model_y, q, T):
        """
        Calculate gradient of x with respect to mean squared error loss function.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            # index at which first heat demand, q_t is.
            q_init_index: int = 3
            # index at which first heat, h_{t-1} is.
            h_init_index: int = 3 + T + 1 - 1
            # initial state model -- first part corresponds to [\tau^{s,in}_{t-1}, \tau^{s,out}_{t-1}, m_{t-1}, q_t]
            # second part corresponds to h_t
            state_model = model_s(
                tf.concat(
                    [
                        x[:, 0 : q_init_index + 1],
                        x[:, h_init_index + 1 : h_init_index + 2],
                    ],
                    axis=1,
                )
            )
            y = (
                model_y(
                    tf.concat(
                        [state_model, x[:, h_init_index : h_init_index + 2]], axis=1
                    )
                )
                - q[0]
            ) ** 2
            for i in range(1, T):
                # dynamic state transition, s_{t+1}<-s_t
                state_model = model_s(
                    tf.concat(
                        [
                            state_model,
                            x[:, q_init_index + i : q_init_index + i + 1],
                            x[:, h_init_index + i + 1 : h_init_index + i + 2],
                        ],
                        axis=1,
                    )
                )
                y = tf.math.add(
                    y,
                    (
                        (
                            model_y(
                                tf.concat(
                                    [
                                        state_model,
                                        x[
                                            :,
                                            h_init_index + i : h_init_index + i + 2,
                                        ],
                                    ],
                                    axis=1,
                                )
                            )
                            - q[i]
                        )
                        ** 2,
                    ),
                )
        grads = tape.gradient(y, x)
        grads_no_update = np.array([0] * (h_init_index + 1)).reshape(1, -1)
        grads_no_update = tf.constant(grads_no_update, dtype="float32")
        grads_update = np.array(
            grads[0][h_init_index + 1 : h_init_index + 1 + T]
        ).reshape(1, -1)
        grads = tf.concat([grads_no_update, grads_update], axis=1)
        return grads


if __name__ == "__main__":
    T: int = 5  # planning horizon
    N: int = 24  # number of times we repeat MPC
    opt_steps: list = [
        234,
        24,
        36,
        48,
        72,
        96,
        120,
        121,
        176,
        225,
        256,
        289,
        296,
        324,
        381,
        # 400,
        # 441,
        576,
        600,
    ]
    result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
    result_p_plnn_gd: Path = result_p.joinpath("plnn_gd")
    for opt_step in opt_steps:
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
        gd_optimizer = GradientDescent(
            result_p=Path(__file__).parents[4] / "results/constraint_opt",
            parent_p=Path(__file__).parent,
            x_s="x_s.csv",
            electricity_price="electricity_price.csv",
            supply_pipe_plugs="supply_pipe_plugs.pickle",
            return_pipe_plugs="return_pipe_plugs.pickle",
        )
        # initial values of internal variables
        tau_in_init = gd_optimizer.get_tau_in(opt_step=opt_step)  # time-step t-1
        tau_out_init = gd_optimizer.get_tau_out(opt_step=opt_step)  # time-step t-1
        m_init = gd_optimizer.get_m(opt_step=opt_step)  # time-step t-1
        h_init = gd_optimizer.get_h(opt_step=opt_step)  # time-step t-1
        plugs = gd_optimizer.get_plugs(opt_step=opt_step)  # time-step t-1
        # initial values of external variables
        demand_ = gd_optimizer.get_heat_demand(
            opt_step=opt_step, T=T
        )  # normalized heat demand between 0 and 1
        demand = gd_optimizer.get_demand(
            opt_step=opt_step, T=T
        )  # original heat demand in MWh
        price = gd_optimizer.get_price(opt_step=opt_step, T=T)  # time-steps t,...,t+T-1
        # initialize starting point
        starting_point = gd_optimizer.get_starting_point(
            tau_in=tau_in_init,
            tau_out=tau_out_init,
            m=m_init,
            h=h_init,
            heat_demand=demand_,
            T=T,
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
        for i in range(N):
            simulator.reset([demand], [price], plugs)
            obj, h, p, exec_time = gd_optimizer.optimize(
                model_s_p="model_state_plnn_delta_5C.h5",
                model_out_p="model_output_plnn_delta_5C.h5",
                layer_size_s=[25, 25, 3],
                layer_size_y=[10, 10, 1],
                opt_step=opt_step + i,
                T=T,
                learning_rate=0.01,
                iteration_number=10000,
                starting_point=starting_point,
                heat_demand=demand_,
                delta_grad=0.001,
            )
            results["Profit"].append(obj)
            results["Produced heat"].append(h[0])
            results["Produced electricity"].append(p[0])
            results["Demand"].append(demand[0])
            results["Price"].append(price[0])
            results["Gaps"].append(0)
            results["Execution time"].append(exec_time)
            simulator.run(heat=[h], electricity=[p])
            grid_status = simulator.get_object_status(
                object_ids=object_ids,
                get_temp=True,
                get_ms=True,
                get_pressure=False,
                get_violation=True,
            )
            # get initial values for the next MPC iteration
            tau_in_init = gd_optimizer.norm_tau_in(
                grid_status[sup_edge_ids]["Temp"][0][0]
            )
            tau_out_init = gd_optimizer.norm_tau_out(
                grid_status[sup_edge_ids]["Temp"][1][0]
            )
            m_init = gd_optimizer.norm_m(grid_status[sup_edge_ids]["Mass flow"][0][0])
            # h_init = gd_optimizer.norm_h(
            #    PhysicalProperties["HeatCapacity"]
            #    * 10 ** (-6)
            #    * m_init
            #    * (tau_in_init - grid_status[ret_edge_ids]["Temp"][1][0])
            # )
            h_init = gd_optimizer.norm_h(h[0])
            plugs = simulator.get_pipe_states(time_step=1)
            demand_ = gd_optimizer.get_heat_demand(opt_step=opt_step + i + 1, T=T)
            demand = gd_optimizer.get_demand(opt_step=opt_step + i + 1, T=T)
            price = gd_optimizer.get_price(opt_step=opt_step + i + 1, T=T)
            # initialize starting point
            starting_point = gd_optimizer.get_starting_point(
                tau_in=tau_in_init,
                tau_out=tau_out_init,
                m=m_init,
                h=h_init,
                heat_demand=demand_,
                T=T,
            )
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
                result_p_plnn_gd.joinpath("results_delta_5C {}".format(opt_step))
            )
            violations_df.to_csv(
                result_p_plnn_gd.joinpath("violations_delta_5C {}".format(opt_step))
            )
        # print(produced_heat, produced_electricity, violations)

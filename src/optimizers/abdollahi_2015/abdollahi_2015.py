import os
import copy
import numpy as np
import pandas as pd
import time

from pathlib import *
from pyscipopt import Model, exp, quicksum

from src.optimizers.constraint_opt.dhn_nn.functions import plugs_to_list
from src.util.config import (
    GridProperties,
    TimeParameters,
    PipePreset1,
    ProducerPreset1,
    ConsumerPreset1,
    PhysicalProperties,
)
from src.util import config
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer
from src.optimizers.constraint_opt.dhn_nn.param import opt_steps
from src.data_processing.dataloader import Dataset

if GridProperties["ConsumerNum"] == 1:
    from src.simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from src.simulator.cases.parallel_consumers import build_grid

constant_supply_inlet_temperature = 90


class Abdollahi:
    """
    Solves CHP economic dispatch, where heat demand is the sum of consumers' heat demand and linear loss approximation through pipeline.
    Simulation is possible only with heat.
    """

    def __init__(self):
        self.result_p: Path = (
            Path(__file__).parents[3]
            / "results/abdollahi_2015/MPC_episode_length_168_hours"
        )
        self.heuristics_p: Path = (
            Path(__file__).parents[3] / "results/constraint_opt" / "heuristics"
        )
        self.dataset_init = pd.read_csv(
            (Path(__file__).parents[3] / "results/constraint_opt").joinpath(
                "data_num_{}_heat_demand_real_world_for_L={}_time_interval_{}_max_Q_{}MW_deltaT_{}C.csv".format(
                    GridProperties["ConsumerNum"],
                    PipePreset1["Length"],
                    TimeParameters["TimeInterval"],
                    ProducerPreset1["Generators"][0]["MaxHeatProd"],
                    ProducerPreset1["Generators"][0]["MaxRampRateTemp"],
                )
            )
        )
        self.model_p: Path = Path(__file__).parents[3] / "models/abdollahi_2015"
        self.average_soil_temp: int = PipePreset1["EnvironmentTemperature"]
        self.num_chp_points: int = ProducerPreset1["Generators"][0][
            "CharacteristicPointsNum"
        ]
        self.extreme_points_heat: list = [
            chp_point[0]
            for chp_point in ProducerPreset1["Generators"][0]["OperationRegion"]
        ]
        self.extreme_points_power: list = [
            chp_point[1]
            for chp_point in ProducerPreset1["Generators"][0]["OperationRegion"]
        ]

    def get_heat_electricity(self, alpha):
        """
        Extract produced heat and electricity from alpha variables.
        """
        heat, electricity = [], []
        for t in range(TimeParameters["PlanningHorizon"]):
            produced_heat = sum(
                alpha[t][k] * self.extreme_points_heat[k]
                for k in range(self.num_chp_points)
            )
            produced_electricity = sum(
                alpha[t][k] * self.extreme_points_power[k]
                for k in range(self.num_chp_points)
            )
            heat.append(produced_heat)
            electricity.append(produced_electricity)
        return heat, electricity

    def get_alpha(self, m):
        """
        Get alpha actions.
        """
        alpha = []
        for v in m.getVars():
            if "alpha" in v.name:
                alpha.append(m.getVal(v))
        alpha = np.array(alpha).reshape(
            TimeParameters["PlanningHorizon"], self.num_chp_points
        )
        return alpha

    @staticmethod
    def obj_fun(heat, electricity, price):
        """
        Calculate the profit for the array of produced heat and electricity.
        """
        profit = []
        for t in range(TimeParameters["PlanningHorizon"]):
            profit.append(
                ProducerPreset1["Generators"][0]["FuelCost"][0] * heat[t]
                + ProducerPreset1["Generators"][0]["FuelCost"][1] * electricity[t]
                - float(price[t]) * electricity[t]
            )
        return profit

    def calculate_heat_loss(self, supply_temp):
        """
        Heat loss estimate following:
        Gu, Wei & Lu, Shuai & Wang, Jun & Xiang, Yin & Chenglong, Zhang & Wang, Zhihe. (2017). Modeling of the Heating Network for Multi-district Integrated Energy System
        and Its Operation Optimization. Proceedings of the CSEE. 3737. 10.13334/j.0258-8013.pcsee.160991. and
        Huang, B.; Zheng, C.; Sun, Q.; Hu, R. Optimal Economic Dispatch for Integrated Power
        and Heating Systems Considering Transmission Losses. Energies 2019, 12, 2502. https://doi.org/10.3390/en12132502
        """
        return (
            2
            * 3.14
            * (supply_temp - self.average_soil_temp)
            * PipePreset1["Length"]
            * 10 ** (-6)
        ) / PipePreset1["ThermalResistance"]

    def cumulative_heat_loss(
        self,
        simulator,
        object_ids,
        sup_edge_ids,
        alpha,
        plugs,
        demand,
        price,
        history_mass_flow,
    ):
        """
        Calculate the heat loss for each time-step.
        """
        plugs_, heat, electricity, profit, tau_in, grid_status = self.run_simulator(
            simulator,
            object_ids,
            sup_edge_ids,
            alpha,
            plugs,
            demand,
            price,
            history_mass_flow,
        )
        heat_loss = []
        for t in range(TimeParameters["PlanningHorizon"]):
            heat_loss.append(
                self.calculate_heat_loss(constant_supply_inlet_temperature)
            )
        return heat_loss

    def solve_chpd_model(self, heat_demand, electricity_price, heat_loss):
        """
        Optimize CHP model.
        """
        m = Model("CHPED")
        # variable connected to CHP
        (alpha) = {}
        for t in range(TimeParameters["PlanningHorizon"]):
            for k in range(self.num_chp_points):
                alpha[t, k] = m.addVar(
                    lb=0, ub=1, vtype="C", name="alpha(%s,%s)" % (t, k)
                )

        # defining constraints
        for t in range(TimeParameters["PlanningHorizon"]):
            m.addCons(
                quicksum(alpha[t, k] for k in range(self.num_chp_points)) == 1,
                "alpha_sum_constraint(%s)" % (t),
            )
        for t in range(TimeParameters["PlanningHorizon"]):
            m.addCons(
                quicksum(
                    alpha[t, k] * self.extreme_points_heat[k]
                    for k in range(self.num_chp_points)
                )
                == heat_demand[t] + heat_loss[t],
                "heat_demand(%s)" % (t),
            )
        objvar = m.addVar(name="objvar", vtype="C", lb=None, ub=None)
        m.setObjective(objvar, "minimize")
        m.addCons(
            objvar
            >= (
                quicksum(
                    ProducerPreset1["Generators"][0]["FuelCost"][0]
                    * quicksum(
                        alpha[t, k] * self.extreme_points_heat[k]
                        for k in range(self.num_chp_points)
                    )
                    + ProducerPreset1["Generators"][0]["FuelCost"][1]
                    * quicksum(
                        alpha[t, k] * self.extreme_points_power[k]
                        for k in range(self.num_chp_points)
                    )
                    for t in range(TimeParameters["PlanningHorizon"])
                )
                - quicksum(
                    float(electricity_price[t])
                    * (
                        quicksum(
                            alpha[t, k] * self.extreme_points_power[k]
                            for k in range(self.num_chp_points)
                        )
                    )
                    for t in range(TimeParameters["PlanningHorizon"])
                )
            ),
            name="objconst",
        )
        m.writeProblem(os.path.join(self.model_p, "abdollahi_2015.cip"))
        m.optimize()
        return m

    def run_simulator(
        self,
        simulator,
        object_ids,
        sup_edge_ids,
        alpha,
        plugs,
        demand,
        price,
        history_mass_flow,
    ):
        """
        Run the simulator, and get the produced heat, electricity, profit and supply inlet temperature for the length of planning horizon.
        """
        heat, electricity = self.get_heat_electricity(alpha)
        simulator.reset([demand], [price], plugs, history_mass_flow=history_mass_flow)
        simulator.run(heat=[np.asarray(heat)], electricity=[np.asarray(electricity)])
        # get status
        grid_status = simulator.get_object_status(
            object_ids=object_ids,
            start_step=0,
            end_step=TimeParameters["PlanningHorizon"],
            get_temp=True,
            get_ms=False,
            get_pressure=False,
            get_violation=True,
        )
        plugs = copy.copy(
            simulator.get_pipe_states(time_step=TimeParameters["ActionHorizon"])
        )
        profit = Abdollahi.obj_fun(heat=heat, electricity=electricity, price=price)
        tau_in = grid_status[sup_edge_ids]["Temp"][0]
        return plugs, heat, electricity, profit, tau_in, grid_status

    def abdollahi_heuristic(self, N, num_iter=2):
        """
        Using results of Abdollahi optimization as heuristic for warming up training of DNNs.
        """
        dataset = Dataset()
        results = {
            "Q_optimized": [],
            "Produced_electricity": [],
            "Profit": [],
            "T_supply_optimized": [],
            "Heat_demand": [],
            "Electricity_price": [],
        }
        # initialization should be done with the last plug from the Giraud simulation
        plugs = [
            [
                [344010.6, 90.0, 90.0, 1, -1],
                [354981.07, 84.82, 85.0, 0, -2],
                [374941.26, 89.62, 90, -1, -3],
            ],
            [
                [344010.6, 47.88, 47.88, 1, -3.0],
                [354981.07, 47.81, 47.9, 0, -3.0],
                [374941.26, 49.81, 50, -1, -3],
            ],
        ]
        while dataset.get_counter_test() < N:
            heat_demand, electricity_price = dataset.next()
            simulator = build_grid(
                consumer_demands=[heat_demand],
                electricity_prices=[electricity_price],
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
            heat_loss = [0] * TimeParameters["PlanningHorizon"]
            for i in range(num_iter):
                m = self.solve_chpd_model(heat_demand, electricity_price, heat_loss)
                alpha = self.get_alpha(m)
                heat_loss = self.cumulative_heat_loss(
                    # when calculating heat loss initial plugs remain the same through iterations
                    simulator=simulator,
                    object_ids=object_ids,
                    sup_edge_ids=sup_edge_ids,
                    alpha=alpha,
                    plugs=plugs,
                    demand=heat_demand,
                    price=electricity_price,
                )

            (
                plugs,
                produced_heat,
                produced_electricity,
                profit,
                tau_in,
                grid_status,
            ) = self.run_simulator(  # initial plugs are moved here for action horizon
                simulator=simulator,
                object_ids=object_ids,
                sup_edge_ids=sup_edge_ids,
                alpha=alpha,
                plugs=plugs,
                demand=heat_demand,
                price=electricity_price,
            )
            results["Q_optimized"].append(produced_heat[0])
            results["Produced_electricity"].append(produced_electricity[0])
            results["Heat_demand"].append(heat_demand[0])
            results["Electricity_price"].append(electricity_price[0])
            results["Profit"].append(profit[0])
            results["T_supply_optimized"].append(tau_in[0])
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.heuristics_p.joinpath("abdollahi_old.csv"))

    def abdollahi_opt(self, coeff_days, num_iter=1):
        """
        Optimization using linear model of CPH economic dispatch + linear approximation of heat loss.
        """
        dataset = Dataset()
        for coeff_day in coeff_days:
            result = {
                "Q_optimized": [],
                "T_supply_optimized": [],
                "Produced_electricity": [],
                "Profit": [],
                "Heat_demand": [],
                "Electricity_price": [],
                "Supply_inlet_violation": [],
                "Supply_inlet_violation_percent": [],
                "Supply_outlet_violation": [],
                "Supply_outlet_violation_percent": [],
                "Mass_flow_violation": [],
                "Mass_flow_violation_percent": [],
                "Delivered_heat_violation": [],
                "Delivered_heat_violation_percent": [],
                "Runtime": [],
                "Optimality_gap": [],
            }
            # initialization plugs
            plugs_supply: list = plugs_to_list(
                self.dataset_init["Supply plugs 1"][coeff_day]
            )
            plugs_return: list = plugs_to_list(
                self.dataset_init["Ret plugs 1"][coeff_day]
            )
            plugs = [plugs_supply, plugs_return]
            # heat demand, electricity price
            heat_demand, electricity_price = (
                dataset.heat_demand_data[
                    coeff_day : coeff_day + TimeParameters["PlanningHorizon"]
                ],
                dataset.electricity_price_data[
                    coeff_day : coeff_day + TimeParameters["PlanningHorizon"]
                ],
            )
            simulator = build_grid(
                consumer_demands=[heat_demand],
                electricity_prices=[electricity_price],
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
            heat_loss = [0] * TimeParameters["PlanningHorizon"]
            for j in range(1, TimeParameters["PlanningHorizon"] + 1):
                start = time.time()
                history_mass_flow = {
                    sup_edge_ids: plugs[0][0][0] / TimeParameters["TimeInterval"],
                    ret_edge_ids: plugs[0][0][0] / TimeParameters["TimeInterval"],
                }
                for i in range(num_iter):
                    m = self.solve_chpd_model(heat_demand, electricity_price, heat_loss)
                    alpha = self.get_alpha(m)
                    heat_loss = self.cumulative_heat_loss(
                        # when calculating heat loss initial plugs remain the same through iterations
                        simulator=simulator,
                        object_ids=object_ids,
                        sup_edge_ids=sup_edge_ids,
                        alpha=alpha,
                        plugs=plugs,
                        demand=heat_demand,
                        price=electricity_price,
                        history_mass_flow=history_mass_flow,
                    )
                (
                    plugs,
                    produced_heat,
                    produced_electricity,
                    profit,
                    tau_in,
                    grid_status,
                ) = self.run_simulator(  # initial plugs are moved here for action horizon
                    simulator=simulator,
                    object_ids=object_ids,
                    sup_edge_ids=sup_edge_ids,
                    alpha=alpha,
                    plugs=plugs,
                    demand=heat_demand,
                    price=electricity_price,
                    history_mass_flow=history_mass_flow,
                )
                end = time.time()
                gap = m.getGap() * 100
                result["Q_optimized"].append(produced_heat[0])
                result["T_supply_optimized"].append(tau_in[0])
                result["Produced_electricity"].append(produced_electricity[0])
                result["Profit"].append(profit[0])
                result["Heat_demand"].append(heat_demand[0])
                result["Electricity_price"].append(electricity_price[0])
                result["Supply_inlet_violation"].append(
                    abs(grid_status[producer_id]["Violation"]["supply temp"][0])
                )
                result["Supply_inlet_violation_percent"].append(
                    (
                        abs(grid_status[producer_id]["Violation"]["supply temp"][0])
                        / PhysicalProperties["MaxTemp"]
                    )
                    * 100
                )
                result["Supply_outlet_violation"].append(
                    abs(grid_status[consumer_ids]["Violation"]["supply temp"][0])
                )
                result["Supply_outlet_violation_percent"].append(
                    (
                        abs(grid_status[consumer_ids]["Violation"]["supply temp"][0])
                        / ConsumerPreset1["MinTempSupplyPrimary"]
                    )
                    * 100
                )
                result["Mass_flow_violation"].append(
                    abs(grid_status[sup_edge_ids]["Violation"]["flow speed"][0])
                )
                result["Mass_flow_violation_percent"].append(
                    (
                        abs(grid_status[sup_edge_ids]["Violation"]["flow speed"][0])
                        / ConsumerPreset1["MaxMassFlowPrimary"]
                    )
                    * 100
                )
                result["Delivered_heat_violation"].append(
                    abs(grid_status[consumer_ids]["Violation"]["heat delivered"][0])
                )
                result["Delivered_heat_violation_percent"].append(
                    (
                        abs(grid_status[consumer_ids]["Violation"]["heat delivered"][0])
                        / heat_demand[0]
                    )
                    * 100
                )
                result["Runtime"].append(
                    end - start
                )  # runtime of one optimization in seconds
                result["Optimality_gap"].append(gap)
                results_df = pd.DataFrame(result)
                results_df.to_csv(
                    self.result_p.joinpath("abdollahi_" + str(coeff_day) + ".csv")
                )
                heat_demand, electricity_price = (
                    dataset.heat_demand_data[
                        coeff_day
                        + j : coeff_day
                        + j
                        + TimeParameters["PlanningHorizon"]
                    ],
                    dataset.electricity_price_data[
                        coeff_day
                        + j : coeff_day
                        + j
                        + TimeParameters["PlanningHorizon"]
                    ],
                )


if __name__ == "__main__":
    abdollahi = Abdollahi()
    abdollahi.abdollahi_opt(coeff_days=opt_steps["math_opt"])

import copy
import os
import math
import time
import numpy as np
import pandas as pd

from pathlib import Path
import pyscipopt
from pyscipopt import Model, exp, quicksum
from typing import Tuple

from src.util import config
from src.util.config import (
    Dataset,
    PhysicalProperties,
    TimeParameters,
    ProducerPreset1,
    ConsumerPreset1,
    PipePreset1,
    GridProperties,
)
from src.optimizers.constraint_opt.dhn_nn.param import LiOpt
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer
from src.optimizers.constraint_opt.dhn_nn.functions import (
    surface_of_cross_sectional_area,
    get_optimal_produced_electricity,
    plugs_to_list,
    percent_tau_in,
    percent_tau_out,
    percent_m,
    percent_y,
)

from src.data_processing.dataloader import Dataset

if GridProperties["ConsumerNum"] == 1:
    from src.simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from src.simulator.cases.parallel_consumers import build_grid

if ProducerPreset1["ControlWithTemp"] == False:
    raise AssertionError("Control must be done with temperature!")


class Li2016:
    def __init__(self):
        self.initializer: int = LiOpt["LiInit"]

        # data paths
        self.path = {
            "model": Path(__file__).parents[3] / "models/li_2016",
            "result": Path(__file__).parents[3] / "results/li_2016",
        }
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
        # physical constraints
        # source: https://www.engineeringtoolbox.com/overall-heat-transfer-coefficient-d_434.html
        self.heat_transfer_coeff = np.array([0.000000735])  # [MW/(m*C)]
        # source: https://en.wikipedia.org/wiki/Darcy-Weisbach_equation, https://www.pipeflow.com/pipe-pressure-drop-calculations/pipe-friction-factors
        Re = 1000  # laminar flow and circular pipe
        self.friction_coefficient = 64 / Re

        # number of elements
        self.i_hs = 1  # number of heat stations
        self.i_chp = self.i_hs  # number of CHP units
        self.i_nd = 2  # number of nodes
        self.i_hes = 1  # number of heat exchange stations
        self.i_pipe = 1  # number of pipes

        # connections
        self.s_hs = {0: [0]}  # set of HS connected to node (key-node, value-HS)
        self.nd_hs = {
            0: 0
        }  # set of indices of nodes connected to HS (key-HS, value-node)
        self.s_hes = {1: [0]}  # set of HES connected to node (key-node, value-HES)
        self.nd_hes = {0: 1}  # index of node connected to HES (key-HES, value-node)
        self.nd_pf = {0: 0}  # index of starting node of pipeline (key-pipe, value-node)
        self.nd_pt = {0: 1}  # index of ending node of pipeline (key-pipe, value-node)
        self.s_pipe_supply_in = {
            0: [0]
        }  # set of indices of pipelines starting at the certain node of supply network (key-node, pipe-value)
        self.s_pipe_supply_out = {
            1: [0]
        }  # set of indices of pipelines ending at the certain node of supply network (key-node, pipe-value)
        self.s_pipe_return_in = {
            1: [0]
        }  # set of indices of pipelines starting at the certain node of return network (key-node, pipe-value)
        self.s_pipe_return_out = {
            0: [0]
        }  # set of indices of pipelines ending at the certain node of return network (key-node, pipe-value)

        # object's parameters
        self.length = np.array([PipePreset1["Length"]])  # [m]
        self.cross_sectional_area = np.array([PipePreset1["Diameter"]])  # [m]
        self.cross_sectional_area_surface = np.array(
            [surface_of_cross_sectional_area(PipePreset1["Diameter"])]
        )
        self.max_flow_speed = PipePreset1["MaxFlowSpeed"]  # [m/s]
        self.max_flow_rate = (
            self.max_flow_speed
            * PhysicalProperties["Density"]
            * self.cross_sectional_area_surface
        )  # [kg/s]
        self.min_flow_rate = PipePreset1["MinFlowSpeed"]
        self.max_node_t_supply_network = {
            0: PhysicalProperties["MaxTemp"],
            1: PhysicalProperties["MaxTemp"],
        }  # max supply T of the node connected to heat station 1
        self.min_node_t_supply_network = {
            0: PhysicalProperties["MinSupTemp"],
            1: PhysicalProperties["MinSupTemp"],
        }  # # min supply T of the node connected to heat station 1
        self.p_hes = {
            0: ConsumerPreset1["FixPressureLoad"]
        }  # minimum heat load pressure of a certain heat exchanger station
        self.max_node_t_return_network = {0: 80, 1: 80}  # max return T of the node
        self.min_node_t_return_network = {
            0: ConsumerPreset1["TempReturnSecondary"],
            1: ConsumerPreset1["TempReturnSecondary"],
        }  # min return T of the node

        self.water_pump_efficiency = 0.8
        self.max_power_consumption_water_pump = 20  # [MWh]
        self.min_power_consumption_water_pump = 0

        self.coefficient_of_pressure_loss = {
            0: (8 * self.friction_coefficient * self.length[0])
            / (
                PhysicalProperties["Density"]
                * pow(self.cross_sectional_area[0], 5)
                * pow(math.pi, 2)
            )
        }
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

        # ambient temperature
        self.tau_am = np.array(
            [PipePreset1["EnvironmentTemperature"]]
            * (TimeParameters["PlanningHorizon"] + self.initializer)
        )

        # temperature ramping constraint
        self.temp_ramp = 5

        # inequality buffer
        self.delta_temperature = 0
        self.delta_mass_flow = 0
        self.delta_pressure = 0
        self.delta_heat_demand = 0.5

    def update_complicating_variables(
        self, m, time_delay_I, time_delay_II, coeff_variable_S, coeff_variable_R
    ):
        """
        Update values of complicating variables in a new iteration.
        """
        ms_pipe_sol = m.data
        ms_pipe = np.empty(
            (self.i_pipe, TimeParameters["PlanningHorizon"] + self.initializer),
            dtype=float,
        )
        for key, value in ms_pipe_sol.items():
            ms_pipe[list(key)[0], list(key)[1]] = m.getVal(value)

        for b in range(self.i_pipe):
            pipe_vol = (
                PhysicalProperties["Density"]
                * self.cross_sectional_area_surface[b]
                * self.length[b]
            )
            for t in range(TimeParameters["PlanningHorizon"] + self.initializer):
                k, cumsum = 0, ms_pipe[b, t] * TimeParameters["TimeInterval"]
                while cumsum < pipe_vol and k < t:
                    k += 1
                    cumsum += ms_pipe[b, t - k] * TimeParameters["TimeInterval"]
                time_delay_II[b, t] = k
                cumsum -= ms_pipe[b, t] * TimeParameters["TimeInterval"]
                while cumsum < pipe_vol and k < t:
                    k += 1
                    cumsum += ms_pipe[b, t - k] * TimeParameters["TimeInterval"]
                time_delay_I[b, t] = k
                coeff_variable_R[b, t] = np.sum(
                    ms_pipe[b, (t - time_delay_II[b, t]) : (t + 1)]
                    * TimeParameters["TimeInterval"]
                )
                if time_delay_I[b, t] >= time_delay_II[b, t] + 1:
                    coeff_variable_S[b, t] = np.sum(
                        ms_pipe[b, (t - time_delay_I[b, t] + 1) : (t + 1)]
                        * TimeParameters["TimeInterval"]
                    )
                else:
                    coeff_variable_S[b, t] = coeff_variable_R[b, t]
        return [
            time_delay_I,
            time_delay_II,
            coeff_variable_R,
            coeff_variable_S,
        ]

    @staticmethod
    def mass_flow_pipe_in(m, i, t, S_PIPE_in):
        """
        Models mass flow going in the pipe (out of the node).
        """
        if i in S_PIPE_in.keys():
            return quicksum(
                m[S_PIPE_in.get(i)[j], t] for j in range(len(S_PIPE_in.get(i)))
            )
        else:
            return 0

    @staticmethod
    def mass_flow_pipe_out(m, i, t, S_PIPE_out):
        """
        Models mass flow going out of the pipe (in the node).
        """
        if i in S_PIPE_out.keys():
            return quicksum(
                m[S_PIPE_out.get(i)[j], t] for j in range(len(S_PIPE_out.get(i)))
            )
        else:
            return 0

    @staticmethod
    def temp_mixing_outlet(tau_out, m_pipe, i, t, S_PIPE_out):
        """
        Sum of products of temperature and mass flow at the outlet of pipeline.
        """
        if i in S_PIPE_out.keys():
            return quicksum(
                tau_out[S_PIPE_out.get(i)[j], t] * m_pipe[S_PIPE_out.get(i)[j], t]
                for j in range(len(S_PIPE_out.get(i)))
            )
        else:
            return 0

    def temp_mixing_inlet(self, m, tau_in, tau, S_PIPE_in, i, t, name):
        """
        Temperatures of mass flowing from a node are equal to that of a mixed mass at that node.
        """
        if i in S_PIPE_in.keys():
            for j in range(len(S_PIPE_in.get(i))):
                m.addCons(
                    tau_in[S_PIPE_in.get(i)[j], t] - tau[i, t]
                    <= self.delta_temperature,
                    name="L " + name % (i, t),
                )
                m.addCons(
                    tau_in[S_PIPE_in.get(i)[j], t] - tau[i, t]
                    >= -self.delta_temperature,
                    name="G " + name % (i, t),
                )

    def mass_flow_hs(self, m_hs, i, t):
        """
        Mass flow of the heat station (CHP).
        """
        if i in self.s_hs.keys():
            return m_hs[t]
        else:
            return 0

    def mass_flow_hes(self, m_hes, i, t):
        """
        Mass flow of the heat exchange station.
        """
        if i in self.s_hes.keys():
            return quicksum(
                m_hes[self.s_hes.get(i)[j], t] for j in range(len(self.s_hes.get(i)))
            )
        else:
            return 0

    def C_chp(self, alpha, t):
        """
        Operation cost of running CHP
        """
        return ProducerPreset1["Generators"][0]["FuelCost"][1] * quicksum(
            alpha[t, k] * self.extreme_points_power[k]
            for k in range(self.num_chp_points)
        ) + ProducerPreset1["Generators"][0]["FuelCost"][0] * quicksum(
            alpha[t, k] * self.extreme_points_heat[k]
            for k in range(self.num_chp_points)
        )

    def electricity_sell(self, alpha, t):
        """
        Profit by selling electricity to external grid.
        """
        return quicksum(
            alpha[t, k] * self.extreme_points_power[k]
            for k in range(self.num_chp_points)
        )

    def obj_fun(self, t, alpha, electricity_price):
        """
        Calculate value of objective function: operation cost-profit
        """
        return (
            ProducerPreset1["Generators"][0]["FuelCost"][0]
            * self.produced_heat(t, alpha)
            + ProducerPreset1["Generators"][0]["FuelCost"][1]
            * self.produced_electricity(t, alpha)
            - electricity_price[t] * self.produced_electricity(t, alpha)
        )

    def produced_heat(self, t, alpha):
        """
        Calculate produced heat in time-step t.
        """
        return sum(
            alpha[t][k] * self.extreme_points_heat[k]
            for k in range(self.num_chp_points)
        )

    def produced_electricity(self, t, alpha):
        """
        Calculate produced electricity in time-step t.
        """
        return sum(
            alpha[t][k] * self.extreme_points_power[k]
            for k in range(self.num_chp_points)
        )

    def solve_chpd_model(
        self,
        tau_in_init,
        complicating_variables,
        heat_demand,
        electricity_price,
        time_limit,
    ):
        """
        Model CHP dispatch with pipeline energy storage as mixed-integer nonlinear program, and solve it.
        """
        time_delay_I = complicating_variables[0]
        time_delay_II = complicating_variables[1]
        coeff_R = complicating_variables[2]
        coeff_S = complicating_variables[3]
        m = Model("CHPED")
        m.resetParams()
        m.setRealParam("limits/time", time_limit)
        m.setRealParam("numerics/feastol", 0.001)
        m.setBoolParam("lp/presolving", True)
        m.setIntParam("presolving/maxrounds", 0)
        # defining variables
        (
            alpha,
            m_hs,
            tau_ns,
            tau_nr,
            p_ns,
            p_nr,
            m_hes,
            ms_pipe,
            mr_pipe,
            d_pump,
            tau_PS_no_out,
            tau_PR_no_out,
            tau_PS_out,
            tau_PR_out,
            tau_PS_in,
            tau_PR_in,
        ) = ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})

        # variables connected to heat station
        for t in range(TimeParameters["PlanningHorizon"] + self.initializer):
            for k in range(self.num_chp_points):
                alpha[t, k] = m.addVar(
                    lb=0, ub=1, vtype="C", name="alpha(%s,%s)" % (t, k)
                )
            m_hs[t] = m.addVar(
                vtype="C", name="m_hs(%s)" % (t), lb=0, ub=self.max_flow_rate
            )
            d_pump[t] = m.addVar(
                lb=self.min_power_consumption_water_pump,
                ub=self.max_power_consumption_water_pump,
                vtype="C",
                name="d_pump(%s)" % (t),
            )

        # variables connected to network nodes
        for i in range(self.i_nd):
            for t in range(TimeParameters["PlanningHorizon"] + self.initializer):
                tau_ns[i, t] = m.addVar(
                    vtype="C",
                    name="tau_ns(%s,%s)" % (i, t),
                    lb=self.min_node_t_supply_network.get(i),
                    ub=self.max_node_t_supply_network.get(i),
                )
                tau_nr[i, t] = m.addVar(
                    vtype="C",
                    name="tau_nr(%s,%s)" % (i, t),
                    lb=self.min_node_t_return_network.get(i),
                    ub=self.max_node_t_return_network.get(i),
                )
                p_ns[i, t] = m.addVar(vtype="C", name="p_ns(%s,%s)" % (i, t))
                p_nr[i, t] = m.addVar(vtype="C", name="p_nr(%s, %s)" % (i, t))

        # variables connected to heat exchanger
        for i in range(self.i_hes):
            for t in range(TimeParameters["PlanningHorizon"] + self.initializer):
                m_hes[i, t] = m.addVar(
                    vtype="C", name="m_hes(%s,%s)" % (i, t), lb=0, ub=self.max_flow_rate
                )  # mass flow rate of heat exchanger station at period t

        # variables connected to pipes
        for i in range(self.i_pipe):
            for t in range(TimeParameters["PlanningHorizon"] + self.initializer):
                ms_pipe[i, t] = m.addVar(
                    ub=self.max_flow_rate,
                    lb=self.min_flow_rate,
                    vtype="C",
                    name="ms_pipe(%s,%s)" % (i, t),
                )  # mass flow rate of pipeline in the supply network
                mr_pipe[i, t] = m.addVar(
                    ub=self.max_flow_rate,
                    lb=self.min_flow_rate,
                    vtype="C",
                    name="mr_pipe(%s,%s)" % (i, t),
                )  # mass flow rate of pipeline in the return network
                tau_PS_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PS_out(%s,%s)" % (i, t),
                )  # mass flow temperature considering T drop at the outlet of pipeline in supply network
                tau_PR_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PR_out(%s,%s)" % (i, t),
                )  # mass flow temperature considering T drop at the outlet of pipeline in return network
                tau_PS_in[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PS_in(%s,%s)" % (i, t),
                )  # mass flow temperature at the inlet of pipeline in supply network
                tau_PR_in[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PR_in(%s,%s)" % (i, t),
                )  # mass flow temperature at the inlet of pipeline in return network
                tau_PS_no_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PS_no_out(%s,%s)" % (i, t),
                )  # outlet T without heat loss
                tau_PR_no_out[i, t] = m.addVar(
                    vtype="C",
                    name="tau_PR_no_out(%s,%s)" % (i, t),
                )  # outlet T without heat loss

        # defining constraints
        m.addCons(tau_ns[0, 0] == tau_in_init, name="initial_supply_inlet_temperature")
        # CHP unit
        for t in range(TimeParameters["PlanningHorizon"] + self.initializer):
            m.addCons(
                quicksum(alpha[t, k] for k in range(self.num_chp_points)) == 1,
                "alpha_sum_constraint(%s)" % (t),
            )
            """
            m.addCons(
                d_pump[t] * PhysicalProperties["EnergyUnitConversion"]
                == m_hs[t]
                * (p_ns[self.nd_hs.get(0), t] - p_nr[self.nd_hs.get(0), t])
                / (self.water_pump_efficiency * PhysicalProperties["Density"]),
            )
            """
            m.addCons(
                quicksum(
                    alpha[t, k] * self.extreme_points_heat[k]
                    for k in range(self.num_chp_points)
                )
                == (
                    PhysicalProperties["HeatCapacity"]
                    / PhysicalProperties["EnergyUnitConversion"]
                )
                * m_hs[t]
                * (tau_ns[self.nd_hs.get(0), t] - tau_nr[self.nd_hs.get(0), t]),
                name="heat_output_CHP_unit_constraint(%s)" % (t),
            )
            if t < TimeParameters["PlanningHorizon"] + self.initializer - 1:
                m.addCons(
                    tau_ns[self.nd_hs.get(0), t + 1] - tau_ns[self.nd_hs.get(0), t]
                    <= self.temp_ramp,
                    name="temperature_ramping_constraint_L(%s)" % (t),
                )
                m.addCons(
                    tau_ns[self.nd_hs.get(0), t + 1] - tau_ns[self.nd_hs.get(0), t]
                    >= -self.temp_ramp,
                    name="temperature_ramping_constraint_G(%s)" % (t),
                )

        # Heat exchanger station
        for i in range(self.i_hes):
            for t in range(TimeParameters["PlanningHorizon"] + self.initializer):
                m.addCons(
                    (
                        PhysicalProperties["HeatCapacity"]
                        / PhysicalProperties["EnergyUnitConversion"]
                    )
                    * m_hes[i, t]
                    * (tau_ns[self.nd_hes.get(i), t] - tau_nr[self.nd_hes.get(i), t])
                    - heat_demand[t]
                    >= -self.delta_heat_demand,
                    name="heat_exchangers_heat_loads_TDHS_G(%s, %s)" % (i, t),
                )
                m.addCons(
                    (
                        PhysicalProperties["HeatCapacity"]
                        / PhysicalProperties["EnergyUnitConversion"]
                    )
                    * m_hes[i, t]
                    * (tau_ns[self.nd_hes.get(i), t] - tau_nr[self.nd_hes.get(i), t])
                    - heat_demand[t]
                    <= self.delta_heat_demand,
                    name="heat_exchangers_heat_loads_TDHS_L(%s, %s)" % (i, t),
                )
                """
                m.addCons(
                    p_ns[self.nd_hes.get(i), t] - p_nr[self.nd_hes.get(i), t]
                    >= self.p_hes.get(i),
                    name="minimum_heat_load_pressure(%s, %s)" % (i, t),
                )
                """

        # District heating network
        for i in range(self.i_nd):
            for t in range(TimeParameters["PlanningHorizon"] + self.initializer):
                # continuity of mass flow
                m.addCons(
                    Li2016.mass_flow_pipe_in(ms_pipe, i, t, self.s_pipe_supply_in)
                    - Li2016.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    - self.mass_flow_hs(m_hs, i, t)
                    + self.mass_flow_hes(m_hes, i, t)
                    <= self.delta_mass_flow,
                    name="continuity_of_supply_network_mass_flow_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    Li2016.mass_flow_pipe_in(ms_pipe, i, t, self.s_pipe_supply_in)
                    - Li2016.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    - self.mass_flow_hs(m_hs, i, t)
                    + self.mass_flow_hes(m_hes, i, t)
                    >= -self.delta_mass_flow,
                    name="continuity_of_supply_network_mass_flow_G(%s, %s)" % (i, t),
                )
                m.addCons(
                    Li2016.mass_flow_pipe_in(mr_pipe, i, t, self.s_pipe_return_in)
                    - Li2016.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    - self.mass_flow_hes(m_hes, i, t)
                    + self.mass_flow_hs(m_hs, i, t)
                    <= self.delta_mass_flow,
                    name="continuity_of_return_network_mass_flow_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    Li2016.mass_flow_pipe_in(mr_pipe, i, t, self.s_pipe_return_in)
                    - Li2016.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    - self.mass_flow_hes(m_hes, i, t)
                    + self.mass_flow_hs(m_hs, i, t)
                    >= -self.delta_mass_flow,
                    name="continuity_of_return_network_mass_flow_G(%s, %s)" % (i, t),
                )

                # temperature mixing
                m.addCons(
                    Li2016.temp_mixing_outlet(
                        tau_PS_out, ms_pipe, i, t, self.s_pipe_supply_out
                    )
                    - tau_ns[i, t]
                    * Li2016.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    <= self.delta_mass_flow,
                    name="temperature_mixing_outlet_supply_network_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    Li2016.temp_mixing_outlet(
                        tau_PS_out, ms_pipe, i, t, self.s_pipe_supply_out
                    )
                    - tau_ns[i, t]
                    * Li2016.mass_flow_pipe_out(ms_pipe, i, t, self.s_pipe_supply_out)
                    >= -self.delta_mass_flow,
                    name="temperature_mixing_outlet_supply_network_G(%s, %s)" % (i, t),
                )
                m.addCons(
                    Li2016.temp_mixing_outlet(
                        tau_PR_out, mr_pipe, i, t, self.s_pipe_return_out
                    )
                    - tau_nr[i, t]
                    * Li2016.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    <= self.delta_mass_flow,
                    name="temperature_mixing_outlet_return_network_L(%s, %s)" % (i, t),
                )
                m.addCons(
                    Li2016.temp_mixing_outlet(
                        tau_PR_out, mr_pipe, i, t, self.s_pipe_return_out
                    )
                    - tau_nr[i, t]
                    * Li2016.mass_flow_pipe_out(mr_pipe, i, t, self.s_pipe_return_out)
                    >= -self.delta_mass_flow,
                    name="temperature_mixing_outlet_return_network_G(%s, %s)" % (i, t),
                )
                self.temp_mixing_inlet(
                    m,
                    tau_PS_in,
                    tau_ns,
                    self.s_pipe_supply_in,
                    i,
                    t,
                    "temperature_mixing_inlet_supply_network(%s, %s)",
                )
                self.temp_mixing_inlet(
                    m,
                    tau_PR_in,
                    tau_nr,
                    self.s_pipe_return_in,
                    i,
                    t,
                    "temperature_mixing_inlet_return_network(%s, %s)",
                )

        for b in range(self.i_pipe):
            for t in range(TimeParameters["PlanningHorizon"] + self.initializer):
                # pressure drop through pipeline
                """
                m.addCons(
                    p_ns[self.nd_pf.get(b), t]
                    - p_ns[self.nd_pt.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * ms_pipe[b, t]
                    * ms_pipe[b, t]
                    <= self.delta_pressure,
                    name="pressure_loss_supply_net_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    p_ns[self.nd_pf.get(b), t]
                    - p_ns[self.nd_pt.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * ms_pipe[b, t]
                    * ms_pipe[b, t]
                    >= -self.delta_pressure,
                    name="pressure_loss_supply_net_G (%s, %s)" % (b, t),
                )
                m.addCons(
                    p_nr[self.nd_pt.get(b), t]
                    - p_nr[self.nd_pf.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * mr_pipe[b, t]
                    * mr_pipe[b, t]
                    <= self.delta_pressure,
                    name="pressure_loss_return_net_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    p_nr[self.nd_pt.get(b), t]
                    - p_nr[self.nd_pf.get(b), t]
                    - self.coefficient_of_pressure_loss.get(b)
                    * mr_pipe[b, t]
                    * mr_pipe[b, t]
                    >= -self.delta_pressure,
                    name="pressure_loss_return_net_G (%s, %s)" % (b, t),
                )
                """
                # temperature propagation without heat loss through supply and return pipeline
                m.addCons(
                    (
                        tau_PS_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - PhysicalProperties["Density"]
                                * self.cross_sectional_area_surface[b]
                                * self.length[b]
                            )
                            * tau_PS_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k]
                                * TimeParameters["TimeInterval"]
                                * tau_PS_in[b, k]
                                for k in range(
                                    t
                                    - time_delay_I[b, t]
                                    + TimeParameters["ActionHorizon"],
                                    t - time_delay_II[b, t],
                                    TimeParameters["ActionHorizon"],
                                )
                            )
                            + (
                                ms_pipe[b, t] * TimeParameters["TimeInterval"]
                                + PhysicalProperties["Density"]
                                * self.cross_sectional_area_surface[b]
                                * self.length[b]
                                - coeff_S[b, t]
                            )
                            * tau_PS_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * TimeParameters["TimeInterval"])
                    )
                    <= self.delta_temperature,
                    name="outlet_supply_T_without_heat_loss_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    (
                        tau_PS_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - PhysicalProperties["Density"]
                                * self.cross_sectional_area_surface[b]
                                * self.length[b]
                            )
                            * tau_PS_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k]
                                * TimeParameters["TimeInterval"]
                                * tau_PS_in[b, k]
                                for k in range(
                                    t
                                    - time_delay_I[b, t]
                                    + TimeParameters["ActionHorizon"],
                                    t - time_delay_II[b, t],
                                    TimeParameters["ActionHorizon"],
                                )
                            )
                            + (
                                ms_pipe[b, t] * TimeParameters["TimeInterval"]
                                + PhysicalProperties["Density"]
                                * self.cross_sectional_area_surface[b]
                                * self.length[b]
                                - coeff_S[b, t]
                            )
                            * tau_PS_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * TimeParameters["TimeInterval"])
                    )
                    >= -self.delta_temperature,
                    name="outlet_supply_T_without_heat_loss_G (%s, %s)" % (b, t),
                )

                m.addCons(
                    (
                        tau_PR_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - PhysicalProperties["Density"]
                                * self.cross_sectional_area_surface[b]
                                * self.length[b]
                            )
                            * tau_PR_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k]
                                * TimeParameters["TimeInterval"]
                                * tau_PR_in[b, k]
                                for k in range(
                                    t
                                    - time_delay_I[b, t]
                                    + TimeParameters["ActionHorizon"],
                                    t - time_delay_II[b, t],
                                    TimeParameters["ActionHorizon"],
                                )
                            )
                            + (
                                ms_pipe[b, t] * TimeParameters["TimeInterval"]
                                + PhysicalProperties["Density"]
                                * self.cross_sectional_area_surface[b]
                                * self.length[b]
                                - coeff_S[b, t]
                            )
                            * tau_PR_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * TimeParameters["TimeInterval"])
                    )
                    <= self.delta_temperature,
                    name="outlet_return_T_without_heat_loss_L (%s, %s)" % (b, t),
                )
                m.addCons(
                    (
                        tau_PR_no_out[b, t]
                        - (
                            (
                                coeff_R[b, t]
                                - PhysicalProperties["Density"]
                                * self.cross_sectional_area_surface[b]
                                * self.length[b]
                            )
                            * tau_PR_in[b, t - time_delay_II[b, t]]
                            + quicksum(
                                ms_pipe[b, k]
                                * TimeParameters["TimeInterval"]
                                * tau_PR_in[b, k]
                                for k in range(
                                    t
                                    - time_delay_I[b, t]
                                    + TimeParameters["ActionHorizon"],
                                    t - time_delay_II[b, t],
                                    TimeParameters["ActionHorizon"],
                                )
                            )
                            + (
                                ms_pipe[b, t] * TimeParameters["TimeInterval"]
                                + PhysicalProperties["Density"]
                                * self.cross_sectional_area_surface[b]
                                * self.length[b]
                                - coeff_S[b, t]
                            )
                            * tau_PR_in[b, t - time_delay_I[b, t]]
                        )
                        / (ms_pipe[b, t] * TimeParameters["TimeInterval"])
                    )
                    >= -self.delta_temperature,
                    name="outlet_return_T_without_heat_loss_G (%s, %s)" % (b, t),
                )
                # updating temperature to account for the heat loss
                m.addCons(
                    tau_PS_out[b, t]
                    - self.tau_am[t]
                    - (tau_PS_no_out[b, t] - self.tau_am[t])
                    * exp(
                        -(self.heat_transfer_coeff[b] * self.length[b])
                        / (
                            ms_pipe[b, t]
                            * (
                                PhysicalProperties["HeatCapacity"]
                                / PhysicalProperties["EnergyUnitConversion"]
                            )
                        )
                    )
                    == 0,
                    name="outlet_supply_T_with_heat_loss (%s, %s)" % (b, t),
                )
                m.addCons(
                    tau_PR_out[b, t]
                    - self.tau_am[t]
                    - (tau_PR_no_out[b, t] - self.tau_am[t])
                    * exp(
                        -(self.heat_transfer_coeff[b] * self.length[b])
                        / (
                            mr_pipe[b, t]
                            * (
                                PhysicalProperties["HeatCapacity"]
                                / PhysicalProperties["EnergyUnitConversion"]
                            )
                        )
                    )
                    == 0,
                    name="outlet_return_T_with_heat_loss (%s, %s)" % (b, t),
                )
        objvar = m.addVar(name="objvar", vtype="C", lb=None, ub=None)
        m.setObjective(objvar, "minimize")
        m.addCons(
            objvar
            >= (
                quicksum(
                    self.C_chp(alpha, t)
                    for t in range(TimeParameters["PlanningHorizon"] + self.initializer)
                )
                - quicksum(
                    float(electricity_price[t]) * self.electricity_sell(alpha, t)
                    for t in range(TimeParameters["PlanningHorizon"] + self.initializer)
                )
            ),
            name="objconst",
        )
        m.data = ms_pipe
        # m.hideOutput(True)
        m.writeProblem(os.path.join(self.path["model"], "li_2016.cip"))
        m.optimize()
        return m

    def run_simulator(
        self,
        produced_heat,
        supply_inlet_temperature,
        produced_electricity,
        heat_demand,
        electricity_price,
        plugs,
    ):
        """
        Verify the feasibility of the solution through the simulator.
        """
        simulator = build_grid(
            consumer_demands=[heat_demand],
            electricity_prices=[electricity_price],
            config=config,
        )
        simulator.reset([heat_demand], [electricity_price], plugs)
        # get object's ids
        (
            object_ids,
            producer_id,
            consumer_ids,
            sup_edge_ids,
            ret_edge_ids,
        ) = Optimizer.get_object_ids(simulator)
        if ProducerPreset1["ControlWithTemp"]:
            simulator.run(
                temp=[np.asarray(supply_inlet_temperature)],
                electricity=[np.asarray(produced_electricity)],
            )
        else:
            simulator.run(
                heat=[np.asarray(produced_heat)],
                electricity=[np.asarray(produced_electricity)],
            )
        # get status
        grid_status = simulator.get_object_status(
            object_ids=object_ids,
            start_step=0,
            end_step=TimeParameters["PlanningHorizon"] + self.initializer,
            get_temp=True,
            get_ms=True,
            get_pressure=False,
            get_violation=True,
        )
        supply_inlet_temperature_sim = grid_status[producer_id]["Temp"][1]
        return_outlet_temperature_sim = grid_status[producer_id]["Temp"][0]
        mass_flow_sim = grid_status[producer_id]["Mass flow"][0]
        produced_heat_sim = [
            (
                PhysicalProperties["HeatCapacity"]
                / PhysicalProperties["EnergyUnitConversion"]
            )
            * mass_flow_sim[i]
            * (supply_inlet_temperature_sim[i] - return_outlet_temperature_sim[i])
            for i in range(TimeParameters["PlanningHorizon"] + self.initializer)
        ]
        plugs = copy.deepcopy(
            simulator.get_pipe_states(
                time_step=TimeParameters["ActionHorizon"] + self.initializer
            )
        )
        if ProducerPreset1["ControlWithTemp"]:
            supply_inlet_violation = []
            for tau_in in supply_inlet_temperature:
                if tau_in > PhysicalProperties["MaxTemp"]:
                    supply_inlet_violation.append(
                        tau_in - PhysicalProperties["MaxTemp"]
                    )
                else:
                    supply_inlet_violation.append(0)
        else:
            supply_inlet_violation = grid_status[producer_id]["Violation"][
                "supply temp"
            ]
        supply_outlet_violation = grid_status[consumer_ids]["Violation"]["supply temp"]
        mass_flow_violation = grid_status[sup_edge_ids]["Violation"]["flow speed"]
        delivered_heat_violation = grid_status[consumer_ids]["Violation"][
            "heat delivered"
        ]
        (
            supply_inlet_violation,
            supply_outlet_violation,
            mass_flow_violation,
            delivered_heat_violation,
        ) = Li2016.calculate_violation_percentage(
            supply_inlet_violation=supply_inlet_violation,
            supply_outlet_violation=supply_outlet_violation,
            mass_flow_violation=mass_flow_violation,
            delivered_heat_violation=delivered_heat_violation,
        )
        return (
            supply_inlet_violation,
            supply_outlet_violation,
            mass_flow_violation,
            delivered_heat_violation,
            produced_heat_sim,
            supply_inlet_temperature_sim,
            plugs,
        )

    def get_initial_parameters(self, hour):
        """
        Get supply inlet temperature and plugs for the initialization.
        """
        tau_in_init = self.dataset_init["Supply in temp 1"][hour - self.initializer - 1]
        plugs_supply: list = plugs_to_list(
            self.dataset_init["Supply plugs 1"][hour - self.initializer]
        )
        plugs_return: list = plugs_to_list(
            self.dataset_init["Ret plugs 1"][hour - self.initializer]
        )
        plugs = [plugs_supply, plugs_return]
        return tau_in_init, plugs

    def li_opt(
        self,
        tau_in_init,
        plugs,
        heat_demand,
        electricity_price,
        max_iter,
        time_limits,
    ):
        """
        Optimize the model using decomposition and multiple iterations as in Li 2016 paper.
        """
        tau_in_init = round(tau_in_init, 1)
        start = time.time()
        alpha = []
        result = {
            "Q_optimized": [],
            "T_supply_optimized": [],
            "Produced_electricity": [],
            "Profit": [],
            "Heat_demand": [],
            "Electricity_price": [],
            "Runtime": [],
            "Optimality_gap": [],
            "Supply_inlet_violation": [],
            "Supply_outlet_violation": [],
            "Delivered_heat_violation": [],
            "Mass_flow_violation": [],
        }
        # complicating variables
        time_delay_I = np.zeros(
            (self.i_pipe, TimeParameters["PlanningHorizon"] + self.initializer),
            dtype=int,
        )  # time delays associating changes in temperature
        time_delay_II = np.zeros(
            (self.i_pipe, TimeParameters["PlanningHorizon"] + self.initializer),
            dtype=int,
        )  # time delays associating changes in temperature
        coeff_variable_R = np.full(
            (self.i_pipe, TimeParameters["PlanningHorizon"] + self.initializer),
            PhysicalProperties["Density"]
            * surface_of_cross_sectional_area(PipePreset1["Diameter"])
            * PipePreset1["Length"],
        )  # coefficient variables R associated with the historic mass flow
        coeff_variable_S = np.full(
            (self.i_pipe, TimeParameters["PlanningHorizon"] + self.initializer),
            PhysicalProperties["Density"]
            * surface_of_cross_sectional_area(PipePreset1["Diameter"])
            * PipePreset1["Length"],
        )  # coefficient variables S associated with the historic mass flow
        complicating_variables = [  # first iteration, initialized
            time_delay_I,
            time_delay_II,
            coeff_variable_R,
            coeff_variable_S,
        ]
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
            m = self.solve_chpd_model(
                tau_in_init=tau_in_init,
                complicating_variables=complicating_variables,
                heat_demand=heat_demand,
                electricity_price=electricity_price,
                time_limit=time_limits[iter_count - 1],
            )
            complicating_variables = self.update_complicating_variables(
                m=m,
                time_delay_I=time_delay_I,
                time_delay_II=time_delay_II,
                coeff_variable_R=coeff_variable_R,
                coeff_variable_S=coeff_variable_S,
            )
        primal_bound = m.getPrimalbound()
        if primal_bound < 10 ** 6:
            # check the feasibility of the last obtained solution
            sol = m.getBestSol()
            is_feasible = m.checkSol(solution=sol, checkbounds=True, printreason=True)
            if is_feasible:
                print("Solution is feasible")
            else:
                print("Solution is infeasible")
            # check the status of model
            status = m.getStatus()
            print("Status of the problem is " + str(status))
            # extract variables from the model
            for v in m.getVars():
                if "alpha" in v.name:
                    alpha.append(m.getVal(v))
                if "tau_PS_in" in v.name:
                    result["T_supply_optimized"].append(m.getVal(v))
            alpha = np.array(alpha).reshape(
                TimeParameters["PlanningHorizon"] + self.initializer,
                self.num_chp_points,
            )
            gap = m.getGap() * 100
            end = time.time()
            runtime = end - start
            # save results
            for t in range(TimeParameters["PlanningHorizon"] + self.initializer):
                result["Q_optimized"].append(self.produced_heat(t, alpha))
                result["Produced_electricity"].append(
                    self.produced_electricity(t, alpha)
                )
                result["Heat_demand"].append(heat_demand[t])
                result["Electricity_price"].append(electricity_price[t])
                result["Runtime"].append(runtime)
                result["Optimality_gap"].append(gap)
            (
                supply_inlet_violation,
                supply_outlet_violation,
                mass_flow_violation,
                delivered_heat_violation,
                produced_heat,
                tau_in_init,
                plugs,
            ) = self.run_simulator(
                produced_heat=result["Q_optimized"],
                supply_inlet_temperature=result["T_supply_optimized"],
                produced_electricity=result["Produced_electricity"],
                heat_demand=heat_demand,
                electricity_price=electricity_price,
                plugs=plugs,
            )
            produced_electricity = get_optimal_produced_electricity(
                produced_heat=produced_heat, electricity_price=electricity_price
            )
            operation_cost = [
                ProducerPreset1["Generators"][0]["FuelCost"][0] * produced_heat[i]
                + ProducerPreset1["Generators"][0]["FuelCost"][1]
                * produced_electricity[i]
                - electricity_price[i] * produced_electricity[i]
                for i in range(TimeParameters["PlanningHorizon"] + self.initializer)
            ]
            result["Q_optimized"] = produced_heat
            result["Produced_electricity"] = produced_electricity
            result["Profit"] = operation_cost
            result["Supply_inlet_violation"] = supply_inlet_violation
            result["Supply_outlet_violation"] = supply_outlet_violation
            result["Mass_flow_violation"] = mass_flow_violation
            result["Delivered_heat_violation"] = delivered_heat_violation
            results_df = pd.DataFrame(result)
            # omitting the first row due to initialization
            result = results_df.iloc[self.initializer :]
        return primal_bound, result, tau_in_init, plugs

    def evaluate(self, file_name, coeff_day):
        """
        Evaluate feasibility of given file name.
        """
        li2016_f = pd.read_csv(self.path["result"].joinpath(file_name))
        plugs_supply: list = plugs_to_list(
            self.dataset_init["Supply plugs 1"][coeff_day]
        )
        plugs_return: list = plugs_to_list(self.dataset_init["Ret plugs 1"][coeff_day])
        plugs = [plugs_supply, plugs_return]
        (
            supply_inlet_violation,
            supply_outlet_violation,
            mass_flow_violation,
            delivered_heat_violation,
            produced_heat,
            supply_inlet_temperature_simulator,
            plugs,
        ) = self.run_simulator(
            produced_heat=li2016_f["Q_optimized"],
            supply_inlet_temperature=li2016_f["T_supply_optimized"],
            produced_electricity=li2016_f["Produced_electricity"],
            heat_demand=li2016_f["Heat_demand"],
            electricity_price=li2016_f["Electricity_price"],
            plugs=plugs,
        )
        print(
            supply_inlet_violation,
            supply_outlet_violation,
            mass_flow_violation,
            delivered_heat_violation,
        )

    @staticmethod
    def calculate_violation_percentage(
        supply_inlet_violation,
        supply_outlet_violation,
        mass_flow_violation,
        delivered_heat_violation,
    ) -> Tuple[list, list, list, list]:
        """
        Transform violations of supply inlet, supply outlet temperatures, mass flow and delivered heat
        into percentage of violations.
        """
        tau_in, tau_out, m, y = [], [], [], []
        for i in range(len(supply_inlet_violation)):
            tau_in.append(percent_tau_in(supply_inlet_violation[i]))
            tau_out.append(percent_tau_out(supply_outlet_violation[i]))
            m.append(percent_m(mass_flow_violation[i]))
            y.append(percent_y(delivered_heat_violation[i]))
        return tau_in, tau_out, m, y


if __name__ == "__main__":
    start_hour: int = 23
    li2016 = Li2016()
    dataset = Dataset()
    # tau_in_init, plugs = li2016.get_initial_parameters(hour=start_hour)
    tau_in_init, plugs = 75.35, [[[918093.24, 75.35, 75.35, 1, -1], [155839.69, 79.93, 80.1, 0, -2]], [[918093.24, 51.73, 51.73, 1, -2.0], [155839.69, 52.13, 52.23, 0, -2.9304865924700474]]]
    for t in range(1):
        LargePrimalBound: bool = True
        # heat demand and electricity price for certain coefficient
        heat_demand, electricity_price = (
            dataset.heat_demand_data[
                start_hour
                - LiOpt["LiInit"]
                + t : start_hour
                + TimeParameters["PlanningHorizon"]
                + t
            ],
            dataset.electricity_price_data[
                start_hour
                - LiOpt["LiInit"]
                + t : start_hour
                + TimeParameters["PlanningHorizon"]
                + t
            ],
        )
        while LargePrimalBound:
            print("New iteration starts ")
            primal_bound, results_df, tau_in_sim, plugs_sim = li2016.li_opt(
                tau_in_init=tau_in_init,
                plugs=plugs,
                heat_demand=heat_demand,
                electricity_price=electricity_price,
                max_iter=3,
                time_limits=[5, 10, 100],
            )
            if primal_bound < 10 ** 6:
                print("Primal bound found")
                LargePrimalBound = False
                tau_in_init = tau_in_sim[1]
                plugs = plugs_sim
                print(tau_in_init)
                print(plugs_sim)
                results_df.to_csv(
                    li2016.path["result"].joinpath(
                        "li__" + str(start_hour + t) + ".csv"
                    )
                )
                break

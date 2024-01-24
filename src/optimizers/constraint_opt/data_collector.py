import abc
import copy
import gzip
import numpy as np
import pandas as pd
import random

from typing import Tuple
from pathlib import Path
from operator import itemgetter

from util import config
from util.shared import *
from util.config import (
    GridProperties,
    Paths,
    PhysicalProperties,
    TimeParameters,
    PipePreset1,
    ProducerPreset1,
)
from ...data_processing.dataloader import Dataset
from .setting import plugs_to_list

if GridProperties["ConsumerNum"] == 1:
    from ...simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from ...simulator.cases.parallel_consumers import build_grid


def name_col() -> list:
    """
    Define names of pandas DataFrame columns.
    """
    columns = []
    for i in range(GridProperties["ConsumerNum"]):
        columns.append("Heat demand {}".format(i + 1))
    columns.extend(
        (
            "Electricity price",
            "Grid input",
            "Produced electricity",
            "Max supply in temp",  # violation of maximum inlet supply temperature
            "Temp ramp",  # temperature ramping violation
            "Q ramp",  # heat production ramping violation
            "E ramp",  # electricity production ramping violation
        )
    )
    for i in range(GridProperties["ConsumerNum"]):
        columns.extend(
            (
                "Delivered heat {}".format(i + 1),
                "Sec supply inlet temp {}".format(i + 1),
                "Min supply out temp {}".format(i + 1),
                "Underdelivered heat {}".format(i + 1),
            )
        )
    columns.append("Produced heat")
    for i in range(int(GridProperties["PipeNum"] / 2)):
        # supply and return edge with id 1 are main edges (if any)
        columns.extend(
            (
                "Supply in temp {}".format(i + 1),
                "Supply out temp {}".format(i + 1),
                "Supply mass flow {}".format(i + 1),
                "Supply in pressure {}".format(i + 1),
                "Supply out pressure {}".format(i + 1),
                "Supply heat loss {}".format(i + 1),
                "Supply plugs {}".format(i + 1),
                "Supply max mass flow {}".format(i + 1),
                "Supply time delay {}".format(i + 1),
                "Supply in temps-time delay {}".format(i + 1),
            )
        )
        columns.extend(
            (
                "Ret in temp {}".format(i + 1),
                "Ret out temp {}".format(i + 1),
                "Ret mass flow {}".format(i + 1),
                "Ret in pressure {}".format(i + 1),
                "Ret out pressure {}".format(i + 1),
                "Ret heat loss {}".format(i + 1),
                "Ret plugs {}".format(i + 1),
                "Ret max mass flow {}".format(i + 1),
                "Ret time delay {}".format(i + 1),
                "Ret in temps-time delay {}".format(i + 1),
            )
        )
    return columns


def get_input_spec(temp, delta) -> float:
    """
    Increase input variable by delta
    """
    temp = temp + delta
    return temp


def generate_input_spec(min_val, max_val, delta, heat_demands) -> list:
    """
    For data analysis and data generation we have an artificial input. This function randomly
    generates input temperature/heat values following constraints
    on upper, lower bound and maximal subsequent change. It starts from the minimum value on temperature/heat.
    """
    input = []
    temp = min_val
    for i in range(TimeParameters["PlanningHorizon"]):
        input.append(temp)
        while True:
            temp = get_input_spec(temp=temp, delta=delta)
            if temp >= min_val and temp <= max_val:
                break
    return [np.asarray(input)]  # (sadly) type of simulator input is list of numpy array


def get_input(temp, delta) -> float:
    """
    Generate new variable either adding or subtracting up to maximum allowed step.
    random.randint: returns random integers from the discrete uniform distribution.
    """
    delta_T = random.randint(1, delta)
    rand = random.randint(0, 1)
    if rand == 0:
        temp -= delta_T
    else:
        temp += delta_T
    return temp


def generate_input(min_val, max_val, delta, heat_demands) -> list:
    """
    For data analysis and data generation we have an artificial input. This function randomly
    generates input temperature/heat values following constraints
    on upper, lower bound and maximal subsequent change.
    """
    input = []
    temp = min_val + (max_val - min_val) / 2
    for i in range(TimeParameters["PlanningHorizon"]):
        input.append(temp)
        while True:
            temp = get_input(temp=temp, delta=delta)
            if temp >= min_val and temp <= max_val:
                break
    return [np.asarray(input)]  # (sadly) type of simulator input is list of numpy array


class DataCollector:
    def __init__(self, case, warm_up):
        self.columns = name_col()
        self.dataframe = pd.DataFrame(columns=self.columns)
        self.result_p = Path(__file__).parents[3] / "results"
        self.warm_up: bool = warm_up
        warm_up_ext: str = "_warm_up" if self.warm_up else ""
        self.dataframe_path = self.result_p.joinpath(
            "constraint_opt/data_rand"
            + warm_up_ext
            + "_num_{}_".format(GridProperties["ConsumerNum"])
            + case
            + "_for_L={}_time_interval_{}_max_Q_70MW_deltaT_{}C".format(
                PipePreset1["Length"],
                TimeParameters["TimeInterval"],
                ProducerPreset1["Generators"][0]["MaxRampRateTemp"],
            )
            + ".csv"
        )
        self.heuristics_path = self.result_p.joinpath("constraint_opt/heuristics")
        for i in range(int(GridProperties["PipeNum"] / 2)):
            self.dataframe["Supply plugs {}".format(i + 1)] = self.dataframe[
                "Supply plugs {}".format(i + 1)
            ].astype(object)
            self.dataframe["Ret plugs {}".format(i + 1)] = self.dataframe[
                "Ret plugs {}".format(i + 1)
            ].astype(object)
        self.dataset = Dataset()
        self.heat_demand, self.electricity_price = self.get_heat_electricity()

        # heat demand for plotting icnn behavior analysis
        # self.heat_demand = []
        # heat_demand_ = 5
        # for i in range(TimeParameters["PlanningHorizon"]):
        #    self.heat_demand.append(heat_demand_)
        #    heat_demand_ = heat_demand_ +0.4
        # self.heat_demand = np.array(self.heat_demand)

        self.counter = self.get_counter()
        self.num_counter = 0
        if GridProperties["ConsumerNum"] == 1:
            self.heat_demand = [self.heat_demand]
        else:
            self.heat_demand = [
                self.heat_demand * 0.28,
                self.heat_demand * 0.40,
                self.heat_demand * 0.32,
            ]
        self.grid = build_grid(
            self.heat_demand,
            [self.electricity_price],
            config,
        )
        self.obj_id_name = self.grid.get_id_name_all_obj()
        self.producer_id = [k for k, v in self.obj_id_name.items() if v == "CHP"][0]
        self.edge_ids = [k for k, v in self.obj_id_name.items() if v == "Edge"]
        if GridProperties["ConsumerNum"] == 1:
            self.consumer_ids = [
                k for k, v in self.obj_id_name.items() if v == "Consumer"
            ][0]
            self.sup_edge_ids = self.edge_ids[0]
            self.ret_edge_ids = self.edge_ids[1]
        else:
            self.consumer_ids = [
                k for k, v in self.obj_id_name.items() if v == "Consumer"
            ]
            self.sup_edge_ids = self.edge_ids[0 : len(self.edge_ids) : 2]
            self.ret_edge_ids = self.edge_ids[1 : len(self.edge_ids) : 2]
            self.main_sup_edge_id = self.sup_edge_ids[0]
            self.main_ret_edge_id = self.ret_edge_ids[0]
            self.side_supply_edge_ids = self.sup_edge_ids[1:]
            self.side_ret_edge_ids = self.ret_edge_ids[1:]
        if GridProperties["ConsumerNum"] == 1:
            self.object_ids = [self.producer_id] + [self.consumer_ids] + self.edge_ids
        else:
            self.object_ids = [self.producer_id] + self.consumer_ids + self.edge_ids
        self.heuristic = self.get_heuristic(
            columns=["T_supply_optimized", "Q_optimized"]
        )

    @abc.abstractmethod
    def get_heat_electricity(self):
        pass

    @abc.abstractmethod
    def get_counter(self):
        pass

    def get_heuristic(self, columns):
        """
        Get data generated by heuristics.
        """
        if self.warm_up:
            heuristic = pd.read_csv(
                self.heuristics_path.joinpath("abdollahi_gaussian_noise.csv"),
                usecols=columns,
            )
        else:
            heuristic = pd.read_csv(
                self.heuristics_path.joinpath("giraud_gaussian_noise.csv"),
                usecols=columns,
            )
        # heuristic = pd.concat(
        #    [giraud_heuristic, abdollahi_heuristic], ignore_index=True
        # )
        return heuristic

    def rnd(self, x) -> float:
        """
        Round variable on two decimals
        """
        return round(x, 2)

    def reformat_plug_entry_step(self, plugs):
        """
        Plug consists of four parts: mass[kg], current temp[C], entry temp[C], and entry step.
        The problem is that entry step of the plug is reset on -1 at the end of planning horizon.
        When collecting the data, we run the simulator in continuity. We also need to track
        the "real" entry step of the plug. This function reformats plug entry step, so that it
        corresponds to running simulator in continuity. The entry step starts from -1, and
        ends on number of iterations - 2.
        """
        for plug in plugs:
            plug[3] += self.num_counter * TimeParameters["PlanningHorizon"]
        return plugs

    def get_delay_inlet_temp(self, time, i, pipe) -> Tuple[list, list]:
        """
        Calculate at which time-steps water mass entered the pipe,
        with respect to the outlet temperature at the time-step t.
        Retrieve respective inlet temperatures.
        """
        if time == 0:
            return [0], [self.dataframe.loc[0, pipe.format(i + 1)][0][2]]
        time_delays = []
        inlet_temps = []
        # mass and the entry step of the last plug at the current time-step t
        mass = self.dataframe.loc[time, pipe.format(i + 1)][-1][0]
        entry_step = self.dataframe.loc[time, pipe.format(i + 1)][-1][3]
        plug = self.dataframe.loc[time - 1, pipe.format(i + 1)]
        plug = copy.deepcopy(plug)
        plug.reverse()
        j = 0
        while plug[j][3] < entry_step:
            time_delays.append(time - plug[j][3] - 1)
            inlet_temps.append(plug[j][2])
            j += 1
            if j == len(plug):
                break
        if j < len(plug):
            if mass < plug[j][0]:
                time_delays.append(time - plug[j][3] - 1)
                inlet_temps.append(plug[j][2])
        return time_delays, inlet_temps

    @staticmethod
    def get_index_of_maximum_mass_plug(plugs, i) -> int:
        """
        Gets supply inlet temperature index of the plug with the largest mass.
        """
        max_mass = max(enumerate(sub[i] for sub in plugs), key=itemgetter(1))
        index_max_mass = int(max_mass[0])
        return plugs[index_max_mass][3]

    def revise_heat(self):
        """
        Create new set of the produced heat, by increasing the produced heat corresponding to the largest mass
        water chunk at the underdelivered heat time step by one MWh.
        """
        heat = []
        for i in range(len(self.dataframe)):
            if self.dataframe["Underdelivered heat 1"][i] < -0.05:
                plugs = self.dataframe["Supply plugs 1"][i]
                index = DataCollector.get_index_of_maximum_mass_plug(plugs=plugs, i=0)
                temp_ = heat[index] + 1
                if temp_ > ProducerPreset1["Generators"][0]["MaxHeatProd"]:
                    temp_ = ProducerPreset1["Generators"][0]["MaxHeatProd"]
                heat[index] = temp_
            heat.append(self.dataframe["Produced heat"][i])
        return [np.asarray(heat)]

    def revise_temp(self):
        """
        Create new set of inlet temperatures, by increasing supply inlet temperature
        that caused underdelivered heat by some threshold.
        """
        temp = []
        for i in range(len(self.dataframe)):
            if self.dataframe["Underdelivered heat 1"][i] < -0.05:
                plugs = self.dataframe["Supply plugs 1"][i]
                index = DataCollector.get_index_of_maximum_mass_plug(plugs=plugs, i=0)
                temp_ = temp[index - 1] + 1
                if temp_ > PhysicalProperties["MaxTemp"]:
                    temp_ = PhysicalProperties["MaxTemp"]
                temp[index] = temp_
            temp.append(self.dataframe["Supply in temp 1"][i])
        return [np.asarray(temp)]

    def set_initial_control(self, heuristics_control):
        """
        Returns initial control variables depending on whether control is carried with the heuristic or with the random data,
        and depending on whether temperature or heat are used as control variables.
        """
        if heuristics_control:
            if ProducerPreset1["ControlWithTemp"]:
                temp = [np.array(self.heuristic["T_supply_optimized"].tolist())]
            else:
                temp = [np.array(self.heuristic["Q_optimized"].tolist())]
        else:
            if ProducerPreset1["ControlWithTemp"]:
                temp = generate_input(
                    min_val=PhysicalProperties["MinSupTemp"],
                    max_val=PhysicalProperties["MaxTemp"],
                    delta=ProducerPreset1["Generators"][0]["MaxRampRateTemp"],
                    heat_demands=self.heat_demand[0],
                )
            else:
                temp = generate_input(
                    # min_val=ProducerPreset1["Generators"][0]["MinHeatProd"],
                    min_val=10,
                    max_val=ProducerPreset1["Generators"][0]["MaxHeatProd"],
                    delta=5,
                    heat_demands=self.heat_demand[0],
                )
        return temp

    def run(self, num_iter, heuristics_control):
        """
        Run the simulator, collect the data, and save the data in pandas DataFrame
        """
        num_iter = num_iter
        repeat_iter = 3  # if the heat is underdelivered, we increase the temperature of the largest mass plug by 1C and re-run simulation. This can be done up to three times.
        start_iter = 0
        temp = self.set_initial_control(heuristics_control=heuristics_control)
        while start_iter < repeat_iter:
            initial_plugs = None
            while self.counter < num_iter:
                self.grid.reset(
                    demands=self.heat_demand,
                    e_price=[self.electricity_price],
                    pipe_states=initial_plugs,
                )
                electricity = [np.ones(TimeParameters["PlanningHorizon"]) * 20]
                if ProducerPreset1["ControlWithTemp"]:
                    self.grid.run(
                        temp=temp,
                        electricity=electricity,
                    )
                else:
                    self.grid.run(
                        heat=temp,
                        electricity=electricity,
                    )
                initial_plugs = self.grid.get_pipe_states(
                    time_step=TimeParameters["PlanningHorizon"]
                )
                grid_status = self.grid.get_object_status(
                    object_ids=self.object_ids,
                    get_temp=True,
                    get_ms=True,
                    get_pressure=True,
                    get_violation=True,
                )

                """
                Dictionary of producer's violations has following keys: Supply temp-if the producer's outlet temp (supply inlet temp)
                is greater than upper bound, Q ramp-if the ramping rate of heat produced is greater than specified constant, 
                E ramp-electricity ramping, temp ramp-temperature ramping.
                """

                producer_violation = grid_status[self.producer_id]["Violation"]

                """
                Store heat demand, electricity price, grid input, produced electricity, and producer violations
                """
                for t in range(TimeParameters["PlanningHorizon"]):
                    time = self.counter - TimeParameters["PlanningHorizon"] + t
                    for i in range(GridProperties["ConsumerNum"]):
                        self.dataframe.loc[
                            time,
                            "Heat demand {}".format(i + 1),
                        ] = self.rnd(self.heat_demand[i][t])
                    self.dataframe.loc[
                        time, "Electricity price"
                    ] = self.electricity_price[t]
                    self.dataframe.loc[time, "Grid input"] = temp[0][t]
                    self.dataframe.loc[time, "Produced electricity"] = electricity[0][t]
                    self.dataframe.loc[time, "Max supply in temp"] = self.rnd(
                        max(
                            0,
                            grid_status[self.producer_id]["Temp"][1][t]
                            - PhysicalProperties["MaxTemp"],
                        ),
                    )
                    self.dataframe.loc[time, "Temp ramp"] = producer_violation[
                        "temp ramp(degree)"
                    ][t]
                    self.dataframe.loc[time, "Q ramp"] = producer_violation[
                        "Q ramp(%)"
                    ][t]
                    self.dataframe.loc[time, "E ramp"] = producer_violation[
                        "E ramp(%)"
                    ][t]

                actual_delivered_heat = self.grid.get_actual_delivered_heat()

                s_supply_in_temp_tot = self.grid.get_sec_supply_in_temp()

                for i in range(GridProperties["ConsumerNum"]):

                    c_id = (
                        self.consumer_ids
                        if type(self.consumer_ids) is int
                        else self.consumer_ids[i]
                    )

                    """
                    Dictionary of consumer's violations has following keys: Supply_temp-if the inlet temperature to the consumer is
                    smaller than lower bound. Lower bound can be: artificially specified constant e.g 70[C]) or
                    minimal temperature at the inlet under the maximal mass flow. Heat delivered-underdelivered heat.
                    """

                    consumer_violation = grid_status[c_id]["Violation"]

                    consumer_delivered_heat = actual_delivered_heat[c_id]

                    s_supply_in_temp = s_supply_in_temp_tot[c_id]

                    for t in range(TimeParameters["PlanningHorizon"]):
                        time = self.counter - TimeParameters["PlanningHorizon"] + t

                        self.dataframe.loc[
                            time, "Delivered heat {}".format(i + 1)
                        ] = self.rnd(consumer_delivered_heat[t])
                        self.dataframe.loc[
                            time, "Sec supply inlet temp {}".format(i + 1)
                        ] = self.rnd(s_supply_in_temp[t])
                        self.dataframe.loc[
                            time, "Min supply out temp {}".format(i + 1)
                        ] = self.rnd(consumer_violation["supply temp"][t])
                        self.dataframe.loc[
                            time, "Underdelivered heat {}".format(i + 1)
                        ] = self.rnd(consumer_violation["heat delivered"][t])

                heat_dict = self.grid.get_edge_heat_and_loss(
                    edge_ids=self.edge_ids, level=2, level_time=1
                )

                for i in range(int(GridProperties["PipeNum"] / 2)):
                    sup_edge_id = (
                        self.sup_edge_ids
                        if type(self.sup_edge_ids) is int
                        else self.sup_edge_ids[i]
                    )

                    min_id = (
                        self.sup_edge_ids
                        if type(self.sup_edge_ids) is int
                        else min(self.sup_edge_ids)
                    )

                    s_inlet_temp = grid_status[sup_edge_id]["Temp"][0]

                    s_outlet_temp = grid_status[sup_edge_id]["Temp"][1]

                    s_mass_flow = grid_status[sup_edge_id]["Mass flow"][0]

                    s_inlet_pressure = grid_status[sup_edge_id]["Pressure"][0]

                    s_outlet_pressure = grid_status[sup_edge_id]["Pressure"][1]

                    s_edge_violation = grid_status[sup_edge_id]["Violation"]

                    s_heat_loss = heat_dict[sup_edge_id][1]

                    ret_edge_id = (
                        self.ret_edge_ids
                        if type(self.ret_edge_ids) is int
                        else self.ret_edge_ids[i]
                    )

                    r_inlet_temp = grid_status[ret_edge_id]["Temp"][0]

                    r_outlet_temp = grid_status[ret_edge_id]["Temp"][1]

                    r_mass_flow = grid_status[ret_edge_id]["Mass flow"][0]

                    r_inlet_pressure = grid_status[ret_edge_id]["Pressure"][0]

                    r_outlet_pressure = grid_status[ret_edge_id]["Pressure"][1]

                    r_edge_violation = grid_status[ret_edge_id]["Violation"]

                    r_heat_loss = heat_dict[ret_edge_id][1]

                    for t in range(TimeParameters["PlanningHorizon"]):
                        time = self.counter - TimeParameters["PlanningHorizon"] + t
                        # Produced heat is derived as: c*m*(\tau^{s, in}-\tau^{r, out})
                        # as parameter c unit is [J\(kg*C)], we multiply by 10^(-6) to get [MJ\(kg*C)]
                        if i == 0:
                            self.dataframe.loc[time, "Produced heat"] = self.rnd(
                                PhysicalProperties["HeatCapacity"]
                                * 10 ** (-6)
                                * s_mass_flow[t]
                                * (s_inlet_temp[t] - r_outlet_temp[t]),
                            )
                        self.dataframe.loc[
                            time, "Supply in temp {}".format(i + 1)
                        ] = self.rnd(s_inlet_temp[t])
                        self.dataframe.loc[
                            time, "Supply out temp {}".format(i + 1)
                        ] = self.rnd(s_outlet_temp[t])
                        self.dataframe.loc[
                            time, "Supply mass flow {}".format(i + 1)
                        ] = self.rnd(s_mass_flow[t])
                        self.dataframe.loc[
                            time, "Supply in pressure {}".format(i + 1)
                        ] = self.rnd(s_inlet_pressure[t])
                        self.dataframe.loc[
                            time, "Supply out pressure {}".format(i + 1)
                        ] = self.rnd(s_outlet_pressure[t])
                        self.dataframe.loc[
                            time, "Supply heat loss {}".format(i + 1)
                        ] = self.rnd(s_heat_loss[t])
                        self.dataframe.loc[
                            time, "Supply max mass flow {}".format(i + 1)
                        ] = self.rnd(s_edge_violation["flow speed"][t])
                        self.dataframe.loc[
                            time, "Ret in temp {}".format(i + 1)
                        ] = self.rnd(r_inlet_temp[t])
                        self.dataframe.loc[
                            time, "Ret out temp {}".format(i + 1)
                        ] = self.rnd(r_outlet_temp[t])
                        self.dataframe.loc[
                            time, "Ret mass flow {}".format(i + 1)
                        ] = self.rnd(r_mass_flow[t])
                        self.dataframe.loc[
                            time, "Ret in pressure {}".format(i + 1)
                        ] = self.rnd(r_inlet_pressure[t])
                        self.dataframe.loc[
                            time, "Ret out pressure {}".format(i + 1)
                        ] = self.rnd(r_outlet_pressure[t])
                        self.dataframe.loc[
                            time, "Ret heat loss {}".format(i + 1)
                        ] = self.rnd(r_heat_loss[t])
                        self.dataframe.loc[
                            time, "Ret max mass flow {}".format(i + 1)
                        ] = self.rnd(r_edge_violation["flow speed"][t])
                        plugs = self.grid.get_pipe_states(time_step=t)

                        sup_plugs = plugs[sup_edge_id - min_id]

                        ret_plugs = plugs[ret_edge_id - min_id]
                        self.reformat_plug_entry_step(sup_plugs)
                        self.reformat_plug_entry_step(ret_plugs)
                        self.dataframe.loc[
                            time, "Supply plugs {}".format(i + 1)
                        ] = sup_plugs
                        time_delay, inlet_temps = self.get_delay_inlet_temp(
                            time=time, i=i, pipe="Supply plugs {}"
                        )
                        self.dataframe.loc[
                            time, "Supply time delay {}".format(i + 1)
                        ] = time_delay
                        self.dataframe.loc[
                            time, "Supply in temps-time delay {}".format(i + 1)
                        ] = inlet_temps
                        self.dataframe.loc[
                            time, "Ret plugs {}".format(i + 1)
                        ] = ret_plugs
                        time_delay, inlet_temps = self.get_delay_inlet_temp(
                            time=time, i=i, pipe="Ret plugs {}"
                        )
                        self.dataframe.loc[
                            time, "Ret time delay {}".format(i + 1)
                        ] = time_delay
                        self.dataframe.loc[
                            time, "Ret in temps-time delay {}".format(i + 1)
                        ] = inlet_temps

                self.heat_demand, self.electricity_price = self.get_heat_electricity()
                self.counter = self.get_counter()
                self.num_counter = (
                    int(self.counter / TimeParameters["PlanningHorizon"]) - 1
                )
                if GridProperties["ConsumerNum"] == 1:
                    self.heat_demand = [self.heat_demand]
                else:
                    self.heat_demand = [
                        self.heat_demand * 0.28,
                        self.heat_demand * 0.40,
                        self.heat_demand * 0.32,
                    ]
            self.dataframe.to_csv(self.dataframe_path, index=True)
            start_iter += 1
            if np.any(
                np.array(self.dataframe["Underdelivered heat 1"].tolist()) < -0.05
            ):
                print("Underdelivered heat in iteration {}".format(start_iter))
                if ProducerPreset1["ControlWithTemp"]:
                    temp = self.revise_temp()
                else:
                    temp = self.revise_heat()
                self.dataset.reset_()
                self.heat_demand, self.electricity_price = self.get_heat_electricity()
                if GridProperties["ConsumerNum"] == 1:
                    self.heat_demand = [self.heat_demand]
                self.counter = self.dataset.get_counter()
                self.num_counter = 0
            else:
                break


class RealDataCollector(DataCollector):
    """
    Real world dataset
    """

    def __init__(self, case, warm_up):
        super().__init__(case, warm_up)

    def get_heat_electricity(self):
        """
        Get heat demand and electricity price from the real-world dataset
        """
        heat_demand, electricity_price = self.dataset.next_()
        return heat_demand, electricity_price

    def get_counter(self):
        """
        Get real-world counter
        """
        counter = self.dataset.get_counter()
        return counter


class SyntheticDataCollector(DataCollector):
    """
    Synthetic dataset
    """

    def __init__(self, heat_demand, electricity_price, case, warm_up):
        self.num = 0
        self.heat_demand_syn = heat_demand
        self.electricity_price_syn = electricity_price
        super().__init__(case, warm_up)

    def get_heat_electricity(self):
        """
        Synthesize your own heat demand and electricity price data
        """
        return self.heat_demand_syn, self.electricity_price_syn

    def get_counter(self):
        """
        Synthetic counter equals num*planning horizon
        """
        self.num += 1
        return self.num * TimeParameters["PlanningHorizon"]

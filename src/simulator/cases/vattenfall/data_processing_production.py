import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

from util.config import Dataset as ConfigDataset, Paths
from util.config import TimeParameters as ConfigTimeParameters
from util import config

from ..vattenfall_full import build_grid_short_or_long, get_consumers_scale
from ..vattenfall_clustered import build_grid_clustered_short
from .data_processing_ams import DataProcessor as DP


def load_production_data():
    # 0: timestamp 1: supply temp 2: return temp 3: production 1 (GJ)
    # 4: production 2 (GJ) 5: environment temp
    path = Path(__file__).parents[4] / "data/vattenfall/Vattenfall production 2021.csv"

    timestamps = []
    data = []
    with open(path, "r") as file:
        csvreader = csv.reader(file)
        i = 0
        for row in csvreader:
            i += 1
            if i == 1:
                continue
            if row[0].endswith("2020") | row[0].endswith("2021"):
                row[0] += " 0:00"
                timestamps.append(datetime.strptime(row[0], "%d/%m/%Y %H:%M"))
            else:
                timestamps.append(datetime.strptime(row[0], "%m/%d/%y %H:%M"))
            data.append(np.array(row[1:], dtype=float))

    return timestamps, np.array(data)


def distribute_demand(total_demand, consumers_scale):
    demands = np.repeat(total_demand[np.newaxis, :], len(consumers_scale), axis=0)
    for i in range(len(demands)):
        demands[i] = demands[i] * consumers_scale[i] / np.sum(consumers_scale)

    return demands

def retrieve_demand(consumers_scale):
    timestamps, production_data = load_production_data()
    total_demand = (production_data[:, 2] + production_data[:, 3] - 4) / 3.6
    return timestamps, distribute_demand(total_demand, consumers_scale)


def calibrate_hx():
    timestamps, production_data = load_production_data()
    # substract a estimated waste (from vattenfall dataset), convert to MWh
    total_demand = (production_data[:, 2] + production_data[:, 3] - 4) / 3.6

    demand_per_day = np.reshape(total_demand, (-1, 24))
    demand_avg_day = np.mean(demand_per_day, axis=1)
    temp_sup_avg = np.mean(np.reshape(production_data[:, 0], (-1, 24)), axis=1) - 1
    temp_ret_avg = np.mean(np.reshape(production_data[:, 1], (-1, 24)), axis=1) + 1

    demands = np.random.uniform(low=70, high=73, size=(1, int(24 * 4))) / (118 - 2)
    demands = np.repeat(demands, 118 - 2, axis=0)
    electricity_prices = np.random.uniform(low=10, high=60, size=(1, int(24 * 4)))

    consumers_scale = get_consumers_scale(config)
    consumer_demands_capacity = max(total_demand)*consumers_scale

    grid, [
        nodes_heat_ids,
        branches_ids,
        junctions_ids,
        sup_edges_ids,
        ret_edges_ids,
        consumers_scale,
    ] = build_grid_short_or_long(demands, consumer_demands_capacity, electricity_prices, True, config)

    consumers_scale = np.array(list(consumers_scale.values()))[
        np.argsort(list(consumers_scale.keys()))
    ]

    # grid, (_,_,cluster_to_consumer_idx) = build_grid_clustered_short(
    #     3,
    #     demands,
    #     consumer_demands_capacity,
    #     electricity_prices,
    #     config,
    # )

    # consumers_scale = get_consumers_scale(config)
    # scaler = [np.sum(consumers_scale[indices]) for indices in cluster_to_consumer_idx.values()]
    # consumers_scale = np.array(scaler)

    demands = distribute_demand(demand_avg_day, consumers_scale)
    ms_sec = demands * 10 ** 6 / 4181.3 / (70 - 45)

    from ...models.heat_exchanger import solve
    import math

    q = 0.78
    a = 460
    k = 40

    temp_ret = []
    for i, ms_s in enumerate(ms_sec):
        t_ret = []
        for t, ms in enumerate(ms_s[::2]):
            a_c = a
            # a_c = a * ((116 * consumers_scale[i])**0.18)

            _, t_r, _, _ = solve(
                t_supply_p=temp_sup_avg[t*2],  # in degrees C
                setpoint_t_supply_s=70,  # in degrees C
                t_return_s=45,  # in degrees C
                max_mass_flow_p=1000,  # in kg/s
                mass_flow_s=ms,  # in kg/s
                heat_capacity=4181.3,  # in J/kg/K
                surface_area=a_c,  # in m^2
                heat_transfer_q=q,  # See Palsson 1999 p45
                heat_transfer_k=k*np.power(ms/np.max(ms_s),0.3),
                # heat_transfer_k=k,
            )
            t_ret.append(t_r)
        temp_ret.append(t_ret)


    temp_ret = np.array(temp_ret)
    temp_ret = np.sum((temp_ret.T * consumers_scale).T, axis=0) / np.sum(
        consumers_scale
    )

    l1 = np.mean(np.abs(temp_ret_avg[::2] - temp_ret))
    l2 = np.mean(np.power(temp_ret_avg[::2] - temp_ret,2))
    from scipy.stats import linregress
    _, _, rvalue, _, _ = linregress(temp_ret_avg[::2], temp_ret)
    mean_diff = np.mean(temp_ret_avg[::2] - temp_ret)
    print(l1, l2, rvalue**2)
    # print(l1, mean_diff, max(np.abs(temp_ret - temp_ret_avg[::2])))
    # import matplotlib.pyplot as plt 
    # plt.plot(temp_ret, label="simulated")
    # plt.plot(temp_ret_avg[::2], label="reality")
    # plt.xlabel("Time (from Jan till Dec)")
    # plt.ylabel("return temp")
    # plt.title("15 clusters")
    # plt.legend()
    # plt.show()

@dataclass
class DataLoader:
    counter_train = 0
    counter_test = 0
    counter = 0
    data_power = pd.read_csv(
        ConfigDataset["Path"] / ConfigDataset["FileName"],
        usecols=["Season", "Time_day", "Price", "Heat_demand"],
    ).values.tolist()
    data_demand = pd.read_csv(
        ConfigDataset["Path"] / "vattenfall/Vattenfall production 2021.csv",
        usecols=["Heat production1 (GJ)", "Heat production2 (GJ)", "Environment temp"]
    ).values.tolist()
    data_temp = pd.read_csv(
        ConfigDataset["Path"] / "vattenfall/Vattenfall production 2021.csv",
        usecols=["Supply temp", "Return temp"]
    ).values.tolist()
    data_len: int = len(data_demand)
    assert data_len < len(data_power)
    heat_demand_data, electricity_price_data = [], []
    env_temp_data = []
    for i in range(data_len):
        heat_demand_data.append((data_demand[i][0]+ data_demand[i][1])/ 3.6)
        env_temp_data.append(data_demand[i][2])
        electricity_price_data.append(data_power[len(data_power)-data_len+i][2])

    heat_demand_data = np.array(heat_demand_data)
    electricity_price_data = np.array(electricity_price_data)
    max_heat_demand: float = max(heat_demand_data)
    min_heat_demand: float = min(heat_demand_data)
    max_electricity_price: float = max(electricity_price_data)
    min_electricity_price: float = min(electricity_price_data)

    prod_temp_data = np.array(data_temp)

    heat_demand_train = heat_demand_data[
        : int(data_len * ConfigDataset["train_test_split_point"])
    ]
    electricity_price_train = electricity_price_data[
        : int(data_len * ConfigDataset["train_test_split_point"])
    ]
    env_temp_train = env_temp_data[
        : int(data_len * ConfigDataset["train_test_split_point"])
    ]
    prod_temp_train = prod_temp_data[
        : int(data_len * ConfigDataset["train_test_split_point"])
    ]

    heat_demand_test = heat_demand_data[
        int(data_len * ConfigDataset["train_test_split_point"]) :
    ]
    electricity_price_test = electricity_price_data[
        int(data_len * ConfigDataset["train_test_split_point"]) :
    ]
    env_temp_test = env_temp_data[
        int(data_len * ConfigDataset["train_test_split_point"]) :
    ]
    prod_temp_test = prod_temp_data[
        int(data_len * ConfigDataset["train_test_split_point"]) :
    ]
    train_len: int = len(heat_demand_train)
    test_len: int = len(heat_demand_test)

    planning_horizon = ConfigTimeParameters["PlanningHorizon"]
    action_horizon = ConfigTimeParameters["ActionHorizon"]
    time_interval = ConfigTimeParameters["TimeInterval"]

    time_steps_per_hour = 3600 // time_interval

    def next(self, train=False):
        """
        Return demand and electricity price in the length of planning horizon.
        This is dependent whether we want training or testing dataset.
        """
        if train:
            pcounter = self.counter_train
            self.counter_train += self.action_horizon//self.time_steps_per_hour
            demand = self.heat_demand_train
            price = self.electricity_price_train
            env_temp = self.env_temp_train
            prod_temp_sup = self.prod_temp_train[:,0]
            prod_temp_ret = self.prod_temp_train[:,1]
        else:
            pcounter = self.counter_test
            self.counter_test += self.action_horizon//self.time_steps_per_hour
            demand = self.heat_demand_test
            price = self.electricity_price_test
            env_temp = self.env_temp_test
            prod_temp_sup = self.prod_temp_test[:,0]
            prod_temp_ret = self.prod_temp_test[:,1]
        return (
            np.repeat(
                demand[pcounter : pcounter + self.planning_horizon//self.time_steps_per_hour],
                self.time_steps_per_hour,
            ),
            np.repeat(
                price[pcounter : pcounter + self.planning_horizon//self.time_steps_per_hour],
                self.time_steps_per_hour,
            ),
            np.repeat(
                env_temp[pcounter : pcounter + self.planning_horizon//self.time_steps_per_hour],
                self.time_steps_per_hour,
            ),
            np.repeat(
                prod_temp_sup[pcounter : pcounter + self.planning_horizon//self.time_steps_per_hour],
                self.time_steps_per_hour,
            ),
            np.repeat(
                prod_temp_ret[pcounter : pcounter + self.planning_horizon//self.time_steps_per_hour],
                self.time_steps_per_hour,
            ),
        )

    def reset(self, train=False):
        """
        Reset counters on zero.
        """
        if train:
            self.counter_train = 0
        else:
            self.counter_test = 0

    def is_end(self, train=False):
        if train:
            return self.counter_train + self.planning_horizon > self.train_len
        else:
            return self.counter_test + self.planning_horizon > self.test_len
        

if __name__ == "__main__":
    calibrate_hx()

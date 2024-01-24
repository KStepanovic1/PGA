from dataclasses import dataclass
import pandas as pd
from pathlib import *
import numpy as np

from src.util.config import Dataset as ConfigDataset, Paths
from src.util.config import TimeParameters as ConfigTimeParameters


@dataclass
class Dataset:
    counter_train = 0
    counter_test = 0
    counter = 0
    path = Path(__file__).parents[2] / "data"
    data = pd.read_csv(
        path / ConfigDataset["FileName"],
        usecols=["Season", "Time_day", "Price", "Heat_demand"],
    ).values.tolist()
    data_len: int = len(data)
    heat_demand_data, electricity_price_data = [], []
    for i in range(data_len):
        heat_demand_data.append(data[i][3])
        electricity_price_data.append(data[i][2])

    heat_demand_data = np.array(heat_demand_data)

    electricity_price_data = np.array(electricity_price_data)
    max_heat_demand: float = max(heat_demand_data)
    min_heat_demand: float = min(heat_demand_data)
    max_electricity_price: float = max(electricity_price_data)
    min_electricity_price: float = min(electricity_price_data)
    season_train = pd.read_csv(path / "season_train.csv").values.tolist()
    time_of_the_day_train = pd.read_csv(
        path / "time_of_the_day_train.csv"
    ).values.tolist()
    heat_demand_train = heat_demand_data[
        : int(data_len * ConfigDataset["train_test_split_point"])
    ]
    electricity_price_train = electricity_price_data[
        : int(data_len * ConfigDataset["train_test_split_point"])
    ]

    season_test = pd.read_csv(path / "season_test.csv").values.tolist()
    time_of_the_day_test = pd.read_csv(
        path / "time_of_the_day_test.csv"
    ).values.tolist()
    heat_demand_test = heat_demand_data[
        int(data_len * ConfigDataset["train_test_split_point"]) :
    ]
    electricity_price_test = electricity_price_data[
        int(data_len * ConfigDataset["train_test_split_point"]) :
    ]
    train_len: int = len(heat_demand_train)
    test_len: int = len(heat_demand_test)

    planning_horizon = ConfigTimeParameters["PlanningHorizon"]
    action_horizon = ConfigTimeParameters["ActionHorizon"]

    def next(self, train=False):
        """
        Return demand and electricity price in the length of planning horizon.
        This is dependent whether we want training or testing dataset.
        """
        if train:
            pcounter = self.counter_train
            self.counter_train += self.action_horizon
            demand = self.heat_demand_train
            price = self.electricity_price_train
        else:
            pcounter = self.counter_test
            self.counter_test += self.action_horizon
            demand = self.heat_demand_test
            price = self.electricity_price_test
        return (
            demand[pcounter : pcounter + self.planning_horizon],
            price[pcounter : pcounter + self.planning_horizon],
        )

    def get_counter_test(self)->int:
        return self.counter_test

    def get(self, train=False):
        if train:
            counter = self.counter_train
            demand = self.heat_demand_train
            price = self.electricity_price_train
        else:
            counter = self.counter_test
            demand = self.heat_demand_test
            price = self.electricity_price_test
        return (
            demand[counter : counter + self.planning_horizon],
            price[counter : counter + self.planning_horizon],
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

    def next_(self):
        """
        Get heat demand and electricity price in the length equal to the planning horizon.
        The data is accessed without overlapping.
        """
        self.counter += self.planning_horizon
        return (
            self.heat_demand_data[self.counter - self.planning_horizon : self.counter],
            self.electricity_price_data[
                self.counter - self.planning_horizon : self.counter
            ],
        )

    def reset_(self) -> None:
        """
        Reset counter concerning data getter without repetition.
        """
        self.counter = 0

    def get_counter(self) -> int:
        return self.counter

    def is_end_(self) -> bool:
        """
        Is end concerning data without repetition.
        """
        return self.counter + self.planning_horizon < self.data_len

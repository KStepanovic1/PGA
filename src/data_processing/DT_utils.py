import numpy as np
import pandas as pd
from util.util import *
from dataclasses import dataclass
from pathlib import *


@dataclass
class DataPaths:
    PARENT: str = Path(__file__).parents[1]
    CURRENT: str = PARENT / "data_processing"
    IMAGES: str = CURRENT / "data_distribution"
    DATA: str = Path(__file__).parents[2] / "data"
    ELECTRICITY_PRICE: str = DATA / "day_ahead_price"
    HEAT_DEMAND: str = DATA / "Heat_demand.csv"
    PROCESSED_DATA: str = DATA / "processed_data.csv"
    PROCESSED_DATA_DISCRETE: str = DATA / "processed_data_discrete.csv"


@dataclass
class HeatDemand:
    heat_demand: np.array = (
        np.array(pd.DataFrame(pd.read_csv(DataPaths.HEAT_DEMAND, delimiter=";")))
    ).flatten()
    min_heat_demand: float = min(heat_demand)
    max_heat_demand: float = max(heat_demand)


@dataclass
class DataLen(HeatDemand):
    data_len: int = len(HeatDemand.heat_demand)
    daily_data_len: int = int(data_len / TIME_HORIZON)


@dataclass
class ProcessContinuousDataParams:
    heat_demand_upper_bound: int = 100
    heat_demand_lower_bound: int = 0
    month_start_index: int = 5
    month_end_index: int = 7
    day_time_start_index: int = 13
    day_time_end_index: int = 15
    price_start_index: int = 39
    price_end_index: int = 44
    multi_factor = [8664, 8760]
    columns = ["Season", "Time_day", "Price", "Heat_demand"]


@dataclass
class ProcessDiscreteDataParams:
    def __init__(self, N_electricity, N_heat):
        self.season_dict: dict = {
            "12": 0b00,
            "01": 0b00,
            "02": 0b00,
            "03": 0b01,
            "04": 0b01,
            "05": 0b01,
            "06": 0b10,
            "07": 0b10,
            "08": 0b10,
            "09": 0b11,
            "10": 0b11,
            "11": 0b11,
        }
        self.day_time_dict: dict = {
            "00": 0b00,
            "01": 0b00,
            "02": 0b00,
            "03": 0b00,
            "04": 0b00,
            "05": 0b00,
            "06": 0b01,
            "07": 0b01,
            "08": 0b01,
            "09": 0b10,
            "10": 0b10,
            "11": 0b10,
            "12": 0b10,
            "13": 0b10,
            "14": 0b10,
            "15": 0b10,
            "16": 0b10,
            "17": 0b10,
            "18": 0b11,
            "19": 0b11,
            "20": 0b11,
            "21": 0b11,
            "22": 0b11,
            "23": 0b11,
        }

        self.heat_demand_discrete: dict = self.discrete_intervals(
            ProcessContinuousDataParams.heat_demand_lower_bound,
            ProcessContinuousDataParams.heat_demand_upper_bound,
            N_heat,
        )
        self.electricity_price_discrete: dict = self.discrete_intervals(
            -10, 175, N_electricity
        )

    def discrete_intervals(self, start, end, N) -> dict:
        start = int(round(start))
        end = int(round(end))
        discrete_interval = {}
        interval_length = int(round((end - start) / N))
        for i in range(N):
            key = (2 * start + interval_length) / 2
            value = [start, end + 1] if i == N - 1 else [start, start + interval_length]
            discrete_interval[key] = value
            start = int(round(start + interval_length))
        print(discrete_interval)
        return discrete_interval


@dataclass
class TrainTestParams(DataLen):
    def __init__(self, train_ratio, test_ratio):
        self.TRAIN_RATIO: float = train_ratio
        self.TEST_RATIO: float = test_ratio
        self.TRAIN_SIZE: int = int(round(DataLen.daily_data_len * self.TRAIN_RATIO))
        self.TEST_SIZE: int = int(round(DataLen.daily_data_len * self.TEST_RATIO))

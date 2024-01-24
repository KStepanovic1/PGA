import pandas as pd
from pathlib import *
from util.config import Dataset
from abc import ABC, abstractmethod


class DataProcessing(ABC):
    def __init__(self):
        self.default_interval = 3600  # [s]
        self.path = Path(__file__).parents[1] / "data"
        self.columns = ["Season", "Time_day", "Price", "Heat_demand"]

    @abstractmethod
    def create_dataset(self) -> list:
        pass

    def save_data(self, data, name) -> None:
        """
        Store the new dataset.
        """
        df = pd.DataFrame(data, columns=self.columns)
        df.to_csv(
            self.path / name,
            index=False,
        )


class DataRescaling(DataProcessing):
    """
    Rescaling of the heat demand to fit the maximum heat production of the
    CHP operation region.
    """

    def __init__(self, max_heat_demand, min_heat_demand):
        super().__init__()
        self.max_heat_demand = max_heat_demand
        self.min_heat_demand = min_heat_demand
        self.data = pd.read_csv(
            self.path / Dataset["FileName"],
            usecols=self.columns,
        ).values.tolist()
        self.data_len: int = len(self.data)

    def get_max_min_heat_demand(self):
        """
        Calculate maximum and minimum of heat demand of original dataset.
        """
        heat_demand = [self.data[i][3] for i in range(self.data_len)]
        return max(heat_demand), min(heat_demand)

    def create_dataset(self) -> list:
        """
        Create new dataset with rescaled heat demand, while season, time of the day and electricity price
        components remain the same. Count the number of points in which heat demand is greater than 65 MW,
        as those are potential points in which violation might occur (with respect to the maximal heat production
        of the CHP unit -- 70 MW).
        """
        data = []
        num_point = 0
        border_line = 65  # [MW]
        max_heat_demand, min_heat_demand = self.get_max_min_heat_demand()
        for i in range(self.data_len):
            temp = []
            temp.append(self.data[i][0])
            temp.append(self.data[i][1])
            temp.append(self.data[i][2])
            heat_demand_scale = round(
                (self.data[i][3] - max_heat_demand)
                * (self.max_heat_demand - self.min_heat_demand)
                / (max_heat_demand - min_heat_demand)
                + self.max_heat_demand,
                2,
            )
            if heat_demand_scale > border_line:
                num_point += 1
            temp.append(heat_demand_scale)
            data.append(temp)
        print(
            "Number of points in which heat demand is greater than 67 MWh is {}, "
            "which is {} percent of dataset length.".format(
                int(num_point), round(num_point / self.data_len * 100, 5)
            )
        )
        return data


class DataInterpolation(DataProcessing):
    """
    Interpolation of the datapoints depending on the interpolation step.
    """

    def __init__(self, interpolation_interval, max_heat_demand):
        super().__init__()
        self.interpolation_interval = interpolation_interval
        self.interpolation_step: int = int(
            self.default_interval / self.interpolation_interval
        )
        name = "processed_data_max_q_" + str(max_heat_demand) + ".csv"
        self.data = pd.read_csv(
            self.path / name,
            usecols=self.columns,
        ).values.tolist()
        self.data_len: int = len(self.data)

    def create_dataset(self) -> list:
        """
        Create dataset with new, interpolated points. For example, if interpolation step is 2,
        and values on indices 1 and 2 are 10 and 20, respectively,
        then the newly created dataset will have values 10, 15 and 20
        on indices 1,2 and 3, respectively.
        """
        data = []
        for i in range(self.data_len - 1):
            delta_e = round(
                (self.data[i + 1][2] - self.data[i][2]) / self.interpolation_step,
                2,
            )
            delta_q = round(
                (self.data[i + 1][3] - self.data[i][3]) / self.interpolation_step,
                2,
            )
            for j in range(self.interpolation_step):
                temp = []
                temp.append(self.data[i][0])
                temp.append(self.data[i][1])
                interpolate_e = round(self.data[i][2] + j * delta_e, 2)
                interpolate_q = round(self.data[i][3] + j * delta_q, 2)
                temp.append(interpolate_e)
                temp.append(interpolate_q)
                data.append(temp)
        data.append(
            [
                self.data[i + 1][0],
                self.data[i + 1][1],
                self.data[i + 1][2],
                self.data[i + 1][3],
            ]
        )
        return data

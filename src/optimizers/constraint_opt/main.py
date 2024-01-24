import numpy as np
from .data_collector import RealDataCollector, SyntheticDataCollector
from util.shared import *
from util.config import *


if __name__ == "__main__":
    # determines whether we generate data following heuristic or randomly
    heuristics_control: bool = True
    # determines whether we generate data for warming up neural network training or for the actual training
    warm_up: bool = False
    """
    if the SyntheticDataCollector is used, user specifies heat demand and electricity price
    if the RealDataCollector is used, these variables are read from the dataset
    """
    data_collector = RealDataCollector(case=cases["real"], warm_up=warm_up)

    """
    data_collector = SyntheticDataCollector(
        heat_demand=np.array([2] * TimeParameters["PlanningHorizon"]),
        electricity_price=np.array([10] * TimeParameters["PlanningHorizon"]),
        case=cases["low"],
    )
    """

    data_collector.run(num_iter=2 * TimeParameters["PlanningHorizon"], heuristics_control = heuristics_control)

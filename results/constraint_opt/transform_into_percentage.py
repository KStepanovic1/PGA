import os
import pandas as pd

from pathlib import Path


folder = (
    Path(__file__)
    .parent.joinpath("relax_monotonic_icnn_gd")
    .joinpath("initialization_1/MPC_episode_length_24_hours")
)

columns = [
    "Supply_inlet_violation",
    "Supply_outlet_violation",
    "Mass_flow_violation",
    "Delivered_heat_violation",
    "Heat_demand",
]
files = os.listdir(folder)
for file in files:
    if file.endswith(".csv"):
        data = pd.read_csv(folder.joinpath(file), index_col=[0])
        data["Supply_inlet_violation_percent"] = (
            data["Supply_inlet_violation"] / 120
        ) * 100
        data["Supply_outlet_violation_percent"] = (
            data["Supply_outlet_violation"] / 70
        ) * 100
        data["Mass_flow_violation_percent"] = (
            data["Mass_flow_violation"] / 805.15
        ) * 100
        data["Delivered_heat_violation_percent"] = (
            data["Delivered_heat_violation"] / data["Heat_demand"]
        ) * 100
        data.to_csv(folder.joinpath(file), index=False)

"""
As it is not possible to run this program for consequtive time steps, we merge solutions of the
planning horizon length, and check for constraint violations.
"""

import pandas as pd
from pathlib import Path

from src.util import config
from src.util.config import GridProperties, TimeParameters, ProducerPreset1, PipePreset1
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer
from src.optimizers.constraint_opt.dhn_nn.functions import run_simulator, plugs_to_list
from src.optimizers.constraint_opt.dhn_nn.param import LiOpt
from src.data_processing.dataloader import Dataset

if GridProperties["ConsumerNum"] == 1:
    from src.simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from src.simulator.cases.parallel_consumers import build_grid

if ProducerPreset1["ControlWithTemp"] == False:
    raise AssertionError("Control must be done with temperature!")

dataset = Dataset()
dataset_init = pd.read_csv(
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
# optimization steps to be read
opt_steps = [12, 24, 36, 48, 60, 72]
# dictionary of values
dict = {
    "Q_optimized": [],
    "T_supply_optimized": [],
    "Produced_electricity": [],
    "Profit": [],
    "Heat_demand": [],
    "Electricity_price": [],
    "Runtime": [],
    "Optimality_gap": [],
}

# read data
for opt_step in opt_steps:
    li = pd.read_csv("li_{}.csv".format(opt_step))
    for key in dict.keys():
        dict[key].extend(list(li[key]))

dict["Supply_inlet_violation"] = []
dict["Supply_outlet_violation"] = []
dict["Delivered_heat_violation"] = []
dict["Mass_flow_violation"] = []
# heat demand and electricity price for certain coefficient
heat_demand, electricity_price = (
    dataset.heat_demand_data[
        opt_steps[0] : opt_steps[0] + TimeParameters["PlanningHorizon"]
    ],
    dataset.electricity_price_data[
        opt_steps[0] : opt_steps[0] + TimeParameters["PlanningHorizon"]
    ],
)
# initialization plugs
plugs_supply: list = plugs_to_list(dataset_init["Supply plugs 1"][opt_steps[0]])
plugs_return: list = plugs_to_list(dataset_init["Ret plugs 1"][opt_steps[0]])
plugs = [plugs_supply, plugs_return]
# build simulator
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
# run simulator
(
    supply_inlet_violation,
    supply_outlet_violation,
    mass_flow_violation,
    delivered_heat_violation,
    produced_heat_sim,
    supply_inlet_temperature_sim,
    tau_out,
    mass_flow_sim,
    return_outlet_temperature_sim,
    return_inlet_temperature_sim,
    plugs,
) = run_simulator(
    simulator=simulator,
    object_ids=object_ids,
    producer_id=producer_id,
    consumer_ids=consumer_ids,
    sup_edge_ids=sup_edge_ids,
    produced_heat=dict["Q_optimized"],
    supply_inlet_temperature=dict["T_supply_optimized"],
    produced_electricity=dict["Produced_electricity"],
    demand=heat_demand,
    price=electricity_price,
    plugs=plugs,
)
# join violations assesment to dictionary
dict["Supply_inlet_violation"].extend(supply_inlet_violation)
dict["Supply_outlet_violation"].extend(supply_outlet_violation)
dict["Mass_flow_violation"].extend(mass_flow_violation)
dict["Delivered_heat_violation"].extend(delivered_heat_violation)

df = pd.DataFrame(dict)
df.to_csv(
    (Path(__file__).parents[1] / "MPC_episode_length_72_hours").joinpath(
        "li_{}.csv".format(opt_steps[0])
    )
)

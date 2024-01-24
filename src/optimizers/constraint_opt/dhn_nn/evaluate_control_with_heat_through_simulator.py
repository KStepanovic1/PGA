"""
The role of this file is to transform optimization from control with temperature to control with heat.
"""
from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer

if GridProperties["ConsumerNum"] == 1:
    from src.simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from src.simulator.cases.parallel_consumers import build_grid

if ProducerPreset1["ControlWithTemp"] == True:
    raise AssertionError("Control must be done with heat!")


def evaluate_control_with_heat_through_simulator(
    neurons, N, optimizer_type, source, destination
):
    """
    Transform simulator from control with temperature to the simulation with control with heat.
    """
    for opt_step in opt_steps["math_opt"]:
        # get heat demand, electricity price and plugs corresponding to the certain time-step
        heat_demand, electricity_price, plugs = get_demand_price_plugs(
            opt_step=opt_step
        )
        # last element of the list corresponds to the index TimeHorizon+1
        heat_demand = heat_demand[:-1]
        electricity_price = electricity_price[:-1]
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
        # historical mass flow
        history_mass_flow = {
            sup_edge_ids: plugs[0][0][0] / TimeParameters["TimeInterval"],
            ret_edge_ids: plugs[0][0][0] / TimeParameters["TimeInterval"],
        }
        for neuron in neurons:
            for model_num in range(N):
                data = {
                    "Produced_heat_optimizer": [],
                    "Produced_electricity": [],
                    "Profit": [],
                    "Heat_demand": [],
                    "Electricity_price": [],
                    "Supply_inlet_violation": [],
                    "Supply_outlet_violation": [],
                    "Mass_flow_violation": [],
                    "Delivered_heat_violation": [],
                    "Runtime": [],
                    "Optimality_gap": [],
                }
                file_name = (
                    "{}_".format(model_num)
                    + optimizer_type
                    + "_init_"
                    + neurons_ext(neuron)
                    + "_opt_step_{}".format(opt_step)
                    + "_"
                )
                file_name = csv_file_finder(
                    files=os.listdir(source), start=file_name, null_solve=False
                )
                file = pd.read_csv(source.joinpath(file_name))[
                    ["Produced_heat_optimizer", "Runtime", "Optimality_gap"]
                ]
                produced_heat = list(file["Produced_heat_optimizer"])
                runtime = list(file["Runtime"])
                optimality_gap = list(file["Optimality_gap"])
                produced_electricity = get_optimal_produced_electricity(
                    produced_heat=produced_heat, electricity_price=electricity_price
                )
                operation_cost = calculate_operation_cost(
                    produced_heat=produced_heat,
                    produced_electricity=produced_electricity,
                    electricity_price=electricity_price,
                )
                # run through the simulator for feasibility verification
                (
                    supply_inlet_violations,
                    supply_outlet_violations,
                    mass_flow_violations,
                    delivered_heat_violations,
                    produced_heat_sim,
                    tau_in_sim,
                    tau_out,
                    m,
                    ret_tau_out_sim,
                    ret_tau_in_sim,
                    plugs,
                ) = run_simulator(
                    simulator=simulator,
                    object_ids=object_ids,
                    producer_id=producer_id,
                    consumer_ids=consumer_ids,
                    sup_edge_ids=sup_edge_ids,
                    produced_heat=produced_heat,
                    supply_inlet_temperature=[90] * TimeParameters["PlanningHorizon"],
                    produced_electricity=produced_electricity,
                    demand=heat_demand,
                    price=electricity_price,
                    plugs=plugs,
                    history_mass_flow=history_mass_flow,
                )
                data["Produced_heat_optimizer"].extend(produced_heat)
                data["Produced_electricity"].extend(produced_electricity)
                data["Profit"].extend(operation_cost)
                data["Heat_demand"].extend(heat_demand)
                data["Electricity_price"].extend(electricity_price)
                # tranform to percentage
                for supply_inlet_violation in supply_inlet_violations:
                    data["Supply_inlet_violation"].append(
                        percent_tau_in(tau_in=supply_inlet_violation)
                    )
                for supply_outlet_violation in supply_outlet_violations:
                    data["Supply_outlet_violation"].append(
                        percent_tau_out(tau_out=supply_outlet_violation)
                    )
                for mass_flow_violation in mass_flow_violations:
                    data["Mass_flow_violation"].append(percent_m(m=mass_flow_violation))
                for delivered_heat_violation in delivered_heat_violations:
                    data["Delivered_heat_violation"].append(
                        percent_y(y=delivered_heat_violation)
                    )
                data["Runtime"].extend(runtime)
                data["Optimality_gap"].extend(optimality_gap)
                df = pd.DataFrame(data)
                df.to_csv(destination.joinpath(file_name))


if __name__ == "__main__":
    N = 3
    neurons = [[1]]
    optimizer_type = "plnn_milp"
    source = (Path(__file__).parents[4]).joinpath(
        "results\constraint_opt\plnn_milp\MPC_episode_length_72_hours\control with heat"
    )
    destination = (Path(__file__).parents[4]).joinpath(
        "results\constraint_opt\plnn_milp\MPC_episode_length_72_hours\control with heat simulator"
    )
    evaluate_control_with_heat_through_simulator(
        neurons=neurons,
        N=N,
        optimizer_type=optimizer_type,
        source=source,
        destination=destination,
    )

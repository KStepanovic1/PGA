"""
This file contains functions commonly used by all other programs -- through learning and different optimization algorithms.
"""
from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *


def string_to_list(x) -> list:
    """
    Transforms string of shape [x, y, z,...] into list of integers.
    """
    y: list = []
    start_index: int = 1  # index 0 is reserved for [
    end_index: int = len(x) - 1
    if "," not in x:
        y.append(float(x[start_index:end_index]))
    else:
        for m in re.finditer(",", x):
            comma: int = m.start(0)
            y.append(float(x[start_index:comma]))
            start_index: int = comma + 1
        # time-step is int
        y.append(float(x[start_index:end_index]))
    return y


def plugs_to_list(a) -> list:
    """
    Transforms plugs of the format [[x,y,z],[a,b,c,d],[m,n,l,o]] into list of lists.
    """
    plugs: list = []
    start_index: int = 1
    a = a[start_index:]
    for begin_closure in re.finditer("\[", a):
        bracket: int = begin_closure.start(0)
        end_closure: int = a.find("]", bracket) + 1
        a_ = a[bracket:end_closure]
        plug: list = string_to_list(a_)
        plugs.append(plug)
    return plugs


def transform_value_to_percentage():
    """
    Transforms value of violation to percentage of violation.
    """
    # maximum values of violations
    inlet_temp_per: int = 50
    outlet_temp_per: int = 50
    delivered_heat_per: int = 70
    mass_flow_per: int = 810
    # percentage violation files
    result_p: Path = Path(__file__).parents[3] / "results/constraint_opt"

    result_p_icnn_gd: Path = result_p.joinpath("icnn_gd")
    result_p_plnn_milp: Path = result_p.joinpath("plnn_milp")
    for result_p in [result_p_plnn_milp, result_p_icnn_gd]:
        for opt_step in opt_steps:
            violations_percent = {
                "Supply inlet temperature": [],
                "Supply outlet temperature": [],
                "Delivered heat": [],
                "Mass flow": [],
            }
            violation = pd.read_csv(result_p.joinpath("violations " + str(opt_step)))[
                [
                    "Supply inlet temperature",
                    "Supply outlet temperature",
                    "Delivered heat",
                    "Mass flow",
                ]
            ]
            supply_inlet_temperature = [
                round(abs(x) / inlet_temp_per * 100, 2)
                for x in list(violation["Supply inlet temperature"])
            ]
            supply_outlet_temperature = [
                round(abs(x) / outlet_temp_per * 100, 2)
                for x in list(violation["Supply outlet temperature"])
            ]
            delivered_heat = [
                round(abs(x) / delivered_heat_per * 100, 2)
                for x in list(violation["Delivered heat"])
            ]
            mass_flow = [
                round(x / mass_flow_per * 100, 2) for x in list(violation["Mass flow"])
            ]
            violations_percent["Supply inlet temperature"].extend(
                supply_inlet_temperature
            )
            violations_percent["Supply outlet temperature"].extend(
                supply_outlet_temperature
            )
            violations_percent["Delivered heat"].extend(delivered_heat)
            violations_percent["Mass flow"].extend(mass_flow)
            violations_df = pd.DataFrame(violations_percent)
            violations_df.to_csv(
                result_p.joinpath("violations percentage {}".format(opt_step))
            )


def surface_of_cross_sectional_area(diameter):
    """
    Calculate surface of cross-sectional area of the pipe.
    """
    return pow(diameter, 2) * math.pi / 4


def percent_tau_in(tau_in) -> float:
    """
    Calculate percentage of supply inlet temperature violation with respect to maximum violation.
    """
    return (
        abs(tau_in)
        / (PhysicalProperties["MaxTemp"] - ConsumerPreset1["MinTempSupplyPrimary"])
        * 100
    )


def reverse_percent_tau_in(tau_in_percent) -> float:
    """
    Calculate actual violation of supply inlet temperature in C, based on the percentage of violation.
    """
    return (
        tau_in_percent
        * (PhysicalProperties["MaxTemp"] - ConsumerPreset1["MinTempSupplyPrimary"])
    ) / 100


def percent_tau_out(tau_out) -> float:
    """
    Calculate percentage of supply outlet temperature violation with respect to maximum violation.
    """
    return (
        abs(tau_out)
        / (PhysicalProperties["MaxTemp"] - ConsumerPreset1["MinTempSupplyPrimary"])
        * 100
    )


def reverse_percent_tau_out(tau_out_percent) -> float:
    """
    Calculate actual violation of supply outlet temperature in C, based on the percentage of violation.
    """
    return (
        tau_out_percent
        * (PhysicalProperties["MaxTemp"] - ConsumerPreset1["MinTempSupplyPrimary"])
    ) / 100


def percent_y(y) -> float:
    """
    Calculate percentage of delivered heat violation with respect to maximum violation.
    """
    return (
        abs(y)
        / (
            ProducerPreset1["Generators"][0]["MaxHeatProd"]
            - ProducerPreset1["Generators"][0]["MinHeatProd"]
        )
        * 100
    )


def reverse_percent_y(y_percent) -> float:
    """
    Calculate actual violation of delivered heat in MWh, based on the percentage of delivered heat.
    """
    return (
        y_percent
        * (
            ProducerPreset1["Generators"][0]["MaxHeatProd"]
            - ProducerPreset1["Generators"][0]["MinHeatProd"]
        )
        / 100
    )


def percent_m(m) -> float:
    """
    Calculate percentage of mass flow violation with respect to maximum violation.
    """
    return abs(m) / (PipePreset1["MaxFlowSpeed"] - PipePreset1["MinFlowSpeed"]) * 100


def reverse_percent_m(m_percent) -> float:
    """
    Calculate actual violation of mass flow in kg/s, based on the percentage of violation.
    """
    return m_percent * (PipePreset1["MaxFlowSpeed"] - PipePreset1["MinFlowSpeed"]) / 100


def get_optimal_produced_electricity(produced_heat, electricity_price):
    """
    Calculate optimal electricity production based on produced heat and electricity price w.r.t. cost of producing electricity
    """
    produced_electricity = []
    for i in range(len(produced_heat)):
        if electricity_price[i] >= ProducerPreset1["Generators"][0]["FuelCost"][1]:
            electricity = (
                -0.214 * produced_heat[i] + 50
            )  # line between points (0,50) and (70,35)
        else:
            if produced_heat[i] <= 10:
                electricity = (
                    -0.5 * produced_heat[i] + 10
                )  # line between points (0,10) and (10,5)
            else:
                electricity = (
                    0.5 * produced_heat[i]
                )  # line between points (10,5) and (70,35)
        produced_electricity.append(electricity)
    return produced_electricity


def calculate_operation_cost(produced_heat, produced_electricity, electricity_price):
    """
    Calculate operation cost based on produced heat, produced electricity, and electricity price
    """
    profit = []
    for i in range(len(produced_heat)):
        p = (
            ProducerPreset1["Generators"][0]["FuelCost"][0] * produced_heat[i]
            + ProducerPreset1["Generators"][0]["FuelCost"][1] * produced_electricity[i]
            - electricity_price[i] * produced_electricity[i]
        )
        profit.append(p)
    return profit


def get_demand_price_plugs(opt_step: int):
    """
    Based on the time-step t retrieve heat demand and electricity price,
    starting from the time-step t up to the t+planning horizon, and plugs from the time-step t-1.
    """
    # path where constraint_opt results are stored
    result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
    with open(
        result_p.joinpath(
            "data_num_{}_heat_demand_real_world_for_L={}_time_interval_{}_max_Q_{}MW_deltaT_{}C.csv".format(
                GridProperties["ConsumerNum"],
                PipePreset1["Length"],
                TimeParameters["TimeInterval"],
                Generator1["MaxHeatProd"],
                Generator1["MaxRampRateTemp"],
            )
        ),
        "rb",
    ) as f:
        data = pd.read_csv(f)[
            ["Heat demand 1", "Electricity price", "Supply plugs 1", "Ret plugs 1"]
        ]
    demand: np.array = np.array(
        data.loc[
            opt_step : opt_step + TimeParameters["PlanningHorizon"], "Heat demand 1"
        ]
    )
    price: list = list(
        np.array(
            data.loc[
                opt_step : opt_step + TimeParameters["PlanningHorizon"],
                "Electricity price",
            ]
        )
    )
    plugs_supply: list = plugs_to_list(data.loc[opt_step, "Supply plugs 1"])
    plugs_return: list = plugs_to_list(data.loc[opt_step, "Ret plugs 1"])
    plugs: list = [plugs_supply, plugs_return]
    return demand, price, plugs


def run_simulator(
    simulator,
    object_ids,
    producer_id,
    consumer_ids,
    sup_edge_ids,
    produced_heat,
    supply_inlet_temperature,
    produced_electricity,
    demand,
    price,
    plugs,
    history_mass_flow=None,
):
    """
    Verify the feasibility of the solution through the simulator.
    """
    simulator.reset([demand], [price], plugs, history_mass_flow)
    if ProducerPreset1["ControlWithTemp"]:
        simulator.run(
            temp=[supply_inlet_temperature],
            electricity=[produced_electricity],
        )
    else:
        simulator.run(
            heat=[produced_heat],
            electricity=[produced_electricity],
        )
    # get status
    grid_status = simulator.get_object_status(
        object_ids=object_ids,
        start_step=0,
        end_step=TimeParameters["PlanningHorizon"],
        get_temp=True,
        get_ms=True,
        get_pressure=False,
        get_violation=True,
    )
    supply_inlet_temperature_sim = grid_status[producer_id]["Temp"][1]
    return_outlet_temperature_sim = grid_status[producer_id]["Temp"][0]
    return_inlet_temperature_sim = grid_status[consumer_ids]["Temp"][1]
    tau_out = grid_status[consumer_ids]["Temp"][0]
    mass_flow_sim = grid_status[producer_id]["Mass flow"][0]
    produced_heat_sim = [
        (
            PhysicalProperties["HeatCapacity"]
            / PhysicalProperties["EnergyUnitConversion"]
        )
        * mass_flow_sim[i]
        * (supply_inlet_temperature_sim[i] - return_outlet_temperature_sim[i])
        for i in range(TimeParameters["PlanningHorizon"])
    ]
    if ProducerPreset1["ControlWithTemp"]:
        supply_inlet_violation = []
        for tau_in in supply_inlet_temperature:
            if tau_in > PhysicalProperties["MaxTemp"]:
                supply_inlet_violation.append(tau_in - PhysicalProperties["MaxTemp"])
            else:
                supply_inlet_violation.append(0)
    else:
        supply_inlet_violation = grid_status[producer_id]["Violation"]["supply temp"]
    supply_outlet_violation = grid_status[consumer_ids]["Violation"]["supply temp"]
    mass_flow_violation = grid_status[sup_edge_ids]["Violation"]["flow speed"]
    delivered_heat_violation = grid_status[consumer_ids]["Violation"]["heat delivered"]
    plugs = copy.deepcopy(
        simulator.get_pipe_states(time_step=TimeParameters["ActionHorizon"])
    )
    return (
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
    )


def normalize_variable(var, min, max):
    """
    Normalize variable between 0 and 1
    """
    return round(
        (var - min) / (max - min),
        round_dig,
    )


def re_normalize_variable(var, min, max):
    """
    Return normalized variable from [0,1] interval to regular interval.
    """
    return round(min + var * (max - min), 2)


def neurons_ext(layer_size) -> str:
    """
    Form a string indicating number of neurons in each hidden layer.
    """
    neurons = "neurons"
    for i in range(len(layer_size)):
        neurons += "_" + str(layer_size[i])
    return neurons


def save_results(keys, values, path):
    """
    Function for saving results depending on their keys, values and path for storing results.
    """
    results = {}
    for i, key in enumerate(keys):
        results[key] = values[i]
    results_df = pd.DataFrame(results)
    results_df.to_csv(path)


def csv_file_finder(files, start, null_solve):
    """
    Find the file in files that matches start (optimizer) and end (.csv).
    If more such files exist, we throw error.
    """
    if null_solve:
        csv_files = [
            file
            for file in files
            if file.startswith(start + "_time_step_0") and file.endswith(".csv")
        ]
    else:
        csv_files = [
            file for file in files if file.startswith(start) and file.endswith(".csv")
        ]
    if len(csv_files) == 0:
        warnings.warn("We found no files with this name")
        csv_files = [None]
    elif len(csv_files) > 1:
        warnings.warn("Number of files with the same name is greater then one")
        warnings.simplefilter("error")
        warnings.resetwarnings()
    return csv_files[0]


def calculate_plot_bounds(mean, deviation):
    """
    Given mean and standard deviation calculate upper and lower bounds.
    """
    up_bound, down_bound = [], []
    for i in range(len(mean)):
        if mean[i] == None and deviation[i] == None:
            up_bound.append(np.nan)
            down_bound.append(np.nan)
        else:
            up_bound.append(mean[i] + deviation[i])
            down_bound.append(mean[i] - deviation[i])
    return up_bound, down_bound


def calculate_ninety_percent_confidence_interval(mean, deviation):
    """
    Given mean and standard deviation calculate ninety percent confidence intervals.
    """
    z = 1.645
    up_bound, down_bound = [], []
    for i in range(len(mean)):
        if mean[i] == None and deviation[i] == None:
            up_bound.append(np.nan)
            down_bound.append(np.nan)
        else:
            up_bound.append(
                mean[i]
                + z * (deviation[i] / math.sqrt(TimeParameters["PlanningHorizon"]))
            )
            down_bound.append(
                mean[i]
                - z * (deviation[i] / math.sqrt(TimeParameters["PlanningHorizon"]))
            )
    return up_bound, down_bound


def calculate_giraud_operation_cost():
    """
    Calculate operation cost of Giraud optimizer, starting from the first hour of opt_steps hours.
    """
    # start from the index opt_steps["math_opt"][0]+time_delay[str(PipePreset1["Length"])] to mitigate effect of previous actions
    start = opt_steps["math_opt"][0] + time_delay[str(PipePreset1["Length"])]
    end = opt_steps["math_opt"][-1] + TimeParameters["PlanningHorizon"] - 1
    path: Path = Path(__file__).parents[4] / "results/constraint_opt"
    file_name = "data_num_{}_heat_demand_real_world_for_L={}_time_interval_{}_max_Q_{}MW_deltaT_{}C.csv".format(
        GridProperties["ConsumerNum"],
        PipePreset1["Length"],
        TimeParameters["TimeInterval"],
        ProducerPreset1["Generators"][0]["MaxHeatProd"],
        ProducerPreset1["Generators"][0]["MaxRampRateTemp"],
    )
    data = pd.read_csv(path.joinpath(file_name))
    produced_heat = list(data.loc[start:end, "Produced heat"])
    price = list(data.loc[start:end, "Electricity price"])
    produced_electricity = get_optimal_produced_electricity(
        produced_heat=produced_heat, electricity_price=price
    )
    operation_cost = calculate_operation_cost(
        produced_heat=produced_heat,
        produced_electricity=produced_electricity,
        electricity_price=price,
    )
    operation_cost = sum(operation_cost)
    return operation_cost


def delete_specified_files(
    path=Path(__file__).parents[4]
    / "results/constraint_opt/relax_monotonic_icnn_gd/single_step_results",
    pattern=r".*2023-05-12.csv$",
):
    """
    Function to delete files on specific location if they end up with string specified in pattern.
    """
    for file in path.iterdir():
        if file.is_file() and re.match(pattern, file.name):
            os.remove(file)

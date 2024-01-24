import numpy as np

from ..models import Grid, Consumer, Edge, CHP
from ..models import Branch, Junction
from .vattenfall.data_processing_ams import DataProcessor as DP
from .vattenfall.consumer_grouping import (
    build_connection,
    graph_hierarchical_cluster,
    build_clustered_network,
)
from .vattenfall.data_processing_ams import Pipe_params_loader


def build_grid_clustered_short(
    group_number,
    consumer_demands,  # size of the original grid
    consumer_demands_capacity, # design demand capacity
    electricity_prices,
    config,
):
    connection_lists, distance_lists = build_connection()
    labels = graph_hierarchical_cluster(connection_lists, distance_lists, group_number)
    (clustered_connection, conumser_idx_to_cluster_idx) = build_clustered_network(
        connection_lists, distance_lists, labels
    )

    assert len(labels) == len(consumer_demands)

    producer_params = config.ProducerPreset1
    consumer_params = config.ConsumerPreset1
    sup_pipe_params = config.PipePreset1
    ret_pipe_params = config.PipePreset2
    physical_properties = config.PhysicalProperties
    time_params = config.TimeParameters

    blocks = consumer_demands[0].shape[0]
    grid = Grid(  # empty grid
        interval_length=time_params["TimeInterval"],
    )

    nodes_apperance_list = np.array([[n1, n2] for n1, n2, _, _ in clustered_connection])
    nodes_apperance_list = nodes_apperance_list.flatten()
    nodes_indices, unique_counts = np.unique(nodes_apperance_list, return_counts=True)
    splits_total_connect_counts = {k: v for k, v in zip(nodes_indices, unique_counts)}

    pipe_params_loader = Pipe_params_loader()
    counter_demand = 0
    nodes_dict = {}
    splits_connected_counts = {}
    edges_dict = {}
    cluster_to_consumer_idx = {}

    for (
        upstream_node_idx,
        down_stream_node_idx,
        distance,
        dia,
    ) in clustered_connection:
        for n_idx in [upstream_node_idx, down_stream_node_idx]:
            if nodes_dict.get(n_idx) is None:
                if n_idx in DP.storage_indices:
                    producer = CHP(
                        CHPPreset=producer_params["Generators"][0],
                        blocks=blocks,  # time steps (24 hours -> 24 blocks)
                        heat_capacity=physical_properties[
                            "HeatCapacity"
                        ],  # in J/kg/K # for the water
                        temp_upper_bound=physical_properties["MaxTemp"],
                        pump_efficiency=producer_params["PumpEfficiency"],
                        density=physical_properties["Density"],
                        control_with_temp=producer_params["ControlWithTemp"],
                        energy_unit_conversion=physical_properties[
                            "EnergyUnitConversion"
                        ],
                    )
                    grid.add_node(producer)  # characterized with temperature
                    nodes_dict[n_idx] = [producer, producer]
                elif n_idx < len(DP.nodes_list):
                    unclustered_consumer_indices = (
                        labels == conumser_idx_to_cluster_idx[n_idx]
                    )
                    demand = np.sum(
                        consumer_demands[unclustered_consumer_indices], axis=0
                    )
                    assert len(demand) == blocks
                    consumer = Consumer(
                        demand=demand.copy(),
                        heat_capacity=physical_properties["HeatCapacity"],  # in J/kg/K
                        max_mass_flow_p=consumer_params["MaxMassFlowPrimary"],
                        surface_area=(
                            consumer_params["SurfaceArea"]
                            *(len(consumer_demands)/np.sum(unclustered_consumer_indices))**0.18
                        ),  # in m^2
                        heat_transfer_q=consumer_params["q"], 
                        heat_transfer_k_max=consumer_params["k"],
                        demand_capacity=np.sum(
                            consumer_demands_capacity[unclustered_consumer_indices]
                        ),
                        min_supply_temp=consumer_params["MinTempSupplyPrimary"],
                        pressure_load=consumer_params["FixPressureLoad"],
                        setpoint_t_supply_s=consumer_params[
                            "SetPointTempSupplySecondary"
                        ],
                        t_return_s=consumer_params["TempReturnSeconary"],
                        energy_unit_conversion=physical_properties[
                            "EnergyUnitConversion"
                        ],
                    )
                    grid.add_node(consumer)
                    nodes_dict[n_idx] = [consumer, consumer]
                    counter_demand += 1
                    cluster_to_consumer_idx[consumer.id] = unclustered_consumer_indices
                else:
                    sup_branch = Branch(
                        blocks, out_slots_number=splits_total_connect_counts[n_idx] - 1
                    )
                    grid.add_node(sup_branch)

                    ret_junction = Junction(
                        blocks, in_slots_number=splits_total_connect_counts[n_idx] - 1
                    )
                    grid.add_node(ret_junction)
                    nodes_dict[n_idx] = [sup_branch, ret_junction]
                    splits_connected_counts[n_idx] = 0

        upstream_node_sup, upstream_node_ret = nodes_dict[upstream_node_idx]
        down_stream_node_sup, down_stream_node_ret = nodes_dict[down_stream_node_idx]

        if upstream_node_idx < len(DP.nodes_list):
            upstream_node_sup_slot = 1
            upstream_node_ret_slot = 0
        else:
            upstream_node_sup_slot = splits_connected_counts[upstream_node_idx] + 1
            upstream_node_ret_slot = splits_connected_counts[upstream_node_idx]
            splits_connected_counts[upstream_node_idx] += 1

        if down_stream_node_idx < len(DP.nodes_list):
            down_stream_node_sup_slot = 0
            down_stream_node_ret_slot = 1
        else:
            down_stream_node_sup_slot = 0
            down_stream_node_ret_slot = (
                splits_total_connect_counts[down_stream_node_idx] - 1
            )

        e_k = pipe_params_loader.get_k(dia)
        e_v = pipe_params_loader.get_vmax(dia)

        sup_edge = Edge(
            blocks=blocks,
            diameter=dia,  # in meters
            length=distance,  # in meters
            thermal_resistance=e_k,  # in k*m/W
            historical_t_in=sup_pipe_params["InitialTemperature"],  # in ºC
            heat_capacity=physical_properties["HeatCapacity"],  # in J/kg/K
            density=physical_properties["Density"],  # in kg/m^3
            t_ground=sup_pipe_params["EnvironmentTemperature"],  # °C
            max_flow_speed=e_v,  # m/s
            min_flow_speed=sup_pipe_params["MinFlowSpeed"],
            friction_coefficient=sup_pipe_params["FrictionCoefficient"],  # (kg*m)^-1
            energy_unit_conversion=physical_properties["EnergyUnitConversion"],
        )
        grid.add_edge(sup_edge)
        sup_edge.link(
            nodes=(
                (upstream_node_sup, upstream_node_sup_slot),
                (down_stream_node_sup, down_stream_node_sup_slot),
            )
        )

        ret_edge = Edge(
            blocks=blocks,
            diameter=dia,  # in meters
            length=distance,  # in meters
            thermal_resistance=e_k,  # in k*m/W
            historical_t_in=ret_pipe_params["InitialTemperature"],  # in ºC
            heat_capacity=physical_properties["HeatCapacity"],  # in J/kg/K
            density=physical_properties["Density"],  # in kg/m^3
            t_ground=ret_pipe_params["EnvironmentTemperature"],  # °C
            max_flow_speed=e_v,  # m/s
            min_flow_speed=ret_pipe_params["MinFlowSpeed"],
            friction_coefficient=ret_pipe_params["FrictionCoefficient"],  # (kg*m)^-1
            energy_unit_conversion=physical_properties["EnergyUnitConversion"],
        )
        grid.add_edge(ret_edge)
        ret_edge.link(
            nodes=(
                (down_stream_node_ret, down_stream_node_ret_slot),
                (upstream_node_ret, upstream_node_ret_slot),
            )
        )

        edges_dict[(upstream_node_idx, down_stream_node_idx)] = [sup_edge, ret_edge]

    grid.link_nodes()
    """
    nodes_dict: dictionary with node_idx as key, value is simulator objects:
    [supply node, return node] (they are the same except for splits)
    edges_dict: dictionary with (upstream_node_idx, down_stream_node_idx) as key,
    value is simulator objects: [supply edge, return edge]
    cluster_to_consumer_idx: the key is consumer id, the value is what consumer idx 
    (of the original 100+) is in this clustered consumer
    """
    return grid, [
        nodes_dict,
        edges_dict,
        cluster_to_consumer_idx,
    ]


if __name__ == "__main__":
    demands = np.random.uniform(low=10, high=60, size=(116, 24)) / 116
    electricity_prices = np.random.uniform(low=10, high=60, size=(1, 24))
    from util import config_vt as config

    build_grid_clustered_short(
        5,
        demands,
        electricity_prices,
        config,
    )

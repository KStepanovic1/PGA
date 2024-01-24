import numpy as np

from ..models import Grid, Producer, Consumer, Edge, CHP
from ..models import Branch, Junction
from .vattenfall.data_processing_ams import DataProcessor as DP
from ..models.interpolation_heat_exchanger import generate_values
from ..models.heat_exchanger import HeatExchanger

def get_consumers_scale(config):
    consumers_scale = {}
    pipe_params = config.PipePreset1
    for e_idx, e_n_con in enumerate(DP.edge_to_node_connection):
        e_dia = DP.edges_dia[e_idx] / 1000

        upstream_node_idx, down_stream_node_idx = e_n_con

        # skip edges to the actual producer
        if upstream_node_idx in DP.producers_indices:
            continue
        # skip (supply) edge that is downstream of a empty node
        # because it will be processed along with its upstream edge
        elif upstream_node_idx >= len(DP.nodes_list) + len(DP.splits_coo):
            continue

        # combine edges that are connected through empty nodes into one edge
        up_e_idx = e_idx
        while down_stream_node_idx >= len(DP.nodes_list) + len(DP.splits_coo):
            down_e_idx = DP.edges_connected_through_empty_nodes.get(up_e_idx)
            assert down_e_idx is not None
            _, down_stream_node_idx = DP.edge_to_node_connection[down_e_idx]
            up_e_idx = down_e_idx

        # if downstream is consumer, give pipe dia param to this consumer
        if (
            (down_stream_node_idx < len(DP.nodes_list))
            & (down_stream_node_idx not in DP.storage_indices)
            & (down_stream_node_idx not in DP.producers_indices)
        ):
            consumers_scale[down_stream_node_idx] = e_dia ** 2

    consumers_scale = np.array(list(consumers_scale.values()))[np.argsort(list(consumers_scale.keys()))]
    consumers_scale = consumers_scale/np.sum(consumers_scale)

    return consumers_scale

def build_grid_short_or_long(
    consumer_demands,
    consumer_demands_capacity, # design demand capacity
    electricity_prices,
    is_short,
    config,
):
    """
    In this function the origin storage is treated as the producer
    while the long pipe from the acutal producer to the storage is
    removed
    """
    assert len(consumer_demands) == len(DP.nodes_list) - len(
        DP.producers_indices
    ) - len(DP.storage_indices)

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

    values = generate_values(consumer_params["SetPointTempSupplySecondary"], consumer_params["TempReturnSeconary"],
                            HeatExchanger(physical_properties["HeatCapacity"], consumer_params["MaxMassFlowPrimary"],
                                        consumer_params["SurfaceArea"], consumer_params["q"], consumer_params["k"]),
                            "heat_exchanger_values.txt", [80, 110], [0, 30])

    def add_chp():
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
            energy_unit_conversion=physical_properties["EnergyUnitConversion"],
        )
        grid.add_node(producer)  # characterized with temperature
        nodes_heat.append(producer)
        nodes_heat_ids.append(producer.id)

   


    # nodes that can change heat: producer, consumer, storage
    nodes_heat = []
    nodes_heat_ids = []
    nodes_4_way_exchanger = []
    nodes_4_way_exchanger_id = []
    counter_demand = 0

    for n_idx, node_data in enumerate(DP.nodes_list):
        if n_idx in DP.producers_indices:
            if is_short:
                nodes_heat.append(None)
                nodes_heat_ids.append(None)
            else:
                add_chp()

        elif n_idx in DP.storage_indices:
            if is_short:
                add_chp()
            else:
                nodes_heat.append(None)
                nodes_heat_ids.append(None)
                empty_node1 = Branch(blocks, out_slots_number=1)
                empty_node2 = Junction(blocks, in_slots_number=1)
                grid.add_node(empty_node1)
                grid.add_node(empty_node2)
                nodes_4_way_exchanger.append(empty_node1)
                nodes_4_way_exchanger.append(empty_node2)
                nodes_4_way_exchanger_id.append(empty_node1.id)
                nodes_4_way_exchanger_id.append(empty_node2.id)

        else:
            demand = consumer_demands[counter_demand]
            consumer = Consumer(
                demand=demand.copy(),
                heat_capacity=physical_properties["HeatCapacity"],  # in J/kg/K
                max_mass_flow_p=consumer_params["MaxMassFlowPrimary"],
                surface_area=consumer_params["SurfaceArea"],  # in m^2
                heat_transfer_q=consumer_params["q"],
                heat_transfer_k_max=consumer_params["k"],
                demand_capacity=consumer_demands_capacity[counter_demand],
                min_supply_temp=consumer_params["MinTempSupplyPrimary"],
                pressure_load=consumer_params["FixPressureLoad"],
                setpoint_t_supply_s=consumer_params["SetPointTempSupplySecondary"],
                t_return_s=consumer_params["TempReturnSeconary"],
                energy_unit_conversion=physical_properties["EnergyUnitConversion"],
                interpolation_values=values
            )
            grid.add_node(consumer)
            nodes_heat.append(consumer)
            nodes_heat_ids.append(consumer.id)
            counter_demand += 1

    branches = []
    branches_ids = []
    junctions = []
    junctions_ids = []
    for n_idx, n_e_con in enumerate(DP.split_edge_connections):
        slots_num = len(n_e_con)
        sup_branch = Branch(blocks, out_slots_number=slots_num - 1)
        grid.add_node(sup_branch)
        branches.append(sup_branch)
        branches_ids.append(sup_branch.id)

        ret_junction = Junction(blocks, in_slots_number=slots_num - 1)
        grid.add_node(ret_junction)
        junctions.append(ret_junction)
        junctions_ids.append(ret_junction.id)

    # count how many outlet of branch/inlet of split has been connected
    splits_connected_counts = np.zeros(len(branches_ids))
    sup_edges_ids = {}
    ret_edges_ids = {}
    consumers_scale = {}
    pipe_params = config.PipePreset1
    for e_idx, e_n_con in enumerate(DP.edge_to_node_connection):
        e_dia = DP.edges_dia[e_idx] / 1000
        e_len = DP.edges_len[e_idx]
        e_k = DP.edges_K[e_idx]
        e_v = DP.edges_Vmax[e_idx]

        upstream_node_idx, down_stream_node_idx = e_n_con
        # skip edges to the actual producer
        if (upstream_node_idx in DP.producers_indices) & is_short:
            continue
        # skip (supply) edge that is downstream of a empty node
        # because it will be processed along with its upstream edge
        elif upstream_node_idx >= len(DP.nodes_list) + len(DP.splits_coo):
            continue

        # combine edges that are connected through empty nodes into one edge
        up_e_idx = e_idx
        edges_indices = [e_idx]
        while down_stream_node_idx >= len(DP.nodes_list) + len(DP.splits_coo):
            down_e_idx = DP.edges_connected_through_empty_nodes.get(up_e_idx)
            assert down_e_idx is not None
            _, down_stream_node_idx = DP.edge_to_node_connection[down_e_idx]
            e_len += DP.edges_len[down_e_idx]
            up_e_idx = down_e_idx
            edges_indices.append(down_e_idx)

        # if downstream is consumer, give pipe dia param to this consumer
        if (
            (down_stream_node_idx < len(DP.nodes_list))
            & (down_stream_node_idx not in DP.storage_indices)
            & (down_stream_node_idx not in DP.producers_indices)
        ):
            consumers_scale[down_stream_node_idx] = e_dia ** 2

        def get_nodes_with_idx(idx, upstream=True):
            if (idx < len(DP.nodes_list)) & ((is_short) | (idx not in DP.storage_indices)):
                return [nodes_heat[idx], upstream], [nodes_heat[idx], 1 - upstream]
            elif (idx in DP.storage_indices) & (not is_short):
                return [nodes_4_way_exchanger[0], upstream], [nodes_4_way_exchanger[1], 1-upstream]
            elif idx < len(DP.nodes_list) + len(DP.splits_coo):
                idx = idx - len(DP.nodes_list)
                if upstream:
                    splits_connected_counts[idx] += 1
                    return (
                        [branches[idx], splits_connected_counts[idx]],
                        [junctions[idx], splits_connected_counts[idx]],
                    )
                else:
                    return (
                        [branches[idx], 0],
                        [junctions[idx], 0],
                    )

            raise Exception

        ([up_node_sup, up_slot_sup], [up_node_ret, up_slot_ret]) = get_nodes_with_idx(
            upstream_node_idx, upstream=True
        )
        (
            [down_node_sup, down_slot_sup],
            [down_node_ret, down_slot_ret],
        ) = get_nodes_with_idx(down_stream_node_idx, upstream=False)

        sup_edge = Edge(
            blocks=blocks,
            diameter=e_dia,  # in meters
            length=e_len,  # in meters
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
                (up_node_sup, int(up_slot_sup)),
                (down_node_sup, int(down_slot_sup)),
            )
        )

        for e_idx in edges_indices:
            sup_edges_ids[e_idx] = sup_edge.id

        ret_edge = Edge(
            blocks=blocks,
            diameter=e_dia,  # in meters
            length=e_len,  # in meters
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
                (down_node_ret, int(down_slot_ret)),
                (up_node_ret, int(up_slot_ret)),
            )
        )
        for e_idx in edges_indices:
            ret_edges_ids[e_idx] = ret_edge.id

    grid.link_nodes()
    # the "UNIT_ids" does a mapping from index of unit (consumers, splits, edges) in the DataProcessor to the id of them in the simulator
    # consumers_scale gives sizes of the consumers. Heat demand should be propotional to that
    return grid, [
        nodes_heat_ids,
        branches_ids,
        junctions_ids,
        sup_edges_ids,
        ret_edges_ids,
        nodes_4_way_exchanger_id,
        consumers_scale,
    ]


if __name__ == "__main__":
    demands = np.random.uniform(low=10, high=60, size=(118 - 2, 24*4)) / (118 - 2)
    electricity_prices = np.random.uniform(low=10, high=60, size=(1, 24*4))
    from util import config_vt as config

    consumers_scale = get_consumers_scale(config)
    consumer_demands_capacity = 60*consumers_scale
    build_grid_short_or_long(
        demands,
        consumer_demands_capacity,
        electricity_prices, 
        False, 
        config
    )

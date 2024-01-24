'''
grid with 2 producers and one consumer
p1 mass: 3097963 kg
p1 length: 5057 m
p1 diameter: 0.9
total mass: 3868228 kg
avg distance to first branch: 3343 meter
assumed c diameter: 0.556
HX q, a, k: 0.78, 460, 40
'''
q, a, k = 0.78, 1000, 40
consumer_pipe_dia = 0.556
consumer_pipe_len = 3343
producer_pipe_dia = 0.9
producer1_pipe_len = 5057
producer2_pipe_len = 500

from ...models import Grid, Producer, Consumer, Edge, CHP
from ...models import Branch, Junction
from ..vattenfall.data_processing_ams import Pipe_params_loader


def build_grid(
    consumer_demands,
    electricity_prices,
    config,
):
    """
    Building the grid object with two producers.
    """
    pipe_params_loader = Pipe_params_loader()
    producer_pipe_vmax, consumer_pipe_vmax = pipe_params_loader.get_vmax([producer_pipe_dia, consumer_pipe_dia])
    producer_pipe_k, consumer_pipe_k = pipe_params_loader.get_k([producer_pipe_dia, consumer_pipe_dia])

    producer_params = config.ProducerPreset1
    consumer_params = config.ConsumerPreset1
    sup_main_pipe_params = config.PipePreset1
    ret_main_pipe_params = config.PipePreset2
    sup_side_pipe_params = config.PipePreset3
    ret_side_pipe_params = config.PipePreset4
    physical_properties = config.PhysicalProperties
    time_params = config.TimeParameters

    blocks = consumer_demands[0].shape[0]

    grid = Grid(  # empty grid
        interval_length=time_params["TimeInterval"],  # 60 min
    )

    producers = []
    for _ in range(2):
        if producer_params["Type"] == "CHP":
            producer = CHP(
                CHPPreset=producer_params["Parameters"],
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
            producers.append(producer)
        else:
            raise Exception("Producer type not implemented")

    ret_branch = Branch(blocks, out_slots_number=2)
    grid.add_node(ret_branch)
    sup_junction = Junction(blocks, in_slots_number=2)
    grid.add_node(sup_junction)

    consumer = Consumer(
        demand=consumer_demands[0],
        heat_capacity=physical_properties["HeatCapacity"],  # in J/kg/K
        max_mass_flow_p=consumer_params["MaxMassFlowPrimary"],
        surface_area=a,  # in m^2
        heat_transfer_q=q,  # See Palsson 1999 p45
        heat_transfer_k=k,  # See Palsson 1999 p51
        min_supply_temp=consumer_params["MinTempSupplyPrimary"],
        pressure_load=consumer_params["FixPressureLoad"],
        setpoint_t_supply_s=consumer_params["SetPointTempSupplySecondary"],
        t_return_s=consumer_params["TempReturnSeconary"],
        energy_unit_conversion=physical_properties["EnergyUnitConversion"],
    )
    grid.add_node(consumer)


    edges = []
    """
    Add the main supply and return edge. The main supply edge connects Producer and Branch.
    The main return edge connects Junction and Producer.
    """
    for p, length in zip(producers, [producer1_pipe_len, producer2_pipe_len]):
        for pipe_params in [sup_main_pipe_params, ret_main_pipe_params]:
            edge = Edge(
                blocks=blocks,
                diameter=producer_pipe_dia,  # in meters
                length=length,  # in meters
                thermal_resistance=producer_pipe_k,  # in k*m/W
                historical_t_in=pipe_params["InitialTemperature"],  # in ºC
                heat_capacity=physical_properties["HeatCapacity"],  # in J/kg/K
                density=physical_properties["Density"],  # in kg/m^3
                t_ground=pipe_params["EnvironmentTemperature"],  # °C
                max_flow_speed=producer_pipe_vmax,  # m/s
                min_flow_speed=pipe_params["MinFlowSpeed"],
                friction_coefficient=pipe_params["FrictionCoefficient"],  # (kg*m)^-1
                energy_unit_conversion=physical_properties["EnergyUnitConversion"],
            )
            edges.append(edge)
            grid.add_edge(edge)

    edges[0].link(
        nodes=(
            (producers[0], 1),
            (sup_junction, 1),
        )
    )
    edges[1].link(
        nodes=(
            (ret_branch, 1),
            (producers[0], 0),
        )
    )
    edges[2].link(
        nodes=(
            (producers[1], 1),
            (sup_junction, 2),
        )
    )
    edges[3].link(
        nodes=(
            (ret_branch, 2),
            (producers[1], 0),
        )
    )

    for pipe_params in [sup_side_pipe_params, ret_side_pipe_params]:
        edge = Edge(
            blocks=blocks,
            diameter=consumer_pipe_dia,  # in meters
            length=consumer_pipe_len,  # in meters
            thermal_resistance=consumer_pipe_k,  # in k*m/W
            historical_t_in=pipe_params["InitialTemperature"],  # in ºC
            heat_capacity=physical_properties["HeatCapacity"],  # in J/kg/K
            density=physical_properties["Density"],  # in kg/m^3
            t_ground=pipe_params["EnvironmentTemperature"],  # °C
            max_flow_speed=consumer_pipe_vmax,  # m/s
            min_flow_speed=pipe_params["MinFlowSpeed"],
            friction_coefficient=pipe_params["FrictionCoefficient"],  # (kg*m)^-1
            energy_unit_conversion=physical_properties["EnergyUnitConversion"],
        )
        edges.append(edge)
        grid.add_edge(edge)

    edges[4].link(
        nodes=(
            (sup_junction, 0),
            (consumer, 0),
        )
    )
    edges[5].link(
        nodes=(
            (consumer, 1),
            (ret_branch, 0),
        )
    )

    grid.link_nodes(False)

    return grid

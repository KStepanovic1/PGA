from gurobipy import *
import numpy as np
import os
import pandas as pd
from pathlib import *
from datetime import datetime
from pathlib import Path
import copy

from ...simulator.cases.one_consumer import build_grid
from ...simulator.models.grid import Grid
from .linearization import Linearization
from util import config, math_functions
from ..baseline_strategy.constant_temp import Optimizer as ConstantTempOpt

DEBUG_MODE = True

MIN_TEMP_INCREASE = 2
CONSUMER_SUPPLY_TEMP_SAFTY_MARGIN = 2

DP_PMP_MIN = 0
DP_C_MIN = 100
PUMP_POWER_MIN = 0

MIN_MASS_FLOW = 0

# convergence
delta_T = 4  # [C]


PLANNING_HORIZON = config.TimeParameters["PlanningHorizon"]  # [h]
ACTION_HORIZON = config.TimeParameters["ActionHorizon"]  # [h]
TIME_INTERVAL = config.TimeParameters["TimeInterval"]  # [s]
MW_W = config.PhysicalProperties["EnergyUnitConversion"]
GENERATORS = [g for g in config.ProducerPreset1["Generators"]]
# heat and electricity production cost
FUEL_PRICE = [[x["FuelCost"] for x in GENERATORS]]
# start up cost
STARTUP_COST = [[x["StartupCost"] for x in GENERATORS]]
INITIAL_STATUS = [[x["InitialStatus"] for x in GENERATORS]]
# efficiency
EFF_GEN = [[x["Efficiency"] for x in GENERATORS]]
EFF_PMP = config.ProducerPreset1["PumpEfficiency"]
# temperature
T_SUP_MAX = config.PhysicalProperties["MaxTemp"]
T_SUP_MIN = config.PhysicalProperties["MinSupTemp"] + MIN_TEMP_INCREASE
T_SUP_MIN_SIM = config.PhysicalProperties["MinSupTemp"]
T_ENV = config.PipePreset1["EnvironmentTemperature"]
HIS_T_SUP = config.PipePreset1["InitialTemperature"]
HIS_T_RET = config.PipePreset2["InitialTemperature"]
# differential pressure
DP_C_MAX = config.ConsumerPreset1["FixPressureLoad"]
NOM_DP = DP_C_MAX
# mass flow, flow speed
MAX_MASS_FLOW = config.ConsumerPreset1["MaxMassFlowPrimary"]
MAX_FLOW_SPD = config.PipePreset1["MaxFlowSpeed"]
MIN_FLOW_SPD = config.PipePreset1["MinFlowSpeed"]
NOM_MASS_FLOW_C = MAX_MASS_FLOW
# district heating network parameters
PIPE_LEN = config.PipePreset1["Length"]
PIPE_DIAMETER = config.PipePreset1["Diameter"]
CHP_KEY_PTS = [
    [math_functions.points_sort_clockwise(x["OperationRegion"]) for x in GENERATORS]
]

# power [MWh]
Q_MAX = [
    [max(ckp_per_unit[:, 0]) for ckp_per_unit in ckp_per_plant]
    for ckp_per_plant in CHP_KEY_PTS
]
Q_MIN = [
    [min(ckp_per_unit[:, 0]) for ckp_per_unit in ckp_per_plant]
    for ckp_per_plant in CHP_KEY_PTS
]
E_MAX = [
    [max(ckp_per_unit[:, 1]) for ckp_per_unit in ckp_per_plant]
    for ckp_per_plant in CHP_KEY_PTS
]
E_MIN = [
    [min(ckp_per_unit[:, 1]) for ckp_per_unit in ckp_per_plant]
    for ckp_per_plant in CHP_KEY_PTS
]
Q_MAX = np.sum(Q_MAX)
Q_MIN = np.sum(Q_MIN)
E_MAX = np.sum(E_MAX)
E_MIN = np.sum(E_MIN)


# physic constraints
HEAT_CAPACITY = config.PhysicalProperties["HeatCapacity"]
DENSITY = config.PhysicalProperties["Density"]
FRICTION_COEFF = (
    config.PipePreset1["FrictionCoefficient"] * NOM_MASS_FLOW_C
)  # [Pa/(kg/s)] !!! DIMENSION SHOULD BE EQUAL TO NUMBER OF CONSUMERS

MASS_IN_PIPE = (np.pi * PIPE_DIAMETER ** 2) / 4 * PIPE_LEN * DENSITY

# heat_demand = np.random.uniform(
#     10 * 10 ** 6, 70 * 10 ** 6, size=24
# )  # heat demand !!! DIMENSION SHOULD BE EQUAL TO NUMBER OF CONSUMERS
# piece-wise linear function
# When changing L, code in linearization.py has to be changed as well
L = 5
COEFF_LINEAR = 1 / ((HEAT_CAPACITY * NOM_MASS_FLOW_C) / math.sqrt(NOM_DP))

# current time as string
now = datetime.now().strftime("%H_%M_%S")

DATA_PATH = Path(__file__).parents[3] / "data"
DATA_PATH_STORE = (
    Path(__file__).parents[3] / "results/giraud_2017/rolling_horizon_dataset_reverse"
)


def get_control_split_id(simulator):
    ids = []
    for p in simulator.producers:
        split, _ = p.edges[0].nodes[0]
        ids.append(split.id)
        split, _ = p.edges[1].nodes[1]
        ids.append(split.id)

    return np.unique(ids)


class SimulatorEnhancedInterface(Grid):
    def run_sei(
        self,
        temp,
        electricity,
        # valve_pos
    ):
        super(SimulatorEnhancedInterface, self).run(
            temp=temp,
            electricity=electricity,
            # valve_pos=valve_pos,
        )
        node_out = self.get_object_status(
            None,
            get_temp=True,
            get_ms=True,
            get_pressure=True,
            get_violation=True,
        )

        sup_edge_ids = [e.id for e in self.edges if e.is_supply]
        edge_out = self.get_object_status(
            sup_edge_ids,
            get_temp=True,
            get_ms=True,
        )

        pipe_heat_out = self.get_edge_heat_and_loss(
            level=1,
            level_time=1,
        )
        pipe_plug_states = self.get_pipe_states()
        self.ms_node_sei = {
            c_id: node_out[c_id]["Mass flow"][0, :] for c_id in self.consumers_id
        }
        self.ms_node_sei.update(
            {p_id: node_out[p_id]["Mass flow"][0, :] for p_id in self.producers_id}
        )
        self.ms_edge_sei = {
            e_id: edge_out[e_id]["Mass flow"][0, :] for e_id in sup_edge_ids
        }

        self.delay_matrix_sei = {
            edge.id: edge.delay_matrix for edge in self.edges if edge.is_supply
        }

        self.min_sup_temp_sei = {c.id: c.minimum_t_supply_p for c in self.consumers}

        self.heat_loss_ret_sei = pipe_heat_out["return"][1]
        self.T_c_sei = {c.id: node_out[c.id]["Temp"] for c in self.consumers}
        self.diff_pressure_production_plant_sei = np.diff(
            node_out[self.producers_id[0]]["Pressure"], axis=0
        )[0]
        self.T_sup_sei = np.array(
            [node_out[p_id]["Temp"][1, :] for p_id in self.producers_id]
        )
        self.T_ret_sei = np.array(
            [node_out[p_id]["Temp"][0, :] for p_id in self.producers_id]
        )
        self.Q_sei = np.array([p.q for p in self.producers])
        self.his_T_sup_sei = {
            edge.id: [plug.entry_temp for plug in edge.initial_plug_cache][::-1]
            for edge in self.edges
            if edge.is_supply
        }

    @property
    def connection_list(self):
        return [
            [
                edge.nodes[0][0].id,
                edge.nodes[1][0].id,
                edge.id,
                edge._thermal_time_constant,
            ]
            for edge in self.edges
            if edge.is_supply
        ]

    @property
    def connection_list_splits(self):
        connection_list = []
        for node in self.nodes:
            if node.is_supply & (type(node).__name__ == "Branch"):
                connection_list.append(
                    [[node.edges[0].id], [e.id for e in node.edges[1:]]]
                )
            elif node.is_supply & (type(node).__name__ == "Junction"):
                connection_list.append(
                    [[e.id for e in node.edges[:-1]], [node.edges[-1].id]]
                )

        return connection_list


def model(
    heat_demand,
    electricity_price,
    connection_list,  # [upstream node id, downstream node id, edge id, thermal constant]
    connection_list_splits,  # [upstream node id, downstream node id]
    nodes_id_to_idx,
    edges_id_to_idx,
    producers_indices,
    production_unit_indices,
    consumers_indices,
    previous_Y,
    _ms_node,
    _ms_edge,
    _delay_matrices,
    _min_sup_temp,
    _min_sup_temp_producer,
    _temp_consumer,
    _dP_pmp_sim,
    _T_sup,
    _T_ret,
    _Q,
    _his_T_sup,
    iter,
):

    lin = Linearization(DP_C_MIN, DP_C_MAX, L)
    xpts = list(lin.p[: L + 1])
    ypts = list(lin.p[L + 1 :])
    his_blocks = {key: len(value) for key, value in _his_T_sup.items()}

    if _min_sup_temp_producer is None:
        _min_sup_temp_producer = np.ones(PLANNING_HORIZON) * (
            np.max(list(_min_sup_temp.values())) + 6
        )

    m = Model("MILP")

    Q = m.addVars(
        len(producers_indices),
        PLANNING_HORIZON,
        name="heat production",
        lb=Q_MIN,
        ub=Q_MAX,
    )
    E = m.addVars(
        len(producers_indices),
        PLANNING_HORIZON,
        name="electricity production",
        lb=E_MIN,
        ub=E_MAX,
    )
    T_sup = m.addVars(  # (2)
        len(producers_indices),
        PLANNING_HORIZON,
        name="supply temperature",
        lb=T_SUP_MIN,
        ub=T_SUP_MAX,
    )

    T_node_sup = m.addVars(
        len(nodes_id_to_idx),
        PLANNING_HORIZON,
        name="temperature supply nodes",
        lb=T_SUP_MIN - MIN_TEMP_INCREASE,
        ub=T_SUP_MAX,
    )

    # penalty for consumer supply temperature slightly smaller than allowed
    error_consumer_t = m.addVars(
        len(nodes_id_to_idx),
        PLANNING_HORIZON,
        name="temperature supply nodes",
        lb=0,
        ub=CONSUMER_SUPPLY_TEMP_SAFTY_MARGIN,
    )

    T_edge_sup = m.addVars(
        len(edges_id_to_idx),
        2,
        PLANNING_HORIZON,
        name="temperature supply edges",
        lb=T_SUP_MIN - MIN_TEMP_INCREASE,
        ub=T_SUP_MAX,
    )

    ms_edge = m.addVars(
        len(edges_id_to_idx),
        PLANNING_HORIZON,
        name="edge mass flow",
    )

    ms_node = m.addVars(
        len(nodes_id_to_idx),
        PLANNING_HORIZON,
        name="node mass flow",
        lb=MIN_MASS_FLOW,
        ub=MAX_MASS_FLOW,
    )

    ms_p = m.addVars(
        len(producers_indices),
        PLANNING_HORIZON,
        name="producer mass flow",
        lb=MIN_MASS_FLOW,
        ub=MAX_MASS_FLOW,
    )

    ms_c_auxiliary = m.addVars(
        len(nodes_id_to_idx),
        PLANNING_HORIZON,
        name="consumer mass flow auxiliary",
        lb=-99999,
    )
    ms_c_auxiliary2 = m.addVars(
        len(nodes_id_to_idx),
        PLANNING_HORIZON,
        name="consumer mass flow auxiliary 2",
    )

    dP_pmp = m.addVars(  # (3)
        len(producers_indices),
        PLANNING_HORIZON,
        name="differencial pressure at the production plant",
        lb=DP_PMP_MIN,
    )
    dP_c = m.addVars(
        len(nodes_id_to_idx),
        PLANNING_HORIZON,
        name="pressure difference consumers",
        lb=DP_C_MIN,
        # ub=DP_C_MAX,
    )

    MW_pmp = m.addVars(
        len(producers_indices), PLANNING_HORIZON, name="pump power", lb=PUMP_POWER_MIN
    )

    temp_producer_constraint = m.addConstrs(
        (
            T_node_sup[producer_idx, t] == T_sup[i, t]
            for t in range(PLANNING_HORIZON)
            for i, producer_idx in enumerate(producers_indices)
        ),
        name="producer supply temp constraint",
    )

    for upstream_node_id, downstream_node_id, edge_id, thermal_const in connection_list:
        upstream_node_idx = nodes_id_to_idx[upstream_node_id]
        downstream_node_idx = nodes_id_to_idx[downstream_node_id]
        edge_idx = edges_id_to_idx[edge_id]

        temp_edge_constraint = m.addConstrs(  # (13)
            (
                T_edge_sup[edge_idx, 1, t]
                == quicksum(
                    [
                        _delay_matrices[edge_id][t, t2 + his_blocks[edge_id]]
                        * (
                            T_ENV
                            + (T_edge_sup[edge_idx, 0, t2] - T_ENV)
                            * np.exp(-(t - t2) / thermal_const * TIME_INTERVAL)
                        )
                        for t2 in range(t + 1)
                    ]
                )
                + quicksum(
                    _delay_matrices[edge_id][t, t2]
                    * (
                        T_ENV
                        + (_his_T_sup[edge_id][t2] - T_ENV)
                        * np.exp(
                            -(t + his_blocks[edge_id] - t2)
                            / thermal_const
                            * TIME_INTERVAL
                        )
                    )
                    for t2 in range(his_blocks[edge_id])
                )
                for t in range(PLANNING_HORIZON)
            ),
            name="inlet-outlet(edge id:{}) temp constraint".format(edge_id),
        )

        if downstream_node_idx in consumers_indices:
            consumer_temp_equality_constraint = m.addConstrs(
                (
                    T_node_sup[downstream_node_idx, t] == T_edge_sup[edge_idx, 1, t]
                    for t in range(PLANNING_HORIZON)
                ),
                name="consumer(id:{})-edge outlet(id:{}) temp equal constraint".format(
                    downstream_node_id, edge_id
                ),
            )

            mass_flow_temp_dependency_constraint_auxiliary = m.addConstrs(
                ms_c_auxiliary[downstream_node_idx, t]
                == (
                    1
                    - (
                        T_node_sup[downstream_node_idx, t]
                        - _temp_consumer[downstream_node_id][0, t]
                    )
                    / (
                        _temp_consumer[downstream_node_id][0, t]
                        - _temp_consumer[downstream_node_id][1, t]
                    )
                )
                for t in range(PLANNING_HORIZON)
            )

            mass_flow_temp_dependency_constraint_auxiliary2 = m.addConstrs(
                ms_c_auxiliary2[downstream_node_idx, t]
                == max_(ms_c_auxiliary[downstream_node_idx, t], 0.1)
                for t in range(PLANNING_HORIZON)
            )

            mass_flow_temp_dependency_constraint = m.addConstrs(  # (?)
                ms_node[downstream_node_idx, t]
                == _ms_node[downstream_node_id][t]
                * ms_c_auxiliary2[downstream_node_idx, t]
                for t in range(PLANNING_HORIZON)
            )

            consumer_satisfaction_guarantee_constraint = m.addConstrs(  # (11)
                (
                    T_node_sup[downstream_node_idx, t]
                    >= _min_sup_temp[downstream_node_id][t]
                    + CONSUMER_SUPPLY_TEMP_SAFTY_MARGIN
                    - error_consumer_t[downstream_node_idx, t]
                    for t in range(PLANNING_HORIZON)
                ),
                name="consumer_satisfaction_guarantee_constraint_backup",
            )

            mass_flow_consumer_edge_equality_constraint = m.addConstrs(
                (
                    ms_node[downstream_node_idx, t] == ms_edge[edge_idx, t]
                    for t in range(PLANNING_HORIZON)
                ),
                name="mass_flow_consumer_edge_equality_constraint",
            )

        if upstream_node_idx in producers_indices:
            producer_temp_equality_constraint = m.addConstrs(
                (
                    T_node_sup[upstream_node_idx, t] == T_edge_sup[edge_idx, 0, t]
                    for t in range(PLANNING_HORIZON)
                ),
                name="producer(id:{})-edge inlet(id:{}) temp equal constraint".format(
                    upstream_node_idx, edge_id
                ),
            )

            # temp_min_constraint = m.addConstrs(
            #     (
            #         T_node_sup[upstream_node_idx, t] >= _min_sup_temp_producer[t]
            #         for t in range(1, PLANNING_HORIZON)
            #     ),
            #     name="supply temperature minimum constraint"
            # )
            temp_ramp_constraint1 = m.addConstrs(
                (
                    T_node_sup[upstream_node_idx, t]
                    - T_node_sup[upstream_node_idx, t - 1]
                    <= 5
                    for t in range(1, PLANNING_HORIZON)
                ),
                name="supply temperature ramp constraint 1",
            )

            temp_ramp_constraint2 = m.addConstrs(
                (
                    T_node_sup[upstream_node_idx, t]
                    - T_node_sup[upstream_node_idx, t - 1]
                    >= -5
                    for t in range(1, PLANNING_HORIZON)
                ),
                name="supply temperature ramp constraint 2",
            )

            temp_ramp_constraint1_init = m.addConstr(
                T_node_sup[upstream_node_idx, 0] - _his_T_sup[edge_id][-1] <= 5,
                name="supply temperature ramp constraint 1 init",
            )

            mass_flow_producer_edge_equality_constraint = m.addConstrs(
                (
                    ms_node[upstream_node_idx, t] == ms_edge[edge_idx, t]
                    for t in range(PLANNING_HORIZON)
                ),
                name="mass_flow_consumer_edge_equality_constraint",
            )

            # this is only for easier exporting variables
            mass_flow_producer_to_node = m.addConstrs(
                (
                    ms_node[upstream_node_idx, t]
                    == ms_p[
                        np.where(np.array(producers_indices) == upstream_node_idx)[0][
                            0
                        ],
                        t,
                    ]
                    for t in range(PLANNING_HORIZON)
                ),
                name="transfer_node_mass_flow_to_producer",
            )

    for upstream_edges_ids, downstream_edges_ids in connection_list_splits:
        upstream_edge_indices = [edges_id_to_idx[e_id] for e_id in upstream_edges_ids]
        downstream_edge_indices = [
            edges_id_to_idx[e_id] for e_id in downstream_edges_ids
        ]
        mass_flow_split_continuity_constraint = m.addConstrs(
            (
                quicksum(ms_edge[e_idx, t] for e_idx in upstream_edge_indices)
                == quicksum(ms_edge[e_idx, t] for e_idx in downstream_edge_indices)
                for t in range(PLANNING_HORIZON)
            ),
            name="mass_flow_consumer_edge_equality_constraint",
        )
        temp_split_continuity_constraint = m.addConstrs(
            (
                quicksum(
                    _ms_edge[e_id][t] * T_edge_sup[e_idx, 1, t]
                    for e_id, e_idx in zip(upstream_edges_ids, upstream_edge_indices)
                )
                == quicksum(
                    _ms_edge[e_id][t] * T_edge_sup[e_idx, 0, t]
                    for e_id, e_idx in zip(
                        downstream_edges_ids, downstream_edge_indices
                    )
                )
                for t in range(PLANNING_HORIZON)
            ),
            name="split_temp_merge_constraint",
        )

        downstream_edge_same_temp_constraint = m.addConstrs(
            (
                T_edge_sup[downstream_edge_indices[j], 0, t]
                == T_edge_sup[downstream_edge_indices[j + 1], 0, t]
                for t in range(PLANNING_HORIZON)
                for j in range(len(downstream_edges_ids) - 1)
            ),
            name="downstream_edge_same_temp_constraint",
        )

    heat_balance_constraint = m.addConstrs(  # (17)
        (
            (
                quicksum(Q[i, t] for i in range(len(producers_indices)))
                - np.sum(_Q[:, t])
            )
            == (
                quicksum(
                    _ms_node[p_id][t]
                    * HEAT_CAPACITY
                    * (T_node_sup[p_idx, t] - _T_sup[p_id, t])
                    for p_id, p_idx in nodes_id_to_idx.items()
                    if p_idx in producers_indices
                )
                - quicksum(
                    [
                        _ms_node[c_id][t]
                        * HEAT_CAPACITY
                        * (T_node_sup[c_idx, t] - _temp_consumer[c_id][0, t])
                        for c_id, c_idx in nodes_id_to_idx.items()
                        if c_idx in consumers_indices
                    ]
                )
            )
            / MW_W
            for t in range(PLANNING_HORIZON)
        ),
        name="heat balance constraint",
    )

    m.setObjective(  # (1)
        (
            quicksum(
                Q[k, t]
                for t in range(PLANNING_HORIZON)
                for k in range(len(producers_indices))
            )
            + quicksum(
                error_consumer_t[j, t] * 1000
                for t in range(PLANNING_HORIZON)
                for j in range(len(nodes_id_to_idx))
            )
        ),
        GRB.MINIMIZE,
    )
    # m.setObjective(  # (1)
    #     (
    #         quicksum(
    #             Q[t]*FUEL_PRICE[0]
    #             for t in range(PLANNING_HORIZON)
    #         )
    #     ),
    #     GRB.MINIMIZE,
    # )
    m.setParam("OutputFlag", 0)

    m.optimize()

    # m.write("model%s.lp"%iter)

    def export_variables():

        T_sup_optimized = []
        Q_optimized = []
        Y_optimized = []
        X_optimized = []
        E_optimized = []
        # valve_pos_optimized = []
        MW_pmp_optimized = []
        T_node_sup_opt = []
        margin_operation = []
        margin_startup = []

        for v in T_sup.values():
            T_sup_optimized.append(v.X)

        T_sup_optimized = np.array(T_sup_optimized).reshape(
            (len(producers_indices)), PLANNING_HORIZON
        )

        for v in Q.values():
            Q_optimized.append(v.X)

        Q_optimized = np.array(Q_optimized).reshape(
            len(producers_indices), PLANNING_HORIZON
        )

        for v in E.values():
            E_optimized.append(v.X)

        E_optimized = np.array(E_optimized).reshape(
            len(producers_indices), PLANNING_HORIZON
        )

        ms_p_optimized = []
        for v in ms_p.values():
            ms_p_optimized.append(v.X)

        ms_p_optimized = np.array(ms_p_optimized).reshape(
            (len(producers_indices)), PLANNING_HORIZON
        )

        # valve_pos_optimized = ms_p_optimized/np.sum(ms_p_optimized, axis=0)

        for v in MW_pmp.values():
            MW_pmp_optimized.append(v.X)

        MW_pmp_optimized = np.array(MW_pmp_optimized).reshape(
            (len(producers_indices)), PLANNING_HORIZON
        )

        for v in T_node_sup.values():
            T_node_sup_opt.append(v.X)

        T_node_sup_opt = np.array(T_node_sup_opt).reshape(
            (len(nodes_id_to_idx)), PLANNING_HORIZON
        )

        T_edge_sup_opt = []
        for v in T_edge_sup.values():
            T_edge_sup_opt.append(v.X)

        T_edge_sup_opt = np.array(T_edge_sup_opt).reshape(
            (len(edges_id_to_idx)), 2, PLANNING_HORIZON
        )

        error_consumer_t_opt = []
        for v in error_consumer_t.values():
            error_consumer_t_opt.append(v.X)

        return (
            T_sup_optimized,
            Q_optimized,
            E_optimized,
            # valve_pos_optimized,
            MW_pmp_optimized,
            T_node_sup_opt,
        )

    try:
        (
            T_sup_optimized,
            Q_optimized,
            E_optimized,
            # valve_pos_optimized,
            MW_pmp_optimized,
            T_node_sup_opt,
        ) = export_variables()

        ms_edge_optimized = []
        for v in ms_edge.values():
            ms_edge_optimized.append(v.X)
        ms_edge_optimized = np.array(ms_edge_optimized).reshape(
            (len(edges_id_to_idx)), PLANNING_HORIZON
        )

        ms_node_optimized = []
        for v in ms_node.values():
            ms_node_optimized.append(v.X)
        ms_node_optimized = np.array(ms_node_optimized).reshape(
            (len(nodes_id_to_idx)), PLANNING_HORIZON
        )
        # print(T_node_sup_opt)
        # print(ms_edge_optimized)
        # print(connection_list)
        # print(connection_list_splits)
        # print(_Q)
        # print(T_sup_optimized)
        # print(Q_optimized)
        # print(ms_node_optimized)

        # print(_temp_consumer)

    except AttributeError:
        if not DEBUG_MODE:
            print("model is infeasible, ")
            return None, None, None, None, None, None
        print("Warning, model is infeasible cs2, relaxing constraints")
        m.computeIIS()
        m.write("./flex-heat/src/optimizers/giraud_2017/model_warning.ilp")
        # exit()
        m.feasRelaxS(1, True, True, True)
        m.optimize()

        (
            T_sup_optimized,
            Q_optimized,
            E_optimized,
            # valve_pos_optimized,
            MW_pmp_optimized,
            T_node_sup_opt,
        ) = export_variables()

    except:
        m.computeIIS()
        m.write("./src/optimizers/giraud_2017/model_error.ilp")
        print("no solution found")
        exit()
        # print(_delay_matrices)

    obj = m.getObjective().getValue()
    # print(Q_optimized, _Q/MW_W)
    # print(_ms_tot_sim*HEAT_CAPACITY*(np.array(T_sup_optimized) - _T_sup)/MW_W)
    # print([np.sum([
    #                 _delay_matrices[0][t, t2] *
    #                 (
    #                     _ms_tot_sim[t2]*HEAT_CAPACITY*(T_sup_optimized[t2] - _T_sup[t2])
    #                     * np.exp(-(t - t2) / therm_const * TIME_INTERVAL)
    #                 )
    #                 for t2 in range(t + 1)
    #             ])/MW_W
    #         for t in range(PLANNING_HORIZON)]
    # )
    # print(_ms_tot_sim*HEAT_CAPACITY*(np.array(T_c_in_opt) - _T_c_in[0,0])/MW_W)
    # print(np.sum(np.array(T_sup_optimized[17:])*np.array(ms_tot_optimized[17:])))
    # print(np.sum(np.array(T_c_in_opt[17:])*np.array(ms_tot_optimized[17:])))

    return (
        T_sup_optimized,
        Q_optimized,
        E_optimized,
        # valve_pos_optimized,
        MW_pmp_optimized,
    )


class Optimizer(object):
    def __init__(
        self,
        simulator,
        initial_pipe_states=None,
        his_temp=None,
    ):
        self.simulator = simulator
        self.simulator.__class__ = SimulatorEnhancedInterface
        self.plug_states = initial_pipe_states
        self.his_temp = his_temp

        self.control_split_ids = get_control_split_id(self.simulator)

        self.nodes_id_to_idx = {
            node.id: i for i, node in enumerate(self.simulator.nodes)
        }
        self.edges_id_to_idx = {
            edge.id: i for i, edge in enumerate(self.simulator.edges)
        }
        self.producers_indices = []
        for producer in self.simulator.producers:
            self.producers_indices.append(self.nodes_id_to_idx[producer.id])
        self.consumers_indices = []
        for consumer in self.simulator.consumers:
            self.consumers_indices.append(self.nodes_id_to_idx[consumer.id])
        self.production_unit_indices = np.arange(len(INITIAL_STATUS[0]))
        self.Y_startup = np.array(INITIAL_STATUS)

        self._clear()

    def _clear(self):
        self.step_count = 0
        self.actual_E = []
        self.actual_Q = []
        self.actual_T = []
        self.actual_T_consumer = []
        self.actual_margin_operation = []
        self.actual_margin_startup = []
        self.actual_flags = []
        if self.his_temp is None:
            self.T_supply_optimized = (
                np.ones((len(self.producers_indices), PLANNING_HORIZON)) * HIS_T_SUP
            )
        else:
            self.T_supply_optimized = (
                np.ones((len(self.producers_indices), PLANNING_HORIZON)) * self.his_temp
            )

    def run(self, demand, price, min_sup_temp_producer=None):
        T_supply_optimized = self.T_supply_optimized

        E_optimized = np.zeros((len(self.producers_indices), PLANNING_HORIZON))
        # valve_pos_optimized = np.ones((len(self.producers_indices),PLANNING_HORIZON))
        # valve_pos_optimized[0] = 0.1
        # valve_pos_optimized[1] = 0.9
        # valve_pos_optimized = {i: valve_pos_optimized for i in self.control_split_ids}
        self.simulator.reset(demand, price, self.plug_states)

        sum_T_diff = 0
        self.simulator.run_sei(
            temp=T_supply_optimized,
            electricity=E_optimized,
            # valve_pos=valve_pos_optimized,
        )

        min_diff = np.inf
        backup_t_supply = np.copy(T_supply_optimized)
        backup_E = np.copy(E_optimized)
        iter = 0
        while True:
            T_supply_optimized_previous = T_supply_optimized

            (
                T_supply_optimized,
                Q_optimized,
                E_optimized,
                # valve_pos_optimized,
                MW_pmp_optimized,
            ) = model(
                demand,
                price[0],
                self.simulator.connection_list,
                self.simulator.connection_list_splits,
                self.nodes_id_to_idx,
                self.edges_id_to_idx,
                self.producers_indices,
                self.production_unit_indices,
                self.consumers_indices,
                self.Y_startup,
                self.simulator.ms_node_sei,
                self.simulator.ms_edge_sei,
                self.simulator.delay_matrix_sei,
                self.simulator.min_sup_temp_sei,
                min_sup_temp_producer,
                self.simulator.T_c_sei,
                self.simulator.diff_pressure_production_plant_sei,
                self.simulator.T_sup_sei,
                self.simulator.T_ret_sei,
                self.simulator.Q_sei,
                self.simulator.his_T_sup_sei,
                iter,
            )

            if T_supply_optimized is None:
                auto_E_opt = ConstantTempOpt(self.simulator)
                margin, q, E_optimized = auto_E_opt.run(
                    demand,
                    price,
                    T_supply_optimized_previous,
                    # valve_pos_optimized,
                )
                T_supply_optimized = T_supply_optimized_previous
                plug_state = copy.copy(
                    self.simulator.get_pipe_states(time_step=ACTION_HORIZON)
                )
                break

            self.simulator.reset()
            # valve_pos_optimized = {i: valve_pos_optimized for i in self.control_split_ids}
            self.simulator.run_sei(
                T_supply_optimized,
                E_optimized,
                # valve_pos_optimized,
            )

            # print(np.sum(self.simulator.cost),min_cost)
            diff = np.sum(np.abs(T_supply_optimized - T_supply_optimized_previous))

            if diff < min_diff:
                min_diff = diff
                min_cost = self.simulator.get_detailed_margin()
                backup_t_supply = np.copy(T_supply_optimized)
                backup_E = np.copy(E_optimized)
                backup_plug = copy.copy(
                    self.simulator.get_pipe_states(time_step=ACTION_HORIZON)
                )

            sum_T_diff += diff

            if diff <= delta_T:
                # if np.any([np.any(c.temp[0] < c.minimum_t_supply_p) for c in self.simulator.consumers]):
                #     while np.any([np.any(c.temp[0] < c.minimum_t_supply_p) for c in self.simulator.consumers]):
                #         # print(iter,[c.temp[0] for c in self.simulator.consumers])
                #         # print([c.minimum_t_supply_p for c in self.simulator.consumers])
                #         # print("constraint violated cs2, correcting")
                #         T_supply_optimized += 0.1
                #         self.simulator.reset()
                #         self.simulator.run_sei(
                #             T_supply_optimized,
                #             E_optimized,
                #         )
                #     if iter >= 10:
                #         plug_state = copy.copy(self.simulator.get_pipe_states(time_step=ACTION_HORIZON))
                #         break
                #     continue

                plug_state = copy.copy(
                    self.simulator.get_pipe_states(time_step=ACTION_HORIZON)
                )
                # margin = self.simulator.get_detailed_margin(level=0, level_time=1)
                break
            elif iter >= 10:
                T_supply_optimized = backup_t_supply
                E_optimized = backup_E
                plug_state = backup_plug
                print("warning, not converged", min_diff)
                # margin = self.simulator.get_detailed_margin(level=0, level_time=1)
                break
            else:
                # print(np.sum(np.abs(T_supply_optimized - T_supply_optimized_previous)))
                pass
            iter += 1

        self.T_supply_optimized[
            :, : PLANNING_HORIZON - ACTION_HORIZON
        ] = T_supply_optimized[:, ACTION_HORIZON:]
        for i in range(len(self.T_supply_optimized)):
            self.T_supply_optimized[
                i, PLANNING_HORIZON - ACTION_HORIZON :
            ] = self.T_supply_optimized[i, PLANNING_HORIZON - ACTION_HORIZON - 1]
        # print(cost_optimized, np.sum(margin), T_supply_optimized)
        self.plug_states = plug_state
        # print(np.mean(np.abs(np.sum(Q_optimized[0],axis=0)- self.simulator.Q_sei)))
        self.actual_E.append(E_optimized[0, :ACTION_HORIZON])
        self.actual_Q.append(Q_optimized[0, :ACTION_HORIZON])
        self.actual_T.extend(list(T_supply_optimized[:, :ACTION_HORIZON]))

        for c in self.simulator.consumers:
            self.actual_T_consumer.append(c.temp[0, :ACTION_HORIZON])

        return T_supply_optimized[0]

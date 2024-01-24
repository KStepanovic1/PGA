import numpy as np
import warnings
import pickle

from abc import ABC, abstractmethod
from tensorflow.keras.models import load_model
from typing import Tuple

from src.optimizers.constraint_opt.dhn_nn.tensor_constraint import ParNonNeg
from src.optimizers.constraint_opt.dhn_nn.param import round_dig
from src.optimizers.constraint_opt.dhn_nn.functions import *
from src.util.config import GridProperties, ProducerPreset1, PipePreset1


class Optimizer(ABC):
    def __init__(
        self,
        experiment_learn_type,
        experiment_opt_type,
        x_s,
        electricity_price,
        supply_pipe_plugs,
        return_pipe_plugs,
        T,
        N_w,
        N_w_q,
        ad,
    ):
        self.T: int = T  # planning horizon
        self.result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
        self.model_nn_p: Path = (
            Path(__file__).parents[4] / "models/constraint_opt"
        ).joinpath(experiment_learn_type["folder"])
        self.model_p: Path = (
            Path(__file__).parents[4] / "models/constraint_opt"
        ).joinpath(experiment_opt_type["folder"])
        self.parent_p: Path = Path(__file__).parent
        # maximum produced heat of CHP
        self.H_max: float = ProducerPreset1["Generators"][0]["MaxHeatProd"]
        # minimum produced heat of CHP
        self.H_min: float = ProducerPreset1["Generators"][0]["MinHeatProd"]
        # maximum produced electricity of CHP
        self.P_max: float = ProducerPreset1["Generators"][0]["MaxElecProd"]
        # minimum produced electricity of CHP
        self.P_min: float = ProducerPreset1["Generators"][0]["MinElecProd"]
        self.T_in_max = PhysicalProperties["MaxTemp"]
        self.T_out_min = ConsumerPreset1["MinTempSupplyPrimary"]
        self.m_max = ConsumerPreset1["MaxMassFlowPrimary"]
        self.delta = ProducerPreset1["Generators"][0]["MaxRampRateTemp"]
        self.num_extreme_points: int = len(
            ProducerPreset1["Generators"][0]["OperationRegion"]
        )
        self.N_w: int = N_w  # length of previous actions time window concerning heat
        self.N_w_q: int = (
            N_w_q  # length of previous actions time window concerning heat demand
        )
        self.columns: list = self.formate_columns()
        with open(
            self.result_p.joinpath(x_s),
            "rb",
        ) as f:
            self.x_s = pd.read_csv(f)[self.columns]
        with open(
            self.result_p.joinpath(electricity_price),
            "rb",
        ) as f:
            self.electricity_price = list(pd.read_csv(f)["Electricity price"])
        with open(
            self.result_p.joinpath(supply_pipe_plugs),
            "rb",
        ) as f:
            self.supply_pipe_plugs = pickle.load(f)
        with open(
            self.result_p.joinpath(return_pipe_plugs),
            "rb",
        ) as f:
            self.return_pipe_plugs = pickle.load(f)
        with open(self.parent_p.joinpath("state_dict_" + ad + ".pkl"), "rb") as f:
            self.state_dict = pickle.load(f)
        with open(self.parent_p.joinpath("output_dict_" + ad + ".pkl"), "rb") as f:
            self.output_dict = pickle.load(f)
        self.T_in_max_nor = round(
            (self.T_in_max - self.state_dict["Supply in temp 1 min"])
            / (
                self.state_dict["Supply in temp 1 max"]
                - self.state_dict["Supply in temp 1 min"]
            ),
            round_dig,
        )
        self.T_out_min_nor = round(
            (self.T_out_min - self.state_dict["Supply out temp 1 min"])
            / (
                self.state_dict["Supply out temp 1 max"]
                - self.state_dict["Supply out temp 1 min"]
            ),
            round_dig,
        )
        self.m_max_nor = round(
            (self.m_max - self.state_dict["Supply mass flow 1 min"])
            / (
                self.state_dict["Supply mass flow 1 max"]
                - self.state_dict["Supply mass flow 1 min"]
            ),
            round_dig,
        )

    @abstractmethod
    def optimize(self, opt_step, model_s, model_out, layer_size_s, layer_size_y):
        """
        Model optimization.
        """
        pass

    def formate_columns(self):
        """
        Get column names of x_s file with respect to previous actions time window length.
        """
        col = [
            "in_t_{}".format(self.N_w),
            "out_t_{}".format(self.N_w),
            "m_{}".format(self.N_w),
        ]
        col.extend(["q_{}".format(i) for i in range(1, self.N_w_q + 1 + 1)])
        col.extend(["h_{}".format(i) for i in range(1, self.N_w + 1 + 1)])
        return col

    @property
    def get_x_s(self):
        """
        Retrieve data matrix x_s
        """
        return self.x_s

    def get_tau_in(self, opt_step):
        """
        Get supply inlet temperature corresponding to time step t-1
        """
        return self.x_s["in_t_{}".format(self.N_w)][opt_step]

    def get_tau_out(self, opt_step):
        """
        Get supply outlet temperature corresponding to time step t-1
        """
        return self.x_s["out_t_{}".format(self.N_w)][opt_step]

    def get_m(self, opt_step):
        """
        Get mass flow corresponding to time step t-1
        """
        return self.x_s["m_{}".format(self.N_w)][opt_step]

    def get_plugs(self, opt_step):
        """
        Get plugs from supply and return pipe corresponding to time step t-1
        """
        plugs_supply: list = plugs_to_list(self.supply_pipe_plugs[opt_step + 1])
        plugs_return: list = plugs_to_list(self.return_pipe_plugs[opt_step + 1])
        plugs = [plugs_supply, plugs_return]
        return plugs

    def get_h(self, opt_step):
        """
        Get produced heat corresponding to time steps t-N_w,...,t-1
        """
        return list(self.x_s.loc[opt_step, self.columns[-(self.N_w + 1) : -1]])

    def get_demand_(self, opt_step) -> list:
        """
        Get normalized heat demand for the optimizer. This heat demand corresponds to time steps t-1,...,t+T
        """
        return list(
            self.x_s["q_{}".format(self.N_w_q + 1)][opt_step - 1 : opt_step + self.T]
        )

    def get_demand(self, opt_step) -> np.array:
        """
        Get heat demand (not normalized) corresponding to time steps t,t+1,...,t+T
        """
        demand_ = np.array(
            self.x_s["q_{}".format(self.N_w_q + 1)][opt_step : opt_step + self.T]
        )
        demand = np.array(
            list(
                map(
                    lambda d: d
                    * (
                        self.state_dict["Heat demand 1 max"]
                        - self.state_dict["Heat demand 1 min"]
                    )
                    + self.state_dict["Heat demand 1 min"],
                    demand_,
                )
            )
        )
        return demand

    def get_price(self, opt_step) -> list:
        """
        Get electricity price corresponding to time steps t,t+1,...,t+T
        """
        return self.electricity_price[opt_step : opt_step + self.T]

    def get_state_dict(self) -> dict:
        """
        Get dictionary with minimum and maximum value of each variable.
        """
        return self.state_dict

    @staticmethod
    def get_object_ids(simulator):
        """
        Get id of objects
        """
        obj_id_name = simulator.get_id_name_all_obj()
        producer_id = [k for k, v in obj_id_name.items() if v == "CHP"][0]
        edge_ids = [k for k, v in obj_id_name.items() if v == "Edge"]
        if GridProperties["ConsumerNum"] == 1:
            consumer_ids = [k for k, v in obj_id_name.items() if v == "Consumer"][0]
            sup_edge_ids = edge_ids[0]
            ret_edge_ids = edge_ids[1]
        object_ids = [producer_id] + [consumer_ids] + edge_ids
        return object_ids, producer_id, consumer_ids, sup_edge_ids, ret_edge_ids

    @staticmethod
    def reform_weights(weight):
        """
        Rounds weights on six digits while preserving the structure.
        This is important for numerical stability of MILP.
        """
        n_dim = weight.ndim
        weight_ = []
        if n_dim == 2:
            for i in range(weight.shape[0]):
                t = []
                for j in range(weight.shape[1]):
                    w = round(weight[i][j], round_dig)
                    t.append(w)
                weight_.append(t)
        elif n_dim == 1:
            for i in range(weight.shape[0]):
                w = round(weight[i], round_dig)
                weight_.append(w)
        weight_ = np.array(weight_)
        return weight_

    @staticmethod
    def extract_weights(model, tensor_constraint):
        """
        Extraction of weights from feedforward convex neural network.
        The function is adapted for both regular piecewise linear (state=False)
        and input convex neural networks with passthrough layers (state=True).
        return: weights and biases
        """
        if tensor_constraint:
            model = load_model(
                model, compile=False, custom_objects={"ParNonNeg": ParNonNeg}
            )
        else:
            model = load_model(model, compile=False)
        model.summary()
        theta = {}
        j = 0
        for i, layer in enumerate(model.layers):
            if "dense" in layer.name:
                weights = layer.get_weights()
                if len(weights) == 2:
                    theta["wz " + str(j)] = Optimizer.reform_weights(weights[0])
                    theta["b " + str(j)] = Optimizer.reform_weights(weights[1])
                    j += 1
                elif len(weights) == 1:
                    theta["wx " + str(j - 1)] = weights[0]
                else:
                    warnings.warn(
                        "Implemented weight extraction procedure might be unsuitable for the current network!"
                    )
        return theta

    def extract_variables(self, model, h, p, tau_in) -> Tuple[np.array, np.array, np.array]:
        """
        Extract variables from learned model.
        """
        h = model.getAttr("X", h)
        p = model.getAttr("X", p)
        tau_in = model.getAttr("X", tau_in)
        obj = model.getObjective()
        obj = obj.getValue()
        h = np.array(h.values())
        p = np.array(p.values())
        tau_in = np.array(tau_in.values())
        tau_in = np.array([
            x
            * (
                self.state_dict["Supply in temp 1 max"]
                - self.state_dict["Supply in temp 1 min"]
            )
            + self.state_dict["Supply in temp 1 min"]
            for x in tau_in
        ])
        return h, p, tau_in

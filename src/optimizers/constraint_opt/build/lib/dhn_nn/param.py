"""
Specifies parameters of state transition and system output neural networks.
"""

from dataclasses import dataclass
from pathlib import Path

from src.util.config import PipePreset1


@dataclass
class Par:
    def __init__(
        self,
        time_delay,
        columns,
        output_num,
        layer_size,
        outputs,
        x_p,
        y_p,
        electricity_price_p,
        supply_pipe_plugs_p,
        return_pipe_plugs_p,
    ):
        self.result_p: Path = Path(__file__).parents[4] / "results/constraint_opt"
        self.model_p: Path = Path(__file__).parents[4] / "models/constraint_opt"
        self.x_p: Path = self.result_p.joinpath(x_p)
        self.y_p: Path = self.result_p.joinpath(y_p)
        self.electricity_price_p: Path = self.result_p.joinpath(electricity_price_p)
        self.supply_pipe_plugs_p: Path = self.result_p.joinpath(supply_pipe_plugs_p)
        self.return_pipe_plugs_p: Path = self.result_p.joinpath(return_pipe_plugs_p)
        self.time_delay = time_delay
        self.columns = columns
        self.output_num = output_num
        self.layer_size: list = layer_size
        self.outputs = outputs
        self.err = self.init_err()

    def init_err(self):
        err = {}
        for output in self.outputs:
            err[output] = []
        return err


@dataclass
class StatePar(Par):
    """
    Defines parameters of the state space model.
    """

    def __init__(
        self,
        time_delay,
        time_delay_q,
        columns,
        output_num,
        layer_size,
        outputs,
        x_p,
        y_p,
        electricity_price_p,
        supply_pipe_plugs_p,
        return_pipe_plugs_p,
    ):
        super().__init__(
            time_delay,
            columns,
            output_num,
            layer_size,
            outputs,
            x_p,
            y_p,
            electricity_price_p,
            supply_pipe_plugs_p,
            return_pipe_plugs_p,
        )
        self.feature_num: int = 3 + time_delay + 1 + time_delay_q + 1
        self.time_delay_q: int = time_delay_q


@dataclass
class OutputPar(Par):
    """
    Defines parameters of the output space model.
    """

    def __init__(
        self,
        time_delay,
        columns,
        output_num,
        layer_size,
        outputs,
        x_p,
        y_p,
        electricity_price_p,
        supply_pipe_plugs_p,
        return_pipe_plugs_p,
    ):
        super().__init__(
            time_delay,
            columns,
            output_num,
            layer_size,
            outputs,
            x_p,
            y_p,
            electricity_price_p,
            supply_pipe_plugs_p,
            return_pipe_plugs_p,
        )
        # self.feature_num = 4 * (time_delay + 1) + 1
        self.feature_num = 3 + time_delay + 1


EarlyStopStateParams = [
    {"delta": 0.0001, "patience": 15},
    {"delta": 0.00005, "patience": 15},
    {"delta": 0.00005, "patience": 200},
]

EarlyStopOutputParams = [
    {"delta": 0.00005, "patience": 20},
    {"delta": 0.00001, "patience": 25},
    {"delta": 0.000002, "patience": 25},
    {"delta": 0.000001, "patience": 25},
]

StatePLNNParam = {
    "[1]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[1, 1]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[3]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[5]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[5, 3]": {"delta": 0.00005, "patience": 200, "epochs": 2000},
    "[10]": {"delta": 0.00005, "patience": 200, "epochs": 3000},
    "[10, 10]": {"delta": 0.000015, "patience": 200, "epochs": 3000},
    "[50, 50]": {"delta": 0.000005, "patience": 200, "epochs": 3000},
    "[100, 100, 100]": {"delta": 0.000005, "patience": 200, "epochs": 3000},
}

StateICNNParam = {
    "[1]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[1, 1]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[3]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[5]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[5, 3]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[10]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[10, 10]": {"delta": 0.000015, "patience": 200, "epochs": 2000},
    "[50, 50]": {"delta": 0.000005, "patience": 200, "epochs": 2000},
    "[100, 100, 100]": {"delta": 0.000005, "patience": 200, "epochs": 2000},
}

StateMonotonicICNNParam = {
    "[1]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[1, 1]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[3]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[5]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[5, 3]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[10]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[10, 10]": {"delta": 0.00005, "patience": 200, "epochs": 1000},
    "[50, 50]": {"delta": 0.000005, "patience": 200, "epochs": 3000},
    "[100, 100, 100]": {"delta": 0.000005, "patience": 200, "epochs": 3000},
}

OutputPLNNParam = {
    "[1]": {"delta": 0.000001, "patience": 40, "epochs": 1000},
    "[1, 1]": {"delta": 0.000001, "patience": 40, "epochs": 1000},
    "[3]": {"delta": 0.000001, "patience": 40, "epochs": 1000},
    "[5]": {"delta": 0.000001, "patience": 40, "epochs": 2000},
    "[5, 3]": {"delta": 0.00000025, "patience": 50, "epochs": 2000},
    "[10]": {"delta": 0.00000025, "patience": 50, "epochs": 2000},
    "[10, 10]": {"delta": 0.0000001, "patience": 50, "epochs": 3000},
    "[50, 50]": {"delta": 0.00000015, "patience": 50, "epochs": 3000},
    "[100, 100, 100]": {"delta": 0.00000015, "patience": 50, "epochs": 4000},
}

OutputICNNParam = {
    "[1]": {"delta": 0.000001, "patience": 40, "epochs": 1000},
    "[1, 1]": {"delta": 0.000001, "patience": 40, "epochs": 1000},
    "[3]": {"delta": 0.000001, "patience": 40, "epochs": 1000},
    "[5]": {"delta": 0.000001, "patience": 40, "epochs": 2000},
    "[5, 3]": {"delta": 0.00000025, "patience": 50, "epochs": 2000},
    "[10]": {"delta": 0.00000025, "patience": 50, "epochs": 2000},
    "[10, 10]": {"delta": 0.0000001, "patience": 50, "epochs": 3000},
    "[50, 50]": {"delta": 0.0000001, "patience": 75, "epochs": 3000},
    "[100, 100, 100]": {"delta": 0.0000001, "patience": 75, "epochs": 4000},
}

OutputMonotonicICNNParam = {
    "[1]": {"delta": 0.000002, "patience": 25, "epochs": 1000},
    "[1, 1]": {"delta": 0.000002, "patience": 25, "epochs": 1000},
    "[3]": {"delta": 0.000002, "patience": 25, "epochs": 1000},
    "[5]": {"delta": 0.000002, "patience": 25, "epochs": 1000},
    "[5, 3]": {"delta": 0.000002, "patience": 35, "epochs": 1000},
    "[10]": {"delta": 0.000002, "patience": 35, "epochs": 1000},
    "[10, 10]": {"delta": 0.000002, "patience": 35, "epochs": 1000},
    "[50, 50]": {"delta": 0.000001, "patience": 35, "epochs": 1000},
    "[100, 100, 100]": {"delta": 0.000001, "patience": 35, "epochs": 1000},
}

time_delay = {"4000": 10}
time_delay_q = {"4000": 1}

round_dig: int = 6  # last digits precision for milp optimizer

opt_steps = {"math_opt": [12, 24, 36, 48, 60, 72]}

opt_steps["dnn_opt"] = [
    x - time_delay[str(PipePreset1["Length"])]-1 for x in opt_steps["math_opt"]
]

import pandas as pd
from pathlib import Path

from .optimizer import Optimizer
from ..setting import opt_steps
from util.config import CHPPreset1


class BasicStrategy(Optimizer):
    def __init__(
        self,
        result_p,
        parent_p,
        x_s,
        electricity_price,
        supply_pipe_plugs,
        return_pipe_plugs,
    ):
        super().__init__(
            result_p,
            parent_p,
            x_s,
            electricity_price,
            supply_pipe_plugs,
            return_pipe_plugs,
            "plnn",
        )

    def optimize(self, opt_step, model_s, model_out, layer_size_s, layer_size_y):
        pass

    def basic_strategy(self, opt_step, T):
        """
        Calculating heat and profit of basic strategy.
        """
        demand = self.get_demand(opt_step, T)
        price = self.get_price(opt_step, T)
        heat = self.basic_heat(demand)
        profit = []
        for i in range(len(demand)):
            electricity = Optimizer.calculate_p(
                h=heat[i], delta_c=CHPPreset1["FuelCost"][1] - price[i]
            )
            profit.append(
                CHPPreset1["FuelCost"][0] * heat[i]
                + CHPPreset1["FuelCost"][1] * electricity
                - price[i] * electricity
            )
        return heat, profit


if __name__ == "__main__":
    T = 24
    opt_step = opt_steps  # initial steps
    result_p = Path(__file__).parents[4] / "results/constraint_opt"
    result_p_basic_strategy = result_p.joinpath("basic_strategy")
    for opt_step in opt_step:
        basic_strategy = BasicStrategy(
            result_p=result_p,
            parent_p=Path(__file__).parent,
            x_s="x_s.csv",
            electricity_price="electricity_price.csv",
            supply_pipe_plugs="supply_pipe_plugs.pickle",
            return_pipe_plugs="return_pipe_plugs.pickle",
        )
        heat, profit = basic_strategy.basic_strategy(opt_step=opt_step, T=T)
        result = {"Profit": profit, "Produced heat": heat}
        result = pd.DataFrame(result)
        name = "results {}".format(str(opt_step))
        result.to_csv(result_p_basic_strategy.joinpath(name))

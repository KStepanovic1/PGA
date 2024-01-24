from gurobipy import *
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer


class MILP(Optimizer):
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
        super().__init__(
            experiment_learn_type=experiment_learn_type,
            experiment_opt_type=experiment_opt_type,
            x_s=x_s,
            electricity_price=electricity_price,
            supply_pipe_plugs=supply_pipe_plugs,
            return_pipe_plugs=return_pipe_plugs,
            T=T,
            N_w=N_w,
            N_w_q=N_w_q,
            ad=ad,
        )

    @staticmethod
    def write_inf_m(model):
        """
        Example based on workforce1: https://www.gurobi.com/documentation/9.5/examples/workforce1_py.html.
        Computes set of constraints and bounds that make model infeasible,
        and writes those constraints in .ilp file.
        """
        model.computeIIS()
        model.write("infeasible_constraints.ilp")

    @staticmethod
    def relax_inf_m(model):
        """
        Example based on workforce3: https://www.gurobi.com/documentation/9.5/examples/workforce3_py.html.
        Modifies the Model object to create a feasibility relaxation.
        model.feasRelaxS(relaxobjtype, minrelax, vrelax, crelax ): https://www.gurobi.com/documentation/9.5/refman/py_model_feasrelaxs.html
        relaxobjtype: {0,1,2} specifies the objective of feasibility relaxation.
        minrelax: Bool The type of feasibility relaxation to perform.
        vrelax: Bool Indicates whether variable bounds can be relaxed.
        crelax: Bool Indicates whether constraints can be relaxed.
        """
        model = model.copy()
        orignumvars = model.NumVars
        model.feasRelaxS(1, True, False, True)
        model.optimize()
        status = model.Status
        if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            print(
                "The relaxed model cannot be solved \
                   because it is infeasible or unbounded"
            )
            sys.exit(1)

        if status != GRB.OPTIMAL:
            print("Optimization was stopped with status %d" % status)
            sys.exit(1)

        print("\nSlack values:")
        slacks = model.getVars()[orignumvars:]
        for sv in slacks:
            if sv.X > 1e-6:
                print("%s = %g" % (sv.VarName, sv.X))
        model.optimize()
        obj = model.getObjective()
        obj = obj.getValue()
        return obj, model

from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *
from src.optimizers.constraint_opt.dhn_nn.optimizer import Optimizer

if GridProperties["ConsumerNum"] == 1:
    from src.simulator.cases.one_consumer import build_grid
elif GridProperties["ConsumerNum"] > 1:
    from src.simulator.cases.parallel_consumers import build_grid

N_init: int = 0  # number of runs needed to diminish the effect of initial actions
MPC: int = 1
curr_time = datetime.datetime.now().strftime("%Y-%m-%d")


class GradientDescent(Optimizer):
    """
    Optimizing multi-step mathematical model based on neural network via stochastic gradient descent.
    """

    def __init__(
        self,
        C,
        experiment_learn_type,
        experiment_opt_type,
        x_s,
        electricity_price,
        supply_pipe_plugs,
        return_pipe_plugs,
        T,
        N_w,
        N_w_q,
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
            ad=experiment_learn_type["ad"],
        )
        self.C = C

    def calculate_approximate_objective_function(self, J, constraint_violation):
        """
        Calculate value of approximate objective function.
        """
        hat_J = J
        for i in range(self.T):
            hat_J += self.C * constraint_violation[i] ** 2
        return hat_J

    def calculate_objective_function(self, h):
        """
        Calculate value of objective function.
        """
        J = 0
        for i in range(self.T):
            J += ProducerPreset1["Generators"][0]["FuelCost"][0] * h[i]
        return J

    def projected_gradient_descent(self, h):
        """
        Projecting produced heat into the allowed space
        """
        for i in range(self.T):
            if h[i] > self.H_max:
                h[i] = self.H_max
            elif h[i] < self.H_min:
                h[i] = self.H_min + 5  # [MWh]
        return h

    def calculate_constraint_violations(
        self, state, q_nor_tf, q_nor, var, model_s, model_y
    ):
        """
        Calculate regular values of constraints.
        """
        constraint_values = []
        for i in range(self.T):
            state = model_s(
                tf.concat(
                    [
                        state,
                        q_nor_tf[:, i : self.N_w_q + 1 + i],
                        var[:, i : self.N_w + 1 + i],
                    ],
                    axis=1,
                )
            )
            # system output
            y = (
                model_y(
                    tf.concat(
                        [
                            state,
                            var[
                                :,
                                i : self.N_w + 1 + i,
                            ],
                        ],
                        axis=1,
                    )
                )
                - q_nor[i + self.N_w_q]
            )
            y = y.numpy()[0][0] * (
                self.output_dict["Delivered heat 1 max"]
                - self.output_dict["Delivered heat 1 min"]
            )
            constraint_values.append(y)
        return constraint_values

    def calculate_constraint_violations_nor(
        self, state, q_nor_tf, q_nor, var, model_s, model_y
    ) -> list:
        """
        Calculate normalized values of constraints.
        """
        constraint_values = []
        for i in range(self.T):
            state = model_s(
                tf.concat(
                    [
                        state,
                        q_nor_tf[:, i : self.N_w_q + 1 + i],
                        var[:, i : self.N_w + 1 + i],
                    ],
                    axis=1,
                )
            )
            # system output
            y = (
                model_y(
                    tf.concat(
                        [
                            state,
                            var[
                                :,
                                i : self.N_w + 1 + i,
                            ],
                        ],
                        axis=1,
                    )
                )
                - q_nor[i + self.N_w_q]
            )
            constraint_values.append(y)
        # Extracting values into a list
        values_list = [tensor.numpy()[0][0] for tensor in constraint_values]
        return values_list

    def process_gradient(self, grads):
        """
        Gradients corresponding to previous produced heats are set on zero.
        Only the gradients corresponding to the current produced heat (t,...,t+T) are preserved.
        """
        grads_no_update = np.array([0] * self.N_w).reshape(1, -1)
        grads_no_update = tf.constant(grads_no_update, dtype=tf.float32)
        heat_gradients = grads[0][self.N_w : self.N_w + self.T]
        heat_gradients = np.array(heat_gradients).reshape(1, -1)
        heat_gradients = tf.convert_to_tensor(heat_gradients, dtype=tf.float32)
        grads = tf.concat([grads_no_update, heat_gradients], axis=1)
        return grads, heat_gradients

    @tf.function
    def calculate_gradient(
        self,
        epsilon,
        state_model,
        heat_demand_nor_tf,
        heat_demand_nor,
        var,
        model_s,
        model_y,
    ):
        """
        Calculate gradient of x with respect to mean squared error loss function.
        """
        y = 0
        with tf.GradientTape() as tape:
            tape.watch(var)
            for i in range(self.T):
                # dynamic state transition, s_t -> s_{t+1}
                y = (
                    y
                    + ProducerPreset1["Generators"][0]["FuelCost"][0]
                    * var[:, self.N_w + i]
                )
                state_model = model_s(
                    tf.concat(
                        [
                            state_model,
                            heat_demand_nor_tf[:, i : self.N_w_q + 1 + i],
                            var[:, i : self.N_w + 1 + i],
                        ],
                        axis=1,
                    )
                )
                # system output
                y_ = (
                    self.C
                    * (
                        model_y(
                            tf.concat(
                                [
                                    state_model,
                                    var[
                                        :,
                                        i : self.N_w + 1 + i,
                                    ],
                                ],
                                axis=1,
                            )
                        )
                        - heat_demand_nor[i + self.N_w_q]
                        - epsilon[i]
                    )
                    ** 2
                )
                y = y + y_
        grads = tape.gradient(y, var)
        return grads

    @tf.function
    def calculate_gradient_ipdd(
        self,
        lam,
        state_model,
        heat_demand_nor_tf,
        heat_demand_nor,
        var,
        model_s,
        model_y,
    ):
        """
        Calculate gradient of x with respect to mean squared error loss function.
        """
        y = 0
        with tf.GradientTape() as tape:
            tape.watch(var)
            for i in range(self.T):
                # dynamic state transition, s_t -> s_{t+1}
                y = (
                    y
                    + ProducerPreset1["Generators"][0]["FuelCost"][0]
                    * var[:, self.N_w + i]
                )
                state_model = model_s(
                    tf.concat(
                        [
                            state_model,
                            heat_demand_nor_tf[:, i : self.N_w_q + 1 + i],
                            var[:, i : self.N_w + 1 + i],
                        ],
                        axis=1,
                    )
                )
                # system output
                y_ = (
                    model_y(
                        tf.concat(
                            [
                                state_model,
                                var[
                                    :,
                                    i : self.N_w + 1 + i,
                                ],
                            ],
                            axis=1,
                        )
                    )
                    - heat_demand_nor[i + self.N_w_q]
                )
                y = y + lam[i] * y_ + self.C * y_ ** 2
        grads = tape.gradient(y, var)
        return grads

    def optimize(
        self,
        result_p,
        opt_step,
        control,
        tau_in_init,
        tau_out_init,
        m_init,
        h_init,
        heat_demand_nor,
        price,
        model_s,
        model_out,
        tensor_constraint,
        layer_size_s,
        layer_size_y,
        learning_rate,
        delta,
        patience,
        grad_iter_max,
    ):
        """
        Optimize the model via gradient descent.
        opt_step: int current time-step of the optimization
        tau_in_init: float initial supply inlet temperature
        tau_out_init: float initial supply outlet temperature
        m_init: float initial mass flow
        h_init: list initial produced heats
        heat_demand_nor: list normalized heat demand
        heat_demand: np.array not normalized (regular) heat demand
        price: list electricity price
        model_s: state transition model
        model_out: output model
        tensor_constraint: bool does the learned dnn model have overwritten tensor constraint?
        layer_size_s: list number of neurons per layer for state transition model
        layer_size_y: list number of neurons per layer for output model
        learning_rate: float
        iteration_number:int maximum number of gradient descent iterations
        delta: float
        patience: int
        heat_gradients_path: Path path for saving heat gradients and updated heats
        """
        # the dictionary for saving gradients and updated produced heat
        # the key represent time-step of the produced heat h_1, ..., h_11
        p = [50] * self.T
        constraint_violations_nor, constraint_violations = {}, {}
        for i in range(self.T):
            constraint_violations_nor["constraint {}".format(i)] = []
            constraint_violations["constraint {}".format(i)] = []
        convergence_iter = 0
        execution_time_sum = 0
        # epsilon additions to constraints
        epsilon = [0] * self.T
        J, hat_J, grad_iter_, execution_time = [], [], [], []
        # read state transition model
        model_s = self.load_nn_model(
            model_p=model_s, tensor_constraint=tensor_constraint
        )
        # read system output model
        model_y = self.load_nn_model(
            model_p=model_out, tensor_constraint=tensor_constraint
        )
        # transform normalized heat demand (q) into tf.Variable
        heat_demand_nor_tf = tf.Variable(
            np.array(heat_demand_nor).reshape(1, -1), dtype=tf.float32
        )
        # transform initial produced heats+starting optimization point into tf.Variable
        initial_vector = h_init + control
        # normalized stop parameters
        delta = delta / (self.produced_heat_max - self.produced_heat_min)
        while execution_time_sum < 1000:
            print(execution_time_sum)
            convergence_iter += 1
            grad_iter, grad_patience_iter = 0, 0
            grads_prev = tf.convert_to_tensor(
                np.array([100] * self.T, dtype=np.float32).reshape(1, self.T)
            )
            var = tf.Variable(np.array(initial_vector).reshape(1, -1), dtype=tf.float32)
            state_model = tf.Variable(
                np.array([tau_in_init, tau_out_init, m_init]).reshape(1, -1),
                dtype=tf.float32,
            )
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            start_time = time.time()  # start measuring execution time
            while grad_patience_iter < patience and grad_iter < grad_iter_max:
                gradient = self.calculate_gradient(
                    epsilon=epsilon,
                    state_model=state_model,
                    heat_demand_nor_tf=heat_demand_nor_tf,
                    heat_demand_nor=heat_demand_nor,
                    var=var,
                    model_s=model_s,
                    model_y=model_y,
                )
                gradient, heat_gradients = self.process_gradient(grads=gradient)
                delta_tf = tf.abs(grads_prev - heat_gradients)
                if tf.reduce_all(delta_tf < delta, 1):
                    grad_patience_iter += 1
                else:
                    grad_patience_iter = 0
                grads_prev = heat_gradients
                # zip calculated gradient and previous value of x
                zipped = zip([gradient], [var])
                # update value of input variable according to the calculated gradient
                opt.apply_gradients(zipped)
                grad_iter += 1
            # project variable into the feasible set
            var = tf.where(var < 0, 0, var)
            # calculate normalized constraint violations
            constraint_violation_nor = self.calculate_constraint_violations_nor(
                state=state_model,
                q_nor_tf=heat_demand_nor_tf,
                q_nor=heat_demand_nor,
                var=var,
                model_s=model_s,
                model_y=model_y,
            )
            # update epsilon parameters
            for i in range(self.T):
                """
                epsilon[i] = float(
                    max(
                        (
                            epsilon[i]
                            - (2 / convergence_iter) * constraint_violation_nor[i]
                        ),
                        0,
                    )
                )
                """
                epsilon[i] = float(
                    epsilon[i] - (2 / convergence_iter) * constraint_violation_nor[i]
                )
                # save constraint violations
                constraint_violations_nor["constraint {}".format(i)].append(
                    constraint_violation_nor[i]
                )
            end_time = time.time()  # end measuring execution time
            # calculate regular constraint violations
            constraint_violation = self.calculate_constraint_violations(
                state=state_model,
                q_nor_tf=heat_demand_nor_tf,
                q_nor=heat_demand_nor,
                var=var,
                model_s=model_s,
                model_y=model_y,
            )
            for i in range(self.T):
                constraint_violations["constraint {}".format(i)].append(
                    constraint_violation[i]
                )
            execution_time.append(end_time - start_time)
            grad_iter_.append(grad_iter)
            execution_time_sum += end_time - start_time
            h_nor = var[0][self.N_w : self.N_w + self.T]
            h_nor = list(np.array(h_nor))
            # warm start-up
            # initial_vector = h_init + h_nor
            h = [
                x
                * (
                    self.state_dict["Produced heat max"]
                    - self.state_dict["Produced heat min"]
                )
                + self.state_dict["Produced heat min"]
                for x in h_nor
            ]
            h = self.projected_gradient_descent(h=h)
            J_iter = self.calculate_objective_function(h=h)
            J.append(J_iter)
            hat_J_iter = self.calculate_approximate_objective_function(
                J=J_iter, constraint_violation=constraint_violation
            )
            hat_J.append(hat_J_iter)
            print("Produced heat ", h)
            print("Constraint violations ", constraint_violation)
            print("Constraint violations nor", constraint_violation_nor)
            print("Epsilon ", epsilon)
            print("J ", J_iter)
            print("hat_J ", hat_J_iter)
            operation_cost = calculate_operation_cost(
                produced_heat=h, produced_electricity=p, electricity_price=price
            )
        self.save_results(
            result_p=result_p,
            J=J,
            hat_J=hat_J,
            constraint_violations=constraint_violations,
            execution_time=execution_time,
            grad_iter=grad_iter_,
            extension="gdco_no_neg",
        )
        return operation_cost, h, p, execution_time_sum

    def ipdd_algorithm(
        self,
        result_p,
        opt_step,
        control,
        tau_in_init,
        tau_out_init,
        m_init,
        h_init,
        heat_demand_nor,
        price,
        model_s,
        model_out,
        tensor_constraint,
        layer_size_s,
        layer_size_y,
        learning_rate,
        delta,
        patience,
        grad_iter_max,
        lam,
        eta,
    ):
        """
        Calculate variable through IPDD algorithm.
        """
        p = [50] * self.T
        constraint_violations_nor, constraint_violations = {}, {}
        for i in range(self.T):
            constraint_violations_nor["constraint {}".format(i)] = []
            constraint_violations["constraint {}".format(i)] = []
        convergence_iter = 0
        execution_time_sum = 0
        # epsilon additions to constraints
        epsilon = [0] * self.T
        J, hat_J, grad_iter_, execution_time = [], [], [], []
        # read state transition model
        model_s = self.load_nn_model(
            model_p=model_s, tensor_constraint=tensor_constraint
        )
        # read system output model
        model_y = self.load_nn_model(
            model_p=model_out, tensor_constraint=tensor_constraint
        )
        # transform normalized heat demand (q) into tf.Variable
        heat_demand_nor_tf = tf.Variable(
            np.array(heat_demand_nor).reshape(1, -1), dtype=tf.float32
        )
        # transform initial produced heats+starting optimization point into tf.Variable
        initial_vector = h_init + control
        # normalized stop parameters
        delta = delta / (self.produced_heat_max - self.produced_heat_min)
        while execution_time_sum < 1000:
            convergence_iter += 1
            grad_iter, grad_patience_iter = 0, 0
            grads_prev = tf.convert_to_tensor(
                np.array([100] * self.T, dtype=np.float32).reshape(1, self.T)
            )
            var = tf.Variable(np.array(initial_vector).reshape(1, -1), dtype=tf.float32)
            state_model = tf.Variable(
                np.array([tau_in_init, tau_out_init, m_init]).reshape(1, -1),
                dtype=tf.float32,
            )
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            start_time = time.time()  # start measuring execution time
            while grad_patience_iter < patience and grad_iter < grad_iter_max:
                gradient = self.calculate_gradient_ipdd(
                    lam=lam,
                    state_model=state_model,
                    heat_demand_nor_tf=heat_demand_nor_tf,
                    heat_demand_nor=heat_demand_nor,
                    var=var,
                    model_s=model_s,
                    model_y=model_y,
                )
                gradient, heat_gradients = self.process_gradient(grads=gradient)
                delta_tf = tf.abs(grads_prev - heat_gradients)
                if tf.reduce_all(delta_tf < delta, 1):
                    grad_patience_iter += 1
                else:
                    grad_patience_iter = 0
                grads_prev = heat_gradients
                # zip calculated gradient and previous value of x
                zipped = zip([gradient], [var])
                # update value of input variable according to the calculated gradient
                opt.apply_gradients(zipped)
                grad_iter += 1
            # project variable into the feasible set
            var = tf.where(var < 0, 0, var)
            # calculate normalized constraint violations
            constraint_violation_nor = self.calculate_constraint_violations_nor(
                state=state_model,
                q_nor_tf=heat_demand_nor_tf,
                q_nor=heat_demand_nor,
                var=var,
                model_s=model_s,
                model_y=model_y,
            )
            for i in range(self.T):
                lam[i] = float(lam[i] + self.C * constraint_violation_nor[i])
                constraint_violations_nor["constraint {}".format(i)].append(
                    constraint_violation_nor[i]
                )
            self.C = float(eta * self.C)
            print("Lambda ", lam)
            print("C ", self.C)
            end_time = time.time()
            # calculate regular constraint violations
            constraint_violation = self.calculate_constraint_violations(
                state=state_model,
                q_nor_tf=heat_demand_nor_tf,
                q_nor=heat_demand_nor,
                var=var,
                model_s=model_s,
                model_y=model_y,
            )
            for i in range(self.T):
                constraint_violations["constraint {}".format(i)].append(
                    constraint_violation[i]
                )
            execution_time.append(end_time - start_time)
            grad_iter_.append(grad_iter)
            execution_time_sum += end_time - start_time
            h_nor = var[0][self.N_w : self.N_w + self.T]
            h_nor = list(np.array(h_nor))
            # warm start-up
            # initial_vector = h_init + h_nor
            h = [
                x
                * (
                    self.state_dict["Produced heat max"]
                    - self.state_dict["Produced heat min"]
                )
                + self.state_dict["Produced heat min"]
                for x in h_nor
            ]
            h = self.projected_gradient_descent(h=h)
            J_iter = self.calculate_objective_function(h=h)
            J.append(J_iter)
            hat_J_iter = self.calculate_approximate_objective_function(
                J=J_iter, constraint_violation=constraint_violation
            )
            hat_J.append(hat_J_iter)
            print("Produced heat ", h)
            print("Constraint violations ", constraint_violation)
            print("Constraint violations nor", constraint_violation_nor)
            print("Epsilon ", epsilon)
            print("J ", J_iter)
            print("hat_J ", hat_J_iter)
            operation_cost = calculate_operation_cost(
                produced_heat=h, produced_electricity=p, electricity_price=price
            )
        self.save_results(
            result_p=result_p,
            J=J,
            hat_J=hat_J,
            constraint_violations=constraint_violations,
            execution_time=execution_time,
            grad_iter=grad_iter_,
            extension="ipdd",
        )
        return operation_cost, h, p, execution_time_sum

    def save_results(
        self,
        result_p,
        J,
        hat_J,
        constraint_violations,
        execution_time,
        grad_iter,
        extension,
    ):
        """
        Save results of the optimization.
        """
        J_df = pd.DataFrame(J)
        hat_J_df = pd.DataFrame(hat_J)
        constraint_violations_df = pd.DataFrame(constraint_violations)
        execution_time_df = pd.DataFrame(execution_time)
        grad_iter_df = pd.DataFrame(grad_iter)
        J_df.to_csv(
            result_p.joinpath(
                "J_C={}_eta={}_warm_start_up_".format(self.C, 1.1) + extension + ".csv"
            )
        )
        hat_J_df.to_csv(
            result_p.joinpath(
                "hat_J_C={}_eta={}_warm_start_up_".format(self.C, 1.1)
                + extension
                + ".csv"
            )
        )
        constraint_violations_df.to_csv(
            result_p.joinpath(
                "Constraint_violations_C={}_eta={}_warm_start_up_".format(self.C, 1.1)
                + extension
                + ".csv"
            )
        )
        execution_time_df.to_csv(
            result_p.joinpath(
                "Execution_time_C={}_eta={}_warm_start_up_".format(self.C, 1.1)
                + extension
                + ".csv"
            )
        )
        grad_iter_df.to_csv(
            result_p.joinpath(
                "Number_of_gradient_descent_iterations_C={}_eta={}_warm_start_up_".format(
                    self.C, 1.1
                )
                + extension
                + ".csv"
            )
        )


if __name__ == "__main__":
    N_model: int = 1
    C = 20 # (C=20 is too small and it is not working)
    controls = [1]
    layer_sizes = [[50, 50]]
    experiment_learn_type = experiments_learning["monotonic_icnn"]
    experiment_opt_type = experiments_optimization["monotonic_icnn_gd"]
    tensor_constraint: bool = experiment_opt_type[
        "tensor_constraint"
    ]  # does the neural network model used in optimization have overwritten weight constraint class?
    for control_ in controls:
        result_p: Path = (
            (Path(__file__).parents[4] / "results/constraint_opt")
            .joinpath(experiment_opt_type["folder"])
            .joinpath(experiment_opt_type["sub-folder"])
            .joinpath("initialization_{}/MPC_episode_length_1_hours".format(control_))
        )
        for layer_size in layer_sizes:
            layer_size_s = copy.deepcopy(layer_size)
            layer_size_s.append(3)
            layer_size_y = copy.deepcopy(layer_size)
            layer_size_y.append(1)
            for k in range(N_model):
                # opt_step means that the initial values for tau_in, tau_out, and m will be taken for time-step N_w+opt_step+1
                # +1 is because querying the data was done as data[N_w:]. Instead of this it could have been done as data[(N_w-1):]
                # heat demand and price for the optimization will start from the time-step (N_w+opt_step+1)+1
                for opt_step_index, opt_step in enumerate(opt_steps["dnn_opt"]):
                    results = {
                        "Q_optimized": [],
                        "Produced_heat_optimizer": [],
                        "T_supply_simulator": [],
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
                    gd_optimizer = GradientDescent(
                        C=C,
                        experiment_learn_type=experiment_learn_type,
                        experiment_opt_type=experiment_opt_type,
                        x_s="x_s.csv",
                        electricity_price="electricity_price.csv",
                        supply_pipe_plugs="supply_pipe_plugs.pickle",
                        return_pipe_plugs="return_pipe_plugs.pickle",
                        T=TimeParameters["PlanningHorizon"],
                        N_w=time_delay[str(PipePreset1["Length"])],
                        N_w_q=time_delay_q[str(PipePreset1["Length"])],
                    )
                    # initial values of internal variables
                    tau_in_init: float = round(
                        gd_optimizer.get_tau_in(opt_step=opt_step), round_dig
                    )  # time-step t-1 (normalized)
                    tau_out_init: float = round(
                        gd_optimizer.get_tau_out(opt_step=opt_step), round_dig
                    )  # time-step t-1 (normalized)
                    m_init: float = round(
                        gd_optimizer.get_m(opt_step=opt_step), round_dig
                    )  # time-step t-1 (normalized)
                    h_init: list = gd_optimizer.get_h(
                        opt_step=opt_step
                    )  # time-step t-N_w,...t-1 (normalized)
                    heat_demand_nor: list = gd_optimizer.get_demand_(
                        opt_step=opt_step
                    )  # time-step t-1,...,t+T (normalized)
                    plugs: list = gd_optimizer.get_plugs(
                        opt_step=opt_step
                    )  # time-step t-1
                    # initial values of external variables
                    heat_demand: np.array = gd_optimizer.get_demand(
                        opt_step=opt_step
                    )  # time-steps t,...,t+T (not normalized)
                    plt.plot(heat_demand)
                    plt.xlabel("Time step [h]", fontsize=14)
                    plt.ylabel("Heat demand [MW]", fontsize=14)
                    plt.savefig("Heat_demand.pdf")
                    price: list = gd_optimizer.get_price(
                        opt_step=opt_step
                    )  # time-steps t,...,t+T
                    # build simulator
                    simulator = build_grid(
                        consumer_demands=[heat_demand],
                        electricity_prices=[price],
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
                    state_dict = gd_optimizer.get_state_dict()
                    for i in range(N_init + MPC):
                        history_mass_flow = {
                            sup_edge_ids: re_normalize_variable(
                                var=m_init,
                                min=state_dict["Supply mass flow 1 min"],
                                max=state_dict["Supply mass flow 1 max"],
                            ),
                            ret_edge_ids: re_normalize_variable(
                                var=m_init,
                                min=state_dict["Supply mass flow 1 min"],
                                max=state_dict["Supply mass flow 1 max"],
                            ),
                        }
                        if ProducerPreset1["ControlWithTemp"]:
                            warnings.warn(
                                "For actions optimized via gradient descent, it is possible to control simulator only with the heat"
                            )
                            break
                        model_s: str = (
                            "{}_model_state".format(k)
                            + experiment_learn_type["model_ext"]
                            + "_s_time_delay_{}_".format(
                                time_delay[str(PipePreset1["Length"])]
                            )
                            + neurons_ext(layer_size)
                            + "_"
                            + experiment_opt_type["nn_type"]
                            + ".h5",
                        )
                        model_out: str = (
                            "{}_model_output".format(k)
                            + experiment_learn_type["model_ext"]
                            + "_s_time_delay_{}_".format(
                                time_delay[str(PipePreset1["Length"])]
                            )
                            + neurons_ext(layer_size)
                            + "_"
                            + experiment_opt_type["nn_type"]
                            + ".h5"
                        )
                        # control = gd_optimizer.get_initial_control(opt_step_index, i)
                        control = [control_] * TimeParameters["PlanningHorizon"]
                        # optimize the model for the solution
                        (operation_cost, h, p, exec_time,) = gd_optimizer.optimize(
                            result_p=result_p,
                            opt_step=opt_step + i,
                            control=control,
                            tau_in_init=tau_in_init,
                            tau_out_init=tau_out_init,
                            m_init=m_init,
                            h_init=h_init,
                            heat_demand_nor=heat_demand_nor,
                            price=price,
                            model_s=model_s,
                            model_out=model_out,
                            tensor_constraint=tensor_constraint,
                            layer_size_s=layer_size_s,
                            layer_size_y=layer_size_y,
                            learning_rate=0.01,
                            delta=0.25,  # [MWh]
                            patience=1000,
                            grad_iter_max=10000,
                        )
                        """
                        (
                            operation_cost,
                            h,
                            p,
                            exec_time,
                        ) = gd_optimizer.ipdd_algorithm(
                            result_p=result_p,
                            opt_step=opt_step + i,
                            control=control,
                            tau_in_init=tau_in_init,
                            tau_out_init=tau_out_init,
                            m_init=m_init,
                            h_init=h_init,
                            heat_demand_nor=heat_demand_nor,
                            price=price,
                            model_s=model_s,
                            model_out=model_out,
                            tensor_constraint=tensor_constraint,
                            layer_size_s=layer_size_s,
                            layer_size_y=layer_size_y,
                            learning_rate=0.01,
                            delta=0.25,  # [MWh]
                            patience=1000,
                            grad_iter_max=10000,
                            lam=[0] * TimeParameters["PlanningHorizon"],
                            eta=1.1,
                        )
                        """
                        # run through the simulator for feasibility verification
                        (
                            supply_inlet_violation,
                            supply_outlet_violation,
                            mass_flow_violation,
                            delivered_heat_violation,
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
                            produced_heat=h,
                            supply_inlet_temperature=[90]
                            * TimeParameters["PlanningHorizon"],
                            produced_electricity=p,
                            demand=heat_demand,
                            price=price,
                            plugs=plugs,
                            history_mass_flow=history_mass_flow,
                        )
                        # ensure that the produced heat is not greater that the maximum produced heat
                        if (
                            produced_heat_sim[0]
                            > ProducerPreset1["Generators"][0]["MaxHeatProd"]
                        ):
                            produced_heat_sim[0] = ProducerPreset1["Generators"][0][
                                "MaxHeatProd"
                            ]
                        p = get_optimal_produced_electricity(
                            produced_heat=produced_heat_sim, electricity_price=price
                        )
                        operation_cost = calculate_operation_cost(
                            produced_heat=produced_heat_sim,
                            produced_electricity=p,
                            electricity_price=price,
                        )
                        # save results from each single run before MPC
                        """
                        save_results(
                            keys=[
                                "Profit",
                                "Produced_heat_opt",
                                "Supply_inlet_temperature_sim",
                                "Produced_heat_sim",
                                "Produced_electricity",
                                "Heat_demand",
                                "Electricity_price",
                                "Supply_inlet_violation",
                                "Supply_outlet_violation",
                                "Mass_flow_violation",
                                "Delivered_heat_violation",
                            ],
                            values=[
                                operation_cost,
                                h,
                                tau_in_sim,
                                produced_heat_sim,
                                p,
                                heat_demand,
                                price,
                                supply_inlet_violation,
                                supply_outlet_violation,
                                mass_flow_violation,
                                delivered_heat_violation,
                            ],
                            path=result_p.joinpath(
                                "single_step_results/{}_".format(k)
                                + experiment_opt_type["optimizer_type"]
                                + "_"
                                + neurons_ext(layer_size)
                                + "_opt_step_{}".format(
                                    opt_steps["math_opt"][opt_step_index]
                                )
                                + "_time_step_{}".format(i)
                                + "_"
                                + curr_time
                                + ".csv"
                            ),
                        )
                        """
                        # save only the first results, following MPC framework
                        results["Profit"].append(operation_cost[0])
                        results["Produced_heat_optimizer"].append(h[0])
                        results["Q_optimized"].append(produced_heat_sim[0])
                        results["T_supply_simulator"].append(tau_in_sim[0])
                        results["Produced_electricity"].append(p[0])
                        results["Heat_demand"].append(heat_demand[0])
                        results["Electricity_price"].append(price[0])
                        results["Optimality_gap"].append(0)
                        results["Runtime"].append(exec_time)
                        results["Supply_inlet_violation"].append(
                            abs(supply_inlet_violation[0])
                        )
                        results["Supply_outlet_violation"].append(
                            abs(supply_outlet_violation[0])
                        )
                        results["Mass_flow_violation"].append(
                            abs(mass_flow_violation[0])
                        )
                        results["Delivered_heat_violation"].append(
                            abs(delivered_heat_violation[0])
                        )
                        # get initial values for the next MPC iteration
                        tau_in_init = max(
                            normalize_variable(
                                var=tau_in_sim[0],
                                min=state_dict["Supply in temp 1 min"],
                                max=state_dict["Supply in temp 1 max"],
                            ),
                            0.01,
                        )
                        tau_out_init = max(
                            normalize_variable(
                                var=tau_out[0],
                                min=state_dict["Supply out temp 1 min"],
                                max=state_dict["Supply out temp 1 max"],
                            ),
                            0.01,
                        )
                        m_init = max(
                            normalize_variable(
                                var=m[0],
                                min=state_dict["Supply mass flow 1 min"],
                                max=state_dict["Supply mass flow 1 max"],
                            ),
                            0.01,
                        )
                        # update of previous actions
                        h_init = h_init[TimeParameters["ActionHorizon"] :]
                        h_init.append(
                            max(
                                normalize_variable(
                                    var=produced_heat_sim[0],
                                    min=state_dict["Produced heat min"],
                                    max=state_dict["Produced heat max"],
                                ),
                                0.01,
                            )
                        )
                        heat_demand_nor = gd_optimizer.get_demand_(
                            opt_step=opt_step + i + TimeParameters["ActionHorizon"]
                        )  # time-step t-1,...,t+T (normalized)
                        # update heat demand and electricity price
                        heat_demand = gd_optimizer.get_demand(
                            opt_step=opt_step + i + TimeParameters["ActionHorizon"]
                        )
                        price = gd_optimizer.get_price(
                            opt_step=opt_step + i + TimeParameters["ActionHorizon"]
                        )
                        results_df = pd.DataFrame(results)
                        # save results with initial actions
                        """
                        results_df.to_csv(
                            result_p.joinpath(
                                "{}_".format(k)
                                + experiment_opt_type["optimizer_type"]
                                + "_init_"
                                + neurons_ext(layer_size)
                                + "_opt_step_{}".format(
                                    opt_steps["math_opt"][opt_step_index]
                                )
                                + "_"
                                + curr_time
                                + ".csv"
                            )
                        )
                        """

from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *

from src.optimizers.constraint_opt.dhn_nn.tensor_constraint import ParNonNeg
from src.optimizers.constraint_opt.dhn_nn.nn import NN


class State(NN):
    """
    Learning state function, s_t = g(s_{t-1},...,s_{t-tau}, h_t,...,h_{t-tau}, Q_t,...,Q_{t-tau})
    """

    def __init__(
        self,
        result_p,
        model_p,
        time_delay,
        time_delay_q,
        columns,
        ad,
        delta_s,
        s_ext,
        nor,
        warm_up,
        warm_up_ext,
        param,
    ):
        super().__init__(model_p, delta_s, s_ext, nor, warm_up, warm_up_ext)
        self.time_delay: int = time_delay
        self.time_delay_q: int = time_delay_q
        self.dict_p = self.parent_p.joinpath("state_dict_" + ad + ".pkl")
        self.columns = self.get_columns(columns_=columns)
        self.data = self.read_data(result_p=result_p)
        self.N: int = len(self.data)
        self.ad = ad
        self.warm_up_ext = warm_up_ext
        self.param = param

    def name_columns(self):
        """
        Naming columns of the neural network features dataset depending on the time delay parameter.
        """
        base_x = ["in_t_", "out_t_", "m_", "q_", "h_"]
        columns_x = []
        time_delay_plus = self.time_delay + 1
        time_delay_q_plus = self.time_delay_q + 1
        for col in base_x:
            for i in range(1, time_delay_plus):
                if (col == "in_t_" or col == "out_t_" or col == "m_") and (
                    i == self.time_delay
                ):
                    columns_x.append(col + str(i))
        for i in range(1, time_delay_q_plus + 1):
            columns_x.append("q_" + str(i))
        for i in range(1, time_delay_plus + 1):
            columns_x.append("h_" + str(i))
        columns_y = [
            "in_t_" + str(time_delay_plus),
            "out_t_" + str(time_delay_plus),
            "m_" + str(time_delay_plus),
        ]
        return columns_x, columns_y

    def dataset(self, x_p, y_p, electricity_price_p, plugs_supply_p, plugs_return_p):
        """
        Create two datasets, x--features and y--output. As the y is dependent on
        past features, we specify time delay parameter.
        """
        x, y, electricity_price, plugs_supply, plugs_return = [], [], [], [], []
        columns_x, columns_y = self.name_columns()
        for i in range(self.time_delay, self.N - 1):
            temp = []
            temp.append(self.data["Supply in temp 1"][i])
            temp.append(self.data["Supply out temp 1"][i])
            temp.append(self.data["Supply mass flow 1"][i])
            # temp.extend(self.data["Heat demand 1"][i : i + self.time_delay])
            # temp.extend(self.data["Produced heat"][i : i + self.time_delay])
            temp.extend(self.data["Heat demand 1"][i + 1 - self.time_delay_q : i + 2])
            temp.extend(self.data["Produced heat"][i + 1 - self.time_delay : i + 2])
            x.append(temp)
            if self.delta_s:
                y.append(
                    [
                        self.data["Supply in temp 1"][i + 1]
                        - self.data["Supply in temp 1"][i],
                        self.data["Supply out temp 1"][i + 1]
                        - self.data["Supply out temp 1"][i],
                        self.data["Supply mass flow 1"][i + 1],
                    ]
                )
            else:
                y.append(
                    [
                        self.data["Supply in temp 1"][i + 1],
                        self.data["Supply out temp 1"][i + 1],
                        self.data["Supply mass flow 1"][i + 1],
                    ]
                )
            electricity_price.append(self.data["Electricity price"][i + 1])
            plugs_supply.append(self.data["Supply plugs 1"][i])
            plugs_return.append(self.data["Ret plugs 1"][i])
        x = pd.DataFrame(x, columns=columns_x)
        y = pd.DataFrame(y, columns=columns_y)
        NN.min_max_dict(y)
        electricity_price = pd.DataFrame(
            electricity_price, columns=["Electricity price"]
        )
        scaler_x = 0
        scaler_y = 0
        scaler_s = 0
        if self.nor:
            # normalizes only state variables (if we predict change in the state)
            if self.delta_s:
                x_, scaler_s = NN.state_normalization(x=x[columns_x[:3]])
            # normalizes input and output matrices
            scaler_x, scaler_y, x, y = self.data_normalization(
                x, y, columns_x=columns_x, columns_y=columns_y
            )
        x.to_csv(x_p, index=False)
        y.to_csv(y_p, index=False)
        electricity_price.to_csv(electricity_price_p, index=False)
        with open(plugs_supply_p, "wb") as handle:
            pickle.dump(plugs_supply, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(plugs_return_p, "wb") as handle:
            pickle.dump(plugs_return, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return scaler_x, scaler_y, scaler_s

    def train_nn(
        self,
        experiments_type,
        x_train,
        feature_num,
        y_train,
        layer_size,
        batch_size,
        num_run,
        save_up,
        early_stop,
    ):
        """
        Training of the neural network.
        return: model
        """
        if self.warm_up:
            model = self.neural_net(layer_size=layer_size, feature_num=feature_num)
        else:
            model = self.load_dnn_model(
                result_p_sub=experiments_type["folder"],
                model_ext=experiments_type["model_ext"],
                name=str(num_run) + "_model_state_warm_up",
                time_delay=self.time_delay,
                layer_size=layer_size,
                ad=self.ad,
            )
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        if early_stop:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=self.param[str(layer_size[:-1])]["delta"],
                patience=self.param[str(layer_size[:-1])]["patience"],
            )
            callbacks = [early_stopping]
        else:
            callbacks = []
        model.compile(optimizer=opt, loss=MeanSquaredError())
        history = model.fit(
            x=x_train,
            y=y_train,
            validation_split=0.2,
            batch_size=batch_size,
            epochs=self.param[str(layer_size[:-1])]["epochs"],
            shuffle=False,
            verbose=1,
            callbacks=callbacks,
        )
        if save_up or self.warm_up:
            model.save(
                self.model_p.joinpath(experiments_type["folder"]).joinpath(
                    str(num_run)
                    + "_model_state"
                    + self.warm_up_ext
                    + self.s_ext
                    + "time_delay_"
                    + str(self.time_delay)
                    + experiments_type["model_ext"]
                    + "_"
                    + NN.neurons_ext(layer_size)
                    + "_"
                    + self.ad
                    + ".h5"
                )
            )
        return model, history.history["loss"], history.history["val_loss"]

    def get_min_max(self):
        """
        Creates the dictionary where the key is the name of the variable, and the value is variable's maximum/minimum value.
        """
        super(State, self).get_min_max(dict_p=self.dict_p)

    def get_nn_type(self) -> str:
        """
        Returns type of state neural network: plnn, icnn or monotonic_icnn
        """
        return self.ad


class StatePLNN(State):
    """
    Learning state function as piecewise linear neural network.
    """

    def __init__(
        self,
        result_p,
        model_p,
        time_delay,
        time_delay_q,
        columns,
        delta_s,
        s_ext,
        nor,
        warm_up,
        warm_up_ext,
    ):
        super().__init__(
            result_p,
            model_p,
            time_delay,
            time_delay_q,
            columns,
            "plnn",
            delta_s,
            s_ext,
            nor,
            warm_up,
            warm_up_ext,
            StatePLNNParam,
        )

    def neural_net(self, layer_size, feature_num):
        """
        Creates piecewise linear neural network.
        """
        input_layer = Input(shape=(feature_num,))
        layer = None
        for n, nn in enumerate(layer_size):
            # first hidden layer
            if layer is None:
                layer = Dense(
                    nn,
                    activation="relu",
                    use_bias=True,
                    kernel_regularizer=regularizers.L2(1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                )(input_layer)
            else:
                layer_n = Dense(
                    nn,
                    use_bias=True,
                    kernel_regularizer=regularizers.L2(1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                )(layer)
                if n == len(layer_size) - 1:
                    layer = Activation("linear")(layer_n)
                else:
                    layer = Activation("relu")(layer_n)
        model = ModelNN(inputs=input_layer, outputs=layer)
        return model


class StateICNN(State):
    """
    Learning state function as input convex neural network.
    """

    def __init__(
        self,
        result_p,
        model_p,
        time_delay,
        time_delay_q,
        columns,
        delta_s,
        s_ext,
        nor,
        warm_up,
        warm_up_ext,
    ):
        super().__init__(
            result_p,
            model_p,
            time_delay,
            time_delay_q,
            columns,
            "icnn",
            delta_s,
            s_ext,
            nor,
            warm_up,
            warm_up_ext,
            StateICNNParam,
        )

    def neural_net(self, layer_size, feature_num):
        """
        Create neural network convex in its inputs.
        """
        input_layer = Input(shape=(feature_num,))
        layer = None
        for n, nn in enumerate(layer_size):
            # first hidden layer
            if layer is None:
                layer = Dense(
                    nn,
                    activation="relu",
                    use_bias=True,
                    kernel_regularizer=regularizers.L2(1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                )(input_layer)
            else:
                layer_forward = Dense(
                    nn,
                    use_bias=True,
                    kernel_constraint=non_neg(),
                    kernel_regularizer=regularizers.L2(1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                )(layer)
                layer_pass = Dense(
                    nn,
                    use_bias=False,
                )(input_layer)
                # adding feedforward and passthrough layer
                layer_merge = Add()([layer_forward, layer_pass])
                if n == len(layer_size) - 1:
                    layer = Activation("linear")(layer_merge)
                else:
                    layer = Activation("relu")(layer_merge)
        model = ModelNN(inputs=input_layer, outputs=layer)
        return model


class StateMonotonicICNN(State):
    """
    Learning state function as input convex neural network.
    """

    def __init__(
        self,
        result_p,
        model_p,
        time_delay,
        time_delay_q,
        columns,
        delta_s,
        s_ext,
        nor,
        warm_up,
        warm_up_ext,
    ):
        super().__init__(
            result_p,
            model_p,
            time_delay,
            time_delay_q,
            columns,
            "monotonic_icnn",
            delta_s,
            s_ext,
            nor,
            warm_up,
            warm_up_ext,
            StateMonotonicICNNParam,
        )

    def neural_net(self, layer_size, feature_num):
        """
        Create neural network convex in its inputs.
        """
        input_layer = Input(shape=(feature_num,))
        layer = None
        for n, nn in enumerate(layer_size):
            # first hidden layer
            if layer is None:
                layer = Dense(
                    nn,
                    activation="relu",
                    use_bias=True,
                    kernel_constraint=ParNonNeg(
                        time_delay=self.time_delay,
                        param_start=3,
                        param_end=3 + self.time_delay_q + 1,
                        feature_num=feature_num,
                    ),
                    kernel_regularizer=regularizers.L2(1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                )(input_layer)
            else:
                layer_forward = Dense(
                    nn,
                    use_bias=True,
                    kernel_constraint=non_neg(),
                    kernel_regularizer=regularizers.L2(1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                )(layer)
                layer_pass = Dense(
                    nn,
                    kernel_constraint=ParNonNeg(
                        time_delay=self.time_delay,
                        param_start=3,
                        param_end=3 + self.time_delay_q + 1,
                        feature_num=feature_num,
                    ),
                    use_bias=False,
                )(input_layer)
                # adding feedforward and passthrough layer
                layer_merge = Add()([layer_forward, layer_pass])
                if n == len(layer_size) - 1:
                    layer = Activation("linear")(layer_merge)
                else:
                    layer = Activation("relu")(layer_merge)
        model = ModelNN(inputs=input_layer, outputs=layer)
        return model

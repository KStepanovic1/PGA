import numpy as np
import pandas as pd
import tensorflow as tf
import statistics

from util.config import GridProperties, PipePreset1, CHPPreset1


from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import Model as ModelNN
from tensorflow.keras import regularizers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from .param import *
from tensorflow.keras.models import load_model


def heat_change(model):
    """
    Testing how changing main control variable -- produced heat influences output of the network.
    """
    model = load_model(model)
    input = np.array(
        [
            [
                0.5599999999999998,
                0.47562534234069753,
                0.17976872150939646,
                0.40966757668219045,
                0.2250059651634455,
            ],
            [
                0.5,
                0.5857221106445134,
                0.2993702567414823,
                0.753653444676409,
                0.36284100851029993,
            ],
            [0.8, 0.2, 0.1, 0.5, 0.3],
            [0.8, 0.2, 0.1, 0.5, 0.4],
            [0.8, 0.2, 0.1, 0.5, 0.5],
            [0.8, 0.2, 0.1, 0.5, 0.6],
            [0.8, 0.2, 0.1, 0.5, 0.7],
            [0.8, 0.2, 0.1, 0.5, 0.8],
            [0.8, 0.2, 0.1, 0.5, 0.9],
            [0.8, 0.2, 0.1, 0.5, 1],
        ]
    )
    output = model.predict(input)
    return output


class StateTest:
    def __init__(self, result_p, time_delay, columns):
        self.parent_p: Path = Path(__file__).parents[1] / "dhn_plnn"
        self.time_delay = time_delay
        self.columns = self.get_columns(columns_=columns)
        self.data = self.read_data(result_p=result_p)
        self.N = len(self.data)

    def get_columns(self, columns_):
        """
        Reading columns from the real-world dataset depending on the number of consumers.
        """
        columns = []
        for i in range(int(GridProperties["PipeNum"] / 2)):
            for column in columns_:
                columns.append(column + str(i + 1))
        columns.extend(["Produced heat"])
        return columns

    def read_data(self, result_p):
        """
        Reading data from the main file, corresponding to previously formed columns.
        """
        with open(
            result_p.joinpath(
                "data_num_{}_heat_demand_real_world_for_L = {}_time_interval_3600_max_Q_70MW_deltaT_{}C.csv".format(
                    GridProperties["ConsumerNum"],
                    PipePreset1["Length"],
                    CHPPreset1["MaxRampRateTemp"],
                )
            ),
            "rb",
        ) as f:
            data = pd.read_csv(f)[self.columns]
        return data

    def data_normalization(self, x, y, columns_x, columns_y):
        """
        Normalize input and output data between 0 and 1
        """
        scaler_x = MinMaxScaler()
        x_scaled = scaler_x.fit_transform(x)
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y)
        x_scaled = pd.DataFrame(x_scaled, columns=columns_x)
        y_scaled = pd.DataFrame(y_scaled, columns=columns_y)
        return scaler_x, scaler_y, x_scaled, y_scaled

    def train_test_split(self, x_p, y_p):
        """
        Split the dataset on training and testing dataset.
        """
        with open(x_p) as file:
            x = np.array(pd.read_csv(file))
        with open(y_p) as file:
            y = np.array(pd.read_csv(file))
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, shuffle=True
        )
        y = [
            statistics.mean(x_train[:, 0]),
            statistics.mean(x_train[:, 1]),
            statistics.mean(x_train[:, 2]),
        ]
        y_train = [y] * x_train.shape[0]
        y_train = np.array(y_train)
        return x_train, x_test, y_train, y_test

    def prediction(self, model, x_test, y_test, outputs, scaler, time_delay):
        y_pred = model.predict(x_test)
        y_test = scaler.inverse_transform(y_test)
        y_pred = scaler.inverse_transform(y_pred)
        err = {}
        for i, output in enumerate(outputs):
            err[output] = mean_squared_error(y_pred[:, i], y_test[:, i], squared=False)
        return err

    def name_columns(self):
        """
        Naming columns of the neural network features dataset depending on the time delay parameter.
        """
        base_x = ["in_t_", "out_t_", "m_", "q_", "h_"]
        columns_x = []
        time_delay_plus = self.time_delay + 1
        for col in base_x:
            for i in range(1, time_delay_plus):
                if col == "in_t_" or col == "out_t_" or col == "m_":
                    columns_x.append(col + str(i))
                if (col == "q_" or col == "h_") and (i == self.time_delay):
                    columns_x.append(col + str(i + 1))
        columns_y = [
            "in_t_" + str(time_delay_plus),
            "out_t_" + str(time_delay_plus),
            "m_" + str(time_delay_plus),
        ]
        return columns_x, columns_y

    def dataset(self, x_p, y_p):
        """
        Create two datasets, x--features and y--output. As the y is dependent on
        past features, we specify time delay parameter.
        """
        x, y = [], []
        columns_x, columns_y = self.name_columns()
        for i in range(self.N):
            if i < self.N - self.time_delay:
                temp = []
                temp.extend(self.data["Supply in temp 1"][i : i + self.time_delay])
                temp.extend(self.data["Supply out temp 1"][i : i + self.time_delay])
                temp.extend(self.data["Supply mass flow 1"][i : i + self.time_delay])
                # temp.extend(self.data["Heat demand 1"][i : i + self.time_delay])
                # temp.extend(self.data["Produced heat"][i : i + self.time_delay])
                temp.append(self.data["Heat demand 1"][i + self.time_delay])
                temp.append(self.data["Produced heat"][i + self.time_delay])
                x.append(temp)
                y.append(
                    [
                        self.data["Supply in temp 1"][i + self.time_delay],
                        self.data["Supply out temp 1"][i + self.time_delay],
                        self.data["Supply mass flow 1"][i + self.time_delay],
                    ]
                )
        x = pd.DataFrame(x, columns=columns_x)
        y = pd.DataFrame(y, columns=columns_y)
        scaler_x, scaler_y, x, y = self.data_normalization(
            x, y, columns_x=columns_x, columns_y=columns_y
        )
        x.to_csv(x_p, index=False)
        y.to_csv(y_p, index=False)
        return scaler_x, scaler_y

    def plnn(self, layer_size, feature_num):
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
        model.save_weights(self.parent_p.joinpath("weights_state_plnn"))
        return model

    def train_nn(self, x_train, feature_num, y_train, layer_size, batch_size, epochs):
        """
        Training of the neural network.
        return: model
        """
        model = self.plnn(layer_size=layer_size, feature_num=feature_num)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        early_stopping = EarlyStopping(monitor="loss", min_delta=0.00005, patience=10)
        model.compile(optimizer=opt, loss=MeanSquaredError())
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=False,
            verbose=1,
            callbacks=[early_stopping],
        )
        model.save(self.parent_p.joinpath("model_state_plnn.h5"))
        return model, history.history["loss"][-1]


if __name__ == "__main__":
    # output = heat_change(model="model_state_plnn.h5")
    num_run = 1
    # state parameters
    s_par = StatePar(
        time_delay=1,
        columns=[
            "Supply in temp ",
            "Supply out temp ",
            "Supply mass flow ",
            "Heat demand ",
        ],
        output_num=3,
        outputs=[
            "Supply inlet temperature ",
            "Supply outlet temperature ",
            "Mass flow ",
        ],
        x_p="x_s_mean.csv",
        y_p="y_s_mean.csv",
        electricity_price_p="electricity_price.csv",
        supply_pipe_plugs_p="supply_pipe_plugs.pickle",
    )
    # state model
    for i in range(num_run):
        state = StateTest(
            result_p=s_par.result_p, time_delay=s_par.time_delay, columns=s_par.columns
        )
        scaler_x, scaler_y = state.dataset(x_p=s_par.x_p, y_p=s_par.y_p)
        x_train, x_test, y_train, y_test = state.train_test_split(
            x_p=s_par.x_p, y_p=s_par.y_p
        )
        model, loss = state.train_nn(
            x_train=x_train,
            feature_num=s_par.feature_num,
            y_train=y_train,
            layer_size=[25, 25, s_par.output_num],
            batch_size=32,
            epochs=500,
        )
        err_ = state.prediction(
            model=model,
            x_test=x_test,
            y_test=y_test,
            outputs=s_par.outputs,
            scaler=scaler_y,
            time_delay=s_par.time_delay,
        )
        for output in s_par.outputs:
            s_par.err[output].append(err_[output])
    for i, output in enumerate(s_par.outputs):
        print(output + " error {:.6f}".format(statistics.mean(s_par.err[output])))

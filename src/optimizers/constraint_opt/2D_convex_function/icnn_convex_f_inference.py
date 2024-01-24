import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from datetime import datetime
from gurobipy import *
from pathlib import Path
from tensorflow.keras.layers import Input, Dense, Add, Activation
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from statistics import mean

# the model from tensorflow.keras can not be imported as Model
# because it will overwrite Model from gurobipy
# from gurobipy import * has a Model in itself
from tensorflow.keras import Model as ModelNN
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


"""
Finding the minimum of convex function using gradient descent and linear programming.
"""


def neurons_ext(layer_size) -> str:
    """
    Form a string indicating number of neurons in each hidden layer.
    """
    neurons = "neurons"
    for i in range(len(layer_size) - 1):
        neurons += "_" + str(layer_size[i])
    return neurons


def main(
    size,
    feature_num,
    lower_bound,
    upper_bound,
    layer_size,
    batch_size,
    epochs,
    iterations,
    starting_point,
    learning_rate,
    num_run,
):
    train_mse_, test_mse_, x_gd_, y_gd_, t_gd_, x_lp_, y_lp_, t_lp_ = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    dying_relu = 0
    num_ = 0
    while num_run > 0:
        x = np.linspace(lower_bound, upper_bound, size * feature_num)
        x = x.reshape((size, feature_num))
        y = convex_function(x, feature_num)
        x_train, x_test, y_train, y_test = train_test_dataset(x=x, y=y)
        model, train_loss = train_nn(
            x_train=x_train,
            feature_num=feature_num,
            y_train=y_train,
            layer_size=layer_size,
            batch_size=batch_size,
            epochs=epochs,
        )
        # if the train loss corresponds to all dead ReLus,
        # we skip the current iteraion, and increase the number of dead ReLu networks.
        # if train_loss >= 0.005:
        #    dying_relu += 1
        #    continue
        y_pred = custom_prediction(x_test=x_test, model=model, layer_size=layer_size)
        """
        x, time_gd = optimize_input_via_gradient(
            model=model,
            learning_rate=learning_rate,
            starting_point=starting_point,
            iterations=iterations,
        )
        y_opt = model(x.value())
        y_opt_grad = np.array(y_opt)[0][0]
        x_opt_grad = np.array(x.value())[0]

        x_lp, y_lp, y_m_lp, time_lp = optimize_input_via_lp(
            layer_size=layer_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_nn=model,
        )
        print(
            "Training MSE is {:.5f}. \n Testing MSE is {:.5f}.".format(
                train_loss, mean_squared_error(y_test, y_pred)
            )
        )
        print(
            "Minimum of function approximated by NN, calculated by GD is {:.6f} for x = {:.6f}."
            "\n Minimum of function approximated by NN, calculated by LP is {:.6f} for x = {:.6f}.".format(
                y_opt_grad,
                x_opt_grad,
                y_lp,
                x_lp,
            )
        )
        print(
            "Gradient descent: Elapsed time is {:.5f}.\n"
            "Linear program: Elapsed time is {:.5f}.".format(time_gd, time_lp)
        )
        plot(x_test, y_test, y_pred, x_opt_grad, y_opt_grad, x_lp, y_lp)
        """
        train_mse_.append(train_loss)
        test_mse_.append(mean_squared_error(y_test, y_pred))
        """
        x_gd_.append(x_opt_grad)
        y_gd_.append(y_opt_grad)
        t_gd_.append(time_gd)
        x_lp_.append(x_lp)
        y_lp_.append(y_lp)
        t_lp_.append(time_lp)
        """
        num_run -= 1
        num_ += 1
    df = pd.DataFrame(test_mse_)
    df.to_csv(result_p.joinpath("icnn_mse_" + neurons_ext(layer_size) + ".csv"))

    """
    print(
        "On {:2d} runs, average train MSE is {:.5f}, test MSE is {:.5f}, \n "
        "std train is {:.5f}, std test is {:.5f}"
        "x_GD is {:.5f}, y_GD is {:.5f}, time GD is {:.5f},\n"
        "x_LP is {:.5f}, y_LP is {:.5f}, time LP is {:.5f}. \n "
        "Number of times all neurons were dead is {:2d}"
        .format(
            num_,
            mean(train_mse_),
            mean(test_mse_),
            np.std(np.array(train_mse_)),
            np.std(np.array(test_mse_)),
            mean(x_gd_),
            mean(y_gd_),
            mean(t_gd_),
            mean(x_lp_),
            mean(y_lp_),
            mean(t_lp_),
            dying_relu,
        )
    )
    """


def convex_function(x, feature_num):
    """
    Create convex function on (0,1) space.
    param: size of the dataset
    return: input and output dataset
    """
    # y = (x - 0.5) ** 2
    y = x[:, 0] ** 2
    for i in range(1, feature_num):
        y += x[:, i] ** 2
    return y


def train_test_dataset(x, y):
    """
    Divide original dataset on training and testing.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True
    )
    return x_train, x_test, y_train, y_test


def ficnn(layer_size, feature_num):
    """
    Creating Fully Input Convex Neural Network based on the paper of Amos et. al.
    Using Functional API because of passthrough layers.
    param: number of neurons per layer
    return: created model
    """
    # input layer has only one neuron
    input_layer = Input(shape=(feature_num,))
    layer = None
    for n, nn in enumerate(layer_size):
        # first hidden layer
        if layer is None:
            layer = Dense(
                nn,
                activation="relu",
                use_bias=True,
            )(input_layer)
        else:
            layer_forward = Dense(nn, kernel_constraint=non_neg(), use_bias=True)(layer)
            layer_pass = Dense(nn, use_bias=False)(input_layer)
            # adding feedforward and passthrough layer
            layer_merge = Add()([layer_forward, layer_pass])
            # last layer has linear function
            if n == len(layer_size) - 1:
                layer = Activation("linear")(layer_merge)
            # layers before the last have relu functions
            else:
                layer = Activation("relu")(layer_merge)

    # creation of the model based on Functional API
    model = ModelNN(inputs=input_layer, outputs=layer)
    model.save_weights("weights")
    return model


def train_nn(x_train, feature_num, y_train, layer_size, batch_size, epochs):
    """
    Training of the neural network.
    return: model
    """
    model = ficnn(layer_size=layer_size, feature_num=feature_num)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=MeanSquaredError())
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.00005, patience=10)
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        verbose=1,
        callbacks=[early_stopping],
    )
    model.save("model_icnn.h5")
    return model, history.history["loss"][-1]


def custom_prediction(x_test, model, layer_size):
    """
    Retrieve weights from trained neural network, and reconstruct predictions
    from scratch. The use of this function is to verify
    whether the reconstruction of the neural network,
    performed in the linear program, is correct.
    """
    y_pred_model = model.predict(x_test)
    """
    y_pred_custom = []
    theta = extract_weights()
    for x in x_test:
        z = []
        for i in range(len(layer_size)):
            zz = []
            if i == 0:
                for j in range(layer_size[i]):
                    zz.append(
                        max(
                            (x * theta["wz " + str(i)][0][j] + theta["b " + str(i)][j]),
                            0,
                        )
                    )
            elif i < len(layer_size) - 1:
                for j in range(layer_size[i]):
                    zz.append(
                        max(
                            (
                                sum(
                                    z[i - 1][k] * theta["wz " + str(i)][k][j]
                                    for k in range(layer_size[i - 1])
                                )
                                + x * theta["wx " + str(i)][0][j]
                                + theta["b " + str(i)][j]
                            ),
                            0,
                        )
                    )
            else:
                for j in range(layer_size[i]):
                    y = (
                        sum(
                            z[i - 1][k] * theta["wz " + str(i)][k][0]
                            for k in range(layer_size[i - 1])
                        )
                        + x * theta["wx " + str(i)][0][j]
                        + theta["b " + str(i)][j]
                    )
            z.append(zz)
        y_pred_custom.append(y)
    print(
        "The difference between custom predicted and model predicted is {:.5f}".format(
            mean_squared_error(y_pred_custom, y_pred_model)
        )
    )
    """
    return y_pred_model


def plot(x_test, y_test, y_pred, x_opt, y_opt, x_lp, y_lp):
    """
    Plot real data, predicted data by neural network, optimum of the function
    found by gradient descent, and optimum found by linear program.
    """
    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.plot(x_test, y_test, ".", color="blue", label="Real data")
    plt.plot(x_test, y_pred, ".", color="red", label="Predicted data")
    plt.plot(x_opt, y_opt, marker="*", markersize=20, color="green", label="Optimum GD")
    plt.plot(x_lp, y_lp, marker="o", markersize=10, color="yellow", label="Optimum LP")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Y = (X-0.5)^2")
    plt.legend()
    plt.show()
    # plt.savefig(
    #    "Real and predicted function and optimums found via GD and LP " + now + ".png"
    # )


def optimize_input_via_gradient(model, learning_rate, starting_point, iterations):
    """
    Repeatedly optimize input x for specified number of iterations.
    """
    start_time = time.time()
    starting_point = np.array([starting_point])
    # initial value as numpy array of one element (one feature)
    x = tf.Variable(starting_point)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for i in range(iterations):
        gradient = calculate_gradient(x=x, model=model)
        # zip calculated gradient and previous value of x
        zipped = zip([gradient], [x])
        # update value of input variable according to the calculated gradient
        opt.apply_gradients(zipped)
        # have a look at x, and check whether it violates constraints
        # and if it violates set it on closest feasible value.
    end_time = time.time()
    elapsed_time = end_time - start_time
    return x, elapsed_time


def calculate_gradient(x, model):
    """
    Calculate gradient of x with respect to mean squared error loss function.
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x.value(), training=True)
    grads = tape.gradient(y, x).numpy()
    return grads


def extract_weights():
    """
    Extraction of weights from feedforward convex neural network.
    The function differes from weight extraction of regular neural network,
    because of the convex neural network structure difference -- presence of
    passthrough layers.
    return: weights and biases
    """
    model = load_model("model_icnn.h5")
    model.summary()
    theta = {}
    j = 0
    for i, layer in enumerate(model.layers):
        if "dense" in layer.name:
            weights = layer.get_weights()
            if len(weights) == 2:
                theta["wz " + str(j)] = weights[0]
                theta["b " + str(j)] = weights[1]
                j += 1
            elif len(weights) == 1:
                theta["wx " + str(j - 1)] = weights[0]
            else:
                warnings.warn(
                    "Implemented weight extraction procedure might be unsuitable for the current network!"
                )
    return theta


def optimize_input_via_lp(layer_size, lower_bound, upper_bound, model_nn):
    """
    Construct the linear program of ReLu neural network,
    under the knowledge that one of nonlinear constraints representing ReLu
    must be tight, and found the minimum of such linear program.
    """
    theta = extract_weights()
    start_time = time.time()
    model = Model("lp_icnn_inference")
    model.reset()
    # model.Params.Presolve = 0
    x = model.addVar(lb=lower_bound, ub=upper_bound, name="x")
    z = []
    for i, nn in enumerate(layer_size):
        # Note: default lower bound for variable is 0. Therefore, we need to change a default lower bound.
        z.append(model.addVars(nn, lb=-1000, name="z_" + str(i)))
    for i in range(len(layer_size)):
        if i == 0:
            model.addConstrs(
                (
                    z[i][j] >= x * theta["wz " + str(i)][0][j] + theta["b " + str(i)][j]
                    for j in range(layer_size[i])
                ),
                name="I_layer_" + str(i),
            )
            model.addConstrs(
                (z[i][j] >= 0 for j in range(layer_size[i])), name="II_layer_" + str(i)
            )
        elif i < len(layer_size) - 1:
            model.addConstrs(
                (
                    z[i][j]
                    >= sum(
                        z[i - 1][k] * theta["wz " + str(i)][k][j]
                        for k in range(layer_size[i - 1])
                    )
                    + x * theta["wx " + str(i)][0][j]
                    + theta["b " + str(i)][j]
                    for j in range(layer_size[i])
                ),
                name="I_layer_" + str(i),
            )
            model.addConstrs(
                (z[i][j] >= 0 for j in range(layer_size[i])), name="II_layer_" + str(i)
            )
        else:
            model.addConstrs(
                (
                    z[i][j]
                    == sum(
                        z[i - 1][k] * theta["wz " + str(i)][k][0]
                        for k in range(layer_size[i - 1])
                    )
                    + x * theta["wx " + str(i)][0][j]
                    + theta["b " + str(i)][j]
                    for j in range(layer_size[i])
                ),
                name="I_layer_" + str(i),
            )

    model.setObjective(
        z[len(layer_size) - 1][layer_size[len(layer_size) - 1] - 1], GRB.MINIMIZE
    )
    model.write("lp_icnn_inference.lp")
    model.optimize()
    x = model.getAttr("X", [x])
    obj = model.getObjective()
    obj = obj.getValue()
    y_m_lp = model_nn.predict([x])[0]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return x[0], obj, y_m_lp[0], elapsed_time


def optimize_exact(layer_size, lower_bound, upper_bound):
    """
    Exact representation of ReLu network using maximum functions.
    However, probably because of MAX functions, model is infeasible.
    It can be solved only when constraints are relaxed.
    """
    theta = extract_weights()
    model = Model("nlp_inference")
    model.reset()
    x = model.addVar(lb=lower_bound, ub=upper_bound, name="x")
    z = []
    for i, nn in enumerate(layer_size):
        z.append(model.addVars(nn, name="z_" + str(i)))
    for i in range(len(layer_size)):
        if i == 0:
            temp = model.addVars(layer_size[i], name="temp_" + str(i))
            model.addConstrs(
                temp[j] == x * theta["wz " + str(i)][0][j] + theta["b " + str(i)][j]
                for j in range(layer_size[i])
            )
            model.addConstrs(
                (
                    z[i][j]
                    == max_(
                        temp[j],
                        constant=0,
                    )
                    for j in range(layer_size[i])
                ),
                name="I_layer_" + str(i),
            )
        elif i < len(layer_size) - 1:
            temp_ = model.addVars(layer_size[i], name="_temp_" + str(i))
            model.addConstrs(
                temp_[j]
                == sum(
                    z[i - 1][k] * theta["wz " + str(i)][k][j]
                    for k in range(layer_size[i - 1])
                )
                + x * theta["wx " + str(i)][0][j]
                + theta["b " + str(i)][j]
                for j in range(layer_size[i])
            )
            model.addConstrs(
                (
                    z[i][j]
                    == max_(
                        temp_[j],
                        constant=0,
                    )
                    for j in range(layer_size[i])
                ),
                name="I_layer_" + str(i),
            )
        else:
            model.addConstrs(
                (
                    z[i][j]
                    == sum(
                        z[i - 1][k] * theta["wz " + str(i)][k][0]
                        for k in range(layer_size[i - 1])
                    )
                    + x * theta["wx " + str(i)][0][j]
                    + theta["b " + str(i)][j]
                    for j in range(layer_size[i])
                ),
                name="I_layer_" + str(i),
            )
    model.setObjective(
        z[len(layer_size) - 1][layer_size[len(layer_size) - 1] - 1], GRB.MINIMIZE
    )
    model.write("nlp_inference.lp")
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        model.feasRelaxS(1, True, True, False)
        model.optimize()
    x = model.getAttr("X", [x])
    obj = model.getObjective()
    obj = obj.getValue()
    return x[0], obj


if __name__ == "__main__":
    result_p: Path = Path(__file__).parents[1] / "2D_convex_function"
    layer_sizes = [[100,100,100,1]]
    for layer_size in layer_sizes:
        main(
            size=10000,
            feature_num=10,
            lower_bound=-1,
            upper_bound=1,
            layer_size=layer_size,
            batch_size=64,
            epochs=1000,
            iterations=1000,
            starting_point=0.1,
            learning_rate=0.001,
            num_run=5,
        )

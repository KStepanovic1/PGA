import numpy as np
import warnings
import tensorflow as tf
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

from datetime import datetime
from gurobipy import *
from tensorflow.keras.layers import Input, Dense, Add, Activation
from tensorflow.keras.models import load_model
from statistics import mean

# the model from tensorflow.keras can not be imported as Model
# because it will overwrite Model from gurobipy
# from gurobipy import * has a Model in itself
from tensorflow.keras import Model as ModelNN
from tensorflow.keras import regularizers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def main(
    size,
    feature_num,
    lower_bound_,
    upper_bound_,
    lower_bound,
    upper_bound,
    layer_size,
    batch_size,
    epochs,
    iterations,
    starting_points,
    learning_rate,
    constraint_x,
    constraint_y,
):
    colors = ["green", "cyan"]
    x_gd_, y_gd_, y_starting_point = [], [], []
    x = np.linspace(lower_bound_, upper_bound_, size)
    y = nonconvex_function(x)
    x, y = normalize(x, y, lower_bound, upper_bound)
    x_train, x_test, y_train, y_test = train_test_dataset(x=x, y=y)
    model, training_loss = train_nn(
        x_train=x_train,
        feature_num=feature_num,
        y_train=y_train,
        layer_size=layer_size,
        batch_size=batch_size,
        epochs=epochs,
    )
    y_pred = custom_prediction(x_test=x_test, model=model, layer_size=layer_size)
    for starting_point in starting_points:
        x, time_gd = optimize_input_via_gradient(
            model=model,
            learning_rate=learning_rate,
            starting_point=starting_point,
            iterations=iterations,
            constraint_x=constraint_x,
            constraint_y=constraint_y,
        )
        y_opt = model(x.value())
        y_opt_grad = np.array(y_opt)[0][0]
        x_opt_grad = np.array(x.value())[0]
        x_gd_.append(x_opt_grad)
        y_gd_.append(y_opt_grad)
        y_starting_point.append(np.array(model(np.array([starting_point]))))
    x_ind, y_ind, y_ind_nn, time_ind = optimize_input_via_indicator_constraint(
        layer_size=layer_size,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        model_nn=model,
        constraint_x=constraint_x,
        constraint_y=constraint_y,
    )
    print(
        "Training MSE is {:.5f}. \n Testing MSE is {:.5f}".format(
            training_loss, mean_squared_error(y_test, y_pred)
        )
    )
    print(
        "Gradient descent: Optimal input is {:.5f} and value of objective is {:.5f}. \n"
        "Indicator constraints: Optimal input is {:.5f} and value of objective is {:.5f}.".format(
            x_opt_grad, y_opt_grad, x_ind, y_ind
        )
    )
    print(
        "Gradient descent: Elapsed time is {:.5f}. \n "
        "Indicator constraints: Elapsed time is {:.5f}.".format(time_gd, time_ind)
    )
    plot(
        x_test,
        y_test,
        y_pred,
        x_gd_,
        y_gd_,
        starting_points,
        y_starting_point,
        colors,
        x_ind,
        y_ind,
        constraint_x,
        constraint_y,
    )


def nonconvex_function(x):
    """
    Create nonconvex function.
    """
    y = []
    for i in range(len(x)):
        y.append(
            2 * math.sin(x[i] * (2 * math.pi / 500))
            + math.sin(x[i] * (2 * math.pi / 1000))
            + 3 * math.cos(x[i] * (2 * math.pi / 200))
        )
    return np.array(y)


def normalize(x, y, lower_bound, upper_bound):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
    scaler.fit(x)
    x = scaler.transform(x)
    scaler.fit(y)
    y = scaler.transform(y)
    return x, y


def train_test_dataset(x, y) -> None:
    """
    Divide original dataset on training and testing.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True
    )
    return x_train, x_test, y_train, y_test


def plnn(layer_size, feature_num):
    """
    Creating Piecewise linear neural network.
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
            # last layer has linear function
            if n == len(layer_size) - 1:
                layer = Activation("linear")(layer_n)
            # layers before the last have relu functions
            else:
                layer = Activation("relu")(layer_n)

    # creation of the model based on Functional API
    model = ModelNN(inputs=input_layer, outputs=layer)
    model.save_weights("weights")
    return model


def train_nn(x_train, feature_num, y_train, layer_size, batch_size, epochs):
    """
    Training of the neural network.
    return: model
    """
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    model = plnn(layer_size=layer_size, feature_num=feature_num)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=MeanSquaredError())
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.0001, patience=10)
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        verbose=1,
        callbacks=[early_stopping],
    )
    model.save("model_plnn.h5")
    return model, history.history["loss"][-1]


def custom_prediction(x_test, model, layer_size):
    """
    Retrieve weights from trained neural network, and reconstruct predictions
    from scratch. The use of this function is to verify
    whether the reconstruction of the neural network,
    performed in the linear program, is correct.
    """
    y_pred_model = model.predict(x_test)
    return y_pred_model


def plot(
    x_test,
    y_test,
    y_pred,
    x_opt,
    y_opt,
    starting_points,
    y_starting_point,
    colors,
    x_ind,
    y_ind,
    constraint_x,
    constraint_y,
):
    """
    Plot real data, predicted data by neural network, optimum of the function
    found by gradient descent, and optimum found by linear program.
    """
    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.plot(x_test, y_test, ".", color="blue", label="Real data")
    plt.plot(x_test, y_pred, ".", color="red", label="Predicted data")
    for i, starting_point in enumerate(starting_points):
        plt.plot(
            starting_point,
            y_starting_point[i],
            marker="v",
            markersize=10,
            color=colors[i],
            label="Starting point " + str(i),
        )
        plt.plot(
            x_opt[i],
            y_opt[i],
            marker="*",
            markersize=10,
            color=colors[i],
            label="Optimum GD " + str(i),
        )
    plt.plot(
        x_ind,
        y_ind,
        marker="o",
        markersize=10,
        color="yellow",
        label="Optimum disjunctive MILP",
    )
    # plt.plot(x_test, -0.1 * x_test + 0.1, label="-0.1*x+0.1")
    # plt.axhline(y=constraint_y[0], label="LB")
    plt.axhline(y=constraint_y, label="Q", color="black")
    plt.axvline(x=constraint_x, label="$H_{max}$", color="pink")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Nonconvex function")
    plt.legend()
    plt.savefig(
        "Real and predicted function and optimums found via GD and LP " + now + ".png"
    )
    plt.show()


def optimize_input_via_gradient(
    model, learning_rate, starting_point, iterations, constraint_x, constraint_y
):
    """
    Repeatedly optimize input x for specified number of iterations.
    """
    start_time = time.time()
    starting_point = np.array([starting_point])
    # initial value as numpy array of one element (one feature)
    x = tf.Variable(starting_point)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for i in range(iterations):
        gradient = calculate_gradient(x=x, model=model, constraint_y=constraint_y)
        # zip calculated gradient and previous value of x
        zipped = zip([gradient], [x])
        # update value of input variable according to the calculated gradient
        opt.apply_gradients(zipped)
        # projected gradient descent
        x_ = min(np.array(x.value())[0], constraint_x)
        x.assign(np.array([x_]))
    end_time = time.time()
    elapsed_time = end_time - start_time
    return x, elapsed_time


def calculate_gradient(x, model, constraint_y):
    """
    Calculate gradient of x with respect to mean squared error loss function.
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = (
            model(x.value(), training=True)
            + 25 * (model(x.value(), training=True) - constraint_y) ** 2
        )
    grads = tape.gradient(y, x).numpy()
    return grads


def reform(weight):
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
                w = round(weight[i][j], 6)
                t.append(w)
            weight_.append(t)
    elif n_dim == 1:
        for i in range(weight.shape[0]):
            w = round(weight[i], 6)
            weight_.append(w)
    weight_ = np.array(weight_)
    return weight_


def extract_weights():
    """
    Extraction of weights from feedforward convex neural network.
    The function differes from weight extraction of regular neural network,
    because of the convex neural network structure difference -- presence of
    passthrough layers.
    return: weights and biases
    """
    model = load_model("model_plnn.h5")
    model.summary()
    theta = {}
    j = 0
    for i, layer in enumerate(model.layers):
        if "dense" in layer.name:
            weights = layer.get_weights()
            if len(weights) == 2:
                theta["wz " + str(j)] = reform(weights[0])
                theta["b " + str(j)] = reform(weights[1])
                j += 1
            elif len(weights) == 1:
                theta["wx " + str(j - 1)] = weights[0]
            else:
                warnings.warn(
                    "Implemented weight extraction procedure might be unsuitable for the current network!"
                )
    return theta


def optimize_input_via_indicator_constraint(
    layer_size, lower_bound, upper_bound, model_nn, constraint_x, constraint_y
):
    """
    Piecewise linear neural network implemented as a mixed-integer linear program
    following framework by Fiscetti and Jo et. al. Bilinear equivalent constraints
    whose purpose is to set either z or s variable to zero in order to recover ReLu
    function are transformed to indicator constraints to facillitate mathematical solver.
    """
    a, z, s = [], [], []
    theta = extract_weights()
    start_time = time.time()
    model = Model("ind_con_plnn_inference")
    model.reset()
    # input to the neural network
    x = model.addVar(lb=lower_bound, ub=upper_bound, name="x")
    model.addConstr(x <= constraint_x, name="constraint_x")
    # output of the hidden layers
    for i, nn in enumerate(layer_size[:-1]):
        # Note: default lower bound for variable is 0. Therefore, we need to change a default lower bound.
        z.append(
            model.addVars(
                nn, vtype=GRB.CONTINUOUS, lb=0, ub=+GRB.INFINITY, name="z_" + str(i)
            )
        )
        s.append(
            model.addVars(
                nn, vtype=GRB.CONTINUOUS, lb=0, ub=+GRB.INFINITY, name="s_" + str(i)
            )
        )
        a.append(model.addVars(nn, vtype=GRB.BINARY, name="a_" + str(i)))
    y = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=+GRB.INFINITY, name="y")
    for i in range(len(layer_size)):
        if i == 0:
            model.addConstrs(
                (
                    z[i][j] - s[i][j]
                    == x * theta["wz " + str(i)][0][j] + theta["b " + str(i)][j]
                    for j in range(layer_size[i])
                ),
                name="I_layer_" + str(i),
            )
            for j in range(layer_size[i]):
                model.addGenConstrIndicator(
                    a[i][j],
                    True,
                    z[i][j],
                    GRB.LESS_EQUAL,
                    0.0,
                    name="indicator_constraint_z_" + str(i),
                )
                model.addGenConstrIndicator(
                    a[i][j],
                    False,
                    s[i][j],
                    GRB.LESS_EQUAL,
                    0.0,
                    name="indicator_constraint_s_" + str(i),
                )
        elif i < len(layer_size) - 1:
            model.addConstrs(
                (
                    z[i][j] - s[i][j]
                    == sum(
                        z[i - 1][k] * theta["wz " + str(i)][k][j]
                        for k in range(layer_size[i - 1])
                    )
                    + theta["b " + str(i)][j]
                    for j in range(layer_size[i])
                ),
                name="I_layer_" + str(i),
            )
            for j in range(layer_size[i]):
                model.addGenConstrIndicator(
                    a[i][j],
                    True,
                    z[i][j],
                    GRB.LESS_EQUAL,
                    0.0,
                    name="indicator_constraint_z_" + str(i),
                )
                model.addGenConstrIndicator(
                    a[i][j],
                    False,
                    s[i][j],
                    GRB.LESS_EQUAL,
                    0.0,
                    name="indicator_constraint_s_" + str(i),
                )
        else:
            model.addConstrs(
                (
                    y
                    == sum(
                        z[i - 1][k] * theta["wz " + str(i)][k][0]
                        for k in range(layer_size[i - 1])
                    )
                    + theta["b " + str(i)][j]
                    for j in range(layer_size[i])
                ),
                name="output_layer",
            )
            # constraint on output
            model.addConstr((y >= constraint_y), name="out_high")
    model.setObjective(y, GRB.MINIMIZE)
    model.write("ind_con_plnn_inference.lp")
    model.optimize()
    x = model.getAttr("X", [x])
    obj = model.getObjective()
    obj = obj.getValue()
    y_m_lp = model_nn.predict([x])[0]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return x[0], obj, y_m_lp[0], elapsed_time


if __name__ == "__main__":
    main(
        size=100000,
        feature_num=1,
        lower_bound_=-400,
        upper_bound_=200,
        lower_bound=0,
        upper_bound=1,
        layer_size=[10, 10, 10, 1],
        batch_size=64,
        epochs=10000,
        iterations=1000,
        starting_points=[0.1, 0.95],
        learning_rate=0.001,
        constraint_x=0.9,
        constraint_y=0.5,
    )

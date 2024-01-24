import numpy as np
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from datetime import datetime
from gurobipy import *
from statistics import mean

# the model from tensorflow.keras can not be imported as Model
# because it will overwrite Model from gurobipy
# from gurobipy import * has a Model in itself
from tensorflow.keras import Model as ModelNN
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Input, Dense, Add, Activation
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


"""
Finding the minimum of nonconvex function approximated with ICNN with custom loss.
"""


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
    x_gd_, y_gd_, time_gd_ = [], [], []
    x = np.linspace(lower_bound_, upper_bound_, size)
    y = nonconvex_function(x)
    x, y = normalize(x, y, lower_bound, upper_bound)
    x_train, x_test, y_train, y_test = train_test_dataset(x=x, y=y)
    model, train_loss = train_nn(
        x_train=x_train,
        feature_num=feature_num,
        y_train=y_train,
        layer_size=layer_size,
        batch_size=batch_size,
        epochs=epochs,
        alpha=0.9,
        i=0,
        constraint_y=constraint_y,
    )
    y_pred, y_pred_train = prediction(x_train=x_train, x_test=x_test, model=model)
    plt.plot(x_test, y_test, ".", label="Real data", color="blue")
    plt.plot(x_test, y_pred, ".", label="Predicted data", color="red")
    for i, starting_point in enumerate(starting_points):
        x, time_gd = optimize_input_via_gradient(
            model=model,
            learning_rate=learning_rate,
            starting_point=starting_point,
            iterations=iterations,
            constraint_x=constraint_x,
            constraint_y=constraint_y,
        )
        time_gd_.append(time_gd)
        y_opt = model(x.value())
        y_opt_grad = np.array(y_opt)[0][0]
        x_opt_grad = np.array(x.value())[0]
        x_gd_.append(x_opt_grad)
        y_gd_.append(y_opt_grad)
        plt.plot(
            starting_point,
            np.array(model(np.array([starting_point]))),
            marker="v",
            markersize=10,
            label="Starting point " + str(i),
            color=colors[i],
        )
        plt.plot(
            x_opt_grad,
            y_opt_grad,
            marker="*",
            markersize=20,
            label="Optimum GD " + str(i),
            color=colors[i],
        )
    x_lp, y_lp, y_m_lp, time_lp = optimize_input_via_lp(
        layer_size=layer_size,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        model_nn=model,
        constraint_x=constraint_x,
        constraint_y=constraint_y,
    )
    plt.plot(x_lp, y_lp, marker="o", markersize=10, color="yellow", label="Optimum LP")
    plt.axhline(y=constraint_y, label="Q", color="black")
    plt.axvline(x=constraint_x, label="$H_{max}$", color="pink")
    plt.legend()
    plt.title("Non-convex function")
    plt.savefig("Custom loss functions for ICNNs.png")
    plt.show()
    print(
        "Training MSE is {:.5f}. Testing MSE is {:.5f}.".format(
            mean_squared_error(y_train, y_pred_train),
            mean_squared_error(y_test, y_pred),
        )
    )
    print(
        "Minimum of function approximated by NN, calculated by GD 1 is {:.6f} for x = {:.6f}.\n"
        "Minimum of function approximated by NN, calculated by GD 2 is {:.6f} for x = {:.6f}. \n"
        "\n Minimum of function approximated by NN, calculated by LP is {:.6f} for x = {:.6f}.".format(
            y_gd_[0],
            x_gd_[0],
            y_gd_[1],
            x_gd_[1],
            y_lp,
            x_lp,
        )
    )
    print(
        "Gradient descent 1: Elapsed time is {:.5f}.\n"
        "Gradient descent 2: Elapsed time is {:.5f}.\n"
        "Linear program: Elapsed time is {:.5f}.".format(
            time_gd_[0], time_gd_[1], time_lp
        )
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
    """
    Normalize input and output between 0 and 1.
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
    scaler.fit(x)
    x = scaler.transform(x)
    scaler.fit(y)
    y = scaler.transform(y)
    return x, y


def train_test_dataset(x, y):
    """
    Divide original dataset on training and testing.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True
    )
    return x_train, x_test, y_train, y_test


def activation_(x):
    """
    Specify activation function.
    """
    return tf.nn.relu(x)


def custom_loss(alpha, y_constraint, i):
    """
    Customized MSE that tries to encourage lowest error close to constraint.
    """

    def loss(y_true, y_pred):
        if i == 0:
            mean_sqr_error = K.mean(K.abs(y_true - y_pred))
            p = K.mean(alpha * relu(y_pred - y_constraint))
            error = mean_sqr_error + p
        elif i == 1:
            mean_sqr_error = K.mean(K.square(y_true - y_pred))
            p = K.mean(alpha * K.square(y_pred - y_constraint))
            error = mean_sqr_error + p
        else:
            error = K.mean((1 - y_true) * K.square(y_true - y_pred))
        return error

    return loss


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
                use_bias=True,
            )(input_layer)
            layer = activation_(layer)
        else:
            layer_forward = Dense(
                nn,
                kernel_constraint=non_neg(),
                use_bias=True,
            )(layer)
            layer_pass = Dense(
                nn,
                use_bias=False,
            )(input_layer)
            # adding feedforward and passthrough layer
            layer_merge = Add()([layer_forward, layer_pass])
            # last layer has linear function
            if n == len(layer_size) - 1:
                layer = Activation("linear")(layer_merge)
            # layers before the last have relu functions
            else:
                layer = activation_(x=layer_merge)

    # creation of the model based on Functional API
    model = ModelNN(inputs=input_layer, outputs=layer)
    model.save_weights("weights")
    return model


def train_nn(
    x_train,
    feature_num,
    y_train,
    layer_size,
    batch_size,
    epochs,
    alpha,
    i,
    constraint_y,
):
    """
    Training of the neural network.
    return: model
    """
    model = ficnn(layer_size=layer_size, feature_num=feature_num)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt, loss=custom_loss(alpha=alpha, y_constraint=constraint_y, i=i)
    )
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.00005, patience=15)
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        verbose=1,
        callbacks=[early_stopping],
    )
    model.save("model_non_convex_icnn_control.h5")
    return model, history.history["loss"][-1]


def extract_weights():
    """
    Extraction of weights from feedforward convex neural network.
    The function differes from weight extraction of regular neural network,
    because of the convex neural network structure difference -- presence of
    passthrough layers.
    return: weights and biases
    """
    model = load_model("model_non_convex_icnn_control.h5", compile=False)
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


def prediction(x_train, x_test, model):
    """
    Predicts the output based on the input and the model
    """
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    return y_pred, y_pred_train


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
        # have a look at x, and check whether it violates constraints
        # and if it violates set it on closest feasible value.
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
        y = model(x.value(), training=True)
        y = (
            model(x.value(), training=True)
            + 25 * (model(x.value(), training=True) - constraint_y) ** 2
        )
    grads = tape.gradient(y, x).numpy()
    return grads


def optimize_input_via_lp(
    layer_size, lower_bound, upper_bound, model_nn, constraint_x, constraint_y
):
    """
    Construct the linear program of ReLu neural network,
    under the knowledge that one of nonlinear constraints representing ReLu
    must be tight, and found the minimum of such linear program.
    """
    theta = extract_weights()
    start_time = time.time()
    model = Model("non_convex_lp_icnn_control")
    model.reset()
    x = model.addVar(lb=lower_bound, ub=upper_bound, name="x")
    model.addConstr(x <= constraint_x, name="constraint_x")
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
            model.addConstrs(
                (z[i][j] >= constraint_y for j in range(layer_size[i])),
                name="constraint_y",
            )

    model.setObjective(
        z[len(layer_size) - 1][layer_size[len(layer_size) - 1] - 1], GRB.MINIMIZE
    )
    model.write("non_convex_lp_icnn_inference.lp")
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
        layer_size=[20, 20, 20, 20, 1],
        batch_size=64,
        epochs=500,
        iterations=50,
        starting_points=[0.05, 0.95],
        learning_rate=0.01,
        constraint_x=0.9,
        constraint_y=0.5,
    )

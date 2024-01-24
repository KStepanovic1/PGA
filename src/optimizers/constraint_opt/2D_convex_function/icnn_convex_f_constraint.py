import numpy as np
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from datetime import datetime
from gurobipy import *
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
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
Finding the minimum of convex constraint function using custom loss for training.
"""


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
    colors,
    learning_rate,
    constraint_y,
    constraint_x,
):
    x_gd_, y_gd_, x_lp_, y_lp_ = [], [], [], []
    for start_point in starting_point:
        x = np.linspace(lower_bound, upper_bound, size)
        y = convex_function(x)
        x_train, x_test, y_train, y_test = train_test_dataset(x=x, y=y)
        model, train_loss = train_nn(
            x_train=x_train,
            feature_num=feature_num,
            y_train=y_train,
            layer_size=layer_size,
            batch_size=batch_size,
            epochs=epochs,
        )
        y_pred = model.predict(x_test)
        x = optimize_input_via_gradient(
            model=model,
            learning_rate=learning_rate,
            starting_point=start_point,
            iterations=iterations,
            constraint_x=constraint_x,
            constraint_y=constraint_y,
        )
        y_opt = model(x.value())
        y_opt_grad = np.array(y_opt)[0][0]
        x_opt_grad = np.array(x.value())[0]
        x_lp, y_lp = optimize_input_via_lp(
            layer_size=layer_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            constraint_x=constraint_x,
            constraint_y=constraint_y,
        )
        print(
            "Training MSE is {:.5f}. \n Testing MSE is {:.5f}.".format(
                train_loss, mean_squared_error(y_test, y_pred)
            )
        )
        print(
            "Minimum of function approximated by NN, calculated by GD is {:.6f} for x = {:.6f}.".format(
                y_opt_grad,
                x_opt_grad,
            )
        )
        print(
            "Minimum of function approximated by NN, calculated by LP is {:.6f} for x = {:.6f}.".format(
                y_lp,
                x_lp,
            )
        )
        x_gd_.append(x_opt_grad)
        x_lp_.append(x_lp)
        y_gd_.append(y_opt_grad)
        y_lp_.append(y_lp)
    plot(
        x_test,
        y_test,
        y_pred,
        x_gd_,
        x_lp_,
        y_gd_,
        y_lp_,
        constraint_y,
        constraint_x,
        starting_point,
        model,
        colors,
    )


def convex_function(x):
    """
    Create convex function on (0,1) space.
    param: size of the dataset
    return: input and output dataset
    """
    y = (x - 0.5) ** 2
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


def custom_loss(alpha):
    """
    Customized MSE that tries to encourage lowest error in optimum.
    """

    def loss(data, y_pred):
        input = tf.reshape(data[:, 0], (-1, 1))
        y_true = tf.reshape(data[:, 1], (-1, 1))
        mean_sqr_error = K.mean(K.square(y_true - y_pred))
        p = K.mean(alpha * relu(0.1 - y_pred))
        # -0.1 * input + 0.1 - y_pred)
        # )
        return mean_sqr_error + p

    return loss


def train_nn(x_train, feature_num, y_train, layer_size, batch_size, epochs):
    """
    Training of the neural network.
    return: model
    """
    x_train = np.expand_dims(x_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    data = np.append(x_train, y_train, axis=1)
    model = ficnn(layer_size=layer_size, feature_num=feature_num)
    opt = tf.keras.optimizers.Adam(learning_rate=0.00003)
    model.compile(optimizer=opt, loss=custom_loss(alpha=0.5))
    history = model.fit(
        x=x_train,
        y=data,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        verbose=1,
    )
    model.save("model_icnn.h5")
    return model, history.history["loss"][epochs - 1]


def plot(
    x_test,
    y_test,
    y_pred,
    x_gd_,
    x_lp_,
    y_gd_,
    y_lp_,
    constraint_y,
    constraint_x,
    starting_point,
    model,
    colors,
):
    """
    Plot real data, predicted data by neural network, optimum of the function
    found by gradient descent, and optimum found by linear program.
    """
    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.plot(x_test, y_test, ".", color="blue", label="Real data")
    plt.plot(x_test, y_pred, ".", color="red", label="Predicted data")
    for i, start_point in enumerate(starting_point):
        plt.plot(
            x_gd_[i],
            y_gd_[i],
            marker="*",
            markersize=20,
            color=colors[i],
            label="Optimum GD " + str(i),
        )
        plt.plot(
            x_lp_[i],
            y_lp_[i],
            marker="o",
            markersize=10,
            color=colors[i],
            label="Optimum LP " + str(i),
        )
        plt.plot(
            start_point,
            model(np.array([start_point])),
            marker="v",
            markersize=10,
            color=colors[i],
            label="Start point " + str(i),
        )

    # plt.plot(x_test, -0.1 * x_test + 0.1, label="-0.1*x+0.1")
    plt.axhline(y=0.1, label="LB")
    # plt.axhline(y=constraint_y[1], label="UB")
    plt.axvline(x=constraint_x)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Y = (X-0.5)^2")
    plt.legend()
    plt.savefig(
        "Real and predicted function and optimums found via GD and LP " + now + ".png"
    )
    plt.show()


def optimize_input_via_gradient(
    model, learning_rate, starting_point, iterations, constraint_x, constraint_y
):
    """
    Repeatedly optimize x for specific number of iterations.
    """
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
        # projected gradient descent
        """
        if np.array(x.value())[0] > constraint_x:
            x_ = np.clip(np.array(x.value())[0], 0, constraint_x)
            x.assign(np.array([x_]))
        """
        # Google data center cooling equation 7 (GD optimization in the presence of range constraints)
        x_ = min(np.array(x.value())[0], constraint_x)
        x.assign(np.array([x_]))
    return x


def calculate_gradient(x, model, constraint_y):
    """
    Optimization via gradient with constraints on network's output.
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        # multiobjective optimization if there are constraints on y
        y = (
            model(x.value(), training=True)
            + 1 * (model(x.value(), training=True) - 0.1)
            # + 0.1 * np.array(x.value())[0] - 0.1)
            ** 2
        )
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
    model = load_model("model_icnn.h5", compile=False)
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


def optimize_input_via_lp(
    layer_size, lower_bound, upper_bound, constraint_x, constraint_y
):
    """
    Construct the linear program of ReLu neural network,
    under the knowledge that one of nonlinear constraints representing ReLu
    must be tight, and found the minimum of such linear program.
    """
    theta = extract_weights()
    model = Model("lp_icnn_control")
    model.reset()
    # model.Params.Presolve = 0
    x = model.addVar(lb=lower_bound, ub=upper_bound, name="x")
    z = []
    for i, nn in enumerate(layer_size):
        # Note: default lower bound for variable is 0. Therefore, we need to change a default lower bound.
        z.append(model.addVars(nn, lb=-1000, name="z_" + str(i)))
    # input constraint
    model.addConstr(x <= constraint_x, name="constraint_x")
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
            # constraint on output
            """
            model.addConstrs(
                (z[i][j] >= -0.1 * x + 0.1 for j in range(layer_size[i])),
                name="out_high",
            )
            """
            model.addConstrs(
                (z[i][j] >= 0.1 for j in range(layer_size[i])),
                name="out_high",
            )

    model.setObjective(
        z[len(layer_size) - 1][layer_size[len(layer_size) - 1] - 1], GRB.MINIMIZE
    )
    model.write("lp_icnn_control.lp")
    model.optimize()
    x = model.getAttr("X", [x])
    obj = model.getObjective()
    obj = obj.getValue()
    for v in model.getVars():
        print(v.VarName, v.x)
    return x[0], obj


if __name__ == "__main__":
    main(
        size=10000,
        feature_num=1,
        lower_bound=0,
        upper_bound=1,
        layer_size=[10, 10, 10, 1],
        batch_size=64,
        epochs=2000,
        iterations=10000,
        starting_point=[0.05, 0.95],
        colors=["green", "cyan"],
        learning_rate=0.0001,
        constraint_y=[0.1],
        constraint_x=0.40,
    )

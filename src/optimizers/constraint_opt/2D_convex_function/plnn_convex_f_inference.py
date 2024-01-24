import numpy as np
import pandas as pd
import math
import warnings
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from pathlib import Path
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .keras_extension import ExtendedTensorBoard
from .icnn_convex_f_inference import neurons_ext

"""
Finding the minimum of convex function approximated with piecewise linear network
with gradient descent and mixed-integer linear programming -- big-M and indicator constraints.
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
    learning_rate,
    num_run,
):
    path = pathlib.Path(__file__).parents[1] / "2D_convex_function"
    x_train_p = os.path.join(path, "x_train.csv")
    y_train_p = os.path.join(path, "y_train.csv")
    x_test_p = os.path.join(path, "x_train.csv")
    y_test_p = os.path.join(path, "y_train.csv")
    (
        train_mse_,
        test_mse_,
        x_gd_,
        y_gd_,
        t_gd_,
        x_ind_,
        y_ind_,
        t_ind_,
        t_relax_,
        x_big_,
        y_big_,
        t_big_,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [])
    dying_relu = 0
    num_ = 0
    while num_run > 0:
        x = np.linspace(lower_bound, upper_bound, size * feature_num)
        x = x.reshape((size, feature_num))
        y = convex_function(x, feature_num)
        train_test_dataset(
            x=x,
            y=y,
            x_train_p=x_train_p,
            y_train_p=y_train_p,
            x_test_p=x_test_p,
            y_test_p=y_test_p,
        )
        x_train, y_train, x_test, y_test = read_dataset(
            x_train_p=x_train_p,
            y_train_p=y_train_p,
            x_test_p=x_test_p,
            y_test_p=y_test_p,
        )
        model, training_loss = train_nn(
            x_train=x_train,
            feature_num=feature_num,
            y_train=y_train,
            layer_size=layer_size,
            batch_size=batch_size,
            epochs=epochs,
        )
        # if the train loss corresponds to all dead ReLus,
        # we skip the current iteraion, and increase the number of dead ReLu networks.
        # if training_loss >= 0.005:
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
        x_ind, y_ind, y_ind_nn, time_ind = optimize_input_via_indicator_constraint(
            layer_size=layer_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_nn=model,
        )
        """
        """
        upper_bounds, lower_bounds, time_bound_relax = bounds_via_layer_relaxation(
            layer_size=layer_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        x_big_m, y_big_m, y_big_nn, time_big_m = optimize_input_via_big_M(
            layer_size=layer_size,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            model_nn=model,
        )
        """
        """
        x_big_m, y_big_m, y_big_nn, time_big_m, time_bound_relax = 0, 0, 0, 0, 0
        plt.plot(x_test, y_test, ".", color="red", label="Real")
        plt.plot(x_test, y_pred, ".", color="blue", label="Predicted")
        plt.plot(
            x_big_m,
            y_big_m,
            marker="*",
            markersize=10,
            color="green",
            label="Big M optimum",
        )
        plt.legend()
        plt.show()
        print(
            "Training MSE is {:.5f}. \n Testing MSE is {:.5f}".format(
                training_loss, mean_squared_error(y_test, y_pred)
            )
        )
        print(
            "Gradient descent: Optimal input is {:.5f} and value of objective is {:.5f}. \n"
            "Indicator constraints: Optimal input is {:.5f} and value of objective is {:.5f} \n "
            "Big-M formulation: Optimal input is {:.5f} and value of objective is {:.5f}".format(
                x_opt_grad, y_opt_grad, x_ind, y_ind, x_big_m, y_big_m
            )
        )
        print(
            "Gradient descent: Elapsed time is {:.5f}. \n "
            "Indicator constraints: Elapsed time is {:.5f}. \n "
            "Bound relaxation: Elapsed time is {:.5f} \n "
            "Big_M formulation: Elapsed time is {:.5f}.".format(
                time_gd, time_ind, time_bound_relax, time_big_m
            )
        )
        plot(
            x_test,
            y_test,
            y_pred,
            x_opt_grad,
            y_opt_grad,
            x_ind,
            y_ind,
            x_big_m,
            y_big_m,
        )
        train_mse_.append(training_loss)
        """
        test_mse_.append(mean_squared_error(y_test, y_pred))
        """
        x_gd_.append(x_opt_grad)
        y_gd_.append(y_opt_grad)
        t_gd_.append(time_gd)
        x_ind_.append(x_ind)
        y_ind_.append(y_ind)
        t_ind_.append(time_ind)
        t_relax_.append(time_bound_relax)
        x_big_.append(x_big_m)
        y_big_.append(y_big_m)
        t_big_.append(time_big_m)
        """
        num_run -= 1
        num_ += 1
    df = pd.DataFrame(test_mse_)
    df.to_csv(result_p.joinpath("plnn_mse_" + neurons_ext(layer_size) + ".csv"))
    """
    print(
        "On {:2d} runs, average train MSE is {:.5f}, test MSE is {:.5f}, \n "
        "std train is {:.5f}, std test is {:.5f}"
        "x_GD is {:.5f}, y_GD is {:.5f}, time GD is {:.5f},\n"
        "x_ind is {:.5f}, y_ind is {:.5f}, time IND is {:.5f}. \n "
        "Time for bound relaxation is {:.5f}. \n "
        "x_big_M is {:.5f}, y_big_M is {:.5f}, time big-M is {:.5f}. \n "
        "Number of times all neurons were dead is {:2d}".format(
            num_,
            mean(train_mse_),
            mean(test_mse_),
            np.std(np.array(train_mse_)),
            np.std(np.array(test_mse_)),
            mean(x_gd_),
            mean(y_gd_),
            mean(t_gd_),
            mean(x_ind_),
            mean(y_ind_),
            mean(t_ind_),
            mean(t_relax_),
            mean(x_big_),
            mean(y_big_),
            mean(t_big_),
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


def train_test_dataset(x, y, x_train_p, y_train_p, x_test_p, y_test_p) -> None:
    """
    Divide original dataset on training and testing.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True
    )
    pd.DataFrame(x_train).to_csv(x_train_p, index=False)
    pd.DataFrame(y_train).to_csv(y_train_p, index=False)
    pd.DataFrame(x_test).to_csv(x_test_p, index=False)
    pd.DataFrame(y_test).to_csv(y_test_p, index=False)


def read_dataset(x_train_p, y_train_p, x_test_p, y_test_p):
    """
    Read the saved dataset.
    """
    with open(x_train_p) as file:
        x_train = np.array(pd.read_csv(file))
    with open(y_train_p) as file:
        y_train = np.array(pd.read_csv(file))
    with open(x_test_p) as file:
        x_test = np.array(pd.read_csv(file))
    with open(y_test_p) as file:
        y_test = np.array(pd.read_csv(file))
    return x_train, y_train, x_test, y_test


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
    board_callback = ExtendedTensorBoard(log_dir=log_dir, histogram_freq=1)
    model = plnn(layer_size=layer_size, feature_num=feature_num)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=MeanSquaredError())
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        verbose=1,
        callbacks=[board_callback],
    )
    model.save("model_plnn.h5")
    return model, history.history["loss"][epochs - 1]


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


def plot(x_test, y_test, y_pred, x_opt, y_opt, x_ind, y_ind, x_big_m, y_big_m):
    """
    Plot real data, predicted data by neural network, optimum of the function
    found by gradient descent, and optimum found by linear program.
    """
    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.plot(x_test, y_test, ".", color="blue", label="Real data")
    plt.plot(x_test, y_pred, ".", color="red", label="Predicted data")
    plt.plot(x_opt, y_opt, marker="*", markersize=20, color="green", label="Optimum GD")
    plt.plot(
        x_ind,
        y_ind,
        marker="o",
        markersize=10,
        color="yellow",
        label="Optimum disjunctive MILP",
    )
    plt.plot(
        x_big_m,
        y_big_m,
        marker="v",
        markersize=10,
        color="cyan",
        label="Optimum big_M MILP",
    )
    # plt.plot(x_test, -0.1 * x_test + 0.1, label="-0.1*x+0.1")
    # plt.axhline(y=constraint_y[0], label="LB")
    # plt.axhline(y=constraint_y[1], label="UB")
    # plt.axvline(x=0.4)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Y = (X-0.5)^2")
    plt.legend()
    plt.savefig(
        "Real and predicted function and optimums found via GD and LP " + now + ".png"
    )
    plt.show()


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


def optimize_input_via_lp(layer_size, lower_bound, upper_bound, model_nn):
    """
    Construct the linear program of ReLu neural network,
    under the knowledge that one of nonlinear constraints representing ReLu
    must be tight, and found the minimum of such linear program.
    However, as input x, and the output of the hidden layer z_0
    is not convex with respect to the output of the last layer z_2
    due to presence of negative weights, inequality constraints will not be tight.
    Therefore, formulation as linear program do not work.
    """
    theta = extract_weights()
    model = Model("lp_plnn_inference")
    model.reset()
    # model.Params.Presolve = 0
    x = model.addVar(lb=lower_bound, ub=upper_bound, name="x")
    z = []
    for i, nn in enumerate(layer_size):
        # Note: default lower bound for variable is 0. Therefore, we need to change a default lower bound.
        z.append(model.addVars(nn, name="z_" + str(i)))
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
                    + theta["b " + str(i)][j]
                    for j in range(layer_size[i])
                ),
                name="I_layer_" + str(i),
            )

    model.setObjective(
        z[len(layer_size) - 1][layer_size[len(layer_size) - 1] - 1], GRB.MINIMIZE
    )
    model.write("lp_plnn_inference.lp")
    model.optimize()
    x = model.getAttr("X", [x])
    obj = model.getObjective()
    obj = obj.getValue()
    y_m_lp = model_nn.predict([x])[0]
    return x[0], obj, y_m_lp[0]


def optimize_input_via_indicator_constraint(
    layer_size, lower_bound, upper_bound, model_nn
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
    # model.addConstr(x <= 0.4, name="constraint_x")
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
            # model.addConstr(
            #    (y >= -0.1 * x + 0.1),
            #    name="out_high",
            # )
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


def initialize_bounds(layer_size):
    """
    Initialization of upper and lower bound.
    """
    up, down = [], []
    U = [+1000]
    L = [-1000]
    for layer in layer_size:
        up.append(U * layer)
        down.append(L * layer)
    return up, down


def write_inf_m(model, i, j):
    """
    Example based on workforce1: https://www.gurobi.com/documentation/9.5/examples/workforce1_py.html.
    Computes set of constraints and bounds that make model infeasible,
    and writes those constraints in .ilp file.
    """
    model.computeIIS()
    model.write("bounds_%2d_%2d_infeasible_constraints.ilp" % (i, j))


def solve_inf_m(model):
    """
    Example based on workforce2: https://www.gurobi.com/documentation/9.5/examples/workforce2_py.html.
    Removes infeasible constraints until we solve the model to optimality.
    """
    obj = 0
    removed = []
    while True:
        model.computeIIS()
        for c in model.getConstrs():
            if c.IISConstr:
                print("%s" % c.ConstrName)
                # Remove a single constraint from the model
                removed.append(str(c.ConstrName))
                model.remove(c)
                break
        print("")

        model.optimize()
        status = model.Status

        if status == GRB.UNBOUNDED:
            print("The model cannot be solved because it is unbounded")
            sys.exit(1)
        if status == GRB.OPTIMAL:
            print("Bounds submodel is solved to optimality")
            obj = model.getObjective()
            obj = obj.getValue()
            break
        if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
            print("Optimization was stopped with status %d" % status)
            sys.exit(1)

    print("\nThe following constraints were removed to get a feasible LP:")
    print(removed)
    return obj


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
    return obj


def bounds_via_layer_relaxation(layer_size, lower_bound, upper_bound):
    """
    Piecewise linear neural network implemented as a mixed-integer linear program
    following classical big-M formulation based on the layer relaxation.
    Layer relaxation tries to find tight upper and lower bounds
    to easy solving mixed-integer linear program.
    """
    theta = extract_weights()
    start_time = time.time()
    upper_bounds, lower_bounds = initialize_bounds(layer_size)
    maximize = [True, False]
    for i in range(1, len(layer_size) + 1):
        for j in range(layer_size[i - 1]):
            bound_ = []
            for optimization in maximize:
                constraint_name = "_layer_" + str(i) + "_neuron_" + str(j)
                z, s, a = [], [], []
                model = Model("layer_relax_plnn_inference" + constraint_name)
                # outputs whether model is unbounded or infeasible
                model.Params.DualReductions = 0
                # adjusts the feasibility tolerance of primal constraints
                model.Params.FeasibilityTol = 0.01
                # adjusts the feasibility tolerance of dual constraints
                model.Params.OptimalityTol = 0.01
                # resets the model
                model.reset()
                # adding variables to the model
                x = model.addVar(lb=lower_bound, ub=upper_bound, name="x")
                if i > 1:
                    for layer, neuron in enumerate(layer_size[: i - 1]):
                        z.append(
                            model.addVars(
                                neuron,
                                vtype=GRB.CONTINUOUS,
                                lb=0,
                                name="z_" + str(layer),
                            )
                        )
                        s.append(
                            model.addVars(
                                neuron,
                                vtype=GRB.CONTINUOUS,
                                lb=0,
                                name="s_" + str(layer),
                            )
                        )
                        a.append(
                            model.addVars(
                                neuron,
                                vtype=GRB.BINARY,
                                name="a_" + str(layer),
                            )
                        )
                # outer variable, based on which upper and lower bounds are calculated
                z.append(
                    model.addVars(
                        1,
                        vtype=GRB.CONTINUOUS,
                        lb=-GRB.INFINITY,
                        name="z_" + str(i - 1),
                    )
                )
                s.append(
                    model.addVars(
                        1,
                        vtype=GRB.CONTINUOUS,
                        lb=0,
                        name="s_" + str(i - 1),
                    )
                )
                t = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY,
                    name="t",
                )
                a.append(
                    model.addVars(
                        1,
                        vtype=GRB.BINARY,
                        name="a_" + str(i - 1),
                    )
                )

                # adding constraints to the model
                if i == 1:
                    model.addConstr(
                        t
                        == x * theta["wz " + str(i - 1)][0][j]
                        + theta["b " + str(i - 1)][j],
                        name="layer",
                    )
                else:
                    for layer, neuron in enumerate(layer_size[: i - 1]):
                        if layer == 0:
                            model.addConstrs(
                                (
                                    z[layer][jj] - s[layer][jj]
                                    == x * theta["wz " + str(layer)][0][jj]
                                    + theta["b " + str(layer)][jj]
                                    for jj in range(neuron)
                                ),
                                name="layer_" + str(layer),
                            )
                        else:
                            model.addConstrs(
                                (
                                    z[layer][jj] - s[layer][jj]
                                    == sum(
                                        z[layer - 1][k]
                                        * theta["wz " + str(layer)][k][jj]
                                        for k in range(layer_size[layer - 1])
                                    )
                                    + theta["b " + str(layer)][jj]
                                    for jj in range(neuron)
                                ),
                                name="layer_" + str(layer),
                            )
                        model.addConstrs(
                            (
                                z[layer][jj] <= upper_bounds[layer][jj] * a[layer][jj]
                                for jj in range(neuron)
                            ),
                            name="up_bound_layer_" + str(layer),
                        )
                        model.addConstrs(
                            (z[layer][jj] >= 0 for jj in range(neuron)),
                            name="up_zero_bound_layer_" + str(layer),
                        )
                        model.addConstrs(
                            (
                                s[layer][jj]
                                <= -lower_bounds[layer][jj] * (1 - a[layer][jj])
                                for jj in range(neuron)
                            ),
                            name="low_bound_layer_" + str(layer),
                        )
                    if i < len(layer_size):
                        model.addConstr(
                            (
                                z[i - 1][0] - s[i - 1][0]
                                == sum(
                                    z[i - 2][k] * theta["wz " + str(i - 1)][k][j]
                                    for k in range(layer_size[i - 2])
                                )
                                + theta["b " + str(i - 1)][j]
                            ),
                            name="layer_" + str(i),
                        )
                    else:
                        model.addConstr(
                            (
                                z[i - 1][0]
                                == sum(
                                    z[i - 2][k] * theta["wz " + str(i - 1)][k][j]
                                    for k in range(layer_size[i - 2])
                                )
                                + theta["b " + str(i - 1)][j]
                            ),
                            name="layer_" + str(i),
                        )
                if i < len(layer_size):
                    model.addConstr(
                        t == z[i - 1][0] - s[i - 1][0],
                        name="output",
                    )
                    model.addConstr(
                        z[i - 1][0] <= upper_bounds[i - 1][j] * a[i - 1][0],
                        name="up_bound_out",
                    )
                    model.addConstr(
                        z[i - 1][0] >= 0,
                        name="up_zero_bound_out",
                    )
                    model.addConstr(
                        s[i - 1][0] <= -lower_bounds[i - 1][j] * (1 - a[i - 1][0]),
                        name="low_bound_out",
                    )
                    if optimization:
                        model.setObjective(t, GRB.MAXIMIZE)
                    else:
                        model.setObjective(t, GRB.MINIMIZE)
                else:
                    if optimization:
                        model.setObjective(z[i - 1][0], GRB.MAXIMIZE)
                    else:
                        model.setObjective(z[i - 1][0], GRB.MINIMIZE)
                """
                model.write(
                    "layer_relaxation_plnn_inference_layer_"
                    + str(i)
                    + "_neuron_"
                    + str(j)
                    + ".lp"
                )
                """
                model.optimize()
                status = model.status
                if status == GRB.UNBOUNDED:
                    print("Bounds model (%2d, %2d) is unbounded" % (i, j))
                    sys.exit(0)
                elif status == GRB.INFEASIBLE:
                    print("Bounds model (%2d, %2d) is infeasible" % (i, j))
                    write_inf_m(model=model, i=i, j=j)
                    # obj = solve_inf_m(model = model)
                    obj = relax_inf_m(model=model)
                elif status == GRB.OPTIMAL:
                    print("Bounds model (%2d, %2d) is solved to optimality" % (i, j))
                    obj = model.getObjective()
                    obj = obj.getValue()
                bound_.append(obj)
            upper_bounds[i - 1][j] = round(bound_[0], 6)
            lower_bounds[i - 1][j] = round(bound_[1], 6)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return list(upper_bounds), list(lower_bounds), elapsed_time


def optimize_input_via_big_M(
    layer_size, upper_bounds, lower_bounds, upper_bound, lower_bound, model_nn
):
    z, s, a = [], [], []
    n = len(layer_size) - 1
    theta = extract_weights()
    start_time = time.time()
    model = Model("big_M_plnn_inference")
    # adjusts the feasibility tolerance of primal constraints
    model.Params.FeasibilityTol = 0.01
    # adjusts the feasibility tolerance of dual constraints
    model.Params.OptimalityTol = 0.01
    model.reset()
    # input variable
    x = model.addVar(lb=lower_bound, ub=upper_bound, name="x")
    # hidden layers variables
    for i, nn in enumerate(layer_size[:-1]):
        z.append(model.addVars(nn, vtype=GRB.CONTINUOUS, lb=0, name="z_" + str(i)))
        s.append(model.addVars(nn, vtype=GRB.CONTINUOUS, lb=0, name="s_" + str(i)))
        a.append(model.addVars(nn, vtype=GRB.BINARY, name="a_" + str(i)))
    # output variable
    y = model.addVar(
        vtype=GRB.CONTINUOUS,
        lb=lower_bounds[n][layer_size[n] - 1],
        ub=upper_bounds[n][layer_size[n] - 1],
        name="y",
    )
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
                model.addConstr(
                    z[i][j] <= upper_bounds[i][j] * a[i][j],
                    name="up_bound_con_" + str(i),
                )
                model.addConstr(
                    s[i][j] <= -lower_bounds[i][j] * (1 - a[i][j]),
                    name="low_bound_con_" + str(i),
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
                model.addConstr(
                    z[i][j] <= upper_bounds[i][j] * a[i][j],
                    name="up_bound_con_" + str(i),
                )
                model.addConstr(
                    s[i][j] <= -lower_bounds[i][j] * (1 - a[i][j]),
                    name="low_bound_con_" + str(i),
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
    model.setObjective(y, GRB.MINIMIZE)
    model.write("big_M_plnn_inference.lp")
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("big_M_plnn_inference_infeasible_constraints.ilp")
        model.feasRelaxS(1, True, False, True)
        model.optimize()
    x = model.getAttr("X", [x])
    obj = model.getObjective()
    obj = obj.getValue()
    end_time = time.time()
    elapsed_time = end_time - start_time
    y_m_lp = model_nn.predict([x])[0]
    return x[0], obj, y_m_lp[0], elapsed_time


if __name__ == "__main__":
    result_p: Path = Path(__file__).parents[1] / "2D_convex_function"
    layer_sizes = [[100, 100, 100, 1]]
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

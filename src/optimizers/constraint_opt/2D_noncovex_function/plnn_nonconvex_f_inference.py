import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import time

from gurobipy import *
from datetime import datetime

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Add, Activation
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import Model as ModelNN
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
Finding the minimum of nonconvex function approximated with piecewise linear network
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
    starting_points,
    learning_rate,
    num_run,
):
    x = np.linspace(lower_bound, upper_bound, size)
    y = non_convex_function(x)
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
    x_opt_grad_, y_opt_grad_, time_gd_ = [], [], []
    """
    for starting_point in starting_points:
        x, time_gd = optimize_input_via_gradient(
            model=model,
            learning_rate=learning_rate,
            starting_point=starting_point,
            iterations=iterations,
        )
        y_opt = model(x.value())
        y_opt_grad = np.array(y_opt)[0][0]
        x_opt_grad = np.array(x.value())[0]
        x_opt_grad_.append(x_opt_grad)
        y_opt_grad_.append(y_opt_grad)
        time_gd_.append(time_gd)
    """
    x_ind, y_ind, y_ind_nn, time_ind = optimize_input_via_indicator_constraint(
        layer_size=layer_size,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        model_nn=model,
    )
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
    print(
        "Training MSE is {:.5f}. \n Testing MSE is {:.5f}".format(
            training_loss, mean_squared_error(y_test, y_pred)
        )
    )
    print(
        "Indicator constraints: Optimal input is {:.5f} and value of objective is {:.5f}"
        # "Big-M formulation: Optimal input is {:.5f} and value of objective is {:.5f} \n"
        .format(x_ind, y_ind)
    )
    print(
        "Indicator constraints: Elapsed time is {:.5f}. \n "
        # "Big-M formulation: Elapsed time is {:.5f}. \n "
        .format(time_ind)
    )
    plot(
        x_test,
        y_test,
        y_pred,
        model,
        # starting_points,
        # x_opt_grad_,
        # y_opt_grad_,
        x_ind,
        y_ind,
        # x_big_m,
        # y_big_m,
    )


def plot(
    x_test,
    y_test,
    y_pred,
    model,
    # starting_points,
    # x_opt_grad_,
    # y_opt_grad_,
    x_ind,
    y_ind,
    # x_big_m,
    # y_big_m,
):
    """
    Plot real data, predicted data by neural network, optimum of the function
    found by gradient descent, and optimum found by linear program.
    """
    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.plot(x_test, y_test, ".", color="blue", label="Real data")
    plt.plot(x_test, y_pred, ".", color="red", label="Predicted data")
    colors = ["green", "cyan"]
    """
    for i, starting_point in enumerate(starting_points):
        plt.plot(
            x_opt_grad_[i],
            y_opt_grad_[i],
            marker="*",
            markersize=20,
            color=colors[i],
            label="Optimum GD " + str(i),
        )
        plt.plot(
            starting_point,
            model(np.array([starting_point])),
            marker="v",
            markersize=10,
            color=colors[i],
            label="Start point " + str(i),
        )
   """
    plt.plot(
        x_ind,
        y_ind,
        marker="o",
        markersize=10,
        color="yellow",
        label="Optimum indicator MILP",
    )
    """
    plt.plot(
        x_big_m,
        y_big_m,
        marker="s",
        markersize=7,
        color="pink",
        label="Optimum big-M MILP",
    )
    """
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Non-convex function")
    plt.legend()
    plt.savefig(
        "Real and predicted function and optimums found via GD and MILP " + now + ".png"
    )
    plt.show()


def non_convex_function(x):
    """
    Create non-convex function.
    param: size of the dataset
    return: input and output dataset
    """
    y = (
        0.1 * (3.75 * x - 2) ** 4
        + 0.1 * (4 * x - 3) ** 3
        - 0.1 * (4 * x - 2) ** 2
        - 1.35 * x
        + 1
    )
    return y


def train_test_dataset(x, y):
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
    model.save_weights("weights_nonconvex_plnn")
    return model


def train_nn(x_train, feature_num, y_train, layer_size, batch_size, epochs):
    """
    Training of the neural network.
    return: model
    """
    model = plnn(layer_size=layer_size, feature_num=feature_num)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss=MeanSquaredError())
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.0005, patience=10)
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        verbose=1,
        callbacks=[early_stopping],
    )
    model.save("model_nonconvex_plnn.h5")
    return model, history.history["loss"][-1]


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
    return: weights and biases
    """
    model = load_model("model_nonconvex_plnn.h5")
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
    model = Model("ind_non_con_plnn_inference")
    model.reset()
    # input to the neural network
    x = model.addVar(lb=lower_bound, ub=upper_bound, name="x")
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
    model.setObjective(y, GRB.MINIMIZE)
    model.write("ind_non_con_plnn_inference.lp")
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
    model = Model("big_M_nonconvex_plnn_inference")
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
    model.write("big_M_nonconvex_plnn_inference.lp")
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
    main(
        size=500000,
        feature_num=1,
        lower_bound=-0.15,
        upper_bound=1,
        layer_size=[1000, 1000, 1000, 1000, 1],
        batch_size=64,
        epochs=500,
        iterations=1000,
        starting_points=[-0.12, 0.95],
        learning_rate=0.001,
        num_run=1,
    )

import numpy as np
import warnings
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import time

from datetime import datetime
from gurobipy import *
from statistics import mean
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Circle

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
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

"""
Finding the minimum of 3D nonconvex function approximated with ICNN with custom loss and PLNN.
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
    patience_grad,
    delta_grad,
):
    x_ = np.linspace(lower_bound[0], upper_bound[0], size)
    y_ = np.linspace(lower_bound[1], upper_bound[1], size)
    x_train_, x_test_, y_train_, y_test_ = train_test_dataset(x=x_, y=y_)
    x_train, y_train = create_data(x_=x_train_, y_=y_train_, size=int(size * 0.8))
    x_test, y_test = create_data(x_=x_test_, y_=y_test_, size=int(size * 0.2))
    z_train = non_convex_function(x=x_train, y=y_train, size=int(size * 0.8))
    z_train = z_train.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(z_train)
    z_train = scaler.transform(z_train)
    z_train = np.reshape(z_train, -1)
    z_test = non_convex_function(x=x_test, y=y_test, size=int(size * 0.2))
    data_train = np.stack((x_train, y_train), axis=-1)
    data_test = np.stack((x_test, y_test), axis=-1)
    model, training_loss = train_plnn(
        data_train=data_train,
        feature_num=feature_num,
        z_train=z_train,
        layer_size=layer_size,
        batch_size=batch_size,
        epochs=epochs,
    )
    z_pred = model.predict(data_test)
    z_pred = z_pred.reshape(-1, 1)
    z_pred = scaler.inverse_transform(z_pred)
    z_pred = np.reshape(z_pred, -1)
    x_pl, time_gd_pl = optimize_input_via_gradient(
        model=model,
        learning_rate=learning_rate,
        starting_point=starting_point,
        iterations=iterations,
        patience_grad=patience_grad,
        delta_grad=delta_grad,
        upper_bound=upper_bound[0],
        lower_bound=lower_bound[0],
    )
    y_opt_pl = model(x_pl.value())
    y_opt_grad_pl = np.array(y_opt_pl)[0][0]
    y_opt_grad_pl = scaler.inverse_transform(y_opt_grad_pl.reshape(-1, 1))
    x_opt_grad_pl = np.array(x_pl.value())[0]
    x_ind, y_ind, time_ind = optimize_input_via_indicator_constraint(
        feature_num=feature_num,
        layer_size=layer_size,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        scaler=scaler,
    )
    model_, training_loss_ = train_ficnn(
        data_train=data_train,
        feature_num=feature_num,
        z_train=z_train,
        layer_size=layer_size,
        batch_size=batch_size,
        epochs=epochs,
        alpha=0.7,
    )
    z_pred_ = model_.predict(data_test)
    z_pred_ = z_pred_.reshape(-1, 1)
    z_pred_ = scaler.inverse_transform(z_pred_)
    z_pred_ = np.reshape(z_pred_, -1)
    x_lp, y_lp, time_lp = optimize_input_via_lp(
        feature_num=feature_num,
        layer_size=layer_size,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        scaler=scaler,
    )
    print(x_lp, y_lp, time_lp)
    x, time_gd = optimize_input_via_gradient(
        model=model_,
        learning_rate=learning_rate,
        starting_point=starting_point,
        iterations=iterations,
        patience_grad=patience_grad,
        delta_grad=delta_grad,
        upper_bound=upper_bound[0],
        lower_bound=lower_bound[0],
    )
    y_opt = model_(x.value())
    y_opt_grad = np.array(y_opt)[0][0]
    y_opt_grad = scaler.inverse_transform(y_opt_grad.reshape(-1, 1))
    x_opt_grad = np.array(x.value())[0]
    plot(
        x=x_test_,
        y=y_test_,
        z=z_test,
        z_plnn=z_pred,
        z_icnn=z_pred_,
        size=int(size * 0.2),
        x_ind=x_ind,
        y_ind=y_ind,
        x_icnn=x_opt_grad,
        y_icnn=y_opt_grad,
        x_gr_pl=x_opt_grad_pl,
        y_gr_pl=y_opt_grad_pl,
        x_lp=x_lp,
        y_lp=y_lp,
    )
    print(
        "PLNN: Error on a training dataset is {:.5f}, and on testing dataset is {:.5f}. \n "
        "ICNN: Error on a training dataset is {:.5f}, and on testing dataset is {:.5f}.".format(
            training_loss,
            mean_squared_error(z_test, z_pred),
            training_loss_,
            mean_squared_error(z_test, z_pred_),
        )
    )
    print(
        "Optimum found by PLNN+GD is [{:.5f},{:.5f}], and objective value is {:.5f}. \n "
        "Optimum found by PLNN+MILP is [{:.5f},{:.5f}], and objective value is {:.5f}. \n "
        "Optimum found by ICNN+GD is [{:.5f},{:.5f}], and objective value is {:.5f}.\n"
        "Optimum found by ICNN+LP is [{:.5f},{:.5f}], and objective value is {:.5f}.".format(
            x_opt_grad_pl[0],
            x_opt_grad_pl[1],
            y_opt_grad_pl[0][0],
            x_ind[0],
            x_ind[1],
            y_ind[0][0],
            x_opt_grad[0],
            x_opt_grad[1],
            y_opt_grad[0][0],
            x_lp[0],
            x_lp[1],
            y_lp[0][0],
        )
    )
    print(
        "Elapsed time for PLNN+GD is {:.5f}. \n "
        "Elapsed time for PLNN+MILP is {:.5f}. \n "
        "Elapsed time for ICNN+GD is {:.5f}.\n"
        "Elapsed time for ICNN+LP is {:.5f}".format(
            time_gd_pl, time_ind, time_gd, time_lp
        )
    )


def create_data(x_, y_, size):
    x, y = [], []
    for i in range(size):
        x += [x_[i]] * size
        y += list(y_)
    x = np.array(x)
    y = np.array(y)
    return x, y


def non_convex_function(x, y, size):
    z = []
    for i in range(size * size):
        z.append(x[i] ** 3 * math.exp(-x[i] ** 2 - y[i] ** 2) * (y[i] + 0.05) ** 3)
        # z.append(x[i] * math.exp(-x[i] ** 2 - y[i] ** 2))
    z = np.array(z)
    return z


def train_test_dataset(x, y):
    """
    Divide original dataset on training and testing.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True
    )
    return x_train, x_test, y_train, y_test


def plot(
    x,
    y,
    z,
    z_plnn,
    z_icnn,
    size,
    x_ind,
    y_ind,
    x_icnn,
    y_icnn,
    x_gr_pl,
    y_gr_pl,
    x_lp,
    y_lp,
):
    fig = plt.figure()
    z = z.reshape(size, size)
    z_plnn = z_plnn.reshape(size, size)
    z_icnn = z_icnn.reshape(size, size)
    ma = np.nanmax(z)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=ma, clip=True)
    ax = Axes3D(fig, computed_zorder=False)
    x1, y1 = np.meshgrid(x, y)
    c1 = ax.plot_surface(
        x1,
        y1,
        z,
        cmap="viridis_r",
        linewidth=0.1,
        alpha=0.2,
        edgecolor="k",
        norm=norm,
        label="Real data",
    )
    c1._facecolors2d = c1._facecolor3d
    c1._edgecolors2d = c1._edgecolor3d
    """
    c2 = ax.plot_surface(
        x1,
        y1,
        z_plnn,
        rstride=1,
        cstride=1,
        cmap="viridis",
        label="Predicted data by PLNN",
    )
    c2._facecolors2d = c2._facecolor3d
    c2._edgecolors2d = c2._edgecolor3d
    c3 = ax.plot_surface(
        x1,
        y1,
        z_icnn,
        rstride=1,
        cstride=1,
        cmap="plasma",
        label="Predicted data by ICNN",
    )
    c3._facecolors2d = c3._facecolor3d
    c3._edgecolors2d = c3._edgecolor3d
    """
    ax.scatter(
        x_ind[0],
        x_ind[1],
        y_ind,
        c="k",
        depthshade=False,
        alpha=1,
        s=100,
        label="Optimum by PLNN+MILP",
    )
    ax.scatter(
        x_icnn[0],
        x_icnn[1],
        y_icnn,
        c="r",
        depthshade=False,
        alpha=1,
        s=100,
        marker="*",
        label="Optimum by ICNN+GD",
    )
    ax.scatter(
        x_gr_pl[0],
        x_gr_pl[1],
        y_gr_pl,
        c="b",
        depthshade=False,
        alpha=1,
        s=100,
        marker="^",
        label="Optimum by PLNN+GD",
    )
    ax.scatter(
        x_lp[0],
        x_lp[1],
        y_lp,
        c="g",
        depthshade=False,
        alpha=1,
        s=100,
        marker="v",
        label="Optimum by ICNN+LP",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("$z= x*e^{-x^2-y^2}$")
    title = "$z= x*e^{-x^2-y^2}$"
    plt.title(title)
    ax.legend()
    plt.show()


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


def train_plnn(data_train, feature_num, z_train, layer_size, batch_size, epochs):
    """
    Training of the neural network.
    return: model
    """
    model = plnn(layer_size=layer_size, feature_num=feature_num)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=MeanSquaredError())
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.0005, patience=10)
    history = model.fit(
        x=data_train,
        y=z_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        verbose=1,
        callbacks=[early_stopping],
    )
    model.save("model_nonconvex_3D.h5")
    return model, history.history["loss"][-1]


def custom_loss(alpha, y_min):
    """
    Customized MSE that tries to encourage lowest error in optimum.
    """

    def loss(y_true, y_pred):
        mean_sqr_error = K.mean(K.abs(y_true - y_pred))
        p = K.mean(alpha * relu(y_pred - y_min))
        error = mean_sqr_error + p
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
            layer = Activation("relu")(layer)
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
                layer = Activation("relu")(layer_merge)

    # creation of the model based on Functional API
    model = ModelNN(inputs=input_layer, outputs=layer)
    model.save_weights("weights")
    return model


def train_ficnn(
    data_train, feature_num, z_train, layer_size, batch_size, epochs, alpha
):
    """
    Training of the neural network.
    return: model
    """
    y_min = min(z_train)
    model = ficnn(layer_size=layer_size, feature_num=feature_num)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=custom_loss(alpha=alpha, y_min=y_min))
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.00005, patience=15)
    history = model.fit(
        x=data_train,
        y=z_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        verbose=1,
        callbacks=[early_stopping],
    )
    model.save("model_convex_3D.h5")
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


def extract_weights(convex):
    """
    Extraction of weights from feedforward convex neural network.
    return: weights and biases
    """
    if convex:
        model = load_model("model_convex_3D.h5", compile=False)
    else:
        model = load_model("model_nonconvex_3D.h5")
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


def optimize_input_via_gradient(
    model,
    learning_rate,
    starting_point,
    iterations,
    patience_grad,
    delta_grad,
    upper_bound,
    lower_bound,
):
    """
    Repeatedly optimize input x for specified number of iterations.
    """
    start_time = time.time()
    starting_point = np.array([starting_point])
    # initial value as numpy array of one element (one feature)
    x = tf.Variable(starting_point)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    update_grad = 0
    for i in range(iterations):
        gradient = calculate_gradient(x=x, model=model)
        # implementing patience and minimum gradient update
        # to consider that the process of finding minimum is still running
        if max(np.absolute(gradient[0])) < delta_grad:
            update_grad += 1
        else:
            update_grad = 0
        # zip calculated gradient and previous value of x
        zipped = zip([gradient], [x])
        # update value of input variable according to the calculated gradient
        opt.apply_gradients(zipped)
        # have a look at x, and check whether it violates constraints
        # and if it violates set it on closest feasible value.
        if update_grad == patience_grad:
            print("The last iteration is " + str(i))
            break
        x_ = np.array(x.value())[0]
        x_[x_ > upper_bound] = upper_bound
        x_[x_ < lower_bound] = lower_bound
        x_ = np.array([x_])
        x.assign(x_)
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


def optimize_input_via_lp(feature_num, layer_size, lower_bound, upper_bound, scaler):
    """
    Construct the linear program of ReLu neural network,
    under the knowledge that one of nonlinear constraints representing ReLu
    must be tight, and found the minimum of such linear program.
    """
    theta = extract_weights(convex=True)
    start_time = time.time()
    model = Model("lp_icnn_inference")
    model.reset()
    model.Params.Presolve = 0
    x = model.addVars(feature_num, lb=lower_bound[0], ub=upper_bound[0], name="x")
    z = []
    for i, nn in enumerate(layer_size):
        # Note: default lower bound for variable is 0. Therefore, we need to change a default lower bound.
        z.append(model.addVars(nn, lb=-1000, name="z_" + str(i)))
    for i in range(len(layer_size)):
        if i == 0:
            model.addConstrs(
                (
                    z[i][j]
                    >= sum(
                        x[k] * theta["wz " + str(i)][k][j] for k in range(feature_num)
                    )
                    + theta["b " + str(i)][j]
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
                    + sum(
                        x[k] * theta["wx " + str(i)][k][j] for k in range(feature_num)
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
                    + sum(
                        x[k] * theta["wx " + str(i)][k][j] for k in range(feature_num)
                    )
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
    x = np.array([x[i].X for i in range(feature_num)])
    obj = model.getObjective()
    obj = obj.getValue()
    obj = scaler.inverse_transform(np.array(obj).reshape(-1, 1))
    end_time = time.time()
    elapsed_time = end_time - start_time
    return x, obj, elapsed_time


def optimize_input_via_indicator_constraint(
    feature_num, layer_size, lower_bound, upper_bound, scaler
):
    """
    Piecewise linear neural network implemented as a mixed-integer linear program
    following framework by Fiscetti and Jo et. al. Bilinear equivalent constraints
    whose purpose is to set either z or s variable to zero in order to recover ReLu
    function are transformed to indicator constraints to facillitate mathematical solver.
    """
    a, z, s = [], [], []
    theta = extract_weights(convex=False)
    start_time = time.time()
    model = Model("ind_non_con_plnn_inference")
    model.reset()
    # input to the neural network
    x = model.addVars(feature_num, lb=lower_bound[0], ub=upper_bound[0], name="x")
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
                    == sum(x[k] * theta["wz " + str(i)][k][j] for k in range(2))
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
    model.write("model_nonconvex_3D.lp")
    model.optimize()
    x = np.array([x[i].X for i in range(feature_num)])
    obj = model.getObjective()
    obj = obj.getValue()
    obj = scaler.inverse_transform(np.array(obj).reshape(-1, 1))
    end_time = time.time()
    elapsed_time = end_time - start_time
    return x, obj, elapsed_time


if __name__ == "__main__":
    main(
        size=500,
        feature_num=2,
        lower_bound=[-2, -2],
        upper_bound=[2, 2],
        layer_size=[1000, 1000, 1],
        batch_size=64,
        epochs=100,
        iterations=50,
        starting_point=[1.2, 1.2],
        learning_rate=0.5,
        patience_grad=10,
        delta_grad=0.05,
    )

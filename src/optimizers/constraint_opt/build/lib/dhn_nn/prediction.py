"""
Functions for making single step and multi step predictions for state and delivered heat elements.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from src.util.config import (
    PhysicalProperties,
    ConsumerPreset1,
)
from src.optimizers.constraint_opt.dhn_nn.nn import NN


def verification_rmse(s_scaled, y_scaled, T, scaler_s, scaler_y):
    """
    Calculate root mean squared error if we: mirror the value from the input feature and treat it as we predict the mean value.
    This function serves to inspect whether NN actually learns, just mirrors previous values or just predicts the mean of the array.
    """
    output = ["supply inlet temp", "supply outlet temp", "mass flow"]
    s = scaler_s.inverse_transform(s_scaled)
    y = scaler_y.inverse_transform(y_scaled)
    s = s[:, 0:3]
    s_real = s[:-T]
    y_real = y[:-T]
    s_mean = []
    for i in range(len(output)):
        s_mean.append(np.asarray([np.mean(s_real[i, :])] * len(s_real)))
    s_mean = np.asarray(s_mean).T
    y_mean = np.asarray([np.mean(y_real)] * len(y_real))
    s_pred = s[T:]
    y_pred = y[T:]
    for i, out in enumerate(output):
        err_mirror = mean_squared_error(s_real[:, i], s_pred[:, i], squared=False)
        err_mean = mean_squared_error(s_real[:, i], s_mean[:, i], squared=False)
        print(
            "Mirroring rmse of "
            + out
            + " for T ="
            + str(T)
            + " is {}".format(err_mirror)
        )
        print("Mean rmse of " + out + " for T =" + str(T) + " is {}".format(err_mean))
    err_mirror = mean_squared_error(y_real, y_pred, squared=False)
    err_mean = mean_squared_error(y_mean, y_pred, squared=False)
    print(
        "Mirroring rmse of delivered heat for T=" + str(T) + " is {}".format(err_mirror)
    )
    print("Mean rmse of delivered heat for T=" + str(T) + " is {}".format(err_mean))


def transform_y(s_t, x_s, t, time_delay, time_delay_q):
    """
    Transformation of the output for the next prediction.
    y_t = f(s_t, h_{t-n_w},...,h_t)
    """
    y_t_ = []
    for i in range(len(s_t) - t):
        temp = []
        # append information from the state prediction
        temp.extend(list(s_t[i]))
        # append corresponding produced heat variables
        for j in range(time_delay + 1):
            temp.append(x_s[i + t][3 + time_delay_q + 1 + j])
        temp = np.array(temp)
        y_t_.append(temp)
    y_t_ = np.array(y_t_)
    return y_t_


def transform_s(s_t, x_s, t, time_delay, time_delay_q):
    """
    Transformation of the state space for the next prediction.
    s_t = g(s_{t-1}, q_t, h_{t-n_w},...,h_t)
    """
    s_t_ = []
    for i in range(len(s_t) - t - 1):
        temp = []
        # append the prediction of the state
        temp.extend(list(s_t[i]))
        # append the corresponding heat demand
        for k in range(time_delay_q + 1):
            temp.append(x_s[i + t + 1][3 + k])
        # append the corresponding produced heat
        for j in range(time_delay + 1):
            temp.append(x_s[i + t + 1][3 + time_delay_q + 1 + j])
        temp = np.array(temp)
        s_t_.append(temp)
    s_t_ = np.array(s_t_)
    return s_t_


def bound_variables(tau_s_in, tau_s_out, m):
    """
    If the state prediction exceeds the bound, set it back on the bound.
    """
    if tau_s_in > PhysicalProperties["MaxTemp"]:
        tau_s_in = PhysicalProperties["MaxTemp"]
    elif tau_s_in < PhysicalProperties["MinSupTemp"]:
        tau_s_in = PhysicalProperties["MinSupTemp"]
    if tau_s_out > PhysicalProperties["MaxTemp"]:
        tau_s_out = PhysicalProperties["MaxTemp"]
    elif tau_s_out < PhysicalProperties["MinSupTemp"]:
        tau_s_out = PhysicalProperties["MinSupTemp"]
    if m > ConsumerPreset1["MaxMassFlowPrimary"]:
        m = ConsumerPreset1["MaxMassFlowPrimary"]
    elif m < 0:
        m = 0
    return tau_s_in, tau_s_out, m


def update_delta_prediction(
    s, delta_s, scaler_x, scaler_y, scaler_s, time_step, write_prediction_data, result_p
):
    """
    Predicts the state at the time-step t+1, s_{t+1}, as: s_{t+1} = s_t + \Delta s_t
    """
    s_ = []
    # inverse transformation of state variable
    s = scaler_x.inverse_transform(s)
    # inverse transformation of change in prediction variable
    delta_s = scaler_y.inverse_transform(delta_s)
    # s_{t+1} = s_t + \Delta s_t
    for i in range(len(s)):
        tau_s_in = s[i][0] + delta_s[i][0]
        tau_s_out = s[i][1] + delta_s[i][1]
        m = delta_s[i][2]
        tau_s_in, tau_s_out, m = bound_variables(tau_s_in, tau_s_out, m)
        s_.append([tau_s_in, tau_s_out, m])
    s_ = np.array(s_)
    if write_prediction_data:
        pd.DataFrame(delta_s).to_csv(
            result_p.joinpath("state_change_time_step_" + str(time_step + 1) + ".csv")
        )
        pd.DataFrame(s_).to_csv(
            result_p.joinpath(
                "state_prediction_time_step_" + str(time_step + 1) + ".csv"
            )
        )
    # normalize the state prediction between 0 and 1
    s_ = scaler_s.fit_transform(s_)
    return s_


def state_prediction(
    delta_s,
    s_t_,
    model_s,
    scaler_x,
    scaler_y,
    scaler_s,
    time_step,
    write_prediction_data,
    result_p,
):
    """
    Predict the next state, depending on whether we approximate the state transition function
    or the function of the state change.
    """
    if delta_s:
        # prediction of the change in the next state
        delta_s_t = model_s.predict(s_t_)
        # prediction of the next state based on the change from previous state
        s_t = update_delta_prediction(
            s=s_t_,
            delta_s=delta_s_t,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            scaler_s=scaler_s,
            time_step=time_step,
            write_prediction_data=write_prediction_data,
            result_p=result_p,
        )
    else:
        # feedforward prediction of the next state
        s_t = model_s.predict(s_t_)
    return s_t


def multi_step_prediction(
    delta_s,
    s_ext,
    result_p,
    model_s,
    model_y,
    time_delay,
    time_delay_q,
    T,
    scaler_s_s,
    scaler_x_s,
    scaler_y_s,
    scaler_y_y,
    write_prediction_data,
):
    """
    Rolling out predictions, where the output of the
    previous prediction is used as an input for the next prediction.
    """
    ext: str = s_ext + "time_delay_" + str(time_delay) + ".csv"
    x_s = np.array(pd.read_csv(result_p.joinpath("x_test_s" + ext)))  # real state data
    y_s = np.array(
        pd.read_csv(result_p.joinpath("y_test_s" + ext))
    )  # real state predictions (or change of state predictions)
    y_y = np.array(
        pd.read_csv(result_p.joinpath("y_test_y" + ext))
    )  # real output predictions
    verification_rmse(
        s_scaled=x_s, y_scaled=y_y, T=T, scaler_s=scaler_x_s, scaler_y=scaler_y_y
    )
    output_s = [
        "Supply_inlet_temp",
        "Supply_outlet_temp",
        "Mass_flow",
    ]
    err = {}
    # initialize the state with data
    s_t_ = x_s
    for t in range(T):
        # predict the next state
        s_t = state_prediction(
            delta_s=delta_s,
            s_t_=s_t_,
            model_s=model_s,
            scaler_x=scaler_x_s,
            scaler_y=scaler_y_s,
            scaler_s=scaler_s_s,
            time_step=t,
            write_prediction_data=write_prediction_data,
            result_p=result_p,
        )
        # create suitable input for predicting the output
        y_t_ = transform_y(s_t, x_s, t, time_delay, time_delay_q)
        # prediction of the output
        y_t = model_y.predict(y_t_)
        # create suitable input for predicting the next state
        s_t_ = transform_s(s_t, x_s, t, time_delay, time_delay_q)
    # rescale real-data and predictions to calculate the error
    if delta_s:
        # state prediction
        s_t = scaler_s_s.inverse_transform(s_t)
        # real state at the time-step t
        x_s = scaler_x_s.inverse_transform(x_s)
        # real change of the state
        y_s = scaler_y_s.inverse_transform(y_s)
        if write_prediction_data:
            pd.DataFrame(x_s).to_csv(result_p.joinpath("state_data.csv"))
            pd.DataFrame(y_s).to_csv(result_p.joinpath("state_change.csv"))
        # real state at the time-step t+1
        y_s[:, 0:2] = x_s[:, 0:2] + y_s[:, 0:2]
    else:
        # state prediction
        s_t = scaler_y_s.inverse_transform(s_t)
        # real state at the time-step t+1
        y_s = scaler_y_s.inverse_transform(y_s)
        if write_prediction_data:
            pd.DataFrame(y_s).to_csv(result_p.joinpath("state_data_t_plus_1.csv"))
    y_t = scaler_y_y.inverse_transform(y_t)
    y_y = scaler_y_y.inverse_transform(y_y)
    # the amount of data cut from the back of the dataset.
    # note that in the first iteration (for t=0) no data is lost.
    cut_off = sum(i + 1 for i in range(T - 1))
    # y_y = y_y[1:]
    if cut_off > 0:
        # cut real data to fit predictions front
        y_s = y_s[(T - 1) :]
        y_y = y_y[(T - 1) :]
        # cut real data to fit predictions back
        y_s = y_s[:-cut_off]
        y_y = y_y[: -(cut_off + T - 1)]
        # cut predictions data back (as predicted values exceed corresponding data points in the real-world dataset)
        s_t = s_t[: -(T - 1)]
        y_t = y_t[: -(T - 1)]
    for i, output in enumerate(output_s):
        err[output] = mean_squared_error(y_s[:, i], s_t[:, i], squared=False)
    err["Delivered_heat"] = mean_squared_error(y_y, y_t, squared=False)
    pd.DataFrame(y_y).to_csv(result_p.joinpath("delivered_heat_real.csv"))
    pd.DataFrame(y_t).to_csv(result_p.joinpath("delivered_heat_predicted.csv"))
    return err


def save_loss(result_p, result_p_sub, loss, type_loss, type_nn, layer_size, early_stop):
    """
    Save training and validation loss with and without early stopping.
    Note that each row represents progress of training through epochs.
    """
    if early_stop:
        ext = "_early_stop"
    else:
        ext = ""
    type_funs = ["state", "output"]
    for type_fun in type_funs:
        name = (
            type_loss
            + type_nn
            + "_"
            + type_fun
            + "_"
            + NN.neurons_ext(layer_size)
            + ext
        )
        loss_df = pd.DataFrame(loss[type_fun])
        loss_df.to_csv(result_p.joinpath(result_p_sub).joinpath(name + ".csv"))

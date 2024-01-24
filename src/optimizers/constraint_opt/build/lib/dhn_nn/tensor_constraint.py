import numpy as np
import tensorflow as tf

from keras import backend
from tensorflow.keras.constraints import NonNeg


class ParNonNeg(NonNeg):
    """
    Overriding tensorflow constraint in order to support monotonicity
    with respect to selected inputs.
    """

    def __init__(self, time_delay=0, param_start=3, param_end=6, feature_num=5):
        self.time_delay = time_delay
        self.param_start = param_start
        self.param_end = param_end
        self.feature_num = feature_num

    def __call__(self, w):
        if self.time_delay == 0:
            tensor_non_decreasing = tf.greater_equal(w[0 : self.param_end, :], 0.0)
            tensor_non_constraint = tf.greater_equal(
                w[self.param_end : self.feature_num, :], -1000000.0
            )
            ret = w * tf.cast(
                tf.concat([tensor_non_decreasing, tensor_non_constraint], axis=0),
                backend.floatx(),
            )
        elif self.time_delay > 0:
            tensor_non_constraint_1 = tf.greater_equal(w[0 : self.param_start, :], 0.0)
            tensor_non_decreasing = tf.greater_equal(
                w[self.param_start : self.param_end, :], -1000000.0
            )
            tensor_non_constraint_2 = tf.greater_equal(
                w[self.param_end : self.feature_num, :], 0.0
            )
            ret = w * tf.cast(
                tf.concat(
                    [
                        tensor_non_constraint_1,
                        tensor_non_decreasing,
                        tensor_non_constraint_2,
                    ],
                    axis=0,
                ),
                backend.floatx(),
            )
        elif self.time_delay < 0:
            tensor_non_constraint_1 = tf.greater_equal(w[0 : self.param_start, :], 0.0)
            tensor_non_decreasing_1 = tf.greater_equal(
                w[self.param_start : self.param_start + 1, :], -1000000.0
            )
            tensor_non_constraint_2 = tf.greater_equal(
                w[self.param_start + 1 : self.param_end, :], 0.0
            )
            tensor_non_decreasing_2 = tf.greater_equal(
                w[self.param_end : self.feature_num, :], -1000000.0
            )
            ret = w * tf.cast(
                tf.concat(
                    [
                        tensor_non_constraint_1,
                        tensor_non_decreasing_1,
                        tensor_non_constraint_2,
                        tensor_non_decreasing_2,
                    ],
                    axis=0,
                ),
                backend.floatx(),
            )
        return ret

    def get_config(self):
        return {
            "time_delay": self.time_delay,
            "param_start": self.param_start,
            "param_end": self.param_end,
            "feature_num": self.feature_num,
        }

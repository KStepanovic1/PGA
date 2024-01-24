import csv
import os
import pandas as pd
import pathlib
import numpy as np
import tensorflow as tf


class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    def _log_gradients(self, epoch):
        writer = self._writers["train"]

        with writer.as_default(), tf.GradientTape() as g:
            # here we use test data to calculate the gradients
            path = pathlib.Path(__file__).parents[1] / "2D_convex_function"
            path_x = os.path.join(path, "x_train.csv")
            path_y = os.path.join(path, "y_train.csv")
            with open(path_x) as file:
                x_train = np.array(pd.read_csv(file))
            with open(path_y) as file:
                y_train = np.array(pd.read_csv(file))
            x_train = tf.convert_to_tensor(x_train)
            y_train = tf.convert_to_tensor(y_train)

            y_pred = self.model(x_train)  # forward-propagation
            loss = self.model.compiled_loss(
                y_true=y_train, y_pred=y_pred
            )  # calculate loss
            gradients = g.gradient(
                loss, self.model.trainable_weights
            )  # back-propagation

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                    weights.name.replace(":", "_") + "_grads", data=grads, step=epoch
                )

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
        # but we do need to run the original on_epoch_end, so here we use the super function.
        super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)

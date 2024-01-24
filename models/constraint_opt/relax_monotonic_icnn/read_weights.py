from tensorflow.keras.models import load_model
from dhn_nn.tensor_constraint import ParNonNeg
from dhn_nn.optimizer import Optimizer
import warnings



model = load_model("4_model_output_relax_tau_in_s_time_delay_10_neurons_10_10_monotonic_icnn.h5", compile=False, custom_objects={"ParNonNeg": ParNonNeg})
model.summary()
theta = {}
j = 0
for i, layer in enumerate(model.layers):
    if "dense" in layer.name:
            weights = layer.get_weights()
            if len(weights) == 2:
                    theta["wz " + str(j)] = Optimizer.reform(weights[0])
                    theta["b " + str(j)] = Optimizer.reform(weights[1])
                    j += 1
            elif len(weights) == 1:
                    theta["wx " + str(j - 1)] = weights[0]
            else:
                    warnings.warn(
                        "Implemented weight extraction procedure might be unsuitable for the current network!"
                    )
print(theta)
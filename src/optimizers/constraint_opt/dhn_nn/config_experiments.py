"""
Specifies types of experiments we carry, and subfolders to save results, models and plots of these experiments.
"""

experiments_learning = {
    # determining early stopping parameters
    "validate_early_stopping": {
        "folder": "train_val_loss",
        "early_stop": True,
        "nor": True,
        "delta_s": False,
        "s_ext": {"delta_s": "_s_", "early_stop": "_early_stop"},
    },
    # regular predictions
    "predictions": {
        "folder": "predictions",
        "model_ext": "",
        "early_stop": True,
        "nor": True,
        "delta_s": False,
        "s_ext": {"delta_s": "_s_", "early_stop": ""},
        "ad": "plnn",
    },
    # imposing monotonicity restriction on all produced heat variables
    "monotonic_heat": {
        "folder": "monotonic_heat",
        "early_stop": True,
        "nor": True,
        "delta_s": False,
        "s_ext": {"delta_s": "_s_", "early_stop": ""},
        "model_ext": "_heat_relax_tau_in",
    },
    # omitting variable supply inlet temperature to examine how this would influence prediction accuracy
    "without_tau_in": {
        "folder": "without_tau_in",
        "early_stop": True,
        "nor": True,
        "delta_s": False,
        "s_ext": {"delta_s": "_s_", "early_stop": ""},
        "model_ext": "",
    },
    "monotonic_icnn": {
        "folder": "monotonic_icnn",
        "early_stop": True,
        "nor": True,
        "delta_s": False,
        "s_ext": {"delta_s": "_s_", "early_stop": ""},
        "model_ext": "",
        "ad": "monotonic_icnn",
    },
    "relax_monotonic_icnn": {
        "folder": "relax_monotonic_icnn",
        "early_stop": True,
        "nor": True,
        "delta_s": False,
        "s_ext": {"delta_s": "_s_", "early_stop": ""},
        "model_ext": "_relax_tau_in",
        "ad": "icnn",  # addition when reading dictionary of state and output maximal and minimal values
    },
}


experiments_optimization = {
    "plnn_milp": {
        "folder": "plnn_milp",
        "sub-folder": "MPC_episode_length_72_hours/control with heat",
        "nn_type": "plnn",
        "optimizer_type": "plnn_milp",
        "model_ext": "",
        "tensor_constraint": False,
    },
    "relax_monotonic_icnn_gd": {
        "folder": "relax_monotonic_icnn_gd",
        "sub-folder": "",
        "nn_type": "monotonic_icnn",
        "optimizer_type": "icnn_gd",
        "model_ext": "_relax_tau_in",
        "tensor_constraint": True,
    },
    "monotonic_icnn_gd": {
        "folder": "monotonic_icnn_gd",
        "sub-folder": "",
        "nn_type": "monotonic_icnn",
        "optimizer_type": "icnn_gd",
        "model_ext": "",
        "tensor_constraint": True,
    },
    "plnn_gd": {
        "folder": "plnn_gd",
        "sub-folder": "",
        "nn_type": "plnn",
        "optimizer_type": "plnn_gd",
        "model_ext": "",
        "tensor_constraint": False,
    },
    "monotonic_icnn_plnn":{
        "folder": "monotonic_icnn_plnn",
        "sub-folder": "",
        "nn_type": "monotonic_icnn",
        "optimizer_type": "icnn_plnn",
        "model_ext": "",
        "tensor_constraint": True,
    }
}

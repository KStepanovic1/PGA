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
        "early_stop": True,
        "nor": True,
        "delta_s": False,
        "s_ext": {"delta_s": "_s_", "early_stop": ""},
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
    "relax_monotonic_icnn": {
        "folder": "predictions/relax_monotonic_icnn",
        "early_stop": True,
        "nor": True,
        "delta_s": False,
        "s_ext": {"delta_s": "_s_", "early_stop": ""},
        "model_ext": "_relax_tau_out_m",
    },
}


experiments_optimization = {
    "plnn_milp": {
        "folder": "plnn_milp",
        "nn_type": "plnn",
        "optimizer_type": "plnn_milp",
    }
}

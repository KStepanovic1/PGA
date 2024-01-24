import numpy as np
import copy
import pickle
import pandas as pd
import statistics
import random
from pathlib import Path
from collections import Counter
from matplotlib import pyplot as plt
import gurobipy
import tensorflow as tf
import warnings
from tensorflow.keras.models import load_model

from dhn_nn.param import time_delay
from util.config import PipePreset1

opt_steps = {"math_opt": [12, 24, 36, 48, 60, 72]}

opt_steps["dnn_opt"] = [
    x - time_delay[str(PipePreset1["Length"])]-1 for x in opt_steps["math_opt"]
]

for index, opt_step in enumerate(opt_steps["math_opt"]):
    print(index, opt_step)
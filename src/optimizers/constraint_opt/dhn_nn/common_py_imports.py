import copy
import time
import os
import math
import re
import sys
import pickle
import warnings
from abc import ABC, abstractmethod
from collections import Counter

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import statistics

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras import backend
from keras.constraints import NonNeg
from keras.models import load_model
from keras.layers import Input, Dense, Add, Activation
from keras import Model as ModelNN
from keras import regularizers
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping
from keras.constraints import non_neg
from keras import backend as K


from gurobipy import *

from src.util import config
from src.util.config import (
    TimeParameters,
    PipePreset1,
    GridProperties,
    ProducerPreset1,
    ConsumerPreset1,
    PhysicalProperties,
    Generator1,
)

from src.optimizers.constraint_opt.dhn_nn.config_experiments import (
    experiments_learning,
    experiments_optimization,
)

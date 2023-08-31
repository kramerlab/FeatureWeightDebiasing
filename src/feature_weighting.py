import random
import numpy as np
from loguru import logger

seed = 5

np.random.seed(seed)
random.seed(seed)
random_generator = np.random.RandomState(seed)

def compute_feature_weights():
    pass


if 
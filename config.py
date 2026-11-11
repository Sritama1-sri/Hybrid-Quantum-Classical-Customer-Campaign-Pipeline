
import os
import random
import numpy as np

# Fixed random seeds
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DATA_SPLIT_SEED = 42
MODEL_SEED = 42
OPTIMIZER_SEED = 42
TRANSPILER_SEED = 42
SIMULATOR_SEED = 42
MONTE_CARLO_SEED = 42

SIMULATOR_SHOTS = 1024
MONTE_CARLO_ITERATIONS = 1000

print("Random seeds configured successfully!")

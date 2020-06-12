import datetime
import random
import numpy as np
import torch


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")


def set_seeds(seed: int) -> None:
    """
    set random seeds
    :param seed: int
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

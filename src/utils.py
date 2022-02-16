import os
import random
import torch
from transformers.file_utils import is_torch_available
import numpy as np


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def file_checker(file_format: str, file_path: str) -> bool:
    """
    Check if file path is a file in the wanted"format"
    :param file_format: file format to check
    :param file_path: file path of file to check
    :return:
    """
    valid = False
    if file_path.endswith(file_format) and os.path.isfile(file_path):
        valid = True
    return valid


def set_workers(workers_ratio: int =None):
    """
    Method to set the number of CPU to use given the workers_ratio.
    First default option is to use 50% of CPU otherwise (if CPU only have 1 core)
    it will choose 1 core
    :param workers_ratio:  Ratio between 0 to 1 to decide percentage to CPU cores to use
    :return:
    """
    if workers_ratio is None:
        workers_ratio = 0.5

    num_workers_ = round(os.cpu_count() * workers_ratio)

    return num_workers_ if num_workers_ < os.cpu_count() else 1


def features_checker(feature_names: list, extracted_feature_names: list):
    """

    :param feature_names:
    :param extracted_feature_names:
    :return:
    """
    for feature_name in feature_names:
        if not feature_name in extracted_feature_names:
            raise NotImplementedError
    return True

"""
Utility Functions
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_random_seed(seed):

    np.random.seed(seed)

    random.seed(seed)

    tf.random.set_seed(seed)


def create_directory(path):

    os.makedirs(path, exist_ok=True)


def print_header(title):

    print("\n" + "=" * 70)

    print(title)

    print("=" * 70)


def print_subheader(title):

    print("\n" + "-" * 60)

    print(title)

    print("-" * 60)

"""
This module implements the LRP epsilon-rule for a linear layer, skipping
the activation function.
"""
import numpy as np


def lrp_linear(x: np.ndarray, y: np.ndarray, rel_y: np.ndarray,
               w: np.ndarray = None, eps: float = 0.001) -> np.ndarray:
    """
    Implements the LRP-epsilon rule for a linear layer.
    
    :param x: Layer input (input_size,)
    :param w: Weight matrix (input_size, output_size)
    :param y: Layer output (output_size,)
    :param rel_y: Network output relevance (output_size,)
    :param eps: Stabilizer
    :return: The relevance of x (input_size,)
    """
    y = y + eps * np.where(y >= 0, 1., -1.)
    if w is None:
        return x * (rel_y / y)
    return (w * x[:, np.newaxis]) @ (rel_y / y)

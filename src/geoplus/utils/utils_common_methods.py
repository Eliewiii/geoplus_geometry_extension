"""
Additional common methods.
"""

import numpy as np
from typing import List
import numpy.typing as npt



def weighted_mean(values: np.ndarray, weights: List[float]) -> float:
    """
    Compute the weighted mean of a set of values.

    Parameters:
        values (np.ndarray): Array of values.
        weights (np.ndarray): Array of weights.

    Returns:
        float: Weighted mean.
    """
    values = np.array(values)
    weights = np.array(weights)

    if values.shape[0] != len(weights):
        raise ValueError("Values and weights must be the same length.")

    weighted_sum = np.sum(values * weights)
    total_weight = np.sum(weights)

    if total_weight == 0:
        raise ValueError("Sum of weights must not be zero.")

    return weighted_sum / total_weight


def remove_redundant_vertices(surface_boundary: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Remove redundant vertices from a surface boundary.
    :param surface_boundary:
    :return: List of vertices of the surface boundary with redundant vertices removed.
    """
    return np.array([vertex for i, vertex in enumerate(surface_boundary) if
                     (np.allclose(surface_boundary[i],surface_boundary[(i + 1) % len(surface_boundary)]))])

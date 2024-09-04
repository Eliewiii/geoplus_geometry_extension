"""
Addional functions for numpy arrays planar surfaces operations.
"""

import numpy as np
from numpy import ndarray
from typing import List

from ..utils.utils_2d_projection import compute_planar_surface_boundary_area_and_centroid
from ..utils.utils_adjustements_surface_with_holes import contour_surface_with_holes


def _compute_numpy_array_planar_surface_area_and_centroid(vertex_array: ndarray) -> (float, ndarray):
    """
    Compute the area and centroid of a 3D planar surface defined by a numpy array of vertices.
    :param vertex_array: Numpy array of vertices of face.
    :return area, centroid: Area of the polygon, centroid of the polygon.
    """

    area, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=vertex_array.tolist())
    return area, np.array(centroid)


# =========================================================
# Public Functions
# =========================================================
def compute_numpy_array_planar_surface_area_and_centroid(vertex_array: ndarray) -> (float, ndarray):
    """
    Compute the area and centroid of a 3D planar surface defined by a numpy array of vertices.
    Note that this method is more accurate in specific cases, such as:
        - for 3D planar surfaces (that some libraries like shapely can't handle natively)
        - for surfaces with holes, being part of the contour of the surface, that libraries like
            Pyvista can't handle.
    :param vertex_array: NumPy array of vertices of face. List of list are also accepted.
    :return area, centroid: Area of the polygon, centroid of the polygon.
    """
    return _compute_numpy_array_planar_surface_area_and_centroid(vertex_array=vertex_array)


def compute_numpy_array_planar_surface_area(vertex_array: ndarray) -> float:
    """
    Compute the area and centroid of a 3D planar surface defined by a numpy array of vertices.
    Note that this method is more accurate in specific cases, such as:
        - for 3D planar surfaces (that some libraries like shapely can't handle natively)
        - for surfaces with holes, being part of the contour of the surface, that libraries like
            Pyvista can't handle.
    :param vertex_array: NumPy array of vertices of face. List of list are also accepted.
    :return area, centroid: Area of the polygon, centroid of the polygon.
    """
    area, _ = _compute_numpy_array_planar_surface_area_and_centroid(vertex_array=vertex_array)
    return area


def compute_numpy_array_planar_surface_centroid(vertex_array: ndarray) -> ndarray:
    """
    Compute the centroid of a 3D planar surface defined by a numpy array of vertices.
    Note that this method is more accurate in specific cases, such as:
        - for 3D planar surfaces (that some libraries like shapely can't handle natively)
        - for surfaces with holes, being part of the contour of the surface, that libraries like
            Pyvista can't handle.
    :param vertex_array: NumPy array of vertices of face. List of list are also accepted.
    :return area, centroid: Area of the polygon, centroid of the polygon.
    """
    _, centroid = _compute_numpy_array_planar_surface_area_and_centroid(vertex_array=vertex_array)
    return centroid


def contour_numpy_array_planar_surface_with_holes(vertex_array: ndarray, hole_array_list: List[ndarray]):
    """

    :param vertex_array:
    :param hole_array_list:
    :return:
    """
    hole_list = [hole_array.tolist() for hole_array in hole_array_list]
    contoured_surface = contour_surface_with_holes(surface_boundary=vertex_array.tolist(),
                                                   hole_list=hole_list)
    return np.array(contoured_surface)

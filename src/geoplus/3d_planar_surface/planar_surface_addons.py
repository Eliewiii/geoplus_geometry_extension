"""
Addional functions for numpy arrays planar surfaces operations.
"""

import numpy as np
from numpy import ndarray
from typing import List

from ..utils.utils_2d_projection import compute_planar_surface_boundary_area_and_centroid
from ..utils.utils_adjustements_surface_with_holes import contour_surface_with_holes



# =========================================================
# Public Functions
# =========================================================
def compute_planar_surface_area_and_centroid(vertex_list: List[List[float]]) -> (float, ndarray):
    """
    Compute the area and centroid of a 3D planar surface defined by a numpy array of vertices.
    Note that this method is more accurate in specific cases, such as:
        - for 3D planar surfaces (that some libraries like shapely can't handle natively)
        - for surfaces with holes, being part of the contour of the surface, that libraries like
            Pyvista can't handle.
    :param vertex_list: NumPy array of vertices of face. List of list are also accepted.
    :return area, centroid: Area of the polygon, centroid of the polygon.
    """
    return compute_planar_surface_boundary_area_and_centroid(surface_boundary=vertex_list)


def compute_planar_surface_area(vertex_list: List[List[float]]) -> float:
    """
    Compute the area and centroid of a 3D planar surface defined by a numpy array of vertices.
    Note that this method is more accurate in specific cases, such as:
        - for 3D planar surfaces (that some libraries like shapely can't handle natively)
        - for surfaces with holes, being part of the contour of the surface, that libraries like
            Pyvista can't handle.
    :param vertex_list: NumPy array of vertices of face. List of list are also accepted.
    :return area, centroid: Area of the polygon, centroid of the polygon.
    """
    area, _ = compute_planar_surface_boundary_area_and_centroid(surface_boundary=vertex_list)
    return area


def compute_planar_surface_centroid(vertex_list: List[List[float]]) -> ndarray:
    """
    Compute the centroid of a 3D planar surface defined by a numpy array of vertices.
    Note that this method is more accurate in specific cases, such as:
        - for 3D planar surfaces (that some libraries like shapely can't handle natively)
        - for surfaces with holes, being part of the contour of the surface, that libraries like
            Pyvista can't handle.
    :param vertex_list: NumPy array of vertices of face. List of list are also accepted.
    :return area, centroid: Area of the polygon, centroid of the polygon.
    """
    _, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=vertex_list)
    return centroid


def contour_numpy_array_planar_surface_with_holes(vertex_list: List[List[float]],
                                                  hole_list: List[List[List[float]]]):
    """

    :param vertex_list:
    :param hole_array_list:
    :return:
    """
    return contour_surface_with_holes(surface_boundary=vertex_list.tolist(),
                                                   hole_list=hole_list)

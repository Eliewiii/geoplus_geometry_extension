"""
Addional functions for numpy arrays planar surfaces operations.
"""

from numpy import ndarray
from typing import List

from ..utils.utils_2d_projection import compute_planar_surface_boundary_area_and_centroid
from ..utils.utils_adjustements_surface_with_holes import contour_surface_with_holes
from ..utils.utils_surface_corners import compute_planar_surface_corners_from_existing_points


# =========================================================
# Centroid and Area
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


# =========================================================
# Contour around holes
# =========================================================
def contour_planar_surface_with_holes(vertex_list: List[List[float]],
                                      hole_list: List[List[List[float]]]) -> List[List[float]]:
    """
    Contour a surface with holes. Especially useful for certain applications, such as Radiance, that cannot accept
        geometries/objects with holes, where the holes have to be excluded from the surface.
    Note that:
     - This function does not check if the holes are inside and coplanar with the surface, it is taken from granted.
    :param vertex_list: List of the vertices of the surface (in 3D). #todo: check if it works in 3D
    :param hole_list: List of the list of vertices of the holes (in 3D).
    :return: List of the vertices of the surface contour without the holes.
    """
    return contour_surface_with_holes(surface_boundary=vertex_list,
                                      hole_list=hole_list)

# =========================================================
# Corners
# =========================================================

def compute_planar_surface_corners(vertex_list: List[List[float]]) -> List[List[float]]:
    """
    Compute the corners of a planar surface defined by a list of vertices.
    :param vertex_list: List of vertices of the surface.
    :return: List of the corners of the surface.
    """
    return compute_planar_surface_corners_from_existing_points(vertex_list)

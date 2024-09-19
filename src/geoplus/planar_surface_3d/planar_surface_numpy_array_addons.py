"""
Addional functions for numpy arrays planar surfaces operations.
"""

import numpy as np
from numpy import ndarray

from pyvista import PolyData

from typing import List
import numpy.typing as npt

from ..utils.utils_2d_projection import compute_planar_surface_boundary_area_and_centroid, \
    get_normal_vector_of_planar_surface
from ..utils.utils_adjustements_surface_with_holes import contour_surface_with_holes, \
    compute_exterior_boundary_of_surface_with_contoured_holes
from ..utils.utils_surface_corners import compute_planar_surface_corners_from_existing_points


# =========================================================
# Centroid, Area and Normal
# =========================================================
def compute_numpy_array_planar_surface_area_and_centroid(surface_boundary: npt.NDArray[np.float64]) -> (
        float, npt.NDArray[np.float64]):
    """
    Compute the area and centroid of a 3D planar surface defined by a numpy array of vertices.
    :param surface_boundary: Numpy array of vertices of face.
    :return area, centroid: Area of the polygon, centroid of the polygon.
    """

    area, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=surface_boundary)
    return area, centroid


def compute_numpy_array_planar_surface_area(surface_boundary: npt.NDArray[np.float64]) -> float:
    """
    Compute the area and centroid of a 3D planar surface defined by a numpy array of vertices.
    Note that this method is more accurate in specific cases, such as:
        - for 3D planar surfaces (that some libraries like shapely can't handle natively)
        - for surfaces with holes, being part of the contour of the surface, that libraries like
            Pyvista can't handle.
    :param surface_boundary: NumPy array of vertices of face. List of list are also accepted.
    :return area, centroid: Area of the polygon, centroid of the polygon.
    """
    area, _ = compute_planar_surface_boundary_area_and_centroid(surface_boundary=surface_boundary)
    return area


def compute_numpy_array_planar_surface_centroid(surface_boundary: npt.NDArray[np.float64]) -> npt.NDArray[
    np.float64]:
    """
    Compute the centroid of a 3D planar surface defined by a numpy array of vertices.
    Note that this method is more accurate in specific cases, such as:
        - for 3D planar surfaces (that some libraries like shapely can't handle natively)
        - for surfaces with holes, being part of the contour of the surface, that libraries like
            Pyvista can't handle.
    :param surface_boundary: NumPy array of vertices of face. List of list are also accepted.
    :return area, centroid: Area of the polygon, centroid of the polygon.
    """
    _, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=surface_boundary)
    return centroid


def compute_numpy_array_planar_surface_normal(surface_boundary: npt.NDArray[np.float64]) -> List[float]:
    """
    Compute the normal vector of a 3D planar surface defined by an array of vertices.
    :param surface_boundary: NumPy array of vertices of face. List of list are also accepted.
    :return normal: Normal vector of the polygon.
    """
    return get_normal_vector_of_planar_surface(surface_boundary)


# =========================================================
# Contour around holes
# =========================================================
def contour_numpy_array_planar_surface_with_holes(surface_boundary: npt.NDArray[np.float64],
                                                  hole_array_list: npt.NDArray[np.float64]):
    """
    Contour a planar surface with holes.
    :param surface_boundary: Array of vertices of the surface.
    :param hole_array_list: List of arrays of vertices of the holes.
    :return: Array of vertices of the surface with holes.
    """
    return contour_surface_with_holes(surface_boundary=surface_boundary,
                                      hole_list=hole_array_list)


def compute_exterior_boundary_of_numpy_array_planar_surface_with_contoured_holes(
        surface_boundary: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Remove the holes from the surface boundary to get the exterior boundary of the surface, especially useful if
        a surface is contoured with holes to transform it into a surface format that do not handle holes properly,
        such as Pyvista.
    :param surface_boundary:
    :return:
    """
    return compute_exterior_boundary_of_surface_with_contoured_holes(surface_boundary)


# =========================================================
# Corners
# =========================================================
def compute_numpy_array_planar_surface_corners(surface_boundary: npt.NDArray[np.float64]) -> npt.NDArray[
    np.float64]:
    """
    Compute the corners of a planar surface defined by a list of vertices.
    :param surface_boundary: Array of vertices of the surface.
    :return: Array of the corners of the surface.
    """
    return compute_planar_surface_corners_from_existing_points(surface_boundary)


# =========================================================
# To polydata
# =========================================================
def numpy_array_surface_to_polydata(surface_boundary: npt.NDArray[np.float64]) -> PolyData:
    """
    Convert a numpy array to a PyVista PolyData object.
    :param surface_boundary:
    :return: PyVista PolyData object.
    """
    return PolyData(surface_boundary,np.array(
                    [[len(surface_boundary), *range(len(surface_boundary))]]))
    # vertices = []  # initialize the list of vertices
    # # format of the face expected by PyVista [number of vertices, index vertex 1, index vertex 2 ...] <=> [n, 0,1,2...]
    # face = [len(surface_boundary)]
    # # convert the vertices to a list of coordinates and add the vertices index to the face
    # for index, vertex in enumerate(surface_boundary):
    #     vertices.append([vertex[0], vertex[1], vertex[2]])  # add the coordinates of the vertices
    #     face.append(index)
    # vertices = np.array(vertices)  # convert into numpy array, makes it faster for Pyvista to process
    #
    # return PolyData(vertices, face)


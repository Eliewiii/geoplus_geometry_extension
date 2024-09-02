"""
Callable class to convert a PyVista PolyData object to a GeoPandas GeoDataFrame object.
"""

import pyvista as pv
import numpy as np

from typing import List, Tuple, Union

from ..utils.utils_common_methods import weighted_mean
from ..utils.utils_2d_projection import compute_planar_surface_area_and_centroid


# =========================================================
# Private Helper Functions
# =========================================================

def _get_list_of_vertices_of_each_face(polydata: pv.PolyData) -> List[List[List[float]]]:
    """
    Get the list of vertices for each face of the PolyData object.
    :param polydata_obj: pv.PolyData, the PolyData object.
    :return list_of_list_of_vertices: List of vertices for each face.
    """
    # Extract points from the PolyData object
    points = polydata.points
    # Extract the faces
    faces = polydata.faces
    face_list = []
    """
    The faces array contains the number of vertices for each face followed by the indices of the vertices as follows:
    [num_vertices_face_1, vertex_1_face_1, vertex_2_face_1, ..., vertex_n_face_1, num_vertices_face_2, vertex_1_face_2, ...]
    """
    index = 0
    while index < len(faces):
        num_vertices = faces[index]
        face_list.append(faces[index + 1:index + num_vertices + 1])
        index += num_vertices + 1
    list_of_vertices = [[points[pt_index] for pt_index in face] for face in face_list]

    return list_of_vertices


def _compute_area_and_centroid(polydata: pv.PolyData) -> (float, List[List[float]]):
    """
    Compute the area and centroid of a planar PyVista PolyData object.
    Compared to the build-in method, this method is more accurate in specific cases, such as:
        - for surfaces with holes, being part of the contour of the surface
        - the centroid is computed as the weighted mean of the centroids of the faces
            (and not the weighted mean of the vertices), it works as well for surfaces with holes.
    Works for planar surfaces only !
    :param polydata: PyVista PolyData object.
    :return area, centroid: Area of the surface, centroid of the surface.
    """
    area_list = []
    centroid_list = []
    list_of_face_vertices = _get_list_of_vertices_of_each_face(polydata)
    for face_vertices in list_of_face_vertices:
        face_area, face_centroid = compute_planar_surface_area_and_centroid(face_vertices)
        area_list.append(face_area)
        centroid_list.append(face_centroid)
    centroid = weighted_mean(np.array(centroid_list), area_list)
    return sum(area_list), centroid_list


# =========================================================
# Public Functions
# =========================================================
def compute_area_and_centroid_of_polydata(polydata):
    return _compute_area_and_centroid(polydata)


def compute_area_of_polydata(polydata):
    area, _ = _compute_area_and_centroid(polydata)
    return area


def compute_centroid_of_polydata(polydata):
    _, centroid = _compute_area_and_centroid(polydata)
    return centroid

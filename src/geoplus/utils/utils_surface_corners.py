"""
Utilities for computing the corners of a planar 3D or 2D surface.
"""

import numpy as np

from typing import List, Tuple, Union

from numpy.random import normal

from .utils_2d_projection import get_normal_vector_of_planar_surface, get_planar_surface_plan_vectors_from_normal, \
    transform_3d_vertices_to_2d, transform_2d_vertices_to_3d


def compute_corners_from_existing_points(surface_boundary: List[List[float]]) -> List[List[float]]:
    """
    Computes the corners of a 3D planar surface using existing points of the surface.
    Aims to compute the top, bottom, right, and left corners of the surface.
    The notion of top,bottom, right and left being dependent on the surface and the observer,
    this function has several bias:
      - The top is defined according to the +z vector, except for surfaces in the xy plane, in that case
            the top is set arbitrary to +x
      - The right and left are defined according to the orientation of the surface
      - In many cases, this function is more likely to return top-right, top-left, bottom-right and bottom-left vertices
            instead of top, bottom, right, and left, as all the vertices should be different as far as possible.
    Overall, this functions aims more to return "extreme" corners in general, for instance to check the visibility
        between surface. Getting unique top, bottom, right, and left is not possible for many surfaces.
    Note that the function assumes that the surface is planar.
    :param surface_boundary: List of vertices of the planar surface
    :return: np.ndarray, the corners of the surface in the original coordinate system.
    """

    # Compute the normal
    normal = get_normal_vector_of_planar_surface(surface_boundary=surface_boundary)
    # Define a reference vector for rotation
    x_axis = np.array([1, 0, 0])
    z_axis = np.array([0, 0, 1])
    if np.allclose(normal, z_axis) or np.allclose(normal, -z_axis):
        reference_vector = x_axis
    else:
        reference_vector = z_axis
    # Compute the rotation axis and angle
    v1, v2 = get_planar_surface_plan_vectors_from_normal(surface_boundary=surface_boundary, normal=normal,
                                                         reference_vector=reference_vector)
    rotation_matrix = np.vstack([np.array(v1), np.array(v2), np.array(normal)]).T
    translation_vector = -np.array(surface_boundary[0])
    # Get the points in the 2d local coordinate system
    local_point_2d = np.array(transform_3d_vertices_to_2d(points_3d=surface_boundary, rotation_matrix=rotation_matrix,
                                                          translation_vector=translation_vector))
    # Get the minimum and maximum u and v values of the local points
    min_u = np.min(local_point_2d[:, 0])
    max_u = np.max(local_point_2d[:, 0])
    min_v = np.min(local_point_2d[:, 1])
    max_v = np.max(local_point_2d[:, 1])
    # Potential points for each extreme
    potential_u_min_points = local_point_2d[np.isclose(local_point_2d[:, 0], min_u)]
    potential_u_max_points = local_point_2d[np.isclose(local_point_2d[:, 0], max_u)]
    potential_v_min_points = local_point_2d[np.isclose(local_point_2d[:, 1], min_v)]
    potential_v_max_points = local_point_2d[np.isclose(local_point_2d[:, 1], max_v)]

    # Select unique points based on priority

    min_u_point = select_unique_point(potential_u_min_points, [],
                                      to_be_selected_points=[potential_v_min_points, potential_v_max_points,
                                                             potential_u_max_points])
    max_u_point = select_unique_point(potential_u_max_points, [min_u_point],
                                      to_be_selected_points=[potential_v_min_points, potential_v_max_points])
    max_v_point = select_unique_point(potential_v_max_points, [min_u_point, max_u_point],
                                      to_be_selected_points=[potential_u_max_points])
    min_v_point = select_unique_point(potential_v_min_points, [min_u_point, max_u_point, max_v_point],
                                      to_be_selected_points=[])
    # Corners are the combinations of these extreme points
    corners_local = np.array([
        max_u_point,
        min_u_point,
        max_v_point,
        min_v_point
    ])
    # Transform the corners back to the original coordinate system
    corners_original_coordinate_system = transform_2d_vertices_to_3d(point_2d=corners_local.tolist())

    return corners_original_coordinate_system


def select_unique_point(potential_points: List[np.array], selected_points: np.array,
                        to_be_selected_points: List[np.array]) -> np.array:
    """
    Selects a unique point (as far as possible), from the potential points based on the selected points
        and the points to be selected.
    :param potential_points: List[np.array], the list of potential points.
    :param selected_points: List[np.array], the list of already selected points.
    :param to_be_selected_points: List[np.array], the list of points to be selected.
    :return: np.array, the selected point, unique if possible.
    """
    unique_point = potential_points[0]
    occurrences, min_option_left = get_occurrences(to_be_selected_points, selected_points, unique_point)
    for point in potential_points:
        if not any(np.allclose(point, sel_point) for sel_point in selected_points):
            local_occurrences, local_min_option_left = get_occurrences(to_be_selected_points, selected_points,
                                                                       point)
            if local_occurrences <= occurrences and local_min_option_left >= min_option_left:
                unique_point = point  # The priority is to select a point not already selected.
                occurrences, min_option_left = local_occurrences, local_min_option_left
                if occurrences == 0:
                    return point
    return unique_point  # Fallback if no unique point found


def get_occurrences(to_be_selected_points: List[np.array], selected_points_array: np.array,
                    point: np.array) -> (int, int):
    """
    Get the number of occurrences of a point in an array.
    :param to_be_selected_points:
    :param selected_points_array:
    :param point: np.array, the point to count.
    :return: int, the number of occurrences and the minimum number of options left of a surface with
        the occurence.
    """
    if len(to_be_selected_points) == 0:
        return 0, 0
    occurrence, min_option_left = 0, max([len(array) for array in to_be_selected_points])
    for array in to_be_selected_points:
        if np.sum(np.all(array == point, axis=1)) > 0:
            occurrence += 1
            if len(array) - nb_intersection(array, selected_points_array) < min_option_left:
                min_option_left = len(array) - nb_intersection(array, selected_points_array)
    return occurrence, min_option_left


def nb_intersection(array1: np.array, array2: np.array) -> int:
    """
    Compute the number of intersections between two arrays.
    :param array1: np.array, the first array.
    :param array2: np.array, the second array.
    :return: int, the number of intersections.
    """
    # Convert rows to sets of tuples for intersection
    array1_set = set(map(tuple, array1))
    array2_set = set(map(tuple, array2))

    # Find common rows
    common_rows = array1_set.intersection(array2_set)

    # Count of common rows
    count_common_rows = len(common_rows)
    return count_common_rows

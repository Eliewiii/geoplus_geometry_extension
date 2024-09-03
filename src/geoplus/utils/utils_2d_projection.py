"""

"""

import numpy as np

from shapely.geometry import Polygon
from numpy import ndarray

from typing import List


def compute_planar_surface_boundary_area_and_centroid(surface_boundary: List[List[float]]) -> (float, List[float]):
    """
    Compute the area of a  planar surface defined by a list of vertices.
    :param surface_boundary: List of vertices of face
    :return are,centroid: Area of the polygon.
    """
    point_2d, rotation_matrix, translation_vector = compute_planar_surface_coordinate_in_local_2d_plan(
        surface_boundary)
    # Convert to shapely polygon, with powerful methods for 2D geometries
    polygon_2d_in_local_plan: Polygon = Polygon(point_2d)
    area = polygon_2d_in_local_plan.area  # Isometric transformation, so the area is preserved
    centroid_local_plan = get_polygon_centroid(polygon_2d_in_local_plan)
    centroid = transform_2d_vertices_to_3d(point_2d=centroid_local_plan, rotation_matrix=rotation_matrix,
                                           translation_vector=translation_vector)
    return area, centroid.tolist()


def compute_planar_surface_coordinate_in_local_2d_plan(surface_boundary: List[List[float]]) -> np.ndarray:
    """
    Transform 3D points to a 2D plane using the provided rotation matrix and translation vector.
    :param surface_boundary: List of vertices of the polygon in 3D.
    :return
    """
    rotation_matrix, translation_vector = compute_transformation_to_local_2d_plan(surface_boundary)
    points_2d = transform_3d_vertices_to_2d(surface_boundary, rotation_matrix, translation_vector)
    return points_2d.tolist()


def compute_transformation_to_local_2d_plan(surface_boundary: List[List[float]]) -> [np.ndarray, np.ndarray]:
    """
    Compute the transformation matrix that maps 3D coordinates to the plane of the polygon.
    Ensures that the points are not collinear.
    """
    normal = get_normal_vector_of_planar_surface(surface_boundary=surface_boundary)
    # Calculate the transformation matrix to the plane
    z_axis = normal
    x_axis ,y_axis = get_planar_surface_plan_vectors_from_normal(surface_boundary, normal)
    rotation_matrix = np.vstack([np.array(x_axis), np.array(y_axis), np.array(z_axis)]).T
    translation_vector = -np.array(surface_boundary[0])  # Translate the first point to the origin

    return rotation_matrix, translation_vector


def get_normal_vector_of_planar_surface(surface_boundary: List[List[float]]) -> List[List[float]]:
    """

    :param surface_boundary:
    :return:
    """

    def are_points_collinear(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = np.cross(v1, v2)
        return np.allclose(cross_product, 0.)

    points_3d = np.array(surface_boundary)
    # Try to find a valid set of points to define the plane
    p1 = points_3d[0]
    p2 = points_3d[1]
    for p3 in points_3d[2:]:
        if not are_points_collinear(p1, p2, p3):
            break
        else:
            continue
    if are_points_collinear(p1, p2, p3):
        raise ValueError("Cannot find non-collinear points to define a plane")

    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # Normalize

    return normal.tolist()


def get_planar_surface_plan_vectors_from_normal(surface_boundary: List[List[float]], normal: List[float],
                                                reference_vector: List[float] = None)-> [List[float], List[float]]:
    """

    :param surface_boundary:
    :param normal:
    :param reference_vector:
    :return:
    """
    point_3d = np.array(surface_boundary)
    normal = np.array(normal)
    if reference_vector is None:
        v1 = point_3d[1] - point_3d[0]
    else:
        if np.allclose(np.cross(normal, np.array(reference_vector)), 0.):
            raise ValueError("Reference vector cannot be colinear with the normal vector")
        v1 = np.array(reference_vector) - np.dot(np.array(reference_vector), np.array(normal)) * np.array(normal)

    v1 = np.array(normalize_vector(v1))
    v2 = np.cross(normal, v1)

    return v1.tolist(), v2.tolist()


def transform_3d_vertices_to_2d(points_3d: List[float], rotation_matrix: ndarray,
                                translation_vector: ndarray) -> ndarray:
    """
    Transform 3D points to a 2D plane using the provided rotation matrix and translation vector.
    """
    points_3d = np.array(points_3d)
    transformed_points = (points_3d + translation_vector) @ rotation_matrix
    return transformed_points[:, :2]


def transform_2d_vertices_to_3d(point_2d: List[float], rotation_matrix: ndarray, translation_vector: ndarray):
    """
    Transform a 2D point back to the original 3D space using the inverse of the rotation matrix and translation vector.
    """
    point_2d = np.array(point_2d)
    # Inverse rotation matrix
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    # Apply the inverse transformation
    point_3d = (np.hstack([point_2d, 0]) @ inv_rotation_matrix) - translation_vector
    return point_3d.tolist()


def get_polygon_centroid(polygon: Polygon) -> (float, float):
    # Compute the centroid
    centroid = polygon.centroid
    return centroid.x, centroid.y

def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize a vector.
    :param vector: List[float], the vector to normalize.
    :return: List[float], the normalized vector.
    """
    vector = np.array(vector)
    try :
        return (vector / np.linalg.norm(vector)).tolist()
    except ZeroDivisionError:
        raise ValueError("Cannot normalize a zero vector")

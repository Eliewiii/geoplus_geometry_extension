"""

"""

import numpy as np
from numpy.ma.core import allclose

from shapely.geometry import Polygon
from numpy import ndarray

from typing import List
import numpy.typing as npt

from .utils_common_methods import remove_redundant_vertices


def compute_planar_surface_boundary_area_and_centroid(surface_boundary: npt.NDArray[np.float64]) -> (float, npt.NDArray[np.float64]):
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
    return area, centroid


def compute_planar_surface_coordinate_in_local_2d_plan(surface_boundary: npt.NDArray[np.float64]) -> np.ndarray:
    """
    Transform 3D points to a 2D plane using the provided rotation matrix and translation vector.
    :param surface_boundary: List of vertices of the polygon in 3D.
    :return
    """
    rotation_matrix, translation_vector = compute_transformation_to_local_2d_plan(surface_boundary)
    points_2d = transform_3d_vertices_to_2d(surface_boundary, rotation_matrix, translation_vector)
    return points_2d, rotation_matrix, translation_vector


def compute_transformation_to_local_2d_plan(surface_boundary: npt.NDArray[np.float64]) -> [np.ndarray, np.ndarray]:
    """
    Compute the transformation matrix that maps 3D coordinates to the plane of the polygon.
    Ensures that the points are not collinear.
    """
    surface_boundary_without_redundant_vertices = remove_redundant_vertices(surface_boundary)
    normal = get_normal_vector_of_planar_surface(surface_boundary=surface_boundary_without_redundant_vertices)
    # Calculate the transformation matrix to the plane
    z_axis = normal
    x_axis, y_axis = get_planar_surface_plan_vectors_from_normal(surface_boundary_without_redundant_vertices, normal)
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    translation_vector = surface_boundary_without_redundant_vertices[0]  # Translate the first point to the origin

    return rotation_matrix, translation_vector


def get_normal_vector_of_planar_surface(surface_boundary: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute the normal vector of a planar surface defined by a list of vertices.
    This method ensures that the normal vector is oriented correctly, even with convex surfaces, and theoretically
        self-intersecting surfaces.
    :param surface_boundary:
    :return:
    """

    def are_points_collinear(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = np.cross(v1, v2)
        return np.allclose(cross_product, 0.)

    surface_boundary_without_redundant_vertices = remove_redundant_vertices(surface_boundary)
    # Try to find a valid set of points to define the plane
    p1 = surface_boundary_without_redundant_vertices[0]
    p2 = surface_boundary_without_redundant_vertices[1]
    for p3 in surface_boundary_without_redundant_vertices[2:]:
        if not are_points_collinear(p1, p2, p3):
            break
        else:
            continue
    if are_points_collinear(p1, p2, p3):
        raise ValueError("Cannot find non-collinear points to define a plane")

    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normalize_vector(normal)

    # Calculate the total oriented angle
    total_oriented_angle = 0
    n = len(surface_boundary_without_redundant_vertices)
    for i in range(n):
        v1 = surface_boundary_without_redundant_vertices[(i + 1) % n] - surface_boundary_without_redundant_vertices[i]
        v2 = surface_boundary_without_redundant_vertices[(i + 2) % n] - surface_boundary_without_redundant_vertices[(i + 1) % n]
        total_oriented_angle += compute_oriented_angle(v1, v2, normal)

    # Adjust the normal vector based on the total oriented angle
    if np.allclose(total_oriented_angle, -360):
        normal = -normal  # Flip the normal if the angle is -360 degrees
    elif np.allclose(total_oriented_angle, 360):
        pass
    else:
        raise ValueError(
            "The sum of the oriented angles is not equal to 360 degrees, the might not be a planar surface")

    return normal


def get_planar_surface_plan_vectors_from_normal(surface_boundary: npt.NDArray[np.float64], normal: npt.NDArray[np.float64],
                                                reference_vector: npt.NDArray[np.float64] = None) -> (npt.NDArray[np.float64], npt.NDArray[np.float64]):
    """

    :param surface_boundary:
    :param normal:
    :param reference_vector:
    :return:
    """
    surface_boundary_without_redundant_vertices = remove_redundant_vertices(surface_boundary)
    normal = normal
    if reference_vector is None:
        # necessarilly non null as there is no redundancy
        v1 = surface_boundary_without_redundant_vertices[1] - surface_boundary_without_redundant_vertices[0]
    elif allclose(reference_vector,np.zeros(3)):
        if np.allclose(np.cross(normal, reference_vector), 0.):
            raise ValueError("Reference vector cannot be colinear with the normal vector")
        v1 = reference_vector - np.dot(reference_vector, normal) * normal # projection of the reference vector on the plane
    else:
        raise ValueError(f"Reference vector for surface {surface_boundary} must be a non-null vector")

    v1 = np.array(normalize_vector(v1))
    v2 = np.cross(normal, v1)  # already normalized as v1 and normal are normalized

    return v1, v2


def transform_3d_vertices_to_2d(vertices_3d: npt.NDArray[np.float64], rotation_matrix: npt.NDArray[np.float64],
                                translation_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Transform 3D points to a 2D plane using the provided rotation matrix and translation vector.
    """
    transformed_points = (vertices_3d + translation_vector) @ rotation_matrix
    return transformed_points[:, :2]  # Keep only the first coordinates as it is a 2D projection


def transform_2d_vertices_to_3d(vertices_2d: npt.NDArray[np.float64], rotation_matrix: npt.NDArray[np.float64],
                                translation_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Transform a 2D vertices back to the original 3D space using the inverse of the rotation matrix and translation vector.
    :param vertices_2d: numpy array, the 2D point to transform.
    :param rotation_matrix: numpy array, the rotation matrix used for the transformation.
    :param translation_vector: numpy array, the translation vector used for the transformation.
    :return: numpy array, the vertices in the 3D coordinate system
    """
    # Inverse rotation matrix
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    # Apply the inverse transformation
    return (np.hstack([vertices_2d, 0]) @ inv_rotation_matrix) - translation_vector


def get_polygon_centroid(polygon: Polygon) -> npt.NDArray[np.float64]:
    # Compute the centroid
    centroid = polygon.centroid
    return np.array([centroid.x, centroid.y])


def normalize_vector(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Normalize a vector.
    :param vector: numpy array, the vector to normalize.
    :return: numpy array, the normalized vector.
    """
    try:
        return (vector / np.linalg.norm(vector))
    except ZeroDivisionError:
        raise ValueError("Cannot normalize a zero vector")


def compute_oriented_angle(v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64],
                           normal: npt.NDArray[np.float64]) -> float:
    """
    Compute the oriented angle between two vectors according to a reference normal vector.
    :param v1: numpy array, the first vector.
    :param v2: numpy array, the second vector.
    :param normal: numpy array, the reference normal vector.
    :return: float, the oriented angle between the two vectors.
    """
    # Calculate angle between vectors
    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    # Determine the orientation using the cross product
    if np.dot(np.cross(v1, v2), normal) < 0:
        angle = -angle
    return np.degrees(angle)

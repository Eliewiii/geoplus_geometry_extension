"""

"""

import numpy as np
import pyvista as pv

from typing import List
import numpy.typing as npt



def are_planar_surface_vertices_facing_each_other(vertex_surface_1: npt.NDArray[np.float64],
                                                  vertex_surface_2: npt.NDArray[np.float64],
                                                  normal_1: npt.NDArray[np.float64], normal_2: npt.NDArray[np.float64]):
    """
    Check if two planar surfaces are facing each other. It does not consider obstacles between the two surfaces.
    This method can be run for multiple couples of vertices of the two surfaces for better accuracy.
    :param vertex_surface_1: A vertex of the first surface
    :param vertex_surface_2: A vertex of the second surface
    :param normal_1: Normal vector of the first surface
    :param normal_2: Normal vector of the second surface
    :return: True if the two surfaces are seeing each other, False otherwise.
    """
    vector_1_2 = vertex_surface_2 - vertex_surface_1
    dot_product_1 = np.dot(normal_1, vector_1_2)
    dot_product_2 = np.dot(normal_2, vector_1_2)
    # visibility/facing criteria  (same as PyviewFactor)
    if dot_product_1 > 0 > dot_product_2:
        return True
    else:
        return False


RAY_OFFSET = 0.05  # offset to avoid considering the sender and receiver in the raytracing obstruction detection


def is_ray_intersecting_context(start_point: npt.NDArray[np.float64], end_point: npt.NDArray[np.float64],
                                context_polydata_mesh: pv.PolyData,
                                offset: float = RAY_OFFSET) -> bool:
    """
    Check if a ray intersects a context mesh.

    :param start_point: numpy array vertex of the start point of the ray
    :param end_point: numpy array vertex of the end point of the ray
    :param context_polydata_mesh: PyVista PolyData object of the context mesh
    :param offset: float, offset to avoid considering the sender and receiver in the raytracing obstruction detection
    :return:
    """
    corrected_start_point, corrected_end_point = _excluding_surfaces_from_ray(start_point=start_point,
                                                                              end_point=end_point, offset=offset)
    points, ind = context_polydata_mesh.ray_trace(origin=corrected_start_point[0], end_point=corrected_start_point[1],
                                                  first_point=False,
                                                  plot=False)
    if ind == 0:
        return False
    else:
        return True


# =========================================================
# Private Helper Functions
# =========================================================

def _excluding_surfaces_from_ray(start_point: npt.NDArray[np.float64], end_point: npt.NDArray[np.float64],
                                 offset: float = RAY_OFFSET):
    """
        Return the start and end point of a ray reducing slightly the distance between the vertices to prevent
        considering the sender and receiver in the raytracing obstruction detection
        :param start_point: numpy array, start point of the ray
        :param end_point: numpy array, end point of the ray
        :param offset: float, offset to avoid considering the sender and receiver in the raytracing obstruction detection
        :return: new_start_point, new_end_point: numpy arrays, new start and end points of the ray
    """
    unit_vector = (end_point - start_point) / np.linalg.norm(end_point - start_point)
    # Move the ray boundaries
    new_start_point = start_point + unit_vector * offset  # move the start vertex by 5cm on the toward the end vertex
    new_end_point = end_point - unit_vector * offset  # move the end vertex by 5cm on the toward the start vertex

    return new_start_point, new_end_point

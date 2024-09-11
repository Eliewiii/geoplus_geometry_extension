"""
Function to contour a surface with holes in order to use it in Radiance simulations.
"""


import pyvista as pv
import numpy as np

from typing import List


def contour_surface_with_holes(surface_boundary: List[List[float]], hole_list: List[List[List[float]]]) -> List[
    List[float]]:
    """
    Contour a surface with multiple holes to exclude the holes from the surface, especially useful for Radiance.
    It assumes that the holes are closed loops of vertices with the same orientation as the surface and that the holes
        don't intersect.
    Some weird results may occur with exotic surfaces or holes, with convex surfaces/holes, that might lead to self
        intersecting surfaces. The code can be adjusted to avoid this issue, but it will imply more complexity
        and verifications making the function much slower. It might be implemented in the future.
    :param surface_boundary: List of vertices of the surface in 2 or 3 dimensions, it must be consistent with the holes.
    :param hole_list: List of lists of vertices of the holes in 2 or 3 dimensions, it must be consistent with the surface.
    :return new_surface_boundary: List of vertices of the contoured surface with the holes.
    """
    new_surface_boundary = surface_boundary
    for hole in hole_list:
        new_surface_boundary = _contour_surface_with_hole(new_surface_boundary, hole)

    return new_surface_boundary


def compute_exterior_boundary_of_surface_with_contoured_holes(surface_boundary: List[List[float]]) -> List[List[float]]:
    """
    Remove the holes from the surface boundary to get the exterior boundary of the surface, especially useful if
        a surface is contoured with holes to transform it into a surface format that do not handle holes properly,
        such as Pyvista.
    This function detects the holes by checking the vertices that are repeated in the surface boundary and the vertices
        in between. A special check is done to avoid removing the exterior boundary instead of the hole itself.
    :param surface_boundary:
    :return exterior_surface_boundary: List of vertices of the exterior boundary of the surface.
    """
    num_vertices = len(surface_boundary)
    vertex_index_to_remove = []
    # Check the vertices that are repeated in the surface boundary
    for i in range(num_vertices):
        # The vertex i is not checked as if it was removed but it is the "beginning" of a hole, it will not delete this hole.
        # Worst case it can only create redundant vertices to delete
        for j in range(i+1,num_vertices):
            if j in vertex_index_to_remove:
                continue
            if np.allclose(surface_boundary[j], surface_boundary[i]):  # Maybe add a specific tolerance
                """
                If a hole was removed from the exterior boundary, the vertices just after and before the hole will be
                    the same.
                If they are different, We do not consider it as an iner-hole , but more like a convex surface
                """
                if np.allclose(surface_boundary[j-1], surface_boundary[i+1]):
                    vertex_index_to_remove.extend([k for k in range(i+1,j+1)])
    exterior_surface_boundary = [vertex for i, vertex in enumerate(surface_boundary) if i not in vertex_index_to_remove]
    # make sure two consecutive vertices are not the same
    exterior_surface_boundary = [vertex for i, vertex in enumerate(exterior_surface_boundary) if (exterior_surface_boundary[i] != exterior_surface_boundary[(i+1)%len(exterior_surface_boundary)])]

    return exterior_surface_boundary



def _contour_surface_with_hole(surface_boundary: List[List[float]], hole_vertex_list: List[List[float]]) \
        -> List[List[float]]:
    """
    Contour a surface with a hole to exclude the hole from the surface, especially useful for Radiance. The hole is
        assumed to be a closed loop of vertices with the same orientation as the surface.
    :param surface_boundary: List of vertices of the surface in 2 or 3 dimensions, it must be consistent with the hole.
    :param hole_vertex_list: List of vertices of the hole in 2 or 3 dimensions, it must be consistent with the surface.
    :return new_surface_boundary: List of vertices of the contoured surface with the hole.
    """
    # Convert the input to numpy arrays
    surface_vertices = np.array(surface_boundary)
    hole_vertices = np.array(hole_vertex_list)
    """ Inverse the hole vertices to match as the hole should be "travelled" in the opposite direction not to create a
    surface with an intersection """
    hole_vertices = hole_vertices[::-1]
    # Find the closest points on the surface and hole
    surface_closest_index = _closest_point_index(surface_vertices, hole_vertices[0])
    hole_closest_index = _closest_point_index(hole_vertices, surface_vertices[surface_closest_index])
    # Split and merge the surface boundary
    new_surface_boundary = np.concatenate([
        surface_vertices[:surface_closest_index + 1],
        hole_vertices[hole_closest_index:],
        hole_vertices[:hole_closest_index + 1],
        surface_vertices[surface_closest_index:]
    ])

    return new_surface_boundary.tolist()


def _closest_point_index(vertex_list: List[np.ndarray], target_vertex: np.ndarray) -> int:
    """
    Find the index of the closest vertex to a target vertex in a list of vertices.
    :param vertex_list: List of vertices in 2 or 3 dimensions, it must be consistent with the target vertex.
    :param target_vertex: Target vertex in 2 or 3 dimensions.
    :return closest_index: Index of the closest vertex to the target vertex.
    """
    distances = np.linalg.norm(vertex_list - target_vertex, axis=1)
    return np.argmin(distances)

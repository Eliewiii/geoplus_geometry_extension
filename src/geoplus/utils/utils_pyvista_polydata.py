"""
Additional utility functions for working with PyVista PolyData objects.
"""
from pyvista import PolyData
import numpy as np

from shapely.geometry import Polygon

from typing import List, Tuple, Union


def make_polydata_from_vertices(vertices: np.ndarray) -> PolyData:
    """
    Create a PyVista PolyData object from vertices and faces.
    :param vertices: np.ndarray, the vertices of the PolyData object.
    :return: pv.PolyData, the PolyData object.
    """
    faces = np.array([[len(vertices)] + list(range(len(vertices)))])
    return PolyData(vertices, faces)


def polydata_to_shapely(polydata):
    # Extract points from the PolyData object
    points = polydata.points

    # Assuming there is only one face in the PolyData
    face = polydata.faces.reshape(-1, polydata.faces[0] + 1)[0,
           1:]  # Extracting the first face and ignoring the first number (which is the number of vertices in the face)

    # Extract the points corresponding to the face
    face_points = points[face]

    # Convert to a Shapely Polygon object
    polygon = Polygon(face_points)

    return polygon


def get_faces_list_of_vertices(polydata_obj: PolyData) -> List[List[float]]:
    """
    Get the list of vertices for each face of the PolyData object.
    :param polydata_obj: pv.PolyData, the PolyData object.
    :return list_of_vertices: List of vertices for each face.
    """
    # Extract points from the PolyData object
    points = polydata_obj.points
    # Extract the faces
    faces = polydata_obj.faces
    face_list = []
    index = 0
    while index < len(faces):
        num_vertices = faces[index]
        face_list.append(faces[index + 1:index + num_vertices + 1])
        index += num_vertices + 1
    list_of_vertices = [[points[pt_index] for pt_index in face] for face in face_list]

    return list_of_vertices


def compute_polydata_area(polydata_obj: PolyData) -> float:
    """
    Compute the area of the PolyData.
    :param polydata_obj: pv.PolyData, the PolyData object.
    :return: float, the area of the PolyData.
    """
    cell_sizes = polydata_obj.compute_cell_sizes()
    areas = cell_sizes['Area']
    return sum(areas)


def compute_geometric_centroid(polydata_obj: PolyData) -> np.ndarray:
    """
    Computes the geometric centroid of a planar surface represented as a PyVista PolyData object.
    The geometric centroid is the center of mass of the surface, considering it as a uniform material.
    :param polydata_obj: pv.PolyData, the PolyData object representing the planar surface.
    :return: np.ndarray, the geometric centroid of the surface.

    Notes:
        - The function assumes that the input surface is already a single planar polygon.
        - It automatically triangulates the surface if it's not already made of triangles.
    """
    # Triangulate the polygon to break it into triangles
    triangulated = polydata_obj.triangulate()
    # Get the points and the faces (which should now be triangles)
    points = triangulated.points
    # Reshape the faces array for easier processing
    faces = triangulated.faces.reshape(-1, 4)  # (Number of triangles, 4), where each row is [3, p0, p1, p2]
    """
    The faces array contains information about the triangles in the triangulated PolyData. 
    PyVista stores faces as a flat array. For example, for two triangles, the array might look like this:
        [3, 0, 1, 2, 3, 1, 2, 3]
    This represents:
        - First triangle with vertices at indices 0, 1, and 2.
        - Second triangle with vertices at indices 1, 2, and 3.
    Each face is preceded by the number '3' indicating it's a triangle (three vertices).

    By reshaping with .reshape(-1, 4):
        - The array is converted into a 2D array with 4 columns, where each row represents one triangle.
        - Example: 
            [[3, 0, 1, 2],  # First triangle
             [3, 1, 2, 3]]  # Second triangle
        - The first column (always '3') indicates it's a triangle, and the next three columns are vertex indices.
    """
    total_area = 0.0
    centroid = np.zeros(3)
    # Iterate over each triangle
    for face in faces:
        p0, p1, p2 = points[face[1]], points[face[2]], points[face[3]]
        # Calculate the area of the triangle
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        # Calculate the centroid of the triangle
        triangle_centroid = (p0 + p1 + p2) / 3
        # Add to the weighted sum
        centroid += area * triangle_centroid
        total_area += area
    # Divide by the total area to get the centroid
    centroid /= total_area

    return centroid



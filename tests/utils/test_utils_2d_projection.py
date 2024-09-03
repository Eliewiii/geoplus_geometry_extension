"""
Unit tests for the functions in utils_2d_projection.py
"""
import numpy as np

from src.geoplus.utils.utils_2d_projection import *

# Sample surfaces
surface_0 = [
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 0.],
    [0., 10., 0.]
]

surface_1 = [
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 10.],
    [0., 10., 10.]
]
surface_2 = [
    [0, 0, 0],
    [10, 0, 0],
    [10, 10, 10],
    [0, 10, 10]
]

surface_3 = [
    [0, 0, 0],
    [10, 0, 0],
    [10, 10, 10],
    [0, 10, 10]
]

z_axis = np.array([0, 0, 1])
x_axis = np.array([1, 0, 0])


def test_get_normal_vector_of_planar_surface():
    """

    :return:
    """
    # surface 0
    normal_0 = np.array(get_normal_vector_of_planar_surface(surface_boundary=surface_0))
    assert np.allclose(normal_0, z_axis)
    # surface 1
    normal_1 = np.array(get_normal_vector_of_planar_surface(surface_boundary=surface_1))
    assert np.allclose(normal_1, np.array(normalize_vector([0., -1., 1.])))

def test_get_planar_surface_plan_vectors_from_normal():
    """

    :return:
    """
    # surface 0
    normal_0= get_normal_vector_of_planar_surface(surface_boundary=surface_0)
    v1_0, v2_0 = get_planar_surface_plan_vectors_from_normal(surface_boundary=surface_0, normal=normal_0)
    assert np.allclose(np.dot(v1_0, v2_0),0.)
    assert np.allclose(np.dot(v1_0, normal_0),0.)
    # surface 1
    normal_1 = get_normal_vector_of_planar_surface(surface_boundary=surface_1)
    v1_1, v2_1 = get_planar_surface_plan_vectors_from_normal(surface_boundary=surface_1, normal=normal_1)
    assert np.allclose(np.dot(v1_1, v2_1),0.)
    assert np.allclose(np.dot(v1_1, normal_1),0.)
    # surface 1 with reference vector x_axis
    v1_1, v2_1 = get_planar_surface_plan_vectors_from_normal(surface_boundary=surface_1, normal=normal_1, reference_vector=x_axis)
    assert np.allclose(v1_1,x_axis)
    # surface 1 with reference vector z_axis
    v1_1, v2_1 = get_planar_surface_plan_vectors_from_normal(surface_boundary=surface_1, normal=normal_1, reference_vector=z_axis)
    assert np.allclose(v1_1, normalize_vector([0,1,1]))


def test_centroid_of_surface_with_hole():
    """

    :return:
    """
    # Example usage:
    surface = [
        [0, 0, 0],
        [10, 0, 0],
        [10, 10, 10],
        [0, 10, 10]
    ]
    hole = [
        [3, 3, 3],
        [6, 3, 3],
        [6, 6, 6],
        [3, 6, 6]
    ]
    # Get the contoured surface with multiple holes
    contoured_surface = contour_surface_with_hole(surface, hole)
    polydata_obj = polydata_from_vertices(contoured_surface)

    area, centroid = compute_area_and_centroid_of_polydata(polydata_obj)
    print("")
    print(area, centroid)

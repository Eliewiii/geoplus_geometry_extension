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
# Convex surface
surface_2 = [
    [6., 10., 0.],
    [6., 6., 0.],
    [4., 6., 0.],
    [4., 10., 0.],
    [0., 10., 0.],
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 0.]
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
    # surface 2 with convex surface that can require adjustments of the normal vector
    normal_2 = np.array(get_normal_vector_of_planar_surface(surface_boundary=surface_2))
    assert np.allclose(normal_2, z_axis)

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



def test_compute_planar_surface_coordinate_in_local_2d_plan():
    """

    :return:
    """
    points_2d = compute_planar_surface_coordinate_in_local_2d_plan(surface_boundary=surface_0)
    assert np.allclose(np.array(points_2d[0]), np.array(surface_0)[0][:2])
    #Preservation of the distance
    assert np.allclose(np.linalg.norm(np.array(points_2d[0])-np.array(points_2d[1])),np.linalg.norm(np.array(surface_0[0])-np.array(surface_0[1])))

def test_compute_planar_surface_boundary_area_and_centroid():
    """

    :return:
    """
    area, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=surface_0)
    assert np.allclose(area, 100.)
    assert np.allclose(centroid, [5., 5., 0.])
    area, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=surface_1)
    assert np.allclose(area, 10.0*np.sqrt(200))
    assert np.allclose(centroid, [5., 5., 5.])


















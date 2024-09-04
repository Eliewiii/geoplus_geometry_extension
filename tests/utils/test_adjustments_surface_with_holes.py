"""
Unit tests for the functions in utils_2d_projection.py
"""
import numpy as np

from src.geoplus.utils.utils_2d_projection import *
from src.geoplus.utils.utils_adjustements_surface_with_holes import contour_surface_with_holes, \
    _contour_surface_with_hole

# Sample surfaces
surface_0 = [
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 0.],
    [0., 10., 0.]
]
hole_0_sur_0 = [
    [4., 4., 0.],
    [6., 4., 0.],
    [6., 6., 0.],
    [4., 6., 0.]
]
hole_1_sur_0 = [
    [7., 7., 0.],
    [8., 7., 0.],
    [8., 8., 0.],
    [7., 8., 0.]
]
hole_2_sur_0 = [
    [3., 3., 0.],
    [2., 3., 0.],
    [2., 2., 0.],
    [3., 2., 0.]
]

surface_1 = [
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 10.],
    [0., 10., 10.]
]

hole_0_sur_1 = [
    [4., 4., 4.],
    [6., 4., 4.],
    [6., 6., 6.],
    [4., 6., 6.]
]
hole_1_sur_1 = [
    [7., 7., 7.],
    [8., 7., 7.],
    [8., 8., 8.],
    [7., 8., 8.]
]

hole_2_sur_1 = [
    [3., 3., 3.],
    [2., 3., 3.],
    [2., 2., 2.],
    [3., 2., 2.]
]

z_axis = np.array([0, 0, 1])
x_axis = np.array([1, 0, 0])


def test_contour_surface_with_hole():
    """

    :return:
    """
    ###########
    # Surface 0
    ###########
    area_0, centroid_0 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=surface_0)
    normal_0 = get_normal_vector_of_planar_surface(surface_boundary=surface_0)
    ## Hole 0
    area_hole_0, centroid_hole_0 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=hole_0_sur_0)
    new_boundary = _contour_surface_with_hole(surface_boundary=surface_0, hole_vertex_list=hole_0_sur_0)
    area, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=new_boundary)
    assert np.allclose(area, area_0 - area_hole_0)
    assert np.allclose(centroid, centroid_0)
    # Check the normal vector
    normal = get_normal_vector_of_planar_surface(surface_boundary=new_boundary)
    assert np.allclose(normal, normal_0)
    ## Hole 1
    area_hole_1, centroid_hole_1 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=hole_1_sur_0)
    new_boundary = _contour_surface_with_hole(surface_boundary=surface_0, hole_vertex_list=hole_1_sur_0)
    area, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=new_boundary)
    print("")
    print(new_boundary)
    assert np.allclose(area, area_0 - area_hole_1)
    assert not np.allclose(centroid, centroid_0)
    # Check the normal vector
    normal = get_normal_vector_of_planar_surface(surface_boundary=new_boundary)
    assert np.allclose(normal, normal_0)

    ###########
    # Surface 1
    ###########
    area_1, centroid_1 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=surface_1)
    normal_1 = get_normal_vector_of_planar_surface(surface_boundary=surface_1)
    ## Hole 0
    area_hole_0, centroid_hole_0 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=hole_0_sur_1)
    new_boundary = _contour_surface_with_hole(surface_boundary=surface_1, hole_vertex_list=hole_0_sur_1)
    area, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=new_boundary)
    assert np.allclose(area, area_1 - area_hole_0)
    assert np.allclose(centroid, centroid_1)
    # Check the normal vector
    normal = get_normal_vector_of_planar_surface(surface_boundary=new_boundary)
    assert np.allclose(normal, normal_1)
    ## Hole 1
    area_hole_1, centroid_hole_1 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=hole_1_sur_1)
    new_boundary = _contour_surface_with_hole(surface_boundary=surface_1, hole_vertex_list=hole_1_sur_1)
    area, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=new_boundary)
    assert np.allclose(area, area_1 - area_hole_1)
    assert not np.allclose(centroid, centroid_1)
    # Check the normal vector
    normal = get_normal_vector_of_planar_surface(surface_boundary=new_boundary)
    assert np.allclose(normal, normal_1)







def test_contour_surface_with_holes():
    """

    :return:
    """
    ###########
    # Surface 0
    ###########
    area_0, centroid_0 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=surface_0)
    normal_0 = get_normal_vector_of_planar_surface(surface_boundary=surface_0)
    # Hole 1
    area_hole_1, centroid_hole_1 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=hole_1_sur_0)
    # Hole 2
    area_hole_2, centroid_hole_2 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=hole_2_sur_0)
    new_boundary = contour_surface_with_holes(surface_boundary=surface_0, hole_list=[hole_1_sur_0,
                                                                                     hole_2_sur_0])
    area, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=new_boundary)
    assert np.allclose(area, area_0 - area_hole_1 - area_hole_2)
    assert np.allclose(centroid, centroid_0)
    # Check the normal vector
    normal = get_normal_vector_of_planar_surface(surface_boundary=new_boundary)
    assert np.allclose(normal, normal_0)

    ###########
    # Surface 1
    ###########
    area_1, centroid_1 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=surface_1)
    normal_1 = get_normal_vector_of_planar_surface(surface_boundary=surface_1)
    # Hole 1
    area_hole_1, centroid_hole_1 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=hole_1_sur_1)
    # Hole 2
    area_hole_2, centroid_hole_2 = compute_planar_surface_boundary_area_and_centroid(surface_boundary=hole_2_sur_1)
    new_boundary = contour_surface_with_holes(surface_boundary=surface_1, hole_list=[hole_1_sur_1,
                                                                                     hole_2_sur_1])
    area, centroid = compute_planar_surface_boundary_area_and_centroid(surface_boundary=new_boundary)
    assert np.allclose(area, area_1 - area_hole_1 - area_hole_2)
    assert np.allclose(centroid, centroid_1)
    # Check the normal vector
    normal = get_normal_vector_of_planar_surface(surface_boundary=new_boundary)
    assert np.allclose(normal, normal_1)



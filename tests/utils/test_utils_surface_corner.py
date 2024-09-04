"""
Unit tests for the functions in utils_2d_projection.py
"""

from src.geoplus.utils.utils_adjustements_surface_with_holes import contour_surface_with_holes
from src.geoplus.utils.utils_surface_corners import compute_planar_surface_corners_from_existing_points, nb_intersection

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
surface_2 = [
    [-1., -1., -1.],
    [10., 0., 0.],
    [11., 11., 11.],
    [0., 10., 10.]
]

surface_3 = [
    [0., 0., 0.],
    [10., 0., 0.],
    [11., 11., 11.],
    [0., 10., 10.]
]


def test_compute_planar_surface_corners_from_existing_points():
    """

    :return:
    """
    ###########
    # Surface 0
    ###########
    corners = compute_planar_surface_corners_from_existing_points(surface_boundary=surface_0)
    assert nb_intersection(corners, corners) == 4  # Check unique corners
    surface_with_holes = contour_surface_with_holes(surface_boundary=surface_0, hole_list=[hole_1_sur_0, hole_2_sur_0])
    corners = compute_planar_surface_corners_from_existing_points(surface_boundary=surface_with_holes)
    assert nb_intersection(corners, corners) == 4  # Check unique corners
    ###########
    # Surface 1
    ###########
    corners = compute_planar_surface_corners_from_existing_points(surface_boundary=surface_1)
    assert nb_intersection(corners, corners) == 4  # Check unique corners
    surface_with_holes = contour_surface_with_holes(surface_boundary=surface_1, hole_list=[hole_1_sur_1, hole_2_sur_1])
    corners = compute_planar_surface_corners_from_existing_points(surface_boundary=surface_with_holes)
    assert nb_intersection(corners, corners) == 4  # Check unique corners
    ###########
    # Surface 2
    ###########
    corners = compute_planar_surface_corners_from_existing_points(surface_boundary=surface_2)
    assert nb_intersection(corners, corners) == 2  # Check unique corners
    ###########
    # Surface 3
    ###########
    corners = compute_planar_surface_corners_from_existing_points(surface_boundary=surface_3)
    assert nb_intersection(corners, corners) == 3  # Check unique corners

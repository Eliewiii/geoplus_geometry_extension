"""

"""
import numpy as np
from pyvista import PolyData

from src.geoplus.planar_surface_3d.planar_surface_visibility_check import \
    are_planar_surface_vertices_facing_each_other, is_ray_intersecting_context

from src.geoplus.planar_surface_3d.planar_surface_numpy_array_addons import \
    numpy_array_surface_to_polydata

surface_0 = np.array([
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 0.],
    [0., 10., 0.]
])
centroid_0 = np.array([5., 5., 0.])

surface_1 = np.array([
    [0., 10., 5.],
    [10., 10., 5.],
    [10., 0., 5.],
    [0., 0., 5.],
])
centroid_1 = np.array([5., 5., 5.])

surface_2 = np.array([
    [0., 0., 10.],
    [10., 0., 10.],
    [10., 10., 10.],
    [0., 10., 10.]
])
centroid_2 = np.array([5., 5., 10.])

z_axis = np.array([0., 0., 1.])


def test_are_planar_surfaces_seeing_each_other():
    assert are_planar_surface_vertices_facing_each_other(vertex_surface_1=centroid_0,
                                                         vertex_surface_2=centroid_1,
                                                         normal_1=np.array([0., 0., 1.]),
                                                         normal_2=-np.array([0., 0., 1.]))

    assert not are_planar_surface_vertices_facing_each_other(vertex_surface_1=centroid_0,
                                                             vertex_surface_2=centroid_1,
                                                             normal_1=np.array([0., 0., 1.]),
                                                             normal_2=np.array([0., 0., 1.]))


def test_is_ray_intersecting_context():

    mesh = numpy_array_surface_to_polydata(surface_1) + numpy_array_surface_to_polydata(surface_2)

    print("")
    print(is_ray_intersecting_context(start_point=centroid_0, end_point=centroid_2,
                                      context_polydata_mesh=mesh))



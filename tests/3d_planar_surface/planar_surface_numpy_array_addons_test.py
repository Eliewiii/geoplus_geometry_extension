"""

"""

import numpy as np

from src.geoplus.planar_surface_3d.planar_surface_numpy_array_addons import numpy_array_surface_to_polydata

surface_0 = np.array([
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 0.],
    [0., 10., 0.]
])

surface_1 = [
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 10.],
    [0., 10., 10.]
]

def test_numpy_array_surface_to_polydata():
    """
    Test numpy_array_surface_to_polydata
    :return:
    """
    polydata_0 = numpy_array_surface_to_polydata(surface_0)
    polydata_1 = numpy_array_surface_to_polydata(surface_1)
    mesh = polydata_0 + polydata_1
    print(polydata_0)
    print (mesh)

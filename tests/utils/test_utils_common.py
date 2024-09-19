"""

"""

import numpy as np

from src.geoplus.utils.utils_common_methods import remove_redundant_vertices

surface_0 = np.array([
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 0.],
    [0., 10., 0.]
])

surface_1 = np.array([
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 10.],
    [0., 10., 10.]
])


def test_numpy_array_surface_to_polydata():
    """
    Test numpy_array_surface_to_polydata
    :return:
    """
    surface = remove_redundant_vertices(surface_0)
    print(surface)

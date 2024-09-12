"""

"""

import numpy as np

from typing import List, Tuple



def are_planar_surface_vertices_facing_each_other(vertex_surface_1: List[float], vertex_surface_2: List[float],
                                         normal_1: List[float], normal_2: List[float]):
    """
    Check if two planar surfaces are facing each other. It does not consider obstacles between the two surfaces.
    This method can be run for multiple couples of vertices of the two surfaces for better accuracy.
    :param vertex_surface_1: A vertex of the first surface
    :param vertex_surface_2: A vertex of the second surface
    :param normal_1: Normal vector of the first surface
    :param normal_2: Normal vector of the second surface
    :return: True if the two surfaces are seeing each other, False otherwise.
    """
    vector_1_2 = np.array(vertex_surface_2) - np.array(vertex_surface_1)
    dot_product_1 = np.dot(normal_1, vector_1_2)
    dot_product_2 = np.dot(normal_2, vector_1_2)
    # visibility/facing criteria  (same as PyviewFactor)
    if dot_product_1 > 0 > dot_product_2:
        return True
    else:
        return False

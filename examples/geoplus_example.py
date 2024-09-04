"""
Example of using the geoplus package.
In this example, we will use the geoplus package with surfaces defined by a list of vertices.
The following features will be used:
    - contouring a surface around its holes
    - computing the area and centroid of a surface (with holes)
    - computing the corners of a surface
"""

from geoplus import compute_planar_surface_area_and_centroid, contour_planar_surface_with_holes, compute_planar_surface_corners_from_existing_points

# horizontal surface
surface_1 = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
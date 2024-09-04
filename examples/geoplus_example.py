"""
Example of using the geoplus package.
In this example, we will use the geoplus package with surfaces defined by a list of vertices.
The following features will be used:
    - contouring a surface around its holes
    - computing the area and centroid of a surface (with holes)
    - computing the corners of a surface
"""

from geoplus import compute_planar_surface_area_and_centroid, contour_planar_surface_with_holes, \
    compute_planar_surface_corners_from_existing_points

# horizontal surface
surface_0 = [
    [0., 0., 0.],
    [10., 0., 0.],
    [10., 10., 0.],
    [0., 10., 0.]
]
# hole in the surface
hole_0 = [
    [2., 2., 0.],
    [2., 8., 0.],
    [8., 8., 0.],
    [8., 2., 0.]
]

# compute the area and centroid of the surface
area_s0, centroid_s0 = compute_planar_surface_area_and_centroid(vertex_list=surface_0)
print(f"Area of surface 0: {area_s0}")
print(f"Centroid of surface 0: {centroid_s0}")

# compute the area and centroid of the hole
area_h0, centroid_h0 = compute_planar_surface_area_and_centroid(vertex_list=hole_0)
print(f"Area of hole 0: {area_h0}")
print(f"Centroid of hole 0: {centroid_h0}")

# contour the surface around the hole
contoured_surface = contour_planar_surface_with_holes(vertex_list=surface_0, hole_list=[hole_0])
print(f"Contoured surface: {contoured_surface}")
area_s0_contoured, centroid_s0_contoured = compute_planar_surface_area_and_centroid(vertex_list=contoured_surface)
print(
    f"Area of contoured surface: {area_s0_contoured}, should be equal to the area of the surface minus the area of the hole:"
    f"area_s0 - area_h0 = {area_s0 - area_h0}, it is {area_s0 - area_h0 == area_s0_contoured}")
print(f"Centroid of contoured surface: {centroid_s0_contoured}")

# compute the corners of the surface
corners_s0_contoured = compute_planar_surface_corners_from_existing_points(vertex_list=contoured_surface)
print(f"Corners of surface 0: {corners_s0_contoured}")

print("")
print("You can try with other surfaces and holes, or use the numpy array version of the functions. Other samples "
      "are available at the end of the code")

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

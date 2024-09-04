# GeoPlus

**GeoPlus** is a Python package that provide additional geometry manipulation functions. The included features are
especially useful for surface treatment for certain software, in particular Radiance, which motivated the creation
of this package. The package is still under construction, and new features will be added in the future.


## Features

**Implemented**:

For now, the package focused only on **3D planar** surfaces, and provide the following features:
- Contouring of 3d planar surfaces with holes, for applications such as Radiance, that does not allow holes in
  the geometry;
- Computation of area and centroid of 3d planar surfaces, especially for surfaces with holes, usually
  not considered by libraries such as Pyvista or Shapely (only available in 2D);
- Computation of corners of surfaces.
The package is still under construction. Most of the key features are already implemented.

The package includes as well useful functions for the manipulation of 3D planar surfaces, such as the computation of 
rotation matrices for transformation to local coordinates, true normal vectors.

The package only support lists/tabular and numpy types inputs. 

**To be implemented**:

- Support of Shapely and Pyvista geometries;
- Other geometry manipulation functions when the need arises.

## Pre-requisites

The package is built on top of the following libraries:
- Numpy
- Shapely
- Pyvista

## Installation

You can install the package directly from GitHub using `pip`:

```bash
pip install https://github.com/Eliewiii/geoplus_geometry_extension/archive/refs/tags/**last_tag**.tar.gz
```

## Usage

Examples of usage are available in the `examples` folder. For more detailed usage, check the documentation 
when there be one, though the use is straightforward.

Do not hesitate look at details of the functions in the source code. The functionalities provided by the package
can look basic at first glance, but are more subtle than they seem, justifying the creation of this package. 

The functions developed in this package have some limitation, especially for exotic and some convex geometries, 
keep it in mind, but the functions are developed to be robust for most common cases. Do not hesitate to report any
issue or suggestion to improve the package.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions, feel free to reach out:

* Author: Elie MEIDONI
* Email: elie.medioniwiii@gmail.com

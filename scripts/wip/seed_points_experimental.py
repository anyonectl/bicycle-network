# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: bicycle-network:default (3.14.8)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Create hex grid of seed points
#
# This notebook contains some initial experiments for creating a grid of seed points in the pattern of regular hexagons, over some underlying shape (here, it's a rectangular polygon called `city`, later we will use actual city shapes). 
#
# The solution created builds the hexagonal grid of seed points from the bottomleft corner of a rectangle (its (0,0) point), but it would make more sense to build it from the centre and out (from the underlying shape's centroid).
#
# To do this, I need to read more from [Red Blob Games](https://www.redblobgames.com/grids/hexagons/), which already inspired some of the ideas and code in this notebook.

# %%
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import shapely
from shapely import Polygon, LineString, LinearRing, Point, MultiPoint, GeometryCollection

# %% [markdown]
# ## Seed points
#
# Start with a rectangle shape for a city (we'll switch to a real city and street network later), then create a grid of seed points over that shape.

# %%
import shapely
from shapely import Polygon, LineString, LinearRing, Point

# %%
coords = [(0., 0.), (0., 100.), (100., 100.), (100., 0.)]

# %%
city = Polygon(coords)
city


# %% [markdown]
# ### Hexagons
#
# The code in this section is inspired by ideas and code from Red Blob Games:
# - https://www.redblobgames.com/grids/hexagons/
# - https://www.redblobgames.com/grids/hexagons/implementation.html
# - https://www.redblobgames.com/grids/parts/
#
# ![Hex Sizes](../../red-blob-games/hex_size.png "Hex Sizes")
#
# ![Hex Angles](../../red-blob-games/hex_angles.png "Hex Angles")
#
# ![Cube Coords](../../red-blob-games/cube_coords.png "Cube Coords")
#
# ![Axial Coords](../../red-blob-games/axial_coords.png "Axial Coords")

# %%
def hex_corner(centre: shapely.Point, size: float, i: int, pointy: bool) -> shapely.Point:
    """Produces the ith corner of a regular hexagon as a Shapely point.
    
    Args:
        centre: The centre of the hexagon.
        size: The distance from the hexagon's centre to one of its corners.
        i: Which corner (out of 6).
        pointy: Whether the hexagon should be pointy-top or flat-top.
    
    Returns:
        A shapely.Point representing the ith corner of a hexagon.
    """
    if pointy:
        angle_deg = 60 * (i+1) - 30
    else:
        angle_deg = 60 * i
    angle_rad = np.pi / 180 * angle_deg
    print(f"Point {i}: {angle_deg} degrees")
    return Point(centre.x + size * np.cos(angle_rad),
                 centre.y + size * np.sin(angle_rad))

def hex_corner_flat(centre: shapely.Point, size: float, i: int) -> shapely.Point:
    """Produces the ith corner of a flat-top regular hexagon as a Shapely point.
    
    Args:
        centre: The centre of the hexagon.
        size: The distance from the hexagon's centre to one of its corners.
        i: Which corner (out of 6).
    
    Returns:
        A shapely.Point representing the ith corner of a hexagon.
    """
    hex_corner(centre, size, i, False)

def hex_corner_pointy(centre: shapely.Point, size: float, i: int) -> shapely.Point:
    """Produces the ith corner of a pointy-top regular hexagon as a Shapely point.
    
    Args:
        centre: The centre of the hexagon.
        size: The distance from the hexagon's centre to one of its corners.
        i: Which corner (out of 6).
    
    Returns:
        A shapely.Point.
    """
    hex_corner(centre, size, i, True)


# %%
def get_hex_vertices(centre: shapely.Point, size: float, pointy: bool) -> List[shapely.Point]:
    """Returns a list of hexagon corners as Shapely points.
    
    Args:
        centre: The centre of the hexagon.
        size: The distance from the hexagon's centre to one of its corners.
        pointy: Whether the hexagon should be pointy-top or flat-top.
    
    Returns:
        A list of Shapely points.
    """
    coords = []
    for i in range(0,6):
        coords.append(hex_corner(centre, size, i, pointy))
    return coords


# %%
# TODO: degrees are increasing counterclockwise,
#    I would rather have them increase clockwise

def create_hexagon(centre: shapely.Point, size: float, pointy: bool, polygon: bool) -> shapely.Polygon:
    """Creates a regular hexagon as a Shapely polygon or curve.
    
    Args:
        centre: The centre of the hexagon.
        size: The distance from the hexagon's centre to one of its corners.
        pointy: Whether the hexagon should be pointy-top or flat-top.
        as_polygon: Whether the hexagon should be a Shapely Polygon or LinearRing.
    
    Returns:
        A regular hexagon.
    """
    coords = get_hex_vertices(centre, size, pointy)
    if polygon:
        return Polygon(coords)
    else:
        return LinearRing(coords)


# %%
centre_ex = Point(50., 50.)
size_ex = 5

my_hex_poly = create_hexagon(centre_ex, size_ex, True, True)
my_hex_poly

# %%
my_hex_line = create_hexagon(centre_ex, size_ex, False, False)
my_hex_line

# %%
len(my_hex_line.coords)

# %%
# Next step... create grid of hexagon vertices over the 'city' shape
# See https://www.redblobgames.com/grids/hexagons/ and https://www.redblobgames.com/grids/hexagons/implementation.html


# %% [markdown]
# In a rectangle from (0,0) to (X, Y):
#
# while x<X and y<Y
#
# For pointy
# Set height = 2 * size
# Set width = sqrt(3) * size
#
# Y-AXIS:
#
# 1st col (x=0), and every odd col:
# 1. Set 1st vertex at y=0.25\*height
# 2. Set 2nd vertex at y=0.75\*height (increase by 0.5\*height)
# 3. Set 3rd vertex at y=1.75\height (increase by 1\*height)
# So: Start at y=0.25\*height; when i is even, set vertex at y = y + 0.5\*height; when i is odd, set vertex at y = y + 1\*height
#
# 2nd col (x=0.5\*width), and every even col:
# 1. Set 1st vertex at y=0*height
# 2. Set 2nd vertex at y=1\*height (increase by 1\*height)
# 3. Set 3rd vertex at y=1.5\height (increase by 0.5\*height)
# So: Start at y=0\*height; when i is even, set vertex at y = y + 1\*height; when i is odd, set vertex at y = y + 0.5\*height
#
# X-AXIS:
#
# 1st row (y=0):
# 1. Set 1st vertex at x=0.5\*width
# 2. Set 2nd vertex at x=1.5\*width (increase by 1\*width)
# 3. Set 3rd vertex at x=2.5\width (increase by 1\*width)
# So: Start at x=0.5\*width; set succeeding vertices at x = x + 1\*width (increase x by 1\*width every time)
#
# 2nd row (y=0.25\*height):
# 1. Set 1st vertex at x=0\*width
# 2. Set 2nd vertex at x=1\*width (increase by 1\*width)
# 3. Set 3rd vertex at x=2\width (increase by 1\*width)
# So: Start at x=0\*width; set succeeding vertices at x = x + 1\*width (increase x by 1\*width every time)
#
# 3rd row: same as 2nd row
# 4th row: same as 1st row
# 5th row: same as 1st row
# 6th row: same as 2nd+3rd row
# 7th row: same as 2nd+3rd row
#

# %%
# Try to make just one column first
from shapely import MultiPoint

def create_hex_grid(hex_size: float, Y: float):
    seed_points = []
    hex_height = 2 * hex_size
    # hex_width = math.sqrt(3) * hex_size
    x: float = 0.0
    row: int = 0
    y: float = 0.25 * hex_height
    while y < Y:
        point = (x, y)
        seed_points.append(point)
        if row % 2 == 0:
            y += 0.5 * hex_height
        else:
            y += 1 * hex_height
        row += 1
    return seed_points

seed_points = MultiPoint(create_hex_grid(15.0, 100.0))

for p in list(seed_points.geoms):
    print(p)

seed_points


# %%
# Try to make all even columns first
def create_hex_grid_even(hex_size: float, X: float, Y: float):
    seed_points = []
    hex_height = 2 * hex_size
    hex_width = math.sqrt(3) * hex_size
    col: int = 0
    x: float = 0.0 * hex_width
    while x < X:
        row: int = 0
        y: float = 0.25 * hex_height
        while y < Y:
            point = (x, y)
            seed_points.append(point)
            if row % 2 == 0:
                y += 0.5 * hex_height
            else:
                y += 1 * hex_height
            row += 1
        x += 1 * hex_width
        col += 2
    return seed_points

# Then try to make all odd columns
def create_hex_grid_odd(hex_size: float, X: float, Y: float):
    seed_points = []
    
    hex_height = 2 * hex_size
    hex_width = math.sqrt(3) * hex_size
    
    col: int = 1
    x: float = 0.5 * hex_width
    while x < X:
        row: int = 0
        y: float = 0.0 * hex_height
        while y < Y:
            point = (x, y)
            seed_points.append(point)
            if row % 2 == 0:
                y += 1.0 * hex_height
            else:
                y += 0.5 * hex_height
            row += 1
        
        x += 1 * hex_width
        col += 2
    return seed_points

# Combine them
def create_hex_grid(hex_size: float, X: float, Y: float):
    points_even = create_hex_grid_even(hex_size, X, Y)
    points_odd = create_hex_grid_odd(hex_size, X, Y)
    return MultiPoint(points_even + points_odd)


# %%
seed_points = create_hex_grid(18, 100, 100)
seed_points


# %%
# Try to make both odd and even cols
# TODO: This is incorrect
def create_hex_grid_incorrect(hex_size: float, X: float, Y: float):
    seed_points = []
    hex_height = 2 * hex_size
    hex_width = math.sqrt(3) * hex_size
    col: int = 0
    x: float = 0.0 * hex_width
    while x < X:
        row: int = 0
        if col % 2 == 0:
            y: float = 0.25 * hex_height
            while y < Y:
                point = (x, y)
                seed_points.append(point)
                if row % 2 == 0:
                    y += 0.5 * hex_height
                else:
                    y += 1 * hex_height
                row += 1
        else:
            y: float = 0.0 * hex_height
            while y < Y:
                point = (x, y)
                seed_points.append(point)
                if row % 2 == 0:
                    y += 1 * hex_height
                else:
                    y += 0.5 * hex_height
                row += 1
        x += 1 * hex_width
        col += 1
    return seed_points

points = create_hex_grid_incorrect(18.0, 100.0, 100.0)

for p in points:
    print(p)

seed_points_incorrect = MultiPoint(points)
seed_points_incorrect

# %%
seed_points.bounds

# %%
seed_points.buffer(15)

# %%
print(seed_points.centroid)

# %%
seed_points.envelope

# %%
list(seed_points.geoms)

# %%
city.contains(seed_points)

# %%
gc = GeometryCollection([seed_points, city])
gc

# %% [markdown]
# The above solution builds the hexagonal grid of seed points from the bottomleft corner of a rectangle (its (0,0) point), but it would make more sense to build it from the centre and out (from the underlying shape's centroid). To do this, I need to read more from [Red Blob Games](https://www.redblobgames.com/grids/hexagons/).

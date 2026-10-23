# TODO: Use spatial indexing to find neighbours instead? Sth with trees?

from typing import Dict, List, Tuple
import geohexgrid as ghg
import numpy as np
from shapely import Point


def numeric_id(s: str) -> Tuple[int, int]:
    x, y = s.split(",")
    return int(x), int(y)


def str_id(t: Tuple[int, int]) -> str:
    return f"{t[0]},{t[1]}"


def axial_to_double(a: float, b: float) -> Tuple[int, int]:
    u, v = ghg.axial_to_double(a, b)
    return int(u), int(v)


def make_test_grid(hex_radius: int):
    return ghg.make_grid_from_bounds(-50.0, -50.0, 50.0, 50.0, hex_radius)


# TODO: should only add neighbours to vns if they are in the dataset, i.e. within the bounding box
def vns(node: Tuple[int, int], hex_radius: int) -> List[Tuple[int, int]]:
    q, r = double_to_axial(node, hex_radius)
    vns = []
    for i in range(6):
        a, b = hex_neighbour((q, r), i)
        u, v = axial_to_double(a, b)
        vns.append(str_id((u, v)))
    return vns


def compute_angle(previous: Point, current: Point, next: Point) -> float:
    """
    Only works with cartesian coordinates right now
    """
    # Convert edges to spatial/euclidean vectors
    prev_spatial_v = np.array([previous.x - current.x, previous.y - current.y])
    next_spatial_v = np.array([next.x - current.x, next.y - current.y])

    # Convert to unit vectors (only keep direction)
    prev_unit_v = prev_spatial_v / np.linalg.norm(prev_spatial_v)
    next_unit_v = next_spatial_v / np.linalg.norm(next_spatial_v)

    # Calculate angle
    angle_rad = np.arccos(np.clip(np.dot(prev_unit_v, next_unit_v), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return round(float(angle_deg), 1)


def _direction_bias(angle: float) -> float:
    if angle > 150.0:  # straight line
        return 1.0
    if angle > 90.0:  # slight turn
        return 0.9
    return 0.75  # sharp turn (angle <= 90.0)


def direction_bias(previous: Point, current: Point, next: Point) -> float:
    if previous == current:
        return 1.0  # edge case when this is the first edge being added to a path
    angle = compute_angle(previous, current, next)
    return _direction_bias(angle)


def get_bearing(origin: str, destination: str) -> str:
    """
    Only works with regular grid of flat-top hexagons with double coordinates.
    """
    origin_x, origin_y = numeric_id(origin)

    N = origin_x, origin_y + 2
    S = origin_x, origin_y - 2
    NW = origin_x - 1, origin_y + 1
    SW = origin_x - 1, origin_y - 1
    NE = origin_x + 1, origin_y + 1
    SE = origin_x + 1, origin_y - 1

    bearings = {N: "N", S: "S", NW: "NW", SW: "SW", NE: "NE", SE: "SE"}

    bearing = bearings.get(numeric_id(destination), "Error")
    if bearing == "Error":
        error_msg = f"Could not get bearing from {origin} to {destination}."
        raise ValueError(error_msg)
    return bearing


def vns_list_to_bearing_dict(origin: str, vns) -> Dict[str, str]:
    neighbours = {}
    for neighbour in vns:
        bearing = get_bearing(origin, neighbour)
        if bearing in neighbours:
            error_msg = f"Tried to set {bearing}={neighbour}, but {bearing}={neighbours[bearing]} already exists."
            raise KeyError(f"Origin {origin}: {error_msg}")
        neighbours[bearing] = neighbour
    return neighbours


# Inspired by Red Blob Games:
hex_directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def hex_direction(direction: int) -> Tuple[int, int]:
    return hex_directions[direction]


def hex_add(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def hex_neighbour(hex: Tuple[int, int], direction: int) -> Tuple[int, int]:
    return hex_add(hex, hex_direction(direction))


def axial_to_cube(q: int, r: int) -> Tuple[int, int, int]:
    s = -q - r
    return (q, r, s)


def cube_to_axial(q: int, r: int, s: int) -> Tuple[int, int]:
    assert not (round(q + r + s) != 0), "q + r + s must be 0"
    return (q, r)


def double_to_axial(node: Tuple[int, int], hex_radius: int) -> Tuple[int, int]:
    u, v = node
    x, y = ghg.double_to_cartesian(u, v, hex_radius)
    return ghg.cartesian_to_axial(x, y, hex_radius)

from typing import List, Tuple
import geohexgrid as ghg

def numeric_id(s: str) -> Tuple[int, int]:
    x, y = s.split(",")
    return int(x), int(y)

def str_id(t: Tuple[int, int]) -> str:
    return f"{t[0]},{t[1]}"

def axial_to_double(a: float, b: float) -> Tuple[int, int]:
    u, v = ghg.axial_to_double(a, b)
    return int(u), int(v)

def make_test_grid(hex_radius: int):
    return ghg.make_grid_from_bounds(0.0, 0.0, 100.0, 100.0, hex_radius)

# TODO: should only add neighbours to vns if they are in the dataset, i.e. within the bounding box
def vns(node: Tuple[int, int], hex_radius: int) -> List[Tuple[int, int]]:
    q, r = double_to_axial(node, hex_radius)
    vns = []
    for i in range(6):
        a, b = hex_neighbour((q, r), i)
        u, v = axial_to_double(a, b)
        vns.append(str_id((u, v)))
    return vns


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
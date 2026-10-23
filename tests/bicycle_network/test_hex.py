import bicycle_network.hex
from shapely import Point


def test_compute_angle_south_to_north():
    point = Point(0, -3.187e-15)  # (0,0)
    south = Point(0, -6.928)  # (0,-2)
    north = Point(0, 6.928)  # (0,2)
    result = bicycle_network.hex.compute_angle(south, point, north)
    expected = 180.0
    assert result == expected


def test_compute_angle_north_to_south():
    point = Point(0, -3.187e-15)  # (0,0)
    south = Point(0, -6.928)  # (0,-2)
    north = Point(0, 6.928)  # (0,2)
    result = bicycle_network.hex.compute_angle(north, point, south)
    expected = 180.0
    assert result == expected


def test_compute_angle_south_to_northwest():
    point = Point(0, -3.187e-15)  # (0,0)
    south = Point(0, -6.928)  # (0,-2)
    northwest = Point(-6, 3.464)  # (-1,1)
    result = bicycle_network.hex.compute_angle(south, point, northwest)
    expected = 120.0
    assert result == expected


def test_compute_angle_south_to_northeast():
    point = Point(0, -3.187e-15)  # (0,0)
    south = Point(0, -6.928)  # (0,-2)
    northeast = Point(6, 3.464)  # (1,1)
    result = bicycle_network.hex.compute_angle(south, point, northeast)
    expected = 120.0
    assert result == expected


def test_compute_angle_south_to_southwest():
    point = Point(0, -3.187e-15)  # (0,0)
    south = Point(0, -6.928)  # (0,-2)
    southwest = Point(-6, -3.464)  # (-1,-1)
    result = bicycle_network.hex.compute_angle(south, point, southwest)
    expected = 60.0
    assert result == expected


def test_compute_angle_south_to_southeast():
    point = Point(0, -3.187e-15)  # (0,0)
    south = Point(0, -6.928)  # (0,-2)
    southeast = Point(6, -3.464)  # (1,-1)
    result = bicycle_network.hex.compute_angle(south, point, southeast)
    expected = 60.0
    assert result == expected


def test_compute_angle_same_edge():
    point = Point(0, -3.187e-15)  # (0,0)
    south = Point(0, -6.928)  # (0,-2)
    result = bicycle_network.hex.compute_angle(south, point, south)
    expected = 0.0
    assert result == expected

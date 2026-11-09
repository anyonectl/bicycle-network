import numpy as np
import pytest
from pathlib import Path
import json
import bicycle_network.genetic_algorithm as ga


@pytest.fixture
def start_pop():
    path = (
        Path(__file__).resolve().parents[2]
        / "results/network-data/start_pop_abs_connected.json"
    )
    with path.open("r") as fp:
        data = json.load(fp)
    return ga.json_to_pop(data)


@pytest.fixture
# not connected
def medium_pop():
    pop = {
        0: {
            "demand_covered": 10.5,
            "paths": {
                0: ["-6,14", "-5,15", "-4,14"],
                1: ["-4,10", "-3,9", "-2,8"],
            },
        },
        1: {
            "demand_covered": 20.9,
            "paths": {
                0: ["4,-2", "4,-4", "5,-5", "6,-6"],
                1: ["2,10", "1,9", "0,10", "0,8", "-1,7"],
            },
        },
        2: {
            "demand_covered": 40.0,
            "paths": {
                0: ["8,-6", "8,-4", "8,-2"],
                1: ["2,-12", "1,-13", "0,-12", "0,-10"],
            },
        },
        3: {
            "demand_covered": 25.2,
            "paths": {
                0: ["-5,11", "-5,9", "-5,7"],
                1: ["2,10", "3,9", "3,7"],
            },
        },
        4: {
            "demand_covered": 7.5,
            "paths": {
                0: ["4,-2", "3,-3", "3,-1"],
                1: ["2,0", "3,1", "4,2", "5,3"],
            },
        },
        5: {
            "demand_covered": 19.7,
            "paths": {
                0: ["8,10", "7,11", "6,12"],
                1: ["1,3", "0,2", "1,1"],
            },
        },
        6: {
            "demand_covered": 32.0,
            "paths": {
                0: ["2,14", "3,13", "3,11"],
                1: ["2,6", "1,7", "0,6"],
            },
        },
        7: {
            "demand_covered": 15.0,
            "paths": {
                0: ["-1,11", "-2,10", "-2,12"],
                1: ["0,0", "-1,-1", "-1,1"],
            },
        },
        8: {
            "demand_covered": 1.1,
            "paths": {
                0: ["-6,14", "-6,15"],
                1: ["5,13", "6,15"],
            },
        },
        9: {
            "demand_covered": 29.9,
            "paths": {
                0: ["-6,0", "-5,-1", "-6,-2", "-6,-4", "-6,-6", "-6,-8"],
                1: ["-2,-10", "-2,-12"],
            },
        },
    }
    return pop


@pytest.fixture
# not connected
def small_pop():
    pop = {
        0: {
            "demand_covered": 3.0,
            "paths": {
                0: ["-6,14", "-5,15", "-4,14"],
                1: ["-4,10", "-3,9", "-2,8"],
            },
        },
        1: {
            "demand_covered": 2.0,
            "paths": {
                0: ["4,-2", "4,-4", "5,-5", "6,-6"],
                1: ["2,10", "1,9", "0,10", "0,8", "-1,7"],
            },
        },
        2: {
            "demand_covered": 5.0,
            "paths": {
                0: ["8,-6", "8,-4", "8,-2"],
                1: ["2,-12", "1,-13", "0,-12", "0,-10"],
            },
        },
    }
    return pop


def test_start_pop_returns_dict_structure(start_pop):
    assert isinstance(start_pop, dict)
    assert isinstance(start_pop.get(0), dict)
    assert isinstance(start_pop[0]["demand_covered"], float)
    assert isinstance(start_pop[0]["paths"], dict)
    assert isinstance(start_pop[0]["paths"][0], list)
    assert isinstance(start_pop[0]["paths"][0][0], str)


def test_selection_has_length_n(start_pop):
    n = len(start_pop)
    mating_pool = ga._selection(n, start_pop)
    assert len(mating_pool) == n


def test_selection_has_length_n_small_pop(small_pop):
    n = len(small_pop)
    mating_pool = ga._selection(n, small_pop)
    assert len(mating_pool) == n


def test_selection_returns_dict_structure(start_pop):
    n = len(start_pop)
    mating_pool = ga._selection(n, start_pop)
    assert isinstance(mating_pool, dict)
    assert isinstance(mating_pool.get(0), dict)
    assert isinstance(mating_pool[0]["demand_covered"], float)
    assert isinstance(mating_pool[0]["paths"], dict)
    assert isinstance(mating_pool[0]["paths"][0], list)
    assert isinstance(mating_pool[0]["paths"][0][0], str)


def test_selection_dict_has_id_keys(start_pop):
    n = len(start_pop)
    mating_pool = ga._selection(n, start_pop)
    expected_ids = [i for i in range(n)]
    actual_ids = [id for id in mating_pool]
    assert actual_ids == expected_ids


def test_selection_dict_has_id_keys_small_pop(small_pop):
    n = len(small_pop)
    mating_pool = ga._selection(n, small_pop)
    expected_ids = [i for i in range(n)]
    actual_ids = [id for id in mating_pool]
    assert actual_ids == expected_ids


def test_selection_returns_different_pop(start_pop):
    n = len(start_pop)
    mating_pool = ga._selection(n, start_pop)
    assert mating_pool != start_pop


def test_sum_demand_coverage_returns_correct_sum(small_pop):
    sum = ga._sum_demand_coverage(small_pop)
    assert sum == 10.0


def test_get_probabilities_returns_float_array(small_pop):
    expected = np.array([0.3, 0.2, 0.5])
    actual = ga._get_probabilities(small_pop)
    np.testing.assert_allclose(actual, expected)

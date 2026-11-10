"""Tests of the transformer grouping.

The made-up forests are small trees with loads at given nodes. Every served load must land in exactly one group, no group may
hold more loads than the limit, a group must close before a load goes beyond the radius, and the size must round up to the
next standard size.
"""

from pathlib import Path

import pytest

from dxfeeder import inputs, transformers

PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = inputs.read_settings(PROJECT)


def made_up_forest(parent, length, attached):
    """Return a forest of one tree rooted at node 0 from parent and edge length maps and the attached loads."""
    children = {}
    for node in sorted(parent):
        children.setdefault(parent[node], []).append(node)
    return {"parent": parent, "children": children, "roots": [0], "root_station": {0: "s"}, "length": length,
            "attached": attached}


def loads_for(ids):
    """Return made-up loads of 5 kW at a power factor of 1."""
    return [{"id": load_id, "kw": 5.0, "pf": 1.0} for load_id in ids]


@pytest.mark.parametrize("kva, expected", [(1.0, (10.0, True)), (25.0, (25.0, True)), (25.1, (37.5, True)),
                                           (900.0, (500.0, False))])
def test_standard_size(kva, expected):
    """The size rounds up to the next standard size, and a load above the largest size is flagged."""
    assert transformers.standard_size(kva, SETTINGS["transformer_sizes_kva"]) == expected


def test_group_limit():
    """Thirty loads at one leaf give groups of at most the limit, and every load lands in one group."""
    ids = [f"l{number:02d}" for number in range(30)]
    forest = made_up_forest({1: 0}, {1: 10.0}, {load_id: (1, 5.0) for load_id in ids})
    groups = transformers.group_loads(forest, loads_for(ids), SETTINGS)
    limit = SETTINGS["loads_per_transformer_max"]
    assert all(len(group["loads"]) <= limit for group in groups)
    assert sorted(load_id for group in groups for load_id, _ in group["loads"]) == ids


def test_group_radius():
    """A load that would pass the radius on the next step closes its group where it stands."""
    radius = SETTINGS["transformer_radius_m"]
    forest = made_up_forest({1: 0, 2: 1}, {1: radius, 2: 10.0}, {"far": (2, 0.0), "near": (1, 0.0)})
    groups = transformers.group_loads(forest, loads_for(["far", "near"]), SETTINGS)
    assert [(group["node"], [load_id for load_id, _ in group["loads"]]) for group in groups] == [(1, ["far", "near"])]
    assert all(reach <= radius for group in groups for _, reach in group["loads"])


def test_group_at_root():
    """Loads that never pass the radius close at the substation root, and the reach adds the drop and the tree length."""
    forest = made_up_forest({1: 0}, {1: 20.0}, {"a": (1, 5.0), "b": (0, 3.0)})
    groups = transformers.group_loads(forest, loads_for(["a", "b"]), SETTINGS)
    assert len(groups) == 1 and groups[0]["node"] == 0
    assert groups[0]["loads"] == [("a", 25.0), ("b", 3.0)]


def test_group_size():
    """The size is the apparent load over the loading factor, rounded up."""
    forest = made_up_forest({1: 0}, {1: 20.0}, {"a": (1, 5.0), "b": (1, 5.0)})
    groups = transformers.group_loads(forest, loads_for(["a", "b"]), SETTINGS)
    expected = transformers.standard_size(10.0 / SETTINGS["transformer_loading_factor"], SETTINGS["transformer_sizes_kva"])[0]
    assert groups[0]["kva"] == expected and groups[0]["fits"]


def test_group_deterministic():
    """The same forest gives the same groups twice."""
    ids = [f"l{number:02d}" for number in range(15)]
    forest = made_up_forest({1: 0, 2: 1, 3: 1}, {1: 40.0, 2: 30.0, 3: 30.0},
                            {load_id: (2 + number % 2, float(number)) for number, load_id in enumerate(ids)})
    first = transformers.group_loads(forest, loads_for(ids), SETTINGS)
    assert transformers.group_loads(forest, loads_for(ids), SETTINGS) == first

"""Tests of the radial feeders.

The made-up graphs are small: a path, a square with two sources, and the synthetic site of synthetic_site.py.
The feeder forest must give every node one parent at most, give each node to the nearest source by weight, follow forced routes,
keep the part with no substation out of the feeders while reporting its loads, and merge plain chains into single lines.
"""

from pathlib import Path

import networkx
import numpy
import pytest

import synthetic_site as site
from dxfeeder import buildings, inputs, network, routes, streets

PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = inputs.read_settings(PROJECT)


def made_up_graph(edges, places):
    """Return a graph with nodes at metric places and edges given as (first, second, length, weight)."""
    graph = networkx.Graph()
    for node, (x, y) in places.items():
        graph.add_node(node, x=float(x), y=float(y), lon=float(x), lat=float(y), street=True)
    for first, second, length, weight in edges:
        graph.add_edge(first, second, length=length, weight=weight, forced=weight < length)
    return graph


def test_nearest_nodes_tie():
    """A point at equal distance from two nodes takes the smaller node id."""
    graph = made_up_graph([(5, 3, 10.0, 10.0)], {5: (0, 0), 3: (10, 0)})
    assert network.nearest_nodes(graph, numpy.array([5.0, 1.0]), numpy.array([0.0, 0.0])) == [(3, 5.0), (5, 1.0)]


def test_forest_nearest_source():
    """On a path with a source at each end, each node goes to the source nearer by weight, ties to the smaller node."""
    places = {node: (node * 10, 0) for node in range(1, 6)}
    graph = made_up_graph([(node, node + 1, 10.0, 10.0) for node in range(1, 5)], places)
    parent, distance, owner = network.shortest_path_forest(graph, [1, 5])
    assert owner == {1: 1, 2: 1, 3: 1, 4: 5, 5: 5}
    assert parent == {2: 1, 3: 2, 4: 5} and distance[3] == 20.0


def test_forest_prefers_forced_route():
    """A forced edge with a low weight wins over a shorter street path."""
    places = {1: (0, 0), 2: (10, 0), 3: (20, 0), 4: (10, 30)}
    graph = made_up_graph([(1, 2, 10.0, 10.0), (2, 3, 10.0, 10.0), (1, 4, 32.0, 0.32), (4, 3, 32.0, 0.32)], places)
    parent, _, _ = network.shortest_path_forest(graph, [1])
    assert parent[3] == 4


def test_forest_is_radial():
    """Every reached node except the source has one parent, so the forest edges form a tree."""
    places = {node: (node % 3 * 10, node // 3 * 10) for node in range(9)}
    edges = [(a, b, 10.0, 10.0) for a in range(9) for b in range(a + 1, 9)
             if abs(places[a][0] - places[b][0]) + abs(places[a][1] - places[b][1]) == 10]
    parent, _, _ = network.shortest_path_forest(made_up_graph(edges, places), [0])
    tree = networkx.Graph(list(parent.items()))
    assert networkx.is_tree(tree) and len(parent) == 8


def test_chain_lines():
    """A chain with no branch merges into one line, and a branch node splits the lines."""
    forest = {"parent": {2: 1, 3: 2, 4: 3, 5: 3}, "roots": [1], "length": {2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}}
    assert network.chain_lines(forest, [1], {4, 5}) == [(1, 3, 3.0), (3, 4, 3.0), (3, 5, 4.0)]
    assert network.chain_lines(forest, [1], {4}) == [(1, 4, 6.0)]
    assert network.chain_lines(forest, [3], {4, 5}) == [(3, 4, 3.0), (3, 5, 4.0)]
    assert network.medium_voltage_lines(forest, {5}) == [(1, 5, 7.0)]


@pytest.fixture(scope="module")
def feeders(tmp_path_factory):
    """Return the feeder forest, the report and the unserved loads of the synthetic site."""
    extract, reference = site.write_site(tmp_path_factory.mktemp("network"))
    shape = site.site_shape()
    crs = streets.metric_crs(shape)
    graph = streets.street_graph(extract, shape, SETTINGS["street_highway_values"], crs)
    routes.add_forced_routes(graph, routes.route_lines(reference, "distribution_lines", shape), crs, SETTINGS)
    points = buildings.load_points(extract, shape, SETTINGS["building_skipped_values"])
    loads = buildings.loads_table(points, SETTINGS)
    to_metric = streets.to_metric(crs)
    stations = [{"id": "n800"}]
    station_xy = to_metric.transform(numpy.array([site.place(*site.SUBSTATION[1])[0]]),
                                     numpy.array([site.place(*site.SUBSTATION[1])[1]]))
    return network.build_feeders(graph, loads, buildings.metric_points(points, to_metric), stations, station_xy)


def test_build_feeders_root(feeders):
    """The substation takes node 1, and the forest reaches every node of the grid part and the route node."""
    forest, _, _ = feeders
    assert forest["roots"] == [1] and forest["root_station"] == {1: "n800"}
    assert set(forest["owner"]) == {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_build_feeders_follows_route(feeders):
    """Node 9 hangs from the off-street route node, which hangs from node 1."""
    forest = feeders[0]
    assert forest["parent"][9] == -1 and forest["parent"][-1] == 1


def test_build_feeders_report(feeders):
    """The load near the separate street has no substation path; it is reported, not dropped in silence."""
    _, report, unserved = feeders
    assert unserved == ["w207"]
    assert (report["loads"], report["loads_served"], report["loads_without_substation_path"]) == (6, 5, 1)
    assert (report["graph_parts"], report["parts_kept"], report["parts_with_loads_without_substation"]) == (2, 2, 1)
    assert report["substation_nodes"] == 1 and report["substations_sharing_a_node"] == 0

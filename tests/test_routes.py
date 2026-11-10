"""Tests of the forced routes.

The synthetic site of synthetic_site.py carries one mapped distribution line: node 1, an off-street point 40 m west of node 4,
then node 9. Its two street vertices must snap to nodes 1 and 9, the off-street vertex must become node -1, and both segments
must enter as forced edges whose weight is the length times the forced route factor.
"""

from pathlib import Path

import networkx
import pytest

import synthetic_site as site
from dxfeeder import inputs, routes, streets

PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = inputs.read_settings(PROJECT)


@pytest.fixture()
def built(tmp_path):
    """Return the street graph with the forced routes added, the count report and the route lines."""
    extract, reference = site.write_site(tmp_path)
    shape = site.site_shape()
    crs = streets.metric_crs(shape)
    graph = streets.street_graph(extract, shape, SETTINGS["street_highway_values"], crs)
    lines = routes.route_lines(reference, "distribution_lines", shape)
    return graph, routes.add_forced_routes(graph, lines, crs, SETTINGS), lines


def test_route_lines(built):
    """The one mapped line comes back whole, as three places."""
    assert len(built[2]) == 1 and len(built[2][0]) == 3


def test_forced_edges(built):
    """The off-street vertex becomes node -1, and the two segments join it to nodes 1 and 9."""
    graph, counts, _ = built
    assert counts == {"forced_edges": 2, "route_nodes": 1}
    assert graph.has_edge(1, -1) and graph.has_edge(-1, 9) and not graph.nodes[-1]["street"]
    for edge in ((1, -1), (-1, 9)):
        data = graph.edges[edge]
        assert data["forced"] and data["weight"] == pytest.approx(data["length"] * SETTINGS["forced_route_weight_factor"])
    assert graph.edges[1, -1]["length"] == pytest.approx(107.7, abs=1.0)


def test_streets_stay_open(built):
    """The street edges keep their full weight, so a path can still take any street."""
    graph = built[0]
    assert not graph.edges[1, 2]["forced"] and graph.edges[1, 2]["weight"] == graph.edges[1, 2]["length"]


def test_existing_street_takes_lower_weight():
    """A mapped segment along an existing street edge keeps the street length and takes the forced weight."""
    crs = streets.metric_crs(site.site_shape())
    graph = networkx.Graph()
    places = [site.place(0, 0), site.place(100, 0)]
    for node, (lon, lat) in zip((1, 2), places):
        x, y = streets.to_metric(crs).transform(lon, lat)
        graph.add_node(node, lon=lon, lat=lat, x=x, y=y, street=True)
    graph.add_edge(1, 2, length=100.0, weight=100.0, forced=False)
    counts = routes.add_forced_routes(graph, [places], crs, SETTINGS)
    assert counts == {"forced_edges": 1, "route_nodes": 0}
    assert graph.edges[1, 2]["length"] == 100.0 and graph.edges[1, 2]["forced"]
    assert graph.edges[1, 2]["weight"] == pytest.approx(100.0 * SETTINGS["forced_route_weight_factor"])


def test_no_lines():
    """No mapped line adds nothing."""
    assert routes.add_forced_routes(networkx.Graph(), [], None, SETTINGS) == {"forced_edges": 0, "route_nodes": 0}

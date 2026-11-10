"""Tests of the street graph.

Each test runs on the synthetic site of synthetic_site.py, written to a temporary folder, with no network.
The street graph must keep the drivable streets inside a MultiPolygon site, leave out footways and segments that leave the site,
and keep every connected part, not only the largest one.
"""

from pathlib import Path

import networkx
import pytest

import synthetic_site as site
from dxfeeder import inputs, streets

PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = inputs.read_settings(PROJECT)


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    """Return the street graph of the synthetic site."""
    extract, _ = site.write_site(tmp_path_factory.mktemp("streets"))
    shape = site.site_shape()
    return streets.street_graph(extract, shape, SETTINGS["street_highway_values"], streets.metric_crs(shape))


def test_metric_crs():
    """The site near Halifax gets UTM zone 20 north."""
    assert streets.metric_crs(site.site_shape()).to_epsg() == 32620


def test_street_graph_nodes(graph):
    """The grid and the separate service street enter; the footway and the node outside the site do not."""
    assert sorted(graph.nodes) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21]


def test_street_graph_edges(graph):
    """The grid gives 12 edges of about 100 m and the service street one; the segment to node 30 is left out."""
    assert graph.number_of_edges() == 13 and not graph.has_edge(9, 30)
    assert all(data["length"] == pytest.approx(100.0, rel=0.01) for _, _, data in graph.edges(data=True))
    assert all(data["weight"] == data["length"] and not data["forced"] for _, _, data in graph.edges(data=True))


def test_street_graph_keeps_every_part(graph):
    """Both parts of the MultiPolygon site keep their streets, so the graph has two connected parts."""
    parts = sorted(sorted(part) for part in networkx.connected_components(graph))
    assert parts == [[1, 2, 3, 4, 5, 6, 7, 8, 9], [20, 21]]


def test_graph_from_segments_empty():
    """A site with no segments gives an empty graph."""
    assert streets.graph_from_segments({}, [], streets.metric_crs(site.site_shape())).number_of_nodes() == 0

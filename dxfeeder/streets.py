"""Street graph of a site.

This module reads the drivable streets of a site from a local OpenStreetMap file in PBF or XML format.
It reads the file twice, as the power line filter does: first the street ways and their node lists, then the node places.
It keeps each street segment whose two ends lie inside the site shape, which may be a Polygon or a MultiPolygon.
It builds an undirected graph whose nodes are OpenStreetMap node ids and whose edges carry the length in metres.
It keeps every connected part of the graph; the network step later drops only the parts that hold no load and no substation.
"""

import geopandas
import networkx
import numpy
import osmium
import osmium.filter
import shapely
from pyproj import Transformer

GEOGRAPHIC_CRS = "EPSG:4326"


def metric_crs(site):
    """Return the Universal Transverse Mercator system that pyproj picks for the site shape given in EPSG:4326."""
    return geopandas.GeoSeries([site], crs=GEOGRAPHIC_CRS).estimate_utm_crs()


def to_metric(crs):
    """Return a transformer from longitude and latitude to the metric system, with x before y."""
    return Transformer.from_crs(GEOGRAPHIC_CRS, crs, always_xy=True)


def street_ways(path, highway_values):
    """Return the node list of every way whose highway tag is one of the given values, keyed by way id."""
    wanted = set(highway_values)
    ways = {}
    for way in osmium.FileProcessor(str(path), osmium.osm.WAY).with_filter(osmium.filter.KeyFilter("highway")):
        if way.tags.get("highway") in wanted:
            ways[way.id] = [node.ref for node in way.nodes]
    return ways


def node_places(path, ids):
    """Return the longitude and latitude of each node id that the file holds, keyed by node id."""
    places = {}
    if not ids:
        return places
    for node in osmium.FileProcessor(str(path), osmium.osm.NODE).with_filter(osmium.filter.IdFilter(ids)):
        if node.location.valid():
            places[node.id] = (node.location.lon, node.location.lat)
    return places


def inside(site, places):
    """Return the set of node ids whose place lies inside or on the edge of the site shape."""
    ids = sorted(places)
    if not ids:
        return set()
    coords = numpy.array([places[node] for node in ids])
    flags = shapely.intersects_xy(site, coords[:, 0], coords[:, 1])
    return {node for node, flag in zip(ids, flags) if flag}


def street_graph(path, site, highway_values, crs):
    """Return the street graph of the site, with node places in degrees and metres and edge lengths in metres.

    A segment enters the graph when both of its end nodes lie inside the site shape. A segment that two ways share enters once.
    """
    ways = street_ways(path, highway_values)
    places = node_places(path, {ref for refs in ways.values() for ref in refs})
    kept = inside(site, places)
    segments = set()
    for way_id in sorted(ways):
        refs = [ref for ref in ways[way_id] if ref in places]
        for first, second in zip(refs, refs[1:]):
            if first != second and first in kept and second in kept:
                segments.add((min(first, second), max(first, second)))
    return graph_from_segments(places, sorted(segments), crs)


def graph_from_segments(places, segments, crs):
    """Return a graph built from node places in degrees and segments given as pairs of node ids, in the given metric system."""
    nodes = sorted({node for pair in segments for node in pair})
    graph = networkx.Graph()
    if not nodes:
        return graph
    lon = numpy.array([places[node][0] for node in nodes])
    lat = numpy.array([places[node][1] for node in nodes])
    x, y = to_metric(crs).transform(lon, lat)
    for node, node_lon, node_lat, node_x, node_y in zip(nodes, lon, lat, x, y):
        graph.add_node(node, lon=float(node_lon), lat=float(node_lat), x=float(node_x), y=float(node_y), street=True)
    for first, second in segments:
        length = float(numpy.hypot(graph.nodes[first]["x"] - graph.nodes[second]["x"],
                                   graph.nodes[first]["y"] - graph.nodes[second]["y"]))
        graph.add_edge(first, second, length=length, weight=length, forced=False)
    return graph

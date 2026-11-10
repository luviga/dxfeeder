"""Mapped distribution lines as forced routes.

This module reads mapped distribution lines from a line layer of a GeoPackage and adds them to the street graph.
It clips each line to the site shape and walks its vertices in order.
A vertex within the snap distance of a street node takes the nearest street node, the smallest node id first on a tie.
Any other vertex becomes a node of its own, shared by every line that passes through the same place.
Each segment between two consecutive nodes becomes a forced edge, whose weight is its length times the forced route factor.
The low weight makes the shortest paths prefer mapped routes, and the streets stay open to every path.
A segment between two street nodes that a street already joins keeps the street length and takes the lower weight.
"""

import numpy
import pyogrio
import shapely

from dxfeeder.streets import GEOGRAPHIC_CRS, to_metric


def route_lines(path, layer, site):
    """Return the mapped lines of a GeoPackage layer clipped to the site shape, as lists of (longitude, latitude), in row order."""
    frame = pyogrio.read_dataframe(path, layer=layer).to_crs(GEOGRAPHIC_CRS)
    lines = []
    for shape in frame.geometry.to_numpy():
        if shape is None or shape.is_empty:
            continue
        clipped = shapely.intersection(shape, site)
        for part in shapely.get_parts(clipped):
            if part.geom_type == "LineString" and len(part.coords) > 1:
                lines.append([(float(x), float(y)) for x, y in part.coords])
    return lines


def nearest_ids(tree, ids, x, y, limit):
    """Return, for each point, the id of the nearest indexed node within the limit, the smallest id on a tie, or None."""
    found = [None] * len(x)
    if not len(x) or not len(ids):
        return found
    (asked, matched), _ = tree.query_nearest(shapely.points(x, y), max_distance=limit, return_distance=True, all_matches=True)
    for point, node in zip(asked, matched):
        if found[point] is None or ids[node] < found[point]:
            found[point] = ids[node]
    return found


def node_index(graph, street_only=False):
    """Return the sorted node ids of the graph and a spatial index of their metric places, in the same order."""
    ids = sorted(node for node, data in graph.nodes(data=True) if data.get("street") or not street_only)
    places = shapely.points([graph.nodes[node]["x"] for node in ids], [graph.nodes[node]["y"] for node in ids])
    return ids, shapely.STRtree(places)


def add_forced_routes(graph, lines, crs, settings):
    """Add the mapped lines to the graph as forced edges and return the number of forced edges and of new nodes.

    New nodes take negative ids, counted from minus one in the order of their longitude and latitude.
    """
    places = sorted({place for line in lines for place in line})
    if not places:
        return {"forced_edges": 0, "route_nodes": 0}
    lon = numpy.array([place[0] for place in places])
    lat = numpy.array([place[1] for place in places])
    x, y = to_metric(crs).transform(lon, lat)
    ids, tree = node_index(graph, street_only=True)
    snapped = nearest_ids(tree, ids, x, y, float(settings["snap_m"]))
    node_of, new_nodes = {}, 0
    for place, place_x, place_y, street in zip(places, x, y, snapped):
        if street is None:
            new_nodes += 1
            node_of[place] = -new_nodes
            graph.add_node(-new_nodes, lon=place[0], lat=place[1], x=float(place_x), y=float(place_y), street=False)
        else:
            node_of[place] = street
    factor = float(settings["forced_route_weight_factor"])
    pairs = set()
    for line in lines:
        nodes = [node_of[place] for place in line]
        for first, second in zip(nodes, nodes[1:]):
            if first != second:
                pairs.add((min(first, second), max(first, second)))
    for first, second in sorted(pairs):
        if graph.has_edge(first, second):
            data = graph.edges[first, second]
            data["weight"] = min(data["weight"], data["length"] * factor)
            data["forced"] = True
        else:
            length = float(numpy.hypot(graph.nodes[first]["x"] - graph.nodes[second]["x"],
                                       graph.nodes[first]["y"] - graph.nodes[second]["y"]))
            graph.add_edge(first, second, length=length, weight=length * factor, forced=True)
    return {"forced_edges": len(pairs), "route_nodes": new_nodes}

"""Radial feeders on the street and route graph.

This module joins each load point to its nearest graph node with a straight service drop, and each substation to its nearest node.
It keeps every connected part of the graph that holds at least one load or one substation, never only the largest part.
It assigns each node to the substation nearest by weighted graph distance, and for each substation it keeps the shortest path tree.
Forced routes carry a low weight, so the tree follows them wherever they lead toward the substation.
The result is radial by construction: every node has at most one parent, and each tree holds exactly one substation.
Loads in a part with no substation are reported, with their count and the count of such parts, and are left out of the feeders.
"""

import heapq

import networkx
import numpy
import shapely

from dxfeeder.routes import node_index


def nearest_nodes(graph, x, y):
    """Return, for each metric point, the nearest graph node and the distance to it in metres, the smallest id on a tie."""
    ids, tree = node_index(graph)
    found = [None] * len(x)
    if not len(x) or not ids:
        return found
    (asked, matched), distances = tree.query_nearest(shapely.points(x, y), return_distance=True, all_matches=True)
    for point, node, distance in zip(asked, matched, distances):
        if found[point] is None or ids[node] < found[point][0]:
            found[point] = (ids[node], float(distance))
    return found


def shortest_path_forest(graph, sources):
    """Return the parent, the weighted distance and the source of every node that a source reaches.

    The search runs from all sources at once over the edge weights. A tie in distance goes to the smaller node id, then to the
    smaller source, then to the smaller parent, so the forest does not depend on the order of the graph.
    """
    parent, distance, owner = {}, {}, {}
    heap = [(0.0, source, source, source) for source in sorted(sources)]
    heapq.heapify(heap)
    while heap:
        reach, node, source, previous = heapq.heappop(heap)
        if node in distance:
            continue
        distance[node], owner[node] = reach, source
        if previous != node:
            parent[node] = previous
        for neighbour in sorted(graph.adj[node]):
            if neighbour not in distance:
                heapq.heappush(heap, (reach + graph.edges[node, neighbour]["weight"], neighbour, source, node))
    return parent, distance, owner


def build_feeders(graph, loads, load_xy, substations, substation_xy):
    """Return the feeder forest of a site and a report of what it serves.

    loads and substations are lists of dictionaries with an "id"; load_xy and substation_xy hold their metric x and y arrays.
    The forest is a dictionary with the parent of each node, the children of each node, the roots and the edge length to the
    parent, plus the attached node and the drop length of each served load and the substation id of each root.
    """
    load_at = nearest_nodes(graph, *load_xy)
    station_at = nearest_nodes(graph, *substation_xy)
    load_nodes = {found[0] for found in load_at if found is not None}
    station_nodes = {found[0] for found in station_at if found is not None}
    part_of, parts = {}, []
    for number, part in enumerate(sorted(networkx.connected_components(graph), key=min)):
        parts.append(part)
        for node in part:
            part_of[node] = number
    with_loads = {part_of[node] for node in load_nodes}
    with_stations = {part_of[node] for node in station_nodes}
    roots = {}
    for station, found in sorted(zip((item["id"] for item in substations), station_at)):
        if found is not None:
            roots.setdefault(found[0], station)
    parent, _, owner = shortest_path_forest(graph, roots)
    children = {}
    for node in sorted(parent):
        children.setdefault(parent[node], []).append(node)
    served, unserved = {}, []
    for load, found in zip(loads, load_at):
        if found is not None and found[0] in owner:
            served[load["id"]] = found
        else:
            unserved.append(load["id"])
    forest = {
        "parent": parent,
        "children": children,
        "roots": sorted(roots),
        "root_station": roots,
        "owner": owner,
        "length": {node: graph.edges[node, parent[node]]["length"] for node in parent},
        "attached": served,
    }
    report = {
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "graph_parts": len(parts),
        "parts_kept": len(with_loads | with_stations),
        "parts_dropped_without_load_or_substation": len(parts) - len(with_loads | with_stations),
        "parts_with_loads_without_substation": len(with_loads - with_stations),
        "loads": len(loads),
        "loads_served": len(served),
        "loads_without_substation_path": len(unserved),
        "substations": len(substations),
        "substation_nodes": len(roots),
        "substations_sharing_a_node": len(substations) - len(roots),
        "substation_snap_m_max": max((found[1] for found in station_at if found is not None), default=0.0),
        "drop_m_median": float(numpy.median([found[1] for found in served.values()])) if served else 0.0,
        "drop_m_max": max((found[1] for found in served.values()), default=0.0),
    }
    return forest, report, sorted(unserved)


def post_order(forest, root):
    """Return the nodes of one tree of the forest, each child before its parent, children in node id order."""
    order, stack = [], [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        stack.append((node, True))
        for child in reversed(forest["children"].get(node, [])):
            stack.append((child, False))
    return order


def chain_lines(forest, tops, terminals):
    """Return the lines that join every terminal node up to its top node along the forest, with chains of plain nodes merged.

    Each terminal must lie below one of the top nodes. A line runs between two kept nodes: a top, a terminal, or a node where two
    or more kept branches meet. Each line is a tuple (upper node, lower node, length in metres) in lower node order.
    """
    parent, tops = forest["parent"], set(tops)
    keep = set(tops)
    for node in sorted(terminals):
        while node not in keep:
            keep.add(node)
            node = parent[node]
    branches = {}
    for node in keep - tops:
        branches[parent[node]] = branches.get(parent[node], 0) + 1
    joints = tops | set(terminals) | {node for node, count in branches.items() if count > 1}
    lines = []
    for node in sorted(joints - tops):
        length, upper = 0.0, node
        while True:
            length += forest["length"][upper]
            upper = parent[upper]
            if upper in joints:
                break
        lines.append((upper, node, length))
    return lines


def medium_voltage_lines(forest, terminals):
    """Return the medium voltage lines that join every terminal node to the substation root of its tree."""
    return chain_lines(forest, forest["roots"], terminals)

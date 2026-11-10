"""Route shape of each mapped power line.

This module reads circuits.csv, alberta_lines.csv, lines.gpkg, and the settings file.
For each line and each line source, it keeps the features in the search area that carry the line voltage.
It snaps both end nodes to the kept features and finds the shortest route between the two snap points.
It measures each length on the ellipsoid of the Geodetic Reference System 1980.
A line with a long detour or no route gets one retry from every feature near each end node.
It writes route_shapes.gpkg, with one row per line and per source, and the reason for each line with no shape.
"""

from pathlib import Path

import geopandas
import networkx
import numpy
import pandas
import pyogrio
import shapely
from pyproj import Geod, Transformer

from dxfeeder.inputs import NODE_CRS, OSM_NOTE, InputError, read_settings, write_gpkg
from dxfeeder.lines import MAP_CRS

SOURCES = ("CanVec", "OpenStreetMap")
GEOD = Geod(ellps="GRS80")
TO_MAP = Transformer.from_crs(NODE_CRS, MAP_CRS, always_xy=True)
TO_DEGREES = Transformer.from_crs(MAP_CRS, NODE_CRS, always_xy=True)
FIELDS = ["line_group", "corridor_id", "circuit_id", "part", "record_ids", "voltage_kv", "voltage_values_kv"]
COLUMNS = FIELDS + ["source", "status", "reason", "straight_km", "length_km", "detour_ratio", "submarine_km", "largest_join_m",
                    "feature_ids", "snap_a_km", "snap_b_km", "snap_retry"]
TEXT_COLUMNS = ["line_group", "corridor_id", "circuit_id", "record_ids", "voltage_values_kv", "source", "status", "reason", "feature_ids", "snap_retry"]
NO_FEATURE, BEYOND_LIMIT, SAME_POINT, NO_CONNECTION = "no feature in search area", "snap beyond limit", "same snap point", "no connection"
NOT_SELECTED, KEPT, NOT_KEPT = "not selected", "kept", "not kept"


def ellipsoid_m(x1, y1, x2, y2):
    """Return the ellipsoid distance in meters between map points in EPSG:3347, for single values or for arrays."""
    lon1, lat1 = TO_DEGREES.transform(x1, y1)
    lon2, lat2 = TO_DEGREES.transform(x2, y2)
    return GEOD.inv(lon1, lat1, lon2, lat2)[2]


def voltage_set(text):
    """Return the voltage values of a text with values joined by semicolons, as a set of numbers in kV."""
    try:
        return frozenset(float(value) for value in text.split(";") if value.strip())
    except ValueError as error:
        raise InputError(f"the voltage text {text} does not read as numbers") from error


def line_item(row, group, fields, voltages):
    """Return one line with its fields, its voltage set, its end node codes, and its end nodes in EPSG:3347."""
    a = TO_MAP.transform(float(row["longitude_a"]), float(row["latitude_a"]))
    b = TO_MAP.transform(float(row["longitude_b"]), float(row["latitude_b"]))
    voltage = float(row["voltage_kv"]) if row["voltage_kv"] else None
    fields = {"line_group": group, **fields, "voltage_kv": voltage, "voltage_values_kv": voltages}
    return {"fields": fields, "voltages": voltage_set(voltages), "codes": (row["node_a"], row["node_b"]), "a": a, "b": b,
            "straight_km": ellipsoid_m(*a, *b) / 1000.0}


def read_lines(folder):
    """Return the circuit rows and then the Alberta rows as lines, without a line whose two end node codes are equal."""
    circuits = pandas.read_csv(Path(folder) / "circuits.csv", dtype=str, keep_default_na=False)
    alberta = pandas.read_csv(Path(folder) / "alberta_lines.csv", dtype=str, keep_default_na=False)
    lines = [line_item(row, "CORRIDOR", {"corridor_id": row["corridor_id"], "circuit_id": row["circuit_id"],
                                         "part": int(row["part"]), "record_ids": row["record_ids"]}, row["voltage_values_kv"])
             for row in circuits.to_dict("records")]
    lines += [line_item(row, "ALBERTA", {"corridor_id": None, "circuit_id": None, "part": None, "record_ids": row["record_id"]},
                        row["voltage_kv"]) for row in alberta.to_dict("records")]
    # A line from one node to itself has no length and gives no route shape.
    return [line for line in lines if line["codes"][0] != line["codes"][1]]


def part_ends(index):
    """Return the first and the last coordinate position of each part, from the part index of each coordinate."""
    breaks = numpy.flatnonzero(numpy.diff(index))
    if not len(index):
        return breaks, breaks
    return numpy.r_[0, breaks + 1], numpy.r_[breaks, len(index) - 1]


def join_pairs(x, y, features, join_distance):
    """Return the pairs of part ends closer than the end join distance, with their features, positions, and distance in meters."""
    points = shapely.points(x, y)
    # The map distance stays within a few percent of the ellipsoid distance in Canada, so twice the join distance finds every pair.
    first, second = shapely.STRtree(points).query(points, predicate="dwithin", distance=2.0 * join_distance)
    keep = (first < second) & ((x[first] != x[second]) | (y[first] != y[second]))
    order = numpy.lexsort((second[keep], first[keep]))
    first, second = first[keep][order], second[keep][order]
    distance = ellipsoid_m(x[first], y[first], x[second], y[second])
    close = distance < join_distance
    first, second = first[close], second[close]
    return {"features": (features[first], features[second]), "ends": (x[first], y[first], x[second], y[second]),
            "weight": distance[close]}


def source_network(frame, source, join_distance):
    """Return the features, segments, and join pairs of one line source, with the features in feature id order."""
    frame = frame[frame["source"] == source].sort_values("feature_id").reset_index(drop=True)
    parts, owner = shapely.get_parts(frame.geometry.to_numpy(), return_index=True)
    coords, index = shapely.get_coordinates(parts, return_index=True)
    x, y = coords[:, 0], coords[:, 1]
    start = numpy.flatnonzero(index[:-1] == index[1:])
    segment_feature = owner[index[start]]
    segments = {"x1": x[start], "y1": y[start], "x2": x[start + 1], "y2": y[start + 1], "feature": segment_feature}
    segments["weight"] = ellipsoid_m(segments["x1"], segments["y1"], segments["x2"], segments["y2"])
    ends = numpy.concatenate(part_ends(index))
    values = frame["voltage_values_kv"].to_numpy()
    # CanVec gives no voltage, so every CanVec feature passes the voltage filter.
    voltages = [None if source == SOURCES[0] or pandas.isna(text) else voltage_set(text) for text in values]
    submarine = ((frame["kind"] == "submarine") | (frame["location"] == "underwater")).to_numpy()
    return {"ids": frame["feature_id"].to_numpy(), "tree": shapely.STRtree(frame.geometry.to_numpy()), "voltages": voltages,
            "submarine": submarine, "segments": segments,
            "first_segment": numpy.searchsorted(segment_feature, numpy.arange(len(frame) + 1)),
            "joins": join_pairs(x[ends], y[ends], owner[index[ends]], join_distance)}


def kept_features(network, line, margin):
    """Return the features in the search area of a line that pass the voltage filter, as indices in feature id order."""
    (xa, ya), (xb, yb) = line["a"], line["b"]
    box = shapely.box(min(xa, xb) - margin, min(ya, yb) - margin, max(xa, xb) + margin, max(ya, yb) + margin)
    found = numpy.sort(network["tree"].query(box, predicate="intersects"))
    voltages = network["voltages"]
    if not line["voltages"]:
        return found
    return numpy.array([index for index in found if voltages[index] is None or voltages[index] & line["voltages"]], dtype=int)


def feature_segments(network, kept):
    """Return the segment rows of the kept features, in feature id order and in vertex order inside each feature."""
    first = network["first_segment"]
    return numpy.concatenate([numpy.arange(first[index], first[index + 1]) for index in kept] + [numpy.array([], dtype=int)])


def snap(network, rows, point):
    """Return the nearest point on the segments to an end node, by map distance, with the segment row and the distance in km.

    On equal distances, the smaller feature id wins, then the first segment in vertex order.
    """
    seg = network["segments"]
    x1, y1, x2, y2 = (seg[key][rows] for key in ("x1", "y1", "x2", "y2"))
    dx, dy = x2 - x1, y2 - y1
    length = dx * dx + dy * dy
    t = numpy.clip(((point[0] - x1) * dx + (point[1] - y1) * dy) / numpy.where(length > 0, length, 1.0), 0.0, 1.0)
    distance = numpy.hypot(point[0] - (x1 + t * dx), point[1] - (y1 + t * dy))
    pick = int(numpy.flatnonzero(distance == distance.min())[0])
    # A snap point at either end of the segment takes the vertex position exactly, so it joins the existing graph node.
    if t[pick] <= 0.0 or t[pick] >= 1.0:
        found = (float(x1[pick]), float(y1[pick])) if t[pick] <= 0.0 else (float(x2[pick]), float(y2[pick]))
    else:
        found = (float(x1[pick] + t[pick] * dx[pick]), float(y1[pick] + t[pick] * dy[pick]))
    return {"row": int(rows[pick]), "t": float(t[pick]), "point": found, "km": ellipsoid_m(*point, *found) / 1000.0}


def segment_edges(network, rows, snaps):
    """Return the edges of the segments, keyed by node pair, with each snap point as a split, keeping the first edge of a pair."""
    seg, splits, edges = network["segments"], {}, {}
    for item in snaps:
        splits.setdefault(item["row"], []).append((item["t"], item["point"]))
    columns = [seg[key][rows].tolist() for key in ("x1", "y1", "x2", "y2", "weight", "feature")]
    for row, x1, y1, x2, y2, weight, feature in zip(rows.tolist(), *columns):
        points = [(x1, y1)] + [point for _, point in sorted(splits.get(row, []))] + [(x2, y2)]
        if len(points) == 2:
            weights = [weight]
        else:
            weights = [ellipsoid_m(*u, *v) for u, v in zip(points, points[1:])]
        for u, v, length in zip(points, points[1:], weights):
            key = (u, v) if u < v else (v, u)
            if u != v and key not in edges:
                edges[key] = (length, feature)
    return edges


def route_graph(network, kept, rows, snaps):
    """Return the graph of the kept features, with one node per position, a node at each snap point, and the join edges."""
    graph = networkx.Graph()
    graph.add_nodes_from(item["point"] for item in snaps)
    graph.add_edges_from((u, v, {"weight": length, "feature": feature})
                         for (u, v), (length, feature) in segment_edges(network, rows, snaps).items())
    mask = numpy.zeros(len(network["ids"]), dtype=bool)
    mask[kept] = True
    joins = network["joins"]
    use = mask[joins["features"][0]] & mask[joins["features"][1]]
    # A join edge carries no feature, and an edge of a feature between the same two nodes stays in its place.
    for x1, y1, x2, y2, length in zip(*(column[use].tolist() for column in joins["ends"]), joins["weight"][use].tolist()):
        if not graph.has_edge((x1, y1), (x2, y2)):
            graph.add_edge((x1, y1), (x2, y2), weight=length, feature=None)
    return graph


def route_ids(graph, route, ids):
    """Return the sorted distinct feature ids on a route, as a list of text."""
    return sorted({ids[graph.edges[u, v]["feature"]] for u, v in zip(route, route[1:]) if graph.edges[u, v]["feature"] is not None})


def best_route(graph, start, end, ids, name, limit):
    """Return the shortest route as a node list, the smallest sorted list of feature ids first on a tie, or None if none exists.

    More equal shortest routes than the limit point to a fault in the line source, so the run stops.
    """
    routes = []
    try:
        for route in networkx.all_shortest_paths(graph, start, end, weight="weight"):
            routes.append(route)
            if len(routes) > limit:
                raise InputError(f"{name} has more than {limit} shortest routes of equal length")
    except networkx.NetworkXNoPath:
        return None
    return min(routes, key=lambda route: route_ids(graph, route, ids))


def route_values(graph, route, network, straight_km, snaps):
    """Return the length, detour ratio, submarine length, longest join, feature ids, and snap distances of a route, with its shape."""
    edges = [graph.edges[u, v] for u, v in zip(route, route[1:])]
    length = sum(edge["weight"] for edge in edges) / 1000.0
    submarine = sum(edge["weight"] for edge in edges if edge["feature"] is not None and network["submarine"][edge["feature"]])
    joins = [edge["weight"] for edge in edges if edge["feature"] is None]
    values = {"status": "SHAPE", "length_km": length, "detour_ratio": length / straight_km, "submarine_km": submarine / 1000.0,
              "largest_join_m": max(joins, default=0.0), "feature_ids": ";".join(route_ids(graph, route, network["ids"])),
              "snap_a_km": snaps[0]["km"], "snap_b_km": snaps[1]["km"]}
    return values, shapely.LineString(route)


def pair_route(network, kept, rows, snaps, line, name, limit):
    """Return the values and the shape of the shortest route between two snap points, or None if no route exists."""
    graph = route_graph(network, kept, rows, snaps)
    route = best_route(graph, snaps[0]["point"], snaps[1]["point"], network["ids"], name, limit)
    return None if route is None else route_values(graph, route, network, line["straight_km"], snaps)


def route_shape(network, line, settings, name):
    """Return the values and the route shape of one line on one source. With no route, it returns the reason and no shape."""
    values = {"status": "NO SHAPE", "straight_km": line["straight_km"]}
    kept = kept_features(network, line, settings["search_area_margin_km"] * 1000.0)
    if not len(kept):
        return {**values, "reason": NO_FEATURE}, None
    rows = feature_segments(network, kept)
    snaps = [snap(network, rows, point) for point in (line["a"], line["b"])]
    values.update(snap_a_km=snaps[0]["km"], snap_b_km=snaps[1]["km"])
    if max(values["snap_a_km"], values["snap_b_km"]) > settings["snap_limit_km"]:
        return {**values, "reason": BEYOND_LIMIT}, None
    if snaps[0]["point"] == snaps[1]["point"]:
        return {**values, "reason": SAME_POINT}, None
    found = pair_route(network, kept, rows, snaps, line, name, settings["max_equal_routes"])
    if found is None:
        return {**values, "reason": NO_CONNECTION}, None
    return {**values, **found[0]}, found[1]


def retry_snaps(network, kept, point, distance):
    """Return the snap point on each kept feature within the retry distance of an end node, in feature id order."""
    # Map and ellipsoid distances differ by a few percent in Canada, so the map search with twice the distance finds every feature.
    near = numpy.intersect1d(kept, network["tree"].query(shapely.Point(point), predicate="dwithin", distance=2.0 * distance))
    found = [snap(network, feature_segments(network, [index]), point) for index in near]
    return [item for item in found if item["km"] * 1000.0 <= distance]


def retried_shape(network, line, settings, name):
    """Return the values and the shape of one line on one source after the snap retry, with the outcome of the retry."""
    values, shape = route_shape(network, line, settings, name)
    if not (values["detour_ratio"] > settings["detour_flag"] if shape is not None else values["reason"] in (NO_CONNECTION, SAME_POINT)):
        return {**values, "snap_retry": NOT_SELECTED}, shape
    kept = kept_features(network, line, settings["search_area_margin_km"] * 1000.0)
    rows = feature_segments(network, kept)
    ends = [retry_snaps(network, kept, point, settings["snap_retry_m"]) for point in (line["a"], line["b"])]
    found = [pair_route(network, kept, rows, [first, second], line, name, settings["max_equal_routes"])
             for first in ends[0] for second in ends[1] if first["point"] != second["point"]]
    # On equal lengths, the smaller sorted list of feature ids wins, then the first pair in feature id order.
    found = min(filter(None, found), default=None, key=lambda item: (item[0]["length_km"], item[0]["feature_ids"].split(";")))
    # A retry must improve the first result, and a route much shorter than the straight line does not join the end nodes.
    if found is None or found[0]["detour_ratio"] < settings["snap_retry_ratio_floor"] or (shape is not None and found[0]["detour_ratio"] >= values["detour_ratio"]):
        return {**values, "snap_retry": NOT_KEPT}, shape
    return {"straight_km": line["straight_km"], **found[0], "snap_retry": KEPT}, found[1]


def route_frame(lines, networks, settings):
    """Return one row per line and per source, CanVec first within each line, with the route shape or an empty shape."""
    rows, shapes = [], []
    for line in lines:
        fields = line["fields"]
        name = f"the line {fields['record_ids']} of {fields['corridor_id'] or fields['line_group']}"
        for source in SOURCES:
            values, shape = retried_shape(networks[source], line, settings, f"{name} on {source}")
            rows.append({**fields, "source": source, **values})
            shapes.append(shape)
    frame = pandas.DataFrame(rows, columns=COLUMNS)
    frame["part"] = frame["part"].astype("Int64")
    for column in TEXT_COLUMNS:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return geopandas.GeoDataFrame(frame, geometry=shapes, crs=MAP_CRS)


def run_step(root, paths):
    """Run the route shapes on the line tables and lines.gpkg, write route_shapes.gpkg, and return its change."""
    folder, settings = Path(root) / "outputs", read_settings(root)
    features = pyogrio.read_dataframe(folder / "lines.gpkg")
    networks = {source: source_network(features, source, settings["end_join_m"]) for source in SOURCES}
    frame = route_frame(read_lines(folder), networks, settings)
    return {"outputs/route_shapes.gpkg": write_gpkg(frame, folder / "route_shapes.gpkg", "route_shapes", OSM_NOTE)}

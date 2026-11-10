"""OpenStreetMap filter.

This module reads the OpenStreetMap province files.
It keeps the power lines and power cables of the grid definition, with one copy of each way.
It reads each value of a voltage tag in kV.
It returns the kept ways as one table, with the power, voltage, and location tags of each way.
It writes no file.
"""

import re

import geopandas
import osmium
import osmium.filter
import shapely

# The pattern keeps the exact values line and cable. The ungrouped form ^line|cable$ also matched minor_underground_cable.
LINE_PATTERN = re.compile("^(?:line|cable)$")
# The grid definition query also keeps a minor line at 50 kV or more.
MINOR_LINE_KV = 50.0
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def voltage_values(tag):
    """Return every value of a voltage tag in kV, in tag order, or None if any value does not read as a number."""
    values = [value.strip() for value in str(tag or "").split(";")]
    if not all(NUMBER.fullmatch(value) for value in values):
        return None
    return [float(value) / 1000.0 for value in values]


def voltage_kv(tag):
    """Return the highest value of a voltage tag in kV, or None if any value of the tag does not read as a number."""
    values = voltage_values(tag)
    return None if values is None else max(values)


def keeps_way(tags):
    """Return True if the grid definition counts a way with these tags as a power line or a power cable."""
    # A way under construction or out of use carries a longer key. The filter excludes it, because the project covers existing lines.
    power = tags.get("power", "")
    if power == "minor_line":
        voltage = voltage_kv(tags.get("voltage"))
        return voltage is not None and voltage >= MINOR_LINE_KV
    return bool(LINE_PATTERN.search(power))


def province_lines(path):
    """Return the kept ways of one OpenStreetMap file, keyed by way id, with the node count, tags, and shape."""
    ways = {}
    for way in osmium.FileProcessor(str(path), osmium.osm.WAY).with_filter(osmium.filter.KeyFilter("power")):
        tags = dict(way.tags)
        if keeps_way(tags):
            ways[way.id] = ([node.ref for node in way.nodes], tags)
    needed = {ref for refs, _ in ways.values() for ref in refs}
    nodes = osmium.FileProcessor(str(path), osmium.osm.NODE).with_filter(osmium.filter.IdFilter(needed))
    places = {node.id: (node.location.lon, node.location.lat) for node in nodes}
    lines = {}
    for way_id, (refs, tags) in ways.items():
        points = [places[ref] for ref in refs if ref in places]
        if len(points) > 1:
            lines[way_id] = (len(points), tags["power"], tags.get("voltage"), tags.get("location"), shapely.LineString(points))
    return lines


def filter_osm(paths):
    """Return the kept ways of the province files as one table, with one copy of each way, in way id order."""
    kept = {}
    for path in paths:
        for way_id, line in province_lines(path).items():
            # The paths come in the alphabetical order of the provinces, so a tie keeps the first copy.
            if way_id not in kept or line[0] > kept[way_id][0]:
                kept[way_id] = line
    ids = sorted(kept)
    columns = {"way_id": ids, "power": [kept[way][1] for way in ids], "voltage": [kept[way][2] for way in ids],
               "location": [kept[way][3] for way in ids]}
    return geopandas.GeoDataFrame(columns, geometry=[kept[way][4] for way in ids], crs="EPSG:4326")

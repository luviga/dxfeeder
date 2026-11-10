"""Building footprints and load points of a site.

This module reads the building footprints of a site from a local OpenStreetMap file in PBF or XML format.
A building is a closed way with a building tag, or a multipolygon relation with a building tag whose outer ways close into rings.
A building tag with the value "no" marks a feature that is not a building, so it is skipped.
It reads the file in passes, as the power line filter does: building ways and relations, then outer member ways, then node places.
Each building gives one load point, a point that lies on its footprint, and the point must lie inside the site shape.
Each load carries the demand and the power factor from the settings file.
"""

import numpy
import osmium
import osmium.filter
import shapely

from dxfeeder.streets import node_places


def is_building(tags, skipped_values):
    """Return True if the tags mark a building, that is a building tag whose value is not one of the skipped values."""
    return "building" in tags and tags.get("building") not in set(skipped_values)


def building_records(path, skipped_values):
    """Return the node lists of building ways and the outer member way ids of building relations, each keyed by id."""
    ways, relations = {}, {}
    entities = osmium.osm.WAY | osmium.osm.RELATION
    for item in osmium.FileProcessor(str(path), entities).with_filter(osmium.filter.KeyFilter("building")):
        if not is_building(item.tags, skipped_values):
            continue
        if item.is_way():
            refs = [node.ref for node in item.nodes]
            if len(refs) > 3 and refs[0] == refs[-1]:
                ways[item.id] = refs
        elif item.is_relation() and item.tags.get("type") == "multipolygon":
            relations[item.id] = sorted(member.ref for member in item.members if member.type == "w" and member.role != "inner")
    return ways, relations


def member_ways(path, ids):
    """Return the node list of each way id that the file holds, keyed by way id."""
    found = {}
    if not ids:
        return found
    for way in osmium.FileProcessor(str(path), osmium.osm.WAY).with_filter(osmium.filter.IdFilter(ids)):
        found[way.id] = [node.ref for node in way.nodes]
    return found


def footprint(rings):
    """Return a valid footprint from lists of longitude and latitude pairs, or None if the lists close no ring."""
    lines = [shapely.LineString(ring) for ring in rings if len(ring) > 1]
    if not lines:
        return None
    shape = shapely.make_valid(shapely.union_all(list(shapely.get_parts(shapely.polygonize(lines)))))
    return None if shape.is_empty else shape


def load_points(path, site, skipped_values):
    """Return the load points of a site as a list of (load id, longitude, latitude), sorted by load id.

    The load id is "w" and the way id for a building way, and "r" and the relation id for a building relation.
    """
    ways, relations = building_records(path, skipped_values)
    members = member_ways(path, {ref for refs in relations.values() for ref in refs})
    needed = {ref for refs in ways.values() for ref in refs} | {ref for refs in members.values() for ref in refs}
    places = node_places(path, needed)
    shapes = {}
    for way_id, refs in ways.items():
        shapes[("w", way_id)] = footprint([[places[ref] for ref in refs if ref in places]])
    for relation_id, refs in relations.items():
        shapes[("r", relation_id)] = footprint([[places[node] for node in members[ref] if node in places]
                                                for ref in refs if ref in members])
    keys = sorted(key for key, shape in shapes.items() if shape is not None)
    if not keys:
        return []
    points = shapely.point_on_surface([shapes[key] for key in keys])
    coords = shapely.get_coordinates(points)
    flags = shapely.intersects_xy(site, coords[:, 0], coords[:, 1])
    found = [(f"{kind}{number}", float(x), float(y)) for (kind, number), (x, y), flag in zip(keys, coords, flags) if flag]
    return sorted(found)


def loads_table(points, settings):
    """Return the loads of a site as dictionaries with id, longitude, latitude, kW and power factor, in load id order."""
    return [{"id": load_id, "lon": lon, "lat": lat, "kw": float(settings["load_kw_default"]), "pf": float(settings["load_pf"])}
            for load_id, lon, lat in points]


def metric_points(points, transformer):
    """Return the x and y in metres of each load point as two arrays, in the order of the points."""
    if not points:
        return numpy.empty(0), numpy.empty(0)
    lon = numpy.array([point[1] for point in points])
    lat = numpy.array([point[2] for point in points])
    return transformer.transform(lon, lat)

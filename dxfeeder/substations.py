"""Substations of a site.

This module reads the substations of a site from a point layer of a GeoPackage, or from a local OpenStreetMap file.
In the OpenStreetMap file a substation is a node or a closed way with the tag power=substation; a closed way gives its centroid.
It keeps the substations that lie inside the site shape, sorted by id.
The feeder voltage never comes from a voltage tag of a substation: every feeder runs at the medium voltage of the settings file.
"""

import osmium
import osmium.filter
import pyogrio
import shapely

from dxfeeder.streets import GEOGRAPHIC_CRS, node_places


def from_layer(path, layer, site):
    """Return the substations of a GeoPackage point layer inside the site shape, as (id, longitude, latitude) sorted by id.

    The id is the OpenStreetMap type letter and id when the layer has the columns osm_type and osm_id, else the row number.
    """
    frame = pyogrio.read_dataframe(path, layer=layer).to_crs(GEOGRAPHIC_CRS)
    points = shapely.point_on_surface(frame.geometry.to_numpy())
    if {"osm_type", "osm_id"} <= set(frame.columns):
        ids = [f"{str(kind)[:1]}{int(number)}" for kind, number in zip(frame["osm_type"], frame["osm_id"])]
    else:
        ids = [f"row{number}" for number in range(len(frame))]
    return keep_inside(site, [(name, point.x, point.y) for name, point in zip(ids, points)])


def from_osm(path, site, key, value):
    """Return the substations of an OpenStreetMap file inside the site shape, as (id, longitude, latitude) sorted by id."""
    found, ways = [], {}
    entities = osmium.osm.NODE | osmium.osm.WAY
    for item in osmium.FileProcessor(str(path), entities).with_filter(osmium.filter.TagFilter((key, value))):
        if item.is_node() and item.location.valid():
            found.append((f"n{item.id}", item.location.lon, item.location.lat))
        elif item.is_way():
            refs = [node.ref for node in item.nodes]
            if len(refs) > 3 and refs[0] == refs[-1]:
                ways[item.id] = refs
    places = node_places(path, {ref for refs in ways.values() for ref in refs})
    for way_id in sorted(ways):
        ring = [places[ref] for ref in ways[way_id] if ref in places]
        if len(ring) > 3:
            centre = shapely.Polygon(ring).centroid
            found.append((f"w{way_id}", centre.x, centre.y))
    return keep_inside(site, found)


def keep_inside(site, rows):
    """Return the rows whose longitude and latitude lie inside or on the edge of the site shape, sorted by id."""
    return sorted(row for row in rows if shapely.intersects_xy(site, row[1], row[2]))

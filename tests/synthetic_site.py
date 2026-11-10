"""A small synthetic site for the feeder generator tests.

Everything here is made up and written by the tests into a temporary folder; no part of it comes from OpenStreetMap or any
other data set. The site sits near Halifax so that the metric system is UTM zone 20 north.

Layout, in metres east and north of the origin:
- a 3 by 3 grid of residential streets, nodes 1 to 9, 100 m apart, with a footway that the street filter must leave out;
- a separate service street, nodes 20 and 21, 600 m east, in a second part of the site, with no substation;
- a street segment from node 9 to node 30, which lies outside the site, so the segment is left out;
- buildings near the grid, one building tagged building=no, one building relation made of two outer ways,
  one building outside the site, and one building near the separate street;
- a substation node near node 1, and the same substation in the reference layer;
- one mapped distribution line in the reference layer from node 1 to an off-street point 40 m west of node 4 and on to node 9.
"""

import geopandas
import shapely

from dxfeeder import inputs

ORIGIN = (-63.6, 44.65)
# Degrees per metre near the origin, close enough for a made-up layout.
LON_PER_M, LAT_PER_M = 1 / 79120.0, 1 / 111130.0
GRID = {1: (0, 0), 2: (100, 0), 3: (200, 0), 4: (0, 100), 5: (100, 100), 6: (200, 100), 7: (0, 200), 8: (100, 200), 9: (200, 200)}
OTHER_NODES = {20: (600, 0), 21: (700, 0), 30: (200, 400), 40: (0, -50), 41: (50, -50)}
STREETS = [
    (101, [1, 2, 3], "residential"), (102, [4, 5, 6], "residential"), (103, [7, 8, 9], "residential"),
    (104, [1, 4, 7], "residential"), (105, [2, 5, 8], "residential"), (106, [3, 6, 9], "residential"),
    (107, [20, 21], "service"), (108, [9, 30], "residential"), (109, [40, 41], "footway"),
]
# Each building is a square of 10 m with its south-west corner at the given offset.
BUILDINGS = {201: (10, 10, "house"), 202: (110, 10, "yes"), 203: (110, 110, "house"), 204: (210, 110, "yes"),
             205: (10, 190, "no"), 206: (250, 450, "yes"), 207: (610, 10, "yes")}
RELATION = 301
RELATION_CORNER = (160, 160)
OFF_STREET = (-40, 100)
SUBSTATION = (800, (-20, -20))


def place(east, north):
    """Return the longitude and latitude of a point given in metres east and north of the origin."""
    return (ORIGIN[0] + east * LON_PER_M, ORIGIN[1] + north * LAT_PER_M)


def square(east, north, side=10):
    """Return the corners of a square in longitude and latitude, closed, counter-clockwise."""
    return [place(east, north), place(east + side, north), place(east + side, north + side), place(east, north + side),
            place(east, north)]


def site_shape():
    """Return the site shape: a MultiPolygon of the grid square and the square around the separate street."""
    main = shapely.Polygon(square(-50, -30, 300))
    east = shapely.Polygon(square(550, -50, 200))
    return shapely.MultiPolygon([main, east])


def osm_text():
    """Return the text of the made-up OpenStreetMap file of the site."""
    nodes, ways, relations = [], [], []
    next_id = 1000

    def node(node_id, lon, lat, tags=None):
        """Add one node."""
        tag_text = "".join(f'<tag k="{key}" v="{value}"/>' for key, value in (tags or {}).items())
        nodes.append(f'<node id="{node_id}" version="1" lat="{lat:.7f}" lon="{lon:.7f}">{tag_text}</node>')

    def way(way_id, refs, tags):
        """Add one way."""
        tag_text = "".join(f'<tag k="{key}" v="{value}"/>' for key, value in tags.items())
        ways.append(f'<way id="{way_id}" version="1">' + "".join(f'<nd ref="{ref}"/>' for ref in refs) + tag_text + "</way>")

    for node_id, (east, north) in {**GRID, **OTHER_NODES}.items():
        node(node_id, *place(east, north))
    for way_id, refs, kind in STREETS:
        way(way_id, refs, {"highway": kind})
    for way_id, (east, north, value) in BUILDINGS.items():
        refs = []
        for lon, lat in square(east, north)[:-1]:
            next_id += 1
            node(next_id, lon, lat)
            refs.append(next_id)
        way(way_id, refs + refs[:1], {"building": value})
    corners = []
    for lon, lat in square(*RELATION_CORNER)[:-1]:
        next_id += 1
        node(next_id, lon, lat)
        corners.append(next_id)
    way(401, [corners[0], corners[1], corners[2]], {})
    way(402, [corners[2], corners[3], corners[0]], {})
    relations.append(f'<relation id="{RELATION}" version="1"><member type="way" ref="401" role="outer"/>'
                     '<member type="way" ref="402" role="outer"/><tag k="type" v="multipolygon"/>'
                     '<tag k="building" v="yes"/></relation>')
    node(SUBSTATION[0], *place(*SUBSTATION[1]), {"power": "substation", "voltage": "138000"})
    return '<?xml version="1.0" encoding="UTF-8"?><osm version="0.6">' + "".join(nodes + ways + relations) + "</osm>"


def route_line():
    """Return the made-up mapped distribution line: node 1, an off-street point 40 m west of node 4, node 9."""
    return shapely.LineString([place(*GRID[1]), place(*OFF_STREET), place(*GRID[9])])


def write_site(folder):
    """Write the OpenStreetMap file and the reference GeoPackage of the site to a folder and return their paths."""
    folder.mkdir(parents=True, exist_ok=True)
    extract = folder / "site.osm"
    extract.write_text(osm_text(), encoding="ascii")
    reference = folder / "reference.gpkg"
    frames = {
        "clip": geopandas.GeoDataFrame({"name": ["synthetic"]}, geometry=[site_shape()], crs="EPSG:4326"),
        "substations": geopandas.GeoDataFrame({"osm_type": ["node"], "osm_id": [SUBSTATION[0]], "voltage": ["138000"]},
                                              geometry=[shapely.Point(place(*SUBSTATION[1]))], crs="EPSG:4326"),
        "distribution_lines": geopandas.GeoDataFrame({"osm_id": [501]}, geometry=[route_line()], crs="EPSG:4326"),
    }
    inputs.write_layers(frames, reference)
    return extract, reference


def write_site_file(folder, extract, reference, routes=True):
    """Write a site file for the build command that points at the synthetic inputs, and return its path."""
    lines = ['name = "synthetic"', 'out = "built"', 'substations = "reference"', "",
             "[inputs.extract]", f'path = "{extract.name}"', f'sha256 = "{inputs.file_checksum(extract)}"',
             'url = "made up in the tests"', "",
             "[inputs.reference]", f'path = "{reference.name}"', f'sha256 = "{inputs.file_checksum(reference)}"',
             'url = "made up in the tests"', "",
             "[layers]", 'site = "clip"', 'substations = "substations"']
    if routes:
        lines.append('routes = "distribution_lines"')
    path = folder / "site.toml"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path

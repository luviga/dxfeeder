"""Line sources in one layout.

This module reads the CanVec power lines of the 10 province files and the filtered OpenStreetMap power lines and cables.
It keeps the CanVec lines above ground and under water, with one copy of each feature that two province files share.
It reads the highest value and every value of each OpenStreetMap voltage tag in kV, and the location tag of each way.
It keeps a blank voltage when the source gives none.
It writes both sources to lines.gpkg in one layout, with each shape in EPSG:3347.
"""

from pathlib import Path

import geopandas
import numpy
import pandas
import pyogrio
import shapely

from dxfeeder.inputs import CANVEC_LAYER, CANVEC_PREFIX, OSM_LINES, OSM_NOTE, vector_source, write_gpkg
from dxfeeder.osm_filter import voltage_values

MAP_CRS = "EPSG:3347"
LAYOUT = ["source", "feature_id", "kind", "voltage_kv", "voltage_values_kv", "location"]
# The CanVec domain line_location_cl names 713 aboveground and 714 underwater. Codes 20, not identified, and 731, underground, stay out.
CANVEC_KINDS = {713: "overhead", 714: "submarine"}


def layout_frame(columns, shapes, crs):
    """Return features in the common layout, with each shape as a multi-part line in EPSG:3347."""
    shapes = shapely.multilinestrings(shapely.get_parts(shapes), indices=shapely.get_parts(shapes, return_index=True)[1])
    return geopandas.GeoDataFrame(columns, geometry=shapes, crs=crs)[LAYOUT + ["geometry"]].to_crs(MAP_CRS)


def voltage_columns(tags):
    """Return the highest voltage in kV and the text of every voltage in kV for each tag, both blank for an unread tag."""
    values = [voltage_values(tag) for tag in tags]
    highest = numpy.array([numpy.nan if found is None else max(found) for found in values], dtype=float)
    return highest, [None if found is None else ";".join(f"{value:g}" for value in found) for found in values]


def read_canvec(paths):
    """Return the CanVec power lines above ground and under water, one copy per feature id, in the common layout."""
    names = sorted(name for name in paths if name.startswith(CANVEC_PREFIX))
    canvec = pandas.concat([pyogrio.read_dataframe(vector_source(paths[name]), layer=CANVEC_LAYER) for name in names])
    # A feature in two province files keeps the copy from the province first in alphabetical order.
    canvec = canvec.drop_duplicates("feature_id").sort_values("feature_id")
    canvec = canvec[canvec["line_location"].isin(list(CANVEC_KINDS))]
    blank = [None] * len(canvec)
    columns = {"source": "CanVec", "feature_id": canvec["feature_id"].to_numpy(), "kind": canvec["line_location"].map(CANVEC_KINDS).to_numpy(),
               "voltage_kv": numpy.full(len(canvec), numpy.nan), "voltage_values_kv": blank, "location": blank}
    return layout_frame(columns, canvec.geometry.to_numpy(), canvec.crs)


def read_osm(path):
    """Return the filtered OpenStreetMap power lines and cables, sorted by way id, in the common layout."""
    osm = pyogrio.read_dataframe(path).sort_values("way_id")
    highest, values = voltage_columns(osm["voltage"])
    columns = {"source": "OpenStreetMap", "feature_id": osm["way_id"].astype(str).to_numpy(), "kind": osm["power"].to_numpy(),
               "voltage_kv": highest, "voltage_values_kv": values, "location": osm["location"].to_numpy()}
    return layout_frame(columns, osm.geometry.to_numpy(), osm.crs)


def run_step(root, paths):
    """Run the line layout on the CanVec files and the filtered OpenStreetMap file, write lines.gpkg, and return its change."""
    frame = pandas.concat([read_canvec(paths), read_osm(paths[OSM_LINES])], ignore_index=True)
    return {"outputs/lines.gpkg": write_gpkg(frame[LAYOUT + ["geometry"]], Path(root) / "outputs" / "lines.gpkg", "lines", OSM_NOTE)}

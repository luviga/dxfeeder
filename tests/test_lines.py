"""Tests of Step 3, line sources in one layout.

The sample tests read the CanVec and OpenStreetMap test files around the two test corridors.
The made-up tests write two small CanVec files that share one feature, with each line location code.
A clean case must give one copy of each feature, the kept kinds only, and every shape in EPSG:3347.
Two runs of the step must write the same bytes.
"""

import sqlite3
from pathlib import Path

import geopandas
import numpy
import pyogrio
import shapely

from dxfeeder import inputs, lines, osm_filter

SAMPLE = Path(__file__).resolve().parent / "sample_data"
PATHS = {"canvec_ns": SAMPLE / "canvec_power_lines.gpkg", inputs.OSM_LINES: SAMPLE / "osm_power_lines.gpkg"}


def canvec_file(path, rows):
    """Write a made-up CanVec power line file with the given feature ids and line location codes, and return its path."""
    shapes = [shapely.MultiLineString([[(-63.0 + index * 0.01, 45.0), (-63.0 + index * 0.01, 45.1)]]) for index in range(len(rows))]
    frame = geopandas.GeoDataFrame({"feature_id": [row[0] for row in rows], "line_location": [row[1] for row in rows]},
                                   geometry=shapes, crs="EPSG:4617")
    pyogrio.write_dataframe(frame, path, layer=inputs.CANVEC_LAYER, driver="GPKG")
    return path


def test_read_canvec_made_up(tmp_path):
    """A shared feature keeps one copy, and the codes for not identified and underground stay out."""
    paths = {"canvec_ns": canvec_file(tmp_path / "ns.gpkg", [("a", 713), ("b", 714), ("c", 20), ("d", 731)]),
             "canvec_nb": canvec_file(tmp_path / "nb.gpkg", [("a", 713), ("e", 713)])}
    frame = lines.read_canvec(paths)
    assert frame[["feature_id", "kind"]].values.tolist() == [["a", "overhead"], ["b", "submarine"], ["e", "overhead"]]
    assert frame.crs == lines.MAP_CRS and frame[["voltage_kv", "voltage_values_kv", "location"]].isna().all().all()


def test_read_canvec_sample():
    """The CanVec test file gives one row per feature, with multi-part lines in EPSG:3347 and no voltage."""
    frame = lines.read_canvec(PATHS)
    assert len(frame) == 1396 and frame["feature_id"].is_unique
    assert set(frame["kind"]) <= {"overhead", "submarine"} and set(frame.geom_type) == {"MultiLineString"}
    assert frame[["voltage_kv", "voltage_values_kv", "location"]].isna().all().all() and (frame["source"] == "CanVec").all()


def test_read_osm_sample():
    """The OpenStreetMap test file gives each way once, with its power tag as the kind and its voltage tag in kV."""
    raw = pyogrio.read_dataframe(PATHS[inputs.OSM_LINES])
    frame = lines.read_osm(PATHS[inputs.OSM_LINES])
    assert len(frame) == len(raw) and frame["feature_id"].tolist() == [str(way) for way in sorted(raw["way_id"])]
    expected = {str(way): osm_filter.voltage_kv(tag) for way, tag in zip(raw["way_id"], raw["voltage"])}
    assert all((expected[way] is None and voltage != voltage) or expected[way] == voltage
               for way, voltage in zip(frame["feature_id"], frame["voltage_kv"]))
    assert (frame["voltage_kv"].isna() == frame["voltage_values_kv"].isna()).all()
    locations = dict(zip(raw["way_id"].astype(str), raw["location"]))
    assert [locations[way] for way in frame["feature_id"]] == frame["location"].tolist()
    assert set(frame.geom_type) == {"MultiLineString"} and frame.crs == lines.MAP_CRS


def test_voltage_columns():
    """Each tag gives its highest value and every value in kV, and an unread or missing tag leaves both columns blank."""
    highest, values = lines.voltage_columns(["138000;69000", "12500", "345000;138000;", None])
    assert highest[:2].tolist() == [138.0, 12.5] and numpy.isnan(highest[2:]).all()
    assert values == ["138;69", "12.5", None, None]


def test_run_step_twice(tmp_path):
    """The step writes lines.gpkg in the common layout with the OpenStreetMap note, and a second run changes no byte."""
    assert lines.run_step(tmp_path, PATHS) == {"outputs/lines.gpkg": True}
    path = tmp_path / "outputs" / "lines.gpkg"
    content = path.read_bytes()
    assert lines.run_step(tmp_path, PATHS) == {"outputs/lines.gpkg": False}
    assert path.read_bytes() == content
    info = pyogrio.read_info(path)
    assert list(info["fields"]) == lines.LAYOUT and info["crs"] == lines.MAP_CRS and info["geometry_type"] == "MultiLineString"
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT description FROM gpkg_contents").fetchone()[0] == inputs.OSM_NOTE

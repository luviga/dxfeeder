"""Tests of the route shapes module.

The made-up tests place short lines and features on a flat stretch of the map in EPSG:3347.
They cover each reason for no shape, the voltage filter, the join of close ends, the snap, the tie rule, and the submarine length.
They also cover each outcome of the snap retry.
The sample test builds the circuit table and the line layout from the test files around the two test corridors.
Two runs of the step must write the same bytes, and every shape must end within the snap limit.
"""

import shutil
from pathlib import Path

import geopandas
import networkx
import pyogrio
import pytest
import shapely

from dxfeeder import inputs, lines, shapes
from dxfeeder.inputs import InputError

SAMPLE = Path(__file__).resolve().parent / "sample_data"
PROJECT = Path(__file__).resolve().parents[1]
PATHS = {inputs.CENSUS: SAMPLE / "census_divisions.gpkg", "canvec_ns": SAMPLE / "canvec_power_lines.gpkg",
         inputs.OSM_LINES: SAMPLE / "osm_power_lines.gpkg",
         **{name: SAMPLE / file for name, file in inputs.CODERS_FILES.items()}}
SETTINGS = {"search_area_margin_km": 50.0, "end_join_m": 5.0, "snap_limit_km": 2.0, "max_equal_routes": 100, "detour_flag": 1.5,
            "snap_retry_m": 100.0, "snap_retry_ratio_floor": 0.95}
ORIGIN = shapes.TO_MAP.transform(-63.6, 44.6)


def spot(dx, dy):
    """Return the map position at an offset in meters from the made-up origin near Halifax."""
    return (ORIGIN[0] + dx, ORIGIN[1] + dy)


def feature(feature_id, parts, source="CanVec", values=None, kind="overhead", location=None):
    """Return one made-up feature of the line layout, with each part given as offsets from the origin."""
    shape = shapely.MultiLineString([[spot(*offset) for offset in part] for part in parts])
    return {"source": source, "feature_id": feature_id, "kind": kind, "voltage_kv": None, "voltage_values_kv": values,
            "location": location, "geometry": shape}


def network(rows, source="CanVec"):
    """Return the network of one source from made-up features."""
    frame = geopandas.GeoDataFrame(rows, geometry="geometry", crs=lines.MAP_CRS)
    return shapes.source_network(frame, source, SETTINGS["end_join_m"])


def made_up_line(a, b, voltages=""):
    """Return a made-up line between two offsets from the origin, with a voltage set given as text."""
    a, b = spot(*a), spot(*b)
    fields = {"line_group": "ALBERTA", "corridor_id": None, "circuit_id": None, "part": None, "record_ids": "1",
              "voltage_kv": None, "voltage_values_kv": voltages}
    return {"fields": fields, "voltages": shapes.voltage_set(voltages), "codes": ("A", "B"), "a": a, "b": b,
            "straight_km": shapes.ellipsoid_m(*a, *b) / 1000.0}


def map_km(a, b):
    """Return the ellipsoid distance in km between two offsets from the origin."""
    return shapes.ellipsoid_m(*spot(*a), *spot(*b)) / 1000.0


def route(rows, a, b, source="CanVec", voltages=""):
    """Return the values and the shape of a made-up line on made-up features."""
    return shapes.route_shape(network(rows, source), made_up_line(a, b, voltages), SETTINGS, "the made-up line")


@pytest.mark.parametrize("rows, ends, reason", [
    ([feature("f1", [[(200000, 0), (201000, 0)]])], ((0, 0), (1000, 0)), shapes.NO_FEATURE),
    ([feature("f1", [[(0, 3000), (1000, 3000)]])], ((0, 0), (1000, 0)), shapes.BEYOND_LIMIT),
    ([feature("f1", [[(0, 0), (1000, 0)]])], ((-300, 10), (-200, -10)), shapes.SAME_POINT),
    ([feature("f1", [[(0, 0), (400, 0)]]), feature("f2", [[(600, 0), (1000, 0)]])], ((0, 0), (1000, 0)), shapes.NO_CONNECTION),
])
def test_no_shape_reasons(rows, ends, reason):
    """Each reason for no shape gives an empty shape, and the snap distances stay blank only before the snap runs."""
    values, shape = route(rows, *ends)
    assert (values["status"], values["reason"], shape) == ("NO SHAPE", reason, None)
    assert ("snap_a_km" in values) is (reason != shapes.NO_FEATURE) and "length_km" not in values


def test_voltage_filter():
    """A feature with a blank voltage stays, a feature with a matching second value stays, and a feature with no match goes."""
    rows = [feature("w1", [[(0, 0), (100, 0)]], "OpenStreetMap"), feature("w2", [[(0, 50), (100, 50)]], "OpenStreetMap", "69;138"),
            feature("w3", [[(0, 90), (100, 90)]], "OpenStreetMap", "230")]
    found = network(rows, "OpenStreetMap")
    assert shapes.kept_features(found, made_up_line((0, 0), (100, 0), "138;500"), 50000.0).tolist() == [0, 1]
    assert shapes.kept_features(found, made_up_line((0, 0), (100, 0)), 50000.0).tolist() == [0, 1, 2]
    canvec = network([feature("c1", [[(0, 0), (100, 0)]])])
    assert shapes.kept_features(canvec, made_up_line((0, 0), (100, 0), "138"), 50000.0).tolist() == [0]


@pytest.mark.parametrize("gap, status", [(3.0, "SHAPE"), (6.0, "NO SHAPE")])
def test_join_close_ends(gap, status):
    """Two parts with ends 3 m apart join into one route, and two parts with ends 6 m apart do not."""
    rows = [feature("f1", [[(0, 0), (500, 0)]]), feature("f2", [[(500 + gap, 0), (1000, 0)]])]
    values, _ = route(rows, (0, 0), (1000, 0))
    assert values["status"] == status
    if status == "SHAPE":
        assert values["feature_ids"] == "f1;f2" and values["length_km"] == pytest.approx(map_km((0, 0), (1000, 0)), abs=1e-6)
        assert values["largest_join_m"] == pytest.approx(map_km((500, 0), (500 + gap, 0)) * 1000.0, abs=1e-6)


def test_snap_splits_segment():
    """A snap point inside a segment splits it, and the route starts and ends at the two snap points."""
    rows = [feature("f1", [[(0, 0), (1000, 0), (1000, 1000)]])]
    line = made_up_line((50, 100), (1100, 900))
    found = network(rows)
    values, shape = shapes.route_shape(found, line, SETTINGS, "the made-up line")
    segments = shapes.feature_segments(found, shapes.kept_features(found, line, 50000.0))
    snaps = [shapes.snap(found, segments, point) for point in (line["a"], line["b"])]
    assert shape.coords[0] == snaps[0]["point"] and shape.coords[-1] == snaps[1]["point"]
    assert snaps[0]["point"] == pytest.approx(spot(50, 0)) and snaps[1]["point"] == pytest.approx(spot(1000, 900))
    assert list(shape.coords) == [snaps[0]["point"], spot(1000, 0), snaps[1]["point"]]
    assert values["snap_a_km"] == pytest.approx(map_km((50, 100), (50, 0)), abs=1e-6)
    assert values["length_km"] == pytest.approx(map_km((50, 0), (1000, 0)) + map_km((1000, 0), (1000, 900)), abs=1e-6)


def test_best_route_tie():
    """Two routes of equal length give the route with the smaller sorted list of feature ids."""
    graph = networkx.Graph()
    graph.add_edge("a", "top", weight=1.0, feature=1)
    graph.add_edge("top", "b", weight=1.0, feature=1)
    graph.add_edge("a", "low", weight=1.0, feature=0)
    graph.add_edge("low", "b", weight=1.0, feature=None)
    assert shapes.best_route(graph, "a", "b", ["x1", "x2"], "the made-up line", 100) == ["a", "low", "b"]


def test_best_route_too_many():
    """More than 100 shortest routes of equal length stop the run with an input error."""
    graph = networkx.Graph()
    for step in range(7):
        for side in ("up", "down"):
            graph.add_edge(step, (step, side), weight=1.0, feature=None)
            graph.add_edge((step, side), step + 1, weight=1.0, feature=None)
    with pytest.raises(InputError, match="more than 100"):
        shapes.best_route(graph, 0, 7, [], "the made-up line", 100)


@pytest.mark.parametrize("source, marks", [("CanVec", {"kind": "submarine"}), ("OpenStreetMap", {"location": "underwater"})])
def test_submarine_length(source, marks):
    """The submarine length counts the route length on the submarine feature only."""
    rows = [feature("f1", [[(0, 0), (400, 0)]], source), feature("f2", [[(400, 0), (1000, 0)]], source, **marks)]
    values, _ = route(rows, (0, 0), (1000, 0), source)
    assert values["submarine_km"] == pytest.approx(map_km((400, 0), (1000, 0)), abs=1e-9)
    assert values["length_km"] == pytest.approx(map_km((0, 0), (1000, 0)), abs=1e-6)


def retried(rows, a, b, retry_m=100.0):
    """Return the values and the shape of a made-up CanVec line after the snap retry, with a given retry distance."""
    settings = {**SETTINGS, "snap_retry_m": retry_m}
    return shapes.retried_shape(network(rows), made_up_line(a, b), settings, "the made-up line")


def test_retry_parallel_line():
    """An end node next to two parallel lines gets the short route on the line that the route arrives on."""
    rows = [feature("f1", [[(0, 0), (1000, 0), (3000, 0)]]), feature("f2", [[(500, 40), (2990, 40), (3000, 3)]])]
    first, _ = route(rows, (0, 0), (1000, 36))
    assert first["feature_ids"] == "f1;f2" and first["detour_ratio"] > 4.0
    values, shape = retried(rows, (0, 0), (1000, 36))
    assert (values["snap_retry"], values["status"], values["feature_ids"], values.get("reason")) == (shapes.KEPT, "SHAPE", "f1", None)
    assert values["length_km"] == pytest.approx(map_km((0, 0), (1000, 0)), abs=1e-6)
    assert values["snap_b_km"] == pytest.approx(map_km((1000, 36), (1000, 0)), abs=1e-6)
    assert values["straight_km"] == first["straight_km"] and shape.coords[-1] == pytest.approx(spot(1000, 0))


@pytest.mark.parametrize("retry_m, outcome", [(100.0, shapes.KEPT), (10.0, shapes.NOT_KEPT)])
def test_retry_no_connection(retry_m, outcome):
    """A line with no connection gets a shape from a feature within the retry distance, and keeps no shape without one."""
    rows = [feature("f1", [[(0, 0), (1000, 0)]]), feature("f2", [[(900, 20), (1100, 20)]])]
    first, _ = route(rows, (0, 0), (1000, 15))
    assert first["reason"] == shapes.NO_CONNECTION
    values, shape = retried(rows, (0, 0), (1000, 15), retry_m)
    assert values["snap_retry"] == outcome
    if outcome == shapes.KEPT:
        assert values["feature_ids"] == "f1" and values["detour_ratio"] < 1.0 and shape is not None
    else:
        assert values == {**first, "snap_retry": outcome} and shape is None


def test_retry_no_gain():
    """A long detour with no shorter route keeps the first result, and a route under the flag value gets no retry."""
    rows = [feature("f1", [[(0, 0), (0, 1000), (1000, 1000), (1000, 0)]])]
    first, first_shape = route(rows, (0, 0), (1000, 0))
    values, shape = retried(rows, (0, 0), (1000, 0))
    assert first["detour_ratio"] > 1.5 and values == {**first, "snap_retry": shapes.NOT_KEPT} and shape.equals(first_shape)
    values, _ = retried(rows, (0, 0), (0, 1000))
    assert values["snap_retry"] == shapes.NOT_SELECTED and values["detour_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_retry_tie(monkeypatch):
    """On equal retry lengths, the smaller sorted list of feature ids wins, and on a full tie the first pair in feature id order wins."""
    ends = {spot(0, 0): [{"point": "a1"}, {"point": "a2"}], spot(1000, 0): [{"point": "b1"}, {"point": "b2"}]}
    results = {("a1", "b1"): (1.0, "f2;f3", "first"), ("a1", "b2"): (1.0, "f1;f9", "second"), ("a2", "b1"): (1.0, "f1;f9", "third"),
               ("a2", "b2"): None}
    first = {"status": "NO SHAPE", "straight_km": 1.0, "reason": shapes.NO_CONNECTION, "snap_a_km": 0.0, "snap_b_km": 0.0}

    def made_up_route(network, kept, rows, snaps, line, name, limit):
        """Return the made-up route of one pair of made-up snap points, or None."""
        found = results[snaps[0]["point"], snaps[1]["point"]]
        return found and ({"status": "SHAPE", "length_km": found[0], "detour_ratio": found[0], "feature_ids": found[1]}, found[2])
    monkeypatch.setattr(shapes, "route_shape", lambda *arguments: (first, None))
    monkeypatch.setattr(shapes, "retry_snaps", lambda network, kept, point, distance: ends[point])
    monkeypatch.setattr(shapes, "pair_route", made_up_route)
    values, shape = retried([feature("f1", [[(0, 0), (1000, 0)]])], (0, 0), (1000, 0))
    assert (values["snap_retry"], values["feature_ids"], shape) == (shapes.KEPT, "f1;f9", "second")


@pytest.mark.parametrize("ratio, outcome", [(0.5, shapes.NOT_KEPT), (0.95, shapes.KEPT)])
def test_retry_ratio_floor(monkeypatch, ratio, outcome):
    """A retry route with a detour ratio below the floor gives no shape, and a route at the floor stays."""
    first = {"status": "NO SHAPE", "straight_km": 1.0, "reason": shapes.SAME_POINT, "snap_a_km": 0.0, "snap_b_km": 0.0}
    found = {"status": "SHAPE", "length_km": ratio, "detour_ratio": ratio, "feature_ids": "f1"}
    monkeypatch.setattr(shapes, "route_shape", lambda *arguments: (first, None))
    monkeypatch.setattr(shapes, "retry_snaps", lambda network, kept, point, distance: [{"point": point}])
    monkeypatch.setattr(shapes, "pair_route", lambda *arguments: (found, "made-up shape"))
    values, shape = retried([feature("f1", [[(0, 0), (1000, 0)]])], (0, 0), (1000, 0))
    assert values["snap_retry"] == outcome and values["status"] == ("SHAPE" if outcome == shapes.KEPT else "NO SHAPE")
    assert shape == ("made-up shape" if outcome == shapes.KEPT else None)


def write_line_tables(root, alberta_rows):
    """Write a circuit table with no rows and an Alberta table with the given rows to the outputs folder."""
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "circuits.csv").write_text(circuits.table_text(circuits.CIRCUIT_COLUMNS, []), encoding="ascii")
    text = circuits.table_text(circuits.ALBERTA_COLUMNS, alberta_rows)
    (root / "outputs" / "alberta_lines.csv").write_text(text, encoding="ascii")


@pytest.mark.skip(reason="needs the circuit table module and filled settings, which this package does not hold yet")
def test_run_step_same_node(tmp_path):
    """A line whose two end node codes are equal gets no row, and every other line gets one row per source."""
    shutil.copy(PROJECT / "settings.toml", tmp_path / "settings.toml")
    place = {"longitude_a": "-63.6", "latitude_a": "44.6", "longitude_b": "-63.59", "latitude_b": "44.6"}
    write_line_tables(tmp_path, [{"record_id": "7", "circuit_id": "X1", "voltage_kv": "138", "node_a": "N1", "node_b": "N2", **place},
                                 {"record_id": "8", "circuit_id": "X2", "voltage_kv": "138", "node_a": "N1", "node_b": "N1", **place}])
    rows = [feature("f1", [[(0, 0), (1000, 0)]]), feature("w1", [[(0, 0), (1000, 0)]], "OpenStreetMap", "138")]
    frame = geopandas.GeoDataFrame(rows, geometry="geometry", crs=lines.MAP_CRS)
    inputs.write_gpkg(frame, tmp_path / "outputs" / "lines.gpkg", "lines", inputs.OSM_NOTE)
    assert shapes.run_step(tmp_path, {}) == {"outputs/route_shapes.gpkg": True}
    found = pyogrio.read_dataframe(tmp_path / "outputs" / "route_shapes.gpkg")
    assert found[["record_ids", "source", "status"]].values.tolist() == [["7", "CanVec", "SHAPE"], ["7", "OpenStreetMap", "SHAPE"]]
    assert found.columns.tolist() == shapes.COLUMNS + ["geometry"]
    assert found["snap_retry"].tolist() == [shapes.NOT_SELECTED, shapes.NOT_SELECTED]


def test_voltage_set_bad():
    """A voltage text that does not read as numbers must stop with an input error."""
    with pytest.raises(InputError, match="does not read as numbers"):
        shapes.voltage_set("138;high")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Return a temporary project with the circuit table and the line layout on the test files and the project settings with no link leg."""
    root = tmp_path_factory.mktemp("shapes")
    # The test files hold no Labrador link, so the test settings name no link leg.
    text = (PROJECT / "settings.toml").read_text(encoding="utf-8")
    (root / "settings.toml").write_text(text.replace("[38509, 38510, 38512, 38513]", "[]"), encoding="utf-8")
    for step in (circuits.run_step, lines.run_step):
        step(root, PATHS)
    return root


@pytest.mark.skip(reason="needs the circuit table module, filled settings and the CODERS sample files, which this package does not hold yet")
def test_run_step_sample(built):
    """The step gives one row per circuit row and per source, every shape ends within the snap limit, and a second run keeps the bytes."""
    assert shapes.run_step(built, PATHS) == {"outputs/route_shapes.gpkg": True}
    content = (built / "outputs" / "route_shapes.gpkg").read_bytes()
    frame = pyogrio.read_dataframe(built / "outputs" / "route_shapes.gpkg")
    assert len(frame) == 20 and frame["source"].tolist() == list(shapes.SOURCES) * 10
    found = frame[frame["status"] == "SHAPE"]
    assert len(found) and (found[["snap_a_km", "snap_b_km"]] <= SETTINGS["snap_limit_km"]).all().all()
    assert frame.loc[frame["status"] == "NO SHAPE", "geometry"].isna().all()
    assert shapes.run_step(built, PATHS) == {"outputs/route_shapes.gpkg": False}
    assert (built / "outputs" / "route_shapes.gpkg").read_bytes() == content

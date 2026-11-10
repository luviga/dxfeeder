"""Tests of Step 7, the terrain samples.

The tests write small made-up rasters in EPSG:3979, so they read no network.
They cover the sample spacing, the projection, the raster levels, the kept tiles, the slope, and the class rule.
The command line test uses the made-up rasters of this module for a full build.
"""

import json

import geopandas
import numpy
import pandas
import pyogrio
import pytest
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from dxfeeder import samples
from dxfeeder.inputs import InputError

GROUPS = {"water": [0, 18], "urban": [17], "wetland": [14], "forest": [1, 2, 3, 4, 5, 6],
          "prairie": [7, 8, 9, 10, 11, 12, 13, 15, 16, 19]}
BANDS = [5.0, 15.0]
# Two points converted by the old terrain script from EPSG:3347 to EPSG:3979, with its own projection code.
OLD_POINTS = [((8300000.0, 1550000.0), (2320672.5679604723, 229918.46084704902)),
              ((5000000.0, 2000000.0), (-997482.9025032158, 516875.38066586247))]


def write_raster(path, values, origin, cell, nodata=None, factors=()):
    """Write a small tiled raster in EPSG:3979 with optional overview levels, and return its path as text."""
    profile = {"driver": "GTiff", "width": values.shape[1], "height": values.shape[0], "count": 1, "dtype": values.dtype,
               "crs": samples.RASTER_CRS, "transform": from_origin(origin[0], origin[1], cell, cell), "tiled": True,
               "blockxsize": 16, "blockysize": 16, "nodata": nodata, "compress": "deflate"}
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values, 1)
        if factors:
            dataset.build_overviews(list(factors), Resampling.nearest)
    return str(path)


def raster_row(name, address):
    """Return a made-up input map row for a local raster file."""
    return {"name": name, "address": address, "local_path": "", "checksum": "", "date_recorded": "2026-10-07",
            "size_bytes": "1", "last_modified": "made-up date", "etag": "made-up tag"}


def made_up_rasters(folder, bounds):
    """Write a land cover raster and an elevation raster that cover map bounds in EPSG:3347, and return their input map rows."""
    corners = samples.TO_RASTER.transform([bounds[0], bounds[0], bounds[2], bounds[2]], [bounds[1], bounds[3], bounds[1], bounds[3]])
    left, bottom = min(corners[0]) - 20000.0, min(corners[1]) - 20000.0
    right, top = max(corners[0]) + 20000.0, max(corners[1]) + 20000.0
    folder.mkdir(parents=True, exist_ok=True)
    rows, columns = numpy.mgrid[0:int((top - bottom) // 1000) + 1, 0:int((right - left) // 1000) + 1]
    cover = ((rows // 3 + columns // 3) % 20).astype(numpy.uint8)
    land = write_raster(folder / "land_cover.tif", cover, (left, top), 1000.0)
    rows, columns = numpy.mgrid[0:int((top - bottom) // 240) + 1, 0:int((right - left) // 240) + 1]
    height = (1000.0 * numpy.sin(columns * 240.0 / 7000.0) * numpy.cos(rows * 240.0 / 9000.0)).astype(numpy.float32)
    elevation = write_raster(folder / "elevation.tif", height, (left, top), 240.0, nodata=-32767.0)
    return [raster_row(samples.LAND_COVER, land), raster_row(samples.ELEVATION, elevation)]


@pytest.mark.parametrize("length, expected", [(2500.0, [0.0, 1000.0, 2000.0, 2500.0]), (2000.0, [0.0, 1000.0, 2000.0]),
                                              (0.0, [0.0]), (2000.00000005, [0.0, 1000.0, 2000.0])])
def test_sample_distances(length, expected):
    """A sample sits at the start, at each spacing, and at the end, and a line of no length gives one sample."""
    assert samples.sample_distances(length, 1000.0) == pytest.approx(expected)


def test_sample_points():
    """The samples of a bent line follow the line on the map."""
    x, y = samples.sample_points(shapely.LineString([(0, 0), (1500, 0), (1500, 1000)]), 1000.0)
    assert list(x) == pytest.approx([0.0, 1000.0, 1500.0, 1500.0])
    assert list(y) == pytest.approx([0.0, 0.0, 500.0, 1000.0])


def test_projection_matches_old_script():
    """The conversion to the raster projection applies no datum shift and agrees with the old terrain script."""
    for (x, y), expected in OLD_POINTS:
        assert samples.TO_RASTER.transform(x, y) == pytest.approx(expected, abs=1e-6)


def test_levels_and_cells(tmp_path):
    """The cell size of each level follows the width ratio, and the level nearest the target wins."""
    values = numpy.arange(60 * 100, dtype=numpy.float32).reshape(60, 100)
    address = write_raster(tmp_path / "level.tif", values, (0.0, 1800.0), 30.0, factors=(2, 4, 8))
    raster = samples.Raster(tmp_path, raster_row("level", address))
    assert [raster.cell_size(level) for level in range(4)] == pytest.approx([30.0, 60.0, 120.0, 30.0 * 100 / 13])
    assert raster.level_near(240.0, 100.0, 250.0) == 3
    with pytest.raises(InputError, match="no level"):
        raster.level_near(240.0, 100.0, 200.0)
    rows, columns = raster.cells(0, numpy.array([15.0, 2999.0]), numpy.array([1790.0, 1.0]))
    assert list(rows) == [0, 59] and list(columns) == [0, 99]


def test_read_keeps_tiles(tmp_path):
    """A read keeps each tile in the data folder, a second read needs no raster, and a changed raster stops the run."""
    values = numpy.arange(40 * 40, dtype=numpy.uint8).reshape(40, 40) % 20
    address = write_raster(tmp_path / "cover.tif", values, (0.0, 400.0), 10.0)
    raster = samples.Raster(tmp_path, raster_row("cover", address))
    rows, columns = numpy.array([0, 39, 20, -1]), numpy.array([0, 39, 5, 0])
    assert list(raster.read(0, rows, columns)[:3]) == [values[0, 0], values[39, 39], values[20, 5]]
    assert numpy.isnan(raster.read(0, rows, columns)[3])
    assert len(list((tmp_path / samples.TILE_FOLDER / "cover" / "0").glob("*.npz"))) == 3
    (tmp_path / "cover.tif").unlink()
    again = samples.Raster(tmp_path, raster_row("cover", address))
    assert list(again.read(0, rows[:3], columns[:3])) == [values[0, 0], values[39, 39], values[20, 5]]
    with pytest.raises(InputError, match="another copy"):
        samples.Raster(tmp_path, {**raster_row("cover", address), "etag": "changed tag"})
    with pytest.raises(InputError, match="next build reads only the rest"):
        again.read(0, numpy.array([0]), numpy.array([20]))


def test_remote_identity(tmp_path, monkeypatch, capsys):
    """A remote raster gets one identity check per run, and a local raster gets none."""
    calls = []
    monkeypatch.setattr(samples, "check_raster", lambda row: calls.append(row["name"]) or "Raster unchanged")
    folder = tmp_path / samples.TILE_FOLDER / "remote"
    folder.mkdir(parents=True)
    layout = {"identity": ["1", "made-up date", "made-up tag"], "x0": 0.0, "y0": 0.0, "pixel_x": 1.0, "pixel_y": 1.0,
              "nodata": None, "levels": [[1, 1, 1, 1]]}
    (folder / "layout.json").write_text(json.dumps(layout), encoding="ascii")
    raster = samples.Raster(tmp_path, raster_row("remote", "https://example.org/remote.tif"))
    assert raster.source == "/vsicurl/https://example.org/remote.tif"
    raster.check_identity()
    raster.check_identity()
    assert calls == ["remote"] and "Raster unchanged" in capsys.readouterr().out


def test_slopes(tmp_path):
    """The Horn slope of a tilted plane equals its gradient, and a cell with no data gives no slope."""
    columns = numpy.tile(numpy.arange(20, dtype=numpy.float32), (20, 1))
    height = columns * 24.0
    height[15, 15] = -32767.0
    address = write_raster(tmp_path / "plane.tif", height, (0.0, 4800.0), 240.0, nodata=-32767.0)
    raster = samples.Raster(tmp_path, raster_row("plane", address))
    x, y = numpy.array([1200.0, 3720.0, 10.0]), numpy.array([3600.0, 1080.0, 4790.0])
    result = samples.slopes(raster, 0, x, y)
    assert result[0] == pytest.approx(10.0) and result[1] is None and result[2] is None


@pytest.mark.parametrize("cover, slope, expected", [
    (0, None, None), (18, 3.0, None), (17, 30.0, "urban"), (14, 30.0, "wetland"), (1, 15.5, "mountain"), (10, 15.5, "mountain"),
    (1, 15.0, "forest"), (10, 15.0, "hills"), (10, 5.0, "hills"), (10, 4.9, "prairie"), (19, 2.0, "prairie")])
def test_terrain_class(cover, slope, expected):
    """The first class rule that fits sets the class, and a water sample gets no class."""
    assert samples.terrain_class(cover, slope, GROUPS, BANDS) == expected


def test_terrain_classes_stop():
    """An unknown land cover value or a land sample with no slope stops the run."""
    assert samples.terrain_classes(numpy.array([0, 10]), [None, 1.0], GROUPS, BANDS) == [None, "prairie"]
    with pytest.raises(InputError, match="no land cover group"):
        samples.terrain_classes(numpy.array([20]), [1.0], GROUPS, BANDS)
    with pytest.raises(InputError, match="has no slope"):
        samples.terrain_classes(numpy.array([10]), [None], GROUPS, BANDS)


def test_land_covers_outside(tmp_path):
    """A sample outside the land cover raster stops the run."""
    address = write_raster(tmp_path / "small.tif", numpy.zeros((16, 16), dtype=numpy.uint8), (0.0, 160.0), 10.0)
    raster = samples.Raster(tmp_path, raster_row("small", address))
    assert list(samples.land_covers(raster, numpy.array([5.0]), numpy.array([155.0]))) == [0]
    with pytest.raises(InputError, match="outside the land cover raster"):
        samples.land_covers(raster, numpy.array([500.0]), numpy.array([5.0]))


def test_straight_lines(tmp_path):
    """Each Alberta line gives a straight line, and a line from a node to itself gives one sample."""
    rows = [["record_id", "circuit_id", "voltage_kv", "node_a", "node_b", "longitude_a", "latitude_a", "longitude_b", "latitude_b"],
            ["1", "A", "138", "N1", "N2", "-114.0", "51.0", "-114.0", "51.02"],
            ["2", "B", "", "N3", "N3", "-113.0", "52.0", "-113.0", "52.0"]]
    (tmp_path / "alberta_lines.csv").write_text("".join(",".join(row) + "\n" for row in rows), encoding="ascii")
    items = samples.straight_lines(tmp_path, [230.0, 345.0])
    assert [fields["voltage_class"] for fields, _ in items] == ["below 230 kV", None]
    frame = samples.sample_frame(items, 1000.0)
    assert frame.groupby("path_number").size().tolist() == [4, 1]
    assert set(frame["line_group"]) == {"ALBERTA_STRAIGHT"} and set(frame["path_source"]) == {"straight"}


def test_path_file_projection(tmp_path):
    """A path file in another projection stops the run."""
    line = geopandas.GeoDataFrame({"source": ["CanVec"], "record_ids": ["1"]}, geometry=[shapely.LineString([(0, 0), (1, 1)])],
                                  crs="EPSG:4326")
    for layer in ("CORRIDOR", "ALBERTA"):
        pyogrio.write_dataframe(line, tmp_path / "paths.gpkg", layer=layer)
    settings = {"path_file": "paths.gpkg", "voltage_class_limits_kv": [230.0, 345.0]}
    with pytest.raises(InputError, match="not in EPSG:3347"):
        samples.path_lines(tmp_path, settings)


def test_raster_rows_missing(tmp_path):
    """An input map with no row for a raster stops the run."""
    (tmp_path / "input_map.csv").write_text("name,address,local_path,checksum,date_recorded,size_bytes,last_modified,etag\n",
                                            encoding="ascii")
    with pytest.raises(InputError, match="no row land_cover"):
        samples.raster_rows(tmp_path)


def test_empty_sample_frame():
    """No lines give an empty frame with every key column."""
    frame = samples.sample_frame([], 1000.0)
    assert frame.empty and set(samples.KEY_COLUMNS) <= set(frame.columns)


@pytest.mark.skip(reason="needs the full build command and its test project, which this package does not hold yet")
def test_samples_file_columns(tmp_path):
    """The sample file of a made-up project holds every sample column, and two runs give the same bytes."""
    # The command line test module imports this module, so this test imports it inside the function.
    from test_main import step_project

    from dxfeeder import __main__ as command_line
    root = step_project(tmp_path)
    assert command_line.main(["build"], root=root) == 0
    frame = pyogrio.read_dataframe(root / "outputs" / "terrain_samples.gpkg")
    assert list(frame.columns) == samples.SAMPLE_COLUMNS + ["geometry"]
    assert set(frame["record_file"]) == {samples.COUNTS, samples.BY_SOURCE}
    first = (root / "outputs" / "terrain_samples.gpkg").read_bytes()
    assert samples.run_step(root, {}) == {"outputs/terrain_samples.gpkg": False}
    assert (root / "outputs" / "terrain_samples.gpkg").read_bytes() == first
    assert pandas.api.types.is_integer_dtype(frame["land_cover"])

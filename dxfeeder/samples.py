"""Terrain samples along mapped power lines.

This module reads the path file, alberta_lines.csv, route_shapes.gpkg, the two rasters, and the settings file.
It places a sample at the start of each line, then one at each sample spacing on the map, and one more at the end.
It reads one land cover value and a 3 by 3 block of elevation cells at each sample, and computes the slope.
It gives each land sample a terrain class and keeps each raster tile that it reads in the data folder.
It writes terrain_samples.gpkg with the samples of the paths, the straight Alberta lines, and the route shapes.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas
import numpy
import pandas
import pyogrio
import rasterio
import rasterio.errors
import shapely
from pyproj import Transformer
from rasterio.windows import Window

from dxfeeder.inputs import OSM_NOTE, RASTER_KEYS, InputError, check_raster, read_input_map, read_settings, write_gpkg
from dxfeeder.lines import MAP_CRS
from dxfeeder.shapes import TO_MAP

RASTER_CRS = "EPSG:3979"
TO_RASTER = Transformer.from_crs(MAP_CRS, RASTER_CRS, always_xy=True)
LAND_COVER, ELEVATION = "land_cover", "mrdem_30_dtm"
TILE_FOLDER = Path("data", "tiles")
# The old terrain script reads the tie point as the corner of the first cell, so the reader ignores the point flag of a raster.
GDAL_OPTIONS = {"GTIFF_POINT_GEO_IGNORE": "YES", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "5",
                "GDAL_HTTP_RETRY_DELAY": "2"}
# Parallel reads only shorten the first build. The number of readers changes no value.
READERS = 16
# The old terrain script treats a last spacing point this close to the end, in meters, as the end itself.
END_TOLERANCE_M = 1e-7
NODATA_TOLERANCE = 1e-6
COUNTS, BY_SOURCE = "terrain_counts", "terrain_counts_by_source"
GROUP_NAMES = {"CORRIDOR": "CORRIDOR", "ALBERTA": "ALBERTA_ROUTE"}
KEY_COLUMNS = ["record_file", "path_number", "line_group", "corridor_id", "circuit_id", "part", "record_ids", "voltage_class",
               "path_source", "snap_retry"]
SAMPLE_COLUMNS = KEY_COLUMNS + ["sample_number", "land_cover", "slope_percent", "terrain_class"]


def class_names(limits):
    """Return the three voltage class names from the two class limits in kV."""
    low, high = (format(value, "g") for value in limits)
    return [f"below {low} kV", f"{low} kV to {high} kV", f"above {high} kV"]


def voltage_class(voltage, limits):
    """Return the voltage class of a voltage in kV, with both limits inside the middle class, or None for no voltage."""
    if voltage is None or pandas.isna(voltage):
        return None
    names = class_names(limits)
    return names[0] if voltage < limits[0] else names[1] if voltage <= limits[1] else names[2]


def sample_distances(length, spacing):
    """Return the distances of the samples along a line of a given map length: the start, each spacing, and the end."""
    distances = list(numpy.arange(0.0, length, spacing))
    if not distances or not math.isclose(distances[-1], length, abs_tol=END_TOLERANCE_M):
        distances.append(length)
    return distances


def sample_points(shape, spacing):
    """Return the x and the y values in EPSG:3347 of the samples along one line, with the spacing measured on the map."""
    points = shapely.line_interpolate_point(shape, sample_distances(shape.length, spacing))
    return shapely.get_x(points), shapely.get_y(points)


def key_fields(record_file, group, row, source, limits):
    """Return the key fields of one line for the sample file, with the voltage class from the CODERS voltage."""
    part = row.get("part")
    return {"record_file": record_file, "line_group": group, "corridor_id": row.get("corridor_id"),
            "circuit_id": row.get("circuit_id"), "part": None if part is None or pandas.isna(part) else int(part),
            "record_ids": str(row["record_ids"]), "voltage_class": voltage_class(row.get("voltage_kv"), limits),
            "path_source": source, "snap_retry": row.get("snap_retry")}


def path_lines(root, settings):
    """Return the key fields and the shape of each path in the path file, and stop the run on a path file in another projection."""
    path, limits, items = Path(root) / settings["path_file"], settings["voltage_class_limits_kv"], []
    for layer, group in GROUP_NAMES.items():
        frame = pyogrio.read_dataframe(path, layer=layer)
        if frame.crs is None or not frame.crs.equals(MAP_CRS):
            raise InputError(f"the layer {layer} of the path file is not in {MAP_CRS}, the projection of the sample spacing")
        for row in frame.to_dict("records"):
            items.append((key_fields(COUNTS, group, row, row["source"], limits), row["geometry"]))
    return items


def straight_lines(folder, limits):
    """Return the key fields and the straight line between the two end nodes of each Alberta line, a line of no length included."""
    items = []
    for row in pandas.read_csv(Path(folder) / "alberta_lines.csv", dtype=str, keep_default_na=False).to_dict("records"):
        a = TO_MAP.transform(float(row["longitude_a"]), float(row["latitude_a"]))
        b = TO_MAP.transform(float(row["longitude_b"]), float(row["latitude_b"]))
        fields = {"record_ids": row["record_id"], "voltage_kv": float(row["voltage_kv"]) if row["voltage_kv"] else None}
        items.append((key_fields(COUNTS, "ALBERTA_STRAIGHT", fields, "straight", limits), shapely.LineString([a, b])))
    return items


def shape_lines(folder, limits):
    """Return the key fields and the shape of each route shape of both line sources, for the side record."""
    frame = pyogrio.read_dataframe(Path(folder) / "route_shapes.gpkg")
    return [(key_fields(BY_SOURCE, GROUP_NAMES[row["line_group"]], row, row["source"], limits), row["geometry"])
            for row in frame.to_dict("records") if row["status"] == "SHAPE"]


def raster_layout(source):
    """Return the origin, the cell size, the no-data value, and the size and tile size of each level of one raster."""
    with rasterio.Env(**GDAL_OPTIONS):
        with rasterio.open(source) as dataset:
            transform, nodata, count = dataset.transform, dataset.nodata, len(dataset.overviews(1))
            levels = [[dataset.width, dataset.height, *dataset.block_shapes[0][::-1]]]
        for level in range(count):
            with rasterio.open(source, overview_level=level) as dataset:
                levels.append([dataset.width, dataset.height, *dataset.block_shapes[0][::-1]])
    return {"x0": transform.c, "y0": transform.f, "pixel_x": transform.a, "pixel_y": -transform.e, "nodata": nodata,
            "levels": levels}


class Raster:
    """One raster from the input map, read tile by tile, with each tile kept as a file in the data folder."""

    def __init__(self, root, row):
        """Read the raster layout from the data folder, or from the raster after the identity check of the raster."""
        self.row, self.checked = row, False
        self.folder = Path(root) / TILE_FOLDER / row["name"]
        self.source = f"/vsicurl/{row['address']}" if "://" in row["address"] else row["address"]
        identity = [row[key] for key in RASTER_KEYS]
        path = self.folder / "layout.json"
        if not path.exists():
            self.check_identity()
            self.folder.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"identity": identity, **raster_layout(self.source)}), encoding="ascii")
        self.layout = json.loads(path.read_text(encoding="ascii"))
        if self.layout["identity"] != identity:
            raise InputError(f"the tiles in {self.folder} come from another copy of the raster {row['name']}, "
                             "so a person must delete that folder")

    def check_identity(self):
        """Check the size, the date, and the entity tag of a remote raster once per run, before the first remote read."""
        if not self.checked and "://" in self.row["address"]:
            print(check_raster(self.row), flush=True)
        self.checked = True

    def cell_size(self, level):
        """Return the cell size in meters of one level, from the full cell size and the width ratio, as in the old terrain script."""
        # The order of the product follows the old terrain script, so the cell size agrees to the last bit.
        return self.layout["pixel_x"] * (self.layout["levels"][0][0] / self.layout["levels"][level][0])

    def level_near(self, target, minimum, maximum):
        """Return the level with the cell size nearest the target, and stop the run if that size falls outside the limits."""
        level = min(sorted(range(len(self.layout["levels"])), key=self.cell_size), key=lambda item: abs(self.cell_size(item) - target))
        if not minimum <= self.cell_size(level) <= maximum:
            raise InputError(f"the raster {self.row['name']} has no level with a cell size from {minimum} m to {maximum} m")
        return level

    def cells(self, level, x, y):
        """Return the row and the column of each point at one level, with the width ratio on both axes as in the old terrain script."""
        ratio = self.layout["levels"][0][0] / self.layout["levels"][level][0]
        columns = numpy.floor((x - self.layout["x0"]) / (self.layout["pixel_x"] * ratio)).astype(numpy.int64)
        rows = numpy.floor((self.layout["y0"] - y) / (self.layout["pixel_y"] * ratio)).astype(numpy.int64)
        return rows, columns

    def tile_path(self, level, tile):
        """Return the file that keeps one tile of one level in the data folder."""
        return self.folder / str(level) / f"{tile[0]}_{tile[1]}.npz"

    def fetch(self, level, tile):
        """Read one tile of one level from the raster and keep it as a file in the data folder."""
        width, height, tile_width, tile_height = self.layout["levels"][level]
        window = Window(tile[1] * tile_width, tile[0] * tile_height, min(tile_width, width - tile[1] * tile_width),
                        min(tile_height, height - tile[0] * tile_height))
        options = {"overview_level": level - 1} if level else {}
        try:
            with rasterio.Env(**GDAL_OPTIONS), rasterio.open(self.source, **options) as dataset:
                values = dataset.read(1, window=window)
        except rasterio.errors.RasterioIOError as error:
            raise InputError(f"a tile read of the raster {self.row['name']} failed: {error}. "
                             "The kept tiles stay, so the next build reads only the rest.") from error
        path = self.tile_path(level, tile)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name("partial_" + path.name)
        with partial.open("wb") as handle:
            numpy.savez_compressed(handle, values=values)
        partial.replace(path)

    def read(self, level, rows, columns):
        """Return the cell values at the rows and columns of one level, with the no-data value outside the raster."""
        width, height, tile_width, tile_height = self.layout["levels"][level]
        nodata = self.layout["nodata"]
        values = numpy.full(len(rows), numpy.nan if nodata is None else float(nodata))
        inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        tiles = numpy.stack([rows // tile_height, columns // tile_width], axis=1)[inside]
        needed = sorted({tuple(int(value) for value in tile) for tile in tiles})
        missing = [tile for tile in needed if not self.tile_path(level, tile).exists()]
        if missing:
            self.check_identity()
            with ThreadPoolExecutor(READERS) as pool:
                list(pool.map(lambda tile: self.fetch(level, tile), missing))
        for tile in needed:
            pick = inside & (rows // tile_height == tile[0]) & (columns // tile_width == tile[1])
            with numpy.load(self.tile_path(level, tile)) as data:
                values[pick] = data["values"][rows[pick] % tile_height, columns[pick] % tile_width]
        return values


def slopes(raster, level, x, y):
    """Return the Horn slope in percent at each point from the 3 by 3 block of cells, or None if any cell has no data."""
    rows, columns = raster.cells(level, x, y)
    block = {(dr, dc): raster.read(level, rows + dr, columns + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    nodata = raster.layout["nodata"]
    missing = numpy.zeros(len(x), dtype=bool)
    for values in block.values():
        missing |= ~numpy.isfinite(values)
        if nodata is not None:
            missing |= numpy.abs(values - float(nodata)) <= NODATA_TOLERANCE
    size = raster.cell_size(level)
    dzdx = ((block[-1, 1] + 2 * block[0, 1] + block[1, 1]) - (block[-1, -1] + 2 * block[0, -1] + block[1, -1])) / (8 * size)
    dzdy = ((block[1, -1] + 2 * block[1, 0] + block[1, 1]) - (block[-1, -1] + 2 * block[-1, 0] + block[-1, 1])) / (8 * size)
    return [None if gone else 100.0 * math.hypot(float(a), float(b)) for gone, a, b in zip(missing, dzdx, dzdy)]


def land_covers(raster, x, y):
    """Return the land cover value at each point from the full level, and stop the run if a point falls outside the raster."""
    rows, columns = raster.cells(0, x, y)
    width, height = raster.layout["levels"][0][:2]
    outside = (rows < 0) | (rows >= height) | (columns < 0) | (columns >= width)
    if outside.any():
        raise InputError(f"{int(outside.sum())} samples fall outside the land cover raster")
    return raster.read(0, rows, columns).astype(numpy.int64)


def terrain_class(cover, slope, groups, bands):
    """Return the terrain class of one sample by the first class rule that fits, or None for a water sample."""
    if cover in groups["water"]:
        return None
    if slope is None:
        raise InputError(f"a land sample with land cover {cover} has no slope, because an elevation cell has no data")
    if cover in groups["urban"]:
        return "urban"
    if cover in groups["wetland"]:
        return "wetland"
    if slope > bands[1]:
        return "mountain"
    if cover in groups["forest"]:
        return "forest"
    return "hills" if slope >= bands[0] else "prairie"


def terrain_classes(covers, slope_values, groups, bands):
    """Return the terrain class of each sample, and stop the run on a land cover value outside every land cover group."""
    unknown = sorted(set(covers.tolist()) - {value for values in groups.values() for value in values})
    if unknown:
        raise InputError(f"the land cover values {unknown} belong to no land cover group of the settings file")
    return [terrain_class(cover, slope, groups, bands) for cover, slope in zip(covers.tolist(), slope_values)]


def sample_frame(items, spacing):
    """Return the samples of every line as a frame, with the path number counted within each record file."""
    numbers, parts = {}, []
    for fields, shape in items:
        number = numbers.get(fields["record_file"], 0)
        numbers[fields["record_file"]] = number + 1
        x, y = sample_points(shape, spacing)
        part = pandas.DataFrame({"x": x, "y": y, "sample_number": numpy.arange(len(x))})
        parts.append(part.assign(**fields, path_number=number))
    frame = pandas.concat(parts, ignore_index=True) if parts else pandas.DataFrame(columns=KEY_COLUMNS + ["x", "y", "sample_number"])
    frame["part"] = frame["part"].astype("Int64")
    return frame


def raster_rows(root):
    """Return the input map rows of the land cover raster and the elevation raster."""
    rows = {row["name"]: row for row in read_input_map(root)}
    for name in (LAND_COVER, ELEVATION):
        if name not in rows:
            raise InputError(f"the input map has no row {name}")
    return rows[LAND_COVER], rows[ELEVATION]


def terrain_frame(root, settings, items):
    """Return the samples of the lines with the land cover value, the slope in percent, and the terrain class of each sample."""
    frame = sample_frame(items, settings["sample_spacing_km"] * 1000.0)
    x, y = TO_RASTER.transform(frame["x"].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float))
    land, elevation = (Raster(root, row) for row in raster_rows(root))
    frame["land_cover"] = land_covers(land, x, y)
    overview = settings["elevation_overview_m"]
    level = elevation.level_near(overview["target"], overview["minimum"], overview["maximum"])
    # The slopes stay a list here, because a frame column turns each missing slope into a number.
    slope_values = slopes(elevation, level, x, y)
    frame["slope_percent"] = slope_values
    frame["terrain_class"] = terrain_classes(frame["land_cover"], slope_values, settings["land_cover_groups"],
                                             settings["slope_bands_percent"])
    return frame


def run_step(root, paths):
    """Run the terrain sampling on the path file, the line tables, the route shapes, and the two rasters, write the samples, and return the change."""
    folder, settings = Path(root) / "outputs", read_settings(root)
    limits = settings["voltage_class_limits_kv"]
    items = path_lines(root, settings) + straight_lines(folder, limits) + shape_lines(folder, limits)
    frame = terrain_frame(root, settings, items)
    points = geopandas.points_from_xy(frame["x"], frame["y"], crs=MAP_CRS)
    output = geopandas.GeoDataFrame(frame[SAMPLE_COLUMNS], geometry=points)
    return {"outputs/terrain_samples.gpkg": write_gpkg(output, folder / "terrain_samples.gpkg", "terrain_samples", OSM_NOTE)}

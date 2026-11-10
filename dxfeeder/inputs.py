"""Inputs.

This module reads the input map of dxfeeder and gets each small input from its address.
It checks each file against its pinned SHA-256 (Secure Hash Algorithm 256) checksum.
It checks the size, the date, and the entity tag of each listed raster, and it reads no raster content.
It filters the OpenStreetMap province files to the power lines and power cables of the grid definition.
It holds the shared writers that give the same bytes for the same rows, and helpers that cut small test files to an area.
"""

import csv
import hashlib
import json
import shutil
import tomllib
import urllib.request
import zipfile
from pathlib import Path

import pandas
import pyogrio
import shapely
from pyproj import Transformer

from dxfeeder import osm_filter

INPUT_MAP = "input_map.csv"
SAMPLE_FOLDER = Path("tests", "sample_data")
CENSUS = "census_boundaries"
CODERS_FILES = {"coders_nodes": "coders_nodes.json", "coders_lines": "coders_lines.json", "coders_interties": "coders_interties.json"}
END_NODE_KEYS = ("network_node_code_starting", "network_node_code_ending")
# The CODERS test files omit names, owners, and notes. The project reads codes and numbers, and free text can hold any word.
FREE_TEXT = {"node_name", "owner", "notes", "owner_starting", "owner_ending", "network_node_name_starting", "network_node_name_ending"}
CANVEC_PREFIX, OSM_RAW_PREFIX, OSM_LINES, CANVEC_LAYER = "canvec_", "osm_raw_", "osm_power_lines", "power_line_1"
RASTER_KEYS, RASTER_HEADERS = ("size_bytes", "last_modified", "etag"), ("Content-Length", "Last-Modified", "ETag")
OSM_NOTE = "\u00a9 OpenStreetMap contributors"
# A fixed GeoPackage date makes two writes of the same rows give the same bytes.
GPKG_DATE = "2026-01-01T00:00:00.000Z"
# Reading the positions on the North American Datum 1983 or the World Geodetic System 1984 gives the same map point. The project applies no datum shift.
NODE_CRS = "EPSG:4269"
TIMEOUT_SECONDS = 120


class InputError(Exception):
    """An input problem that stops a command, with a plain message."""


def read_input_map(root):
    """Return the rows of the input map as dictionaries, in file order."""
    path = Path(root) / INPUT_MAP
    if not path.exists():
        raise InputError(f"the input map {INPUT_MAP} is missing")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_settings(root):
    """Return the value of each table in the settings file, keyed by the table name."""
    with (Path(root) / "settings.toml").open("rb") as handle:
        return {name: table["value"] for name, table in tomllib.load(handle).items()}


def file_checksum(path):
    """Return the SHA-256 checksum of a file as hexadecimal text."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(address, target):
    """Copy the bytes at an address to the target file through a partial file, and return the target path."""
    target, partial = Path(target), Path(target).with_name(Path(target).name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(address, timeout=TIMEOUT_SECONDS) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except (OSError, ValueError) as error:
        partial.unlink(missing_ok=True)
        raise InputError(f"the download of {address} to {target} failed: {error}. If the server blocks a program, "
                         "a person can download the file in a browser to that path.") from error
    return partial.replace(target)


def ensure_file(root, row):
    """Check one input file against its checksum, download it first if it is missing, and return a status line."""
    path, status = Path(root) / row["local_path"], "Checked"
    if not row["checksum"]:
        raise InputError(f"the input map row {row['name']} has no checksum")
    if not path.exists():
        if "://" not in row["address"]:
            raise InputError(f"the held copy {row['local_path']} is missing, so a person must copy it to that path")
        download(row["address"], path)
        status = "Downloaded"
    if file_checksum(path) != row["checksum"]:
        raise InputError(f"the checksum of {row['local_path']} differs from the input map")
    return f"{status} {row['name']}"


def check_raster(row):
    """Compare the size, the date, and the entity tag of one raster with the input map, and return a status line."""
    try:
        with urllib.request.urlopen(urllib.request.Request(row["address"], method="HEAD"), timeout=TIMEOUT_SECONDS) as response:
            found = {key: response.headers.get(header, "") for key, header in zip(RASTER_KEYS, RASTER_HEADERS)}
    except (OSError, ValueError) as error:
        raise InputError(f"the identity request for {row['address']} failed: {error}") from error
    text = ", ".join(f"{key} {found[key]}" for key in RASTER_KEYS)
    if any(found[key] != row[key] for key in RASTER_KEYS):
        raise InputError(f"the raster {row['name']} changed: {text}")
    return f"Raster {row['name']} unchanged: {text}"


def write_gpkg(frame, path, layer, description=""):
    """Write a frame to a GeoPackage with the fixed date, keep the old file if the bytes match, and return True on a change."""
    return write_layers({layer: frame}, path, description)


def write_layers(frames, path, description="", geometry_type=None):
    """Write frames as layers of one GeoPackage with the fixed date, keep the old file if the bytes match, and return True on a change."""
    path = Path(path)
    partial = path.with_name("partial_" + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    pyogrio.set_gdal_config_options({"OGR_CURRENT_DATE": GPKG_DATE})
    # Every GeoPackage carries a spatial index. Its bytes can differ between operating systems, so equal bytes hold on one system.
    options = {"SPATIAL_INDEX": "YES", **({"DESCRIPTION": description} if description else {})}
    # A layer with no rows needs a named shape type, because the file cannot take it from the rows.
    extra = {"geometry_type": geometry_type} if geometry_type else {}
    for layer, frame in frames.items():
        pyogrio.write_dataframe(frame.reset_index(drop=True), partial, layer=layer, driver="GPKG", layer_options=options, **extra)
    if path.exists() and path.read_bytes() == partial.read_bytes():
        partial.unlink()
        return False
    partial.replace(path)
    return True


def write_text(path, text, encoding="ascii"):
    """Write ASCII text with LF line endings, keep the old file if the bytes match, and return True on a change.

    A file that holds names from a source can name another encoding.
    """
    path, data = Path(path), text.encode(encoding)
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def vector_source(path):
    """Return the name that the vector reader opens for a file, with the inner folder of a zipped file geodatabase."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            folders = sorted({name.split("/")[0] for name in archive.namelist() if ".gdb/" in name})
        if folders:
            return f"/vsizip/{path.as_posix()}/{folders[0]}"
    return str(path)


def cut_coders_files(root, paths, area):
    """Write the CODERS test files with the nodes inside the cut-out area and the records between them, and return the changes."""
    records = {name: json.loads(paths[name].read_text(encoding="utf-8")) for name in CODERS_FILES}
    places = pandas.DataFrame(records["coders_nodes"])[["longitude", "latitude"]].apply(pandas.to_numeric, errors="coerce")
    x, y = Transformer.from_crs(NODE_CRS, "EPSG:3347", always_xy=True).transform(places["longitude"], places["latitude"])
    kept = {"coders_nodes": [node for node, inside in zip(records["coders_nodes"], shapely.contains_xy(area, x, y)) if inside]}
    codes = {str(node["node_code"]).strip() for node in kept["coders_nodes"]}
    for name in ("coders_lines", "coders_interties"):
        kept[name] = [row for row in records[name] if {str(row.get(key) or "").strip() for key in END_NODE_KEYS} <= codes]
    kept = {name: [{key: value for key, value in row.items() if key not in FREE_TEXT} for row in rows] for name, rows in kept.items()}
    folder = Path(root) / SAMPLE_FOLDER
    return {file: write_text(folder / file, json.dumps(kept[name], indent=1, ensure_ascii=True) + "\n")
            for name, file in CODERS_FILES.items()}


def cut_line_files(root, paths, area):
    """Write the CanVec and OpenStreetMap test files with the features that meet the cut-out area, and return the changes."""
    names = sorted(name for name in paths if name.startswith(CANVEC_PREFIX))
    canvec = pandas.concat([pyogrio.read_dataframe(vector_source(paths[name]), layer=CANVEC_LAYER) for name in names])
    # A feature in two province files keeps the copy from the province first in alphabetical order.
    canvec = canvec.drop_duplicates("feature_id").sort_values("feature_id")
    osm = pyogrio.read_dataframe(paths[OSM_LINES])
    canvec, osm = (frame[frame.to_crs("EPSG:3347").intersects(area).to_numpy()] for frame in (canvec, osm))
    folder = Path(root) / SAMPLE_FOLDER
    return {"canvec_power_lines.gpkg": write_gpkg(canvec, folder / "canvec_power_lines.gpkg", CANVEC_LAYER),
            "osm_power_lines.gpkg": write_gpkg(osm, folder / "osm_power_lines.gpkg", OSM_LINES, OSM_NOTE)}


def fetch_inputs(root, rows):
    """Check or download each input file, build the filtered OpenStreetMap file, check both rasters, and yield status lines."""
    paths = {row["name"]: Path(root) / row["local_path"] for row in rows}
    for row in rows:
        if not row["size_bytes"] and row["name"] != OSM_LINES:
            yield ensure_file(root, row)
    for row in rows:
        if row["name"] == OSM_LINES and not paths[OSM_LINES].exists():
            raw = sorted(name for name in paths if name.startswith(OSM_RAW_PREFIX))
            frame = osm_filter.filter_osm([paths[name] for name in raw])
            write_gpkg(frame, paths[OSM_LINES], OSM_LINES, OSM_NOTE)
            yield f"Built {OSM_LINES} with {len(frame)} ways"
        if row["name"] == OSM_LINES:
            yield ensure_file(root, row)
        elif row["size_bytes"]:
            yield check_raster(row)


def run_download(root):
    """Run the download command on a project folder, print one status line per step, and return the exit code."""
    try:
        rows = read_input_map(root)
        for line in fetch_inputs(root, rows):
            print(line, flush=True)
    except InputError as error:
        print(f"Download failed: {error}")
        return 1
    return 0

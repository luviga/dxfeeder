"""Feeder build of one site.

Run it as  python -m dxfeeder.build --config <site.toml>  to build the OpenDSS feeders of one site.
The site file names the output folder and each input file with its address and its SHA-256 checksum. The build checks every
checksum first, and a single changed byte stops the run before any output is written.

Site file keys:
  name          circuit name, letters, digits and underscores
  out           output folder, relative to the site file unless absolute
  substations   "reference" to read them from a GeoPackage layer, or "osm" to read power=substation from the extract
  [inputs.extract]    path, sha256, url of the OpenStreetMap extract (PBF or XML)
  [inputs.reference]  path, sha256, url of the GeoPackage with the site shape and, if used, substations and mapped lines
  [layers]      site (polygon layer of the site shape), substations (point layer), routes (optional line layer)

Steps: street graph, forced routes, load points, substations, radial feeders, transformers, medium voltage lines, OpenDSS files.
The output folder receives the five OpenDSS files, the manifest of their checksums and build_summary.json. Two builds of the same
inputs with the same settings give the same bytes.
"""

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy
import pyogrio
import shapely

from dxfeeder import buildings, network, routes, streets, substations, transformers, writer
from dxfeeder.inputs import file_checksum, read_settings
from dxfeeder.streets import GEOGRAPHIC_CRS

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "build_summary.json"


class BuildError(Exception):
    """A site file or input problem that stops a build, with a plain message."""


def read_site(path):
    """Return the site file as a dictionary, with every input path and the output folder made absolute."""
    path = Path(path).resolve()
    with path.open("rb") as handle:
        site = tomllib.load(handle)
    for key in ("name", "out", "substations", "inputs", "layers"):
        if key not in site:
            raise BuildError(f"the site file {path.name} has no {key}")
    site["out"] = (path.parent / site["out"]).resolve()
    for item in site["inputs"].values():
        item["path"] = (path.parent / item["path"]).resolve()
    return site


def check_inputs(site):
    """Check every input file of the site against its SHA-256 checksum and return the checksums, keyed by input name."""
    sums = {}
    for name in sorted(site["inputs"]):
        item = site["inputs"][name]
        if not item["path"].exists():
            raise BuildError(f"the input {name} is missing")
        found = file_checksum(item["path"])
        if found != item["sha256"]:
            raise BuildError(f"the checksum of the input {name} differs from the site file, so the build stops")
        sums[name] = found
    return sums


def site_shape(site):
    """Return the site shape in EPSG:4326, the union of every shape of the site layer, as a Polygon or MultiPolygon."""
    frame = pyogrio.read_dataframe(site["inputs"]["reference"]["path"], layer=site["layers"]["site"]).to_crs(GEOGRAPHIC_CRS)
    return shapely.union_all(frame.geometry.to_numpy())


def station_rows(site, shape, settings):
    """Return the substations of the site as (id, longitude, latitude) rows, from the reference layer or the extract."""
    if site["substations"] == "reference":
        return substations.from_layer(site["inputs"]["reference"]["path"], site["layers"]["substations"], shape)
    if site["substations"] == "osm":
        return substations.from_osm(site["inputs"]["extract"]["path"], shape, settings["substation_tag_key"],
                                    settings["substation_tag_value"])
    raise BuildError(f"the substation source {site['substations']} is not reference or osm")


def run_build(site, settings, out=None, log=print):
    """Build the feeders of one site, write the files to the output folder, and return the build summary."""
    started = time.perf_counter()

    def step(text):
        """Print one progress line with the seconds since the start."""
        log(f"[{time.perf_counter() - started:8.1f} s] {text}")

    out = Path(out) if out else site["out"]
    sums = check_inputs(site)
    step("inputs checked")
    shape = site_shape(site)
    crs = streets.metric_crs(shape)
    extract = site["inputs"]["extract"]["path"]
    graph = streets.street_graph(extract, shape, settings["street_highway_values"], crs)
    step(f"street graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    forced = {"forced_edges": 0, "route_nodes": 0}
    if site["layers"].get("routes"):
        lines = routes.route_lines(site["inputs"]["reference"]["path"], site["layers"]["routes"], shape)
        forced = routes.add_forced_routes(graph, lines, crs, settings)
        step(f"forced routes: {len(lines)} lines, {forced['forced_edges']} edges, {forced['route_nodes']} new nodes")
    points = buildings.load_points(extract, shape, settings["building_skipped_values"])
    loads = buildings.loads_table(points, settings)
    step(f"load points: {len(loads)}")
    stations = [{"id": row[0], "lon": row[1], "lat": row[2]} for row in station_rows(site, shape, settings)]
    to_metric = streets.to_metric(crs)
    station_xy = to_metric.transform(numpy.array([row["lon"] for row in stations]), numpy.array([row["lat"] for row in stations]))
    forest, report, unserved = network.build_feeders(graph, loads, buildings.metric_points(points, to_metric), stations, station_xy)
    step(f"feeders: {len(forest['roots'])} substation nodes, {report['loads_served']} loads served")
    units = transformers.group_loads(forest, loads, settings)
    mv_lines = network.medium_voltage_lines(forest, {unit["node"] for unit in units})
    model = writer.feeder_model(graph, forest, units, mv_lines, loads, settings)
    files = writer.write_feeder(out, site["name"], model, settings)
    step(f"written: {len(model['lines'])} lines, {len(model['transformers'])} transformers, {len(model['loads'])} loads")
    metres_per_km = float(settings["metres_per_km"])
    summary = {
        "name": site["name"],
        "input_sha256": sums,
        "metric_crs": crs.to_string(),
        "street_highway_values": list(settings["street_highway_values"]),
        **forced,
        **report,
        "transformers": len(units),
        "transformers_above_largest_size": sum(not unit["fits"] for unit in units),
        "medium_voltage_lines": len(mv_lines),
        "medium_voltage_km": sum(line[2] for line in mv_lines) / metres_per_km,
        "secondary_lines": sum(line[0].startswith("lv_") for line in model["lines"]),
        "secondary_km": sum(line[4] for line in model["lines"] if line[0].startswith("lv_")),
        "service_drops": sum(line[0].startswith("sv_") for line in model["lines"]),
        "service_drop_km": sum(line[4] for line in model["lines"] if line[0].startswith("sv_")),
        "buses": len(model["buses"]),
        "files_sha256": files,
        "loads_without_substation_path_ids": unserved,
    }
    (Path(out) / SUMMARY).write_bytes((json.dumps(summary, indent=1, sort_keys=True) + "\n").encode("ascii"))
    step("summary written")
    return summary


def main(argv=None):
    """Read the command line, build one site, and return 0 on success, 2 when an input or the site file stops the build."""
    parser = argparse.ArgumentParser(prog="python -m dxfeeder.build", description="Build the OpenDSS feeders of one site.")
    parser.add_argument("--config", required=True, type=Path, help="the site file")
    parser.add_argument("--settings", type=Path, default=ROOT, help="folder that holds settings.toml")
    parser.add_argument("--out", type=Path, help="output folder, in place of the one the site file names")
    arguments = parser.parse_args(argv)
    try:
        run_build(read_site(arguments.config), read_settings(arguments.settings), arguments.out)
    except BuildError as error:
        print(f"Build stopped: {error}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

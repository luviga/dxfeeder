"""OpenDSS writer.

This module turns the feeders of a site into OpenDSS files: Master.dss, Lines.dss, Transformers.dss, Loads.dss and Buscoords.dss.
Master.dss defines one circuit with one voltage source per substation node, at the medium voltage of the settings file.
Every line carries its true length in km with units=km and no scaling factor.
Medium voltage lines join the substation to the transformer nodes along the feeder tree. The low voltage secondary of each
transformer follows the same tree down to the nodes where its loads attach, and each load hangs from its node by a service drop.
Chains of tree nodes with no branch, transformer or load are merged into one line with their summed length.
Buscoords.dss gives the longitude and latitude of every bus.
The files are deterministic: every element list is sorted by name, and no file holds a date, a time or a random value.
A manifest beside the files records the SHA-256 checksum of each one, so a later edit can be found.
"""

import hashlib
import json
from pathlib import Path

from dxfeeder.network import chain_lines

DSS_FILES = ("Master.dss", "Lines.dss", "Transformers.dss", "Loads.dss", "Buscoords.dss")
MANIFEST = "manifest.json"
MV_CODE, LV_CODE = "mv_overhead", "lv_service"


def bus_name(node):
    """Return the bus name of a graph node: "n" and the OpenStreetMap node id, or "m" and the number of a route node."""
    return f"n{node}" if node >= 0 else f"m{-node}"


def number(value):
    """Return the shortest text that reads back as the same float."""
    return repr(float(value))


def fixed(value, decimals):
    """Return a float as text with a fixed number of decimals."""
    return f"{float(value):.{int(decimals)}f}"


def feeder_model(graph, forest, transformers, mv_lines, loads, settings):
    """Return the buses, sources, lines, transformers and loads of the feeders as sorted lists of plain values."""
    metres_per_km = float(settings["metres_per_km"])
    minimum = float(settings["line_length_min_m"])
    place = {load["id"]: (load["lon"], load["lat"]) for load in loads}
    by_id = {load["id"]: load for load in loads}
    buses, sources, lines, units, served = {}, [], [], [], []
    for root in forest["roots"]:
        buses[bus_name(root)] = (graph.nodes[root]["lon"], graph.nodes[root]["lat"])
        sources.append((forest["root_station"][root], bus_name(root)))
    for upper, lower, length in mv_lines:
        for node in (upper, lower):
            buses[bus_name(node)] = (graph.nodes[node]["lon"], graph.nodes[node]["lat"])
        lines.append((f"mv_{bus_name(lower)}", bus_name(upper), bus_name(lower), MV_CODE, max(length, minimum) / metres_per_km))
    for unit in transformers:
        top = unit["node"]
        name = f"tx_{bus_name(top)}_{unit['number']}"
        low = f"{name}_lv"

        def low_bus(node, name=name, low=low, top=top):
            """Return the low voltage bus of this transformer at a tree node."""
            return low if node == top else f"{name}_{bus_name(node)}"

        buses[bus_name(top)] = buses[low] = (graph.nodes[top]["lon"], graph.nodes[top]["lat"])
        units.append((name, bus_name(top), low, unit["kva"]))
        attached = {load_id: forest["attached"][load_id] for load_id, _ in unit["loads"]}
        for upper, lower, length in chain_lines(forest, {top}, {node for node, _ in attached.values()}):
            buses[low_bus(lower)] = (graph.nodes[lower]["lon"], graph.nodes[lower]["lat"])
            lines.append((f"lv_{name}_{bus_name(lower)}", low_bus(upper), low_bus(lower), LV_CODE,
                          max(length, minimum) / metres_per_km))
        for load_id, (node, drop) in sorted(attached.items()):
            load_bus = f"ld_{load_id}"
            buses[load_bus] = place[load_id]
            lines.append((f"sv_{load_id}", low_bus(node), load_bus, LV_CODE, max(drop, minimum) / metres_per_km))
            served.append((f"ld_{load_id}", load_bus, by_id[load_id]["kw"], by_id[load_id]["pf"]))
    return {"buses": sorted(buses.items()), "sources": sources, "lines": sorted(lines), "transformers": sorted(units),
            "loads": sorted(served)}


def linecode_text(name, settings, prefix):
    """Return the OpenDSS definition of a three-phase line code from the self and mutual values of the settings file."""
    r_self, r_mutual = number(settings[f"{prefix}_r_self_ohm_per_km"]), number(settings[f"{prefix}_r_mutual_ohm_per_km"])
    x_self, x_mutual = number(settings[f"{prefix}_x_self_ohm_per_km"]), number(settings[f"{prefix}_x_mutual_ohm_per_km"])
    rows = lambda own, other: f"({own} {other} {other} | {other} {own} {other} | {other} {other} {own})"
    return f"New Linecode.{name} nphases=3 units=km rmatrix={rows(r_self, r_mutual)} xmatrix={rows(x_self, x_mutual)}"


def source_text(settings):
    """Return the source settings shared by the circuit and every other voltage source."""
    return (f"basekv={number(settings['mv_kv'])} pu={number(settings['source_pu'])} phases=3 "
            f"r1={number(settings['source_r1_ohm'])} x1={number(settings['source_x1_ohm'])} "
            f"r0={number(settings['source_r0_ohm'])} x0={number(settings['source_x0_ohm'])}")


def file_texts(name, model, settings):
    """Return the text of each OpenDSS file, keyed by file name, with LF line endings."""
    decimals = settings["length_decimals"]
    mv_kv, lv_kv = number(settings["mv_kv"]), number(settings["lv_kv"])
    master = ["Clear"]
    if model["sources"]:
        first_station, first_bus = model["sources"][0]
        master.append(f"! Substation {first_station} feeds the circuit source")
        master.append(f"New Circuit.{name} bus1={first_bus} {source_text(settings)}")
        master += [f"New Vsource.sub_{station} bus1={bus} {source_text(settings)}" for station, bus in model["sources"][1:]]
    master += ["Redirect Lines.dss", "Redirect Transformers.dss", "Redirect Loads.dss",
               f"Set Voltagebases=[{mv_kv} {lv_kv}]", "Calcvoltagebases", "Buscoords Buscoords.dss"]
    lines = [linecode_text(MV_CODE, settings, "mv"), linecode_text(LV_CODE, settings, "lv")]
    lines += [f"New Line.{line} bus1={first} bus2={second} phases=3 linecode={code} length={fixed(km, decimals)} units=km"
              for line, first, second, code, km in model["lines"]]
    units = [f"New Transformer.{unit} phases=3 windings=2 buses=({high}, {low}) conns=(wye, wye) kvs=({mv_kv}, {lv_kv}) "
             f"kvas=({number(kva)}, {number(kva)}) %r={number(settings['transformer_pct_r'])} "
             f"xhl={number(settings['transformer_xhl_pct'])}" for unit, high, low, kva in model["transformers"]]
    loads = [f"New Load.{load} bus1={bus} phases=3 conn=wye kv={lv_kv} kw={number(kw)} pf={number(pf)} model=1"
             for load, bus, kw, pf in model["loads"]]
    places = settings["coordinate_decimals"]
    coords = [f"{bus}, {fixed(lon, places)}, {fixed(lat, places)}" for bus, (lon, lat) in model["buses"]]
    texts = {"Master.dss": master, "Lines.dss": lines, "Transformers.dss": units, "Loads.dss": loads, "Buscoords.dss": coords}
    return {file: "\n".join(rows) + "\n" for file, rows in texts.items()}


def write_feeder(folder, name, model, settings):
    """Write the OpenDSS files and the manifest of checksums to a folder, and return the checksum of each file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sums = {}
    for file, text in file_texts(name, model, settings).items():
        data = text.encode("ascii")
        (folder / file).write_bytes(data)
        sums[file] = hashlib.sha256(data).hexdigest()
    (folder / MANIFEST).write_bytes((json.dumps(sums, indent=1, sort_keys=True) + "\n").encode("ascii"))
    return sums

"""Verification of an OpenDSS feeder.

Run it as  python -m dxfeeder.verify <Master.dss>  with the verify extra installed (opendssdirect.py).
It compiles the master as written, runs one snapshot power flow and reports five criteria and the descriptive numbers.

Criteria:
1 compiles: the master compiles with no OpenDSS error;
2 solves: the snapshot power flow converges;
3 radial: in the graph of enabled lines, switches, reactors and transformers between buses, with parallel elements between the
  same two buses collapsed, every connected part that holds a voltage source holds exactly one and is a tree;
4 connected: every load sits at a bus whose voltage after the solve is above the energized limit of the settings file;
5 no hand edits: every file listed in the manifest beside the master still has its recorded SHA-256 checksum.

Descriptive numbers: circuits, voltage sources, buses, loads, distribution transformers (lowest winding at or below the low
voltage limit of the settings file), line km without switches, the median count of loads per distribution transformer, the
longest path from a source in km, and the lowest and highest load voltage in per unit.
A switch line counts neither in the line km nor among lines without a length unit, and the two legs of a centre-tapped
transformer that end on the same bus count as one branch, not as parallel elements.
"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import networkx

from dxfeeder.inputs import read_settings

ROOT = Path(__file__).resolve().parents[1]
BRANCH_CLASSES = ("line", "transformer", "reactor")
CRITERIA = ("compiles", "solves", "radial", "connected", "no_hand_edits")


def bus_of(entry):
    """Return the bus name of an OpenDSS terminal entry, without its node numbers, in lower case."""
    return entry.split(".")[0].lower()


def nodes_of(entry, phases):
    """Return the node numbers of an OpenDSS terminal entry, or the first nodes up to the phase count when it names none."""
    parts = [part for part in entry.split(".")[1:] if part and part != "0"]
    return [int(part) for part in parts] if parts else list(range(1, phases + 1))


def file_sha256(path):
    """Return the SHA-256 checksum of a file as hexadecimal text."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def no_hand_edits(master, manifest_name="manifest.json"):
    """Return whether every file of the manifest beside the master keeps its checksum, with a plain detail line."""
    path = Path(master).parent / manifest_name
    if not path.exists():
        return False, "no manifest beside the master"
    recorded = json.loads(path.read_text(encoding="ascii"))
    changed = [name for name, digest in sorted(recorded.items())
               if not (path.parent / name).exists() or file_sha256(path.parent / name) != digest]
    return bool(recorded) and not changed, f"{len(recorded)} files checked, {len(changed)} changed"


def length_km(dss, settings):
    """Return the length of the active line in km, from its length and its unit code."""
    unit = settings["length_units"][dss.Lines.Units()]
    return dss.Lines.Length() * settings[f"length_unit_km_{unit}"]


def compile_and_solve(dss, master, settings):
    """Compile a master and run one snapshot solve; return the compile errors, the solve notes and the two flags."""
    words = settings["convergence_words"]
    errors, notes = [], []
    dss.Text.Command("clear")
    # OpenDSS moves the working folder to the folder of the master; the check moves it back.
    folder = os.getcwd()
    try:
        dss.Text.Command(f'compile "{Path(master).resolve()}"')
    except Exception as error:  # noqa: BLE001  OpenDSS reports every failure as a plain exception
        (notes if any(word in str(error).lower() for word in words) else errors).append(str(error))
    finally:
        os.chdir(folder)
    if dss.Error.Number():
        errors.append(dss.Error.Description())
    if not dss.Circuit.Name():
        return errors, notes, not errors, False
    try:
        dss.Text.Command("set mode=snapshot")
        dss.Text.Command("solve")
    except Exception as error:  # noqa: BLE001
        notes.append(str(error))
    return errors, notes, not errors, bool(dss.Solution.Converged())


def branch_graph(dss, settings):
    """Return the branch graph of the compiled circuit, the branch end coordinates and the branch counts."""
    lv_max = float(settings["distribution_transformer_lv_max_kv"])
    coords = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        if dss.Bus.Coorddefined():
            coords[bus.lower()] = (dss.Bus.X(), dss.Bus.Y())
    graph = networkx.Graph()
    graph.add_nodes_from(name.lower() for name in dss.Circuit.AllBusNames())
    counts = {"parallel_elements": 0, "switches": 0, "lines_units_none": 0, "line_km": 0.0}
    segments, transformers = [], []
    for element in dss.Circuit.AllElementNames():
        kind = element.split(".")[0].lower()
        if kind not in BRANCH_CLASSES:
            continue
        dss.Circuit.SetActiveElement(element)
        if not dss.CktElement.Enabled():
            continue
        buses = [bus_of(entry) for entry in dss.CktElement.BusNames()]
        if len(set(buses)) < 2:
            continue
        conductors = dss.CktElement.NumConductors()
        if any(all(dss.CktElement.IsOpen(terminal, conductor) for conductor in range(1, conductors + 1))
               for terminal in range(1, len(buses) + 1)):
            continue
        weight = 0.0
        if kind == "line":
            dss.Lines.Name(element.split(".", 1)[1])
            weight = length_km(dss, settings)
            if dss.Lines.IsSwitch():
                counts["switches"] += 1
                kind = "switch"
            else:
                counts["line_km"] += weight
                counts["lines_units_none"] += dss.Lines.Units() == 0
        elif kind == "transformer":
            dss.Transformers.Name(element.split(".", 1)[1])
            levels = []
            for winding in range(1, dss.Transformers.NumWindings() + 1):
                dss.Transformers.Wdg(winding)
                levels.append(dss.Transformers.kV())
            if min(levels) <= lv_max:
                transformers.append(element.lower())
                kind = "distribution_transformer"
        for other in sorted(set(buses[1:]) - {buses[0]}):
            if graph.has_edge(buses[0], other):
                counts["parallel_elements"] += 1
                graph[buses[0]][other]["elements"].append(element.lower())
            else:
                graph.add_edge(buses[0], other, weight=weight, kind=kind, elements=[element.lower()])
            if buses[0] in coords and other in coords:
                segments.append((kind, *coords[buses[0]], *coords[other]))
    counts["distribution_transformers"] = len(transformers)
    return graph, segments, counts, transformers


def radial_parts(graph, sources):
    """Return whether every part with a source holds exactly one and is a tree, and the count of parts with no source."""
    radial, sourceless = bool(sources), 0
    for part in networkx.connected_components(graph):
        held = sources & part
        if not held:
            sourceless += 1
        elif len(held) != 1 or graph.subgraph(part).number_of_edges() != len(part) - 1:
            radial = False
    return radial, sourceless


def check_master(dss, master, settings):
    """Compile and solve one master and return its facts and its branch end coordinates."""
    errors, notes, compiles, solves = compile_and_solve(dss, master, settings)
    result = {"compile_errors": errors, "solve_notes": notes, "compiles": compiles, "solves": solves}
    if not dss.Circuit.Name():
        result.update({"radial": False, "connected": False})
        return result, []
    sources = set()
    for name in dss.Vsources.AllNames():
        dss.Circuit.SetActiveElement(f"Vsource.{name}")
        sources.add(bus_of(dss.CktElement.BusNames()[0]))
    graph, segments, counts, transformers = branch_graph(dss, settings)
    result.update(counts)
    result["voltage_sources"] = len(dss.Vsources.AllNames())
    result["buses"] = dss.Circuit.NumBuses()
    result["radial"], result["sourceless_parts"] = radial_parts(graph, sources)
    longest, parent = 0.0, {}
    for source in sorted(sources):
        if source in graph:
            lengths, paths = networkx.single_source_dijkstra(graph, source, weight="weight")
            longest = max(longest, max(lengths.values()))
            for node, path in paths.items():
                if len(path) > 1:
                    parent[node] = path[-2]
    result["longest_path_km"] = longest
    limit = float(settings["energized_voltage_min_pu"])
    voltages, dead, unassigned = [], 0, 0
    served = dict.fromkeys(transformers, 0)
    names = dss.Loads.AllNames()
    for name in names:
        dss.Circuit.SetActiveElement(f"Load.{name}")
        entry = dss.CktElement.BusNames()[0]
        nodes = nodes_of(entry, dss.CktElement.NumPhases())
        dss.Circuit.SetActiveBus(bus_of(entry))
        magnitudes = dss.Bus.puVmagAngle()[::2]
        values = [value for node, value in zip(dss.Bus.Nodes(), magnitudes) if node in nodes]
        if not values or max(values) <= limit:
            dead += 1
        else:
            voltages.extend(values)
        node, found = bus_of(entry), None
        while node in parent:
            upper = parent[node]
            if graph[node][upper]["kind"] == "distribution_transformer":
                found = graph[node][upper]["elements"][0]
                break
            node = upper
        if found is None:
            unassigned += 1
        else:
            served[found] += 1
    result.update({"loads": len(names), "loads_deenergized": dead, "connected": dead == 0 and bool(names),
                   "v_min_pu": min(voltages) if voltages else None, "v_max_pu": max(voltages) if voltages else None,
                   "loads_per_transformer": list(served.values()), "loads_without_upstream_transformer": unassigned})
    return result, segments


def verify(master, settings):
    """Return the verification summary of one master: the five criteria, the counts and the branch end coordinates."""
    try:
        import opendssdirect as dss
    except ImportError as error:
        raise SystemExit("verify needs opendssdirect.py; install dxfeeder with the verify extra") from error
    master = Path(master).resolve()
    result, segments = check_master(dss, master, settings)
    edits, detail = no_hand_edits(master)
    per_transformer = result.get("loads_per_transformer", [])
    summary = {
        "master": Path(master).name,
        "criteria": {"compiles": result["compiles"], "solves": result["solves"], "radial": result["radial"],
                     "connected": result["connected"], "no_hand_edits": edits},
        "counts": {
            "circuits": int(bool(dss.Circuit.Name())),
            "voltage_sources": result.get("voltage_sources", 0),
            "buses": result.get("buses", 0),
            "loads": result.get("loads", 0),
            "loads_deenergized": result.get("loads_deenergized", 0),
            "distribution_transformers": result.get("distribution_transformers", 0),
            "line_km": result.get("line_km", 0.0),
            "loads_per_transformer_median": statistics.median(per_transformer) if per_transformer else None,
            "longest_path_km": result.get("longest_path_km"),
            "v_min_pu": result.get("v_min_pu"),
            "v_max_pu": result.get("v_max_pu"),
            "parallel_elements": result.get("parallel_elements", 0),
            "switches": result.get("switches", 0),
            "lines_units_none": result.get("lines_units_none", 0),
            "sourceless_parts": result.get("sourceless_parts", 0),
            "loads_without_upstream_transformer": result.get("loads_without_upstream_transformer", 0),
        },
        "no_hand_edits_detail": detail,
        "compile_errors": result["compile_errors"],
        "solve_notes": result["solve_notes"],
    }
    return summary, segments


def main(argv=None):
    """Verify one master, print the summary as JSON, and return 0 when all five criteria pass, else 1."""
    parser = argparse.ArgumentParser(prog="python -m dxfeeder.verify", description="Verify one OpenDSS feeder master file.")
    parser.add_argument("master", type=Path, help="the Master.dss file to check")
    parser.add_argument("--settings", type=Path, default=ROOT, help="folder that holds settings.toml")
    parser.add_argument("--json", type=Path, help="write the summary to this file")
    parser.add_argument("--segments", type=Path, help="write the branch end coordinates to this CSV file")
    arguments = parser.parse_args(argv)
    summary, segments = verify(arguments.master, read_settings(arguments.settings))
    text = json.dumps(summary, indent=1, sort_keys=True)
    print(text)
    if arguments.json:
        arguments.json.write_text(text + "\n", encoding="ascii")
    if arguments.segments:
        with arguments.segments.open("w", newline="", encoding="ascii") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["kind", "x1", "y1", "x2", "y2"])
            writer.writerows(segments)
    return 0 if all(summary["criteria"][name] for name in CRITERIA) else 1


if __name__ == "__main__":
    sys.exit(main())

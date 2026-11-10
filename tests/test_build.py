"""Tests of the build command, the OpenDSS writer and the verification.

Each test builds the synthetic site of synthetic_site.py in a temporary folder, with no network.
A changed input byte must stop the build before any output; two builds must give the same bytes; the files must carry true
line lengths in km; and, when opendssdirect.py is installed, the built feeder must pass all five verification criteria while a
hand edit or a loop must fail the matching criterion.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

import synthetic_site as site
from dxfeeder import build, inputs, verify, writer

PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = inputs.read_settings(PROJECT)
HAS_OPENDSS = importlib.util.find_spec("opendssdirect") is not None
needs_opendss = pytest.mark.skipif(not HAS_OPENDSS, reason="opendssdirect.py, the verify extra, is not installed")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Return the folder of the synthetic site and the summary of one build into its output folder."""
    folder = tmp_path_factory.mktemp("build")
    extract, reference = site.write_site(folder)
    config = site.write_site_file(folder, extract, reference)
    summary = build.run_build(build.read_site(config), SETTINGS, log=lambda text: None)
    return folder, summary


def test_build_files(built):
    """The build writes the five OpenDSS files, the manifest and the summary."""
    folder, _ = built
    names = sorted(path.name for path in (folder / "built").iterdir())
    assert names == sorted(list(writer.DSS_FILES) + [writer.MANIFEST, build.SUMMARY])


def test_build_summary(built):
    """The summary counts the loads, the forced edges and the unserved load of the synthetic site."""
    _, summary = built
    assert (summary["loads"], summary["loads_served"], summary["loads_without_substation_path_ids"]) == (6, 5, ["w207"])
    assert summary["forced_edges"] == 2 and summary["transformers"] >= 1 and summary["metric_crs"] == "EPSG:32620"
    assert summary["files_sha256"] == json.loads((built[0] / "built" / writer.MANIFEST).read_text(encoding="ascii"))


def test_build_twice_same_bytes(built, tmp_path):
    """A second build of the same inputs into another folder gives the same bytes in every file."""
    folder, _ = built
    build.run_build(build.read_site(folder / "site.toml"), SETTINGS, out=tmp_path / "again", log=lambda text: None)
    for name in list(writer.DSS_FILES) + [writer.MANIFEST, build.SUMMARY]:
        assert (tmp_path / "again" / name).read_bytes() == (folder / "built" / name).read_bytes()


def test_line_lengths_true_km(built):
    """Every line carries units=km and a length in km with no scaling factor, and no length is zero."""
    text = (built[0] / "built" / "Lines.dss").read_text(encoding="ascii")
    lengths = [float(value) for value in re.findall(r"length=([0-9.]+) units=km", text)]
    assert len(lengths) == text.count("New Line.") and min(lengths) > 0 and "*" not in text
    mv = re.search(r"New Line\.mv_n9 .* length=([0-9.]+)", text)
    if mv:
        assert float(mv.group(1)) < 1.0


def test_master_voltage_from_settings(built):
    """The source runs at the medium voltage of the settings file, never at the 138 kV tag of the substation."""
    master = (built[0] / "built" / "Master.dss").read_text(encoding="ascii")
    assert f"basekv={SETTINGS['mv_kv']}" in master and "138" not in master


def test_changed_byte_stops(tmp_path, capsys):
    """One changed byte in an input stops the build with exit code 2 and writes no output."""
    extract, reference = site.write_site(tmp_path)
    config = site.write_site_file(tmp_path, extract, reference)
    data = bytearray(extract.read_bytes())
    data[-2] ^= 1
    extract.write_bytes(bytes(data))
    assert build.main(["--config", str(config), "--settings", str(PROJECT)]) == 2
    assert "checksum" in capsys.readouterr().out and not (tmp_path / "built").exists()


def test_site_file_missing_key(tmp_path):
    """A site file without a required key stops with a build error."""
    (tmp_path / "site.toml").write_text('name = "x"\n', encoding="ascii")
    with pytest.raises(build.BuildError, match="no out"):
        build.read_site(tmp_path / "site.toml")


def test_no_hand_edits(built, tmp_path):
    """The manifest check passes on the written files and fails after one changed byte."""
    folder = tmp_path / "copy"
    folder.mkdir()
    for name in list(writer.DSS_FILES) + [writer.MANIFEST]:
        (folder / name).write_bytes((built[0] / "built" / name).read_bytes())
    assert verify.no_hand_edits(folder / "Master.dss")[0]
    with (folder / "Loads.dss").open("a", encoding="ascii") as handle:
        handle.write("! edited\n")
    passed, detail = verify.no_hand_edits(folder / "Master.dss")
    assert not passed and detail.endswith("1 changed")
    assert verify.no_hand_edits(tmp_path / "Master.dss") == (False, "no manifest beside the master")


@needs_opendss
def test_verify_built_feeder(built):
    """The built feeder compiles, solves, is radial, energizes every load and keeps its bytes."""
    summary, segments = verify.verify(built[0] / "built" / "Master.dss", SETTINGS)
    assert all(summary["criteria"].values()), summary
    counts = summary["counts"]
    assert counts["loads"] == 5 and counts["voltage_sources"] == 1 and counts["loads_without_upstream_transformer"] == 0
    assert counts["distribution_transformers"] == built[1]["transformers"] and segments


@needs_opendss
def test_verify_finds_loop(built, tmp_path):
    """A line added by hand between two buses of the tree makes a loop, so the radial criterion fails."""
    folder = tmp_path / "loop"
    folder.mkdir()
    for name in writer.DSS_FILES:
        (folder / name).write_bytes((built[0] / "built" / name).read_bytes())
    text = (folder / "Lines.dss").read_text(encoding="ascii")
    first, second = re.search(r"New Line\.mv_\S+ bus1=(\S+) bus2=(\S+)", text).groups()
    other = [bus for bus in re.findall(r"New Line\.mv_\S+ bus1=\S+ bus2=(\S+)", text) if bus != second]
    with (folder / "Lines.dss").open("a", encoding="ascii") as handle:
        handle.write(f"New Line.loop bus1={second} bus2={other[0] if other else first} phases=3 linecode=mv_overhead "
                     "length=0.1 units=km\n")
    summary, _ = verify.verify(folder / "Master.dss", SETTINGS)
    assert not summary["criteria"]["radial"] and not summary["criteria"]["no_hand_edits"]

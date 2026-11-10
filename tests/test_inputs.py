"""Tests of the inputs module.

Each test runs on files in a temporary folder or on the test files in the sample data folder.
No test uses the network, because each download test reads a local file through a file address.
A bad case plants one fault, such as one changed byte, and the module must stop with an input error.
The download command test runs on a project without an input map.
"""

import email.utils
import hashlib
import sqlite3
import urllib.error
import zipfile
from pathlib import Path

import geopandas
import pyogrio
import pytest
import shapely

from dxfeeder import inputs
from dxfeeder.inputs import InputError

PROJECT = Path(__file__).resolve().parents[1]
SAMPLE = PROJECT / "tests" / "sample_data"


def test_read_input_map_missing(tmp_path):
    """A project without an input map must stop with an input error."""
    with pytest.raises(InputError, match="input map"):
        inputs.read_input_map(tmp_path)


def test_file_checksum(tmp_path):
    """The checksum of a file equals the SHA-256 checksum of its bytes."""
    (tmp_path / "sample.txt").write_bytes(b"abc")
    assert inputs.file_checksum(tmp_path / "sample.txt") == hashlib.sha256(b"abc").hexdigest()


def test_download_file_address(tmp_path):
    """A download from a file address copies the bytes and leaves no partial file."""
    (tmp_path / "source.txt").write_bytes(b"made-up input")
    target = inputs.download((tmp_path / "source.txt").as_uri(), tmp_path / "inputs" / "copy.txt")
    assert target.read_bytes() == b"made-up input"
    assert [path.name for path in (tmp_path / "inputs").iterdir()] == ["copy.txt"]


def test_download_failure(tmp_path):
    """A failed download stops with a message that names the address and the target, and leaves no file."""
    address, target = (tmp_path / "missing.txt").as_uri(), tmp_path / "inputs" / "copy.txt"
    with pytest.raises(InputError) as caught:
        inputs.download(address, target)
    assert address in str(caught.value) and str(target) in str(caught.value) and "browser" in str(caught.value)
    assert list((tmp_path / "inputs").iterdir()) == []


def test_ensure_file_changed_byte(tmp_path):
    """A copy of one input passes its checksum, and the same copy with one changed byte fails."""
    copy = tmp_path / "data" / "inputs" / "made_up.csv"
    copy.parent.mkdir(parents=True)
    # A synthetic two-row table written by the test; it stands for any held input file.
    copy.write_text("code,value\nA,1\nB,2\n", encoding="ascii")
    row = {"name": "made_up_input", "address": "made-up note", "local_path": "data/inputs/made_up.csv",
           "checksum": inputs.file_checksum(copy)}
    assert inputs.ensure_file(tmp_path, row) == "Checked made_up_input"
    content = bytearray(copy.read_bytes())
    content[0] ^= 1
    copy.write_bytes(bytes(content))
    with pytest.raises(InputError, match="checksum"):
        inputs.ensure_file(tmp_path, row)


def test_ensure_file_download(tmp_path):
    """A missing file with an address downloads once, and the next call only checks it."""
    (tmp_path / "source.txt").write_bytes(b"made-up input")
    row = {"name": "copy", "address": (tmp_path / "source.txt").as_uri(), "local_path": "data/copy.txt",
           "checksum": inputs.file_checksum(tmp_path / "source.txt")}
    assert inputs.ensure_file(tmp_path, row) == "Downloaded copy"
    assert inputs.ensure_file(tmp_path, row) == "Checked copy"


@pytest.mark.parametrize("checksum, message", [("", "no checksum"), ("0" * 64, "held copy")])
def test_ensure_file_bad(tmp_path, checksum, message):
    """A row without a checksum, or a missing held copy, must stop with an input error."""
    row = {"name": "held", "address": "made-up note", "local_path": "data/held.json", "checksum": checksum}
    with pytest.raises(InputError, match=message):
        inputs.ensure_file(tmp_path, row)


@pytest.mark.parametrize("name, code", [(inputs.CENSUS, 403), (inputs.CENSUS, 404), ("made_up_input", 403)])
def test_ensure_file_census_403(tmp_path, monkeypatch, name, code):
    """A made-up refusal stops with an input error that names the status, and adds no pointer to other documents."""
    def refuse(address, timeout):
        """Refuse every request with the made-up status."""
        raise urllib.error.HTTPError(address, code, "made-up refusal", {}, None)

    monkeypatch.setattr(inputs.urllib.request, "urlopen", refuse)
    row = {"name": name, "address": "https://example.com/made_up.zip", "local_path": "data/made_up.zip", "checksum": "0" * 64}
    with pytest.raises(InputError) as caught:
        inputs.ensure_file(tmp_path, row)
    assert "input_sources.md" not in str(caught.value)
    assert f"HTTP Error {code}" in str(caught.value)


def raster_row(path, size=None):
    """Return an input map row for a made-up raster file, with its size and date from the file system."""
    return {"name": "made_up_raster", "address": path.as_uri(), "size_bytes": str(size or path.stat().st_size),
            "last_modified": email.utils.formatdate(path.stat().st_mtime, usegmt=True), "etag": ""}


def test_check_raster(tmp_path):
    """A raster with the size, date, and entity tag of the input map passes, and a changed size fails."""
    (tmp_path / "raster.tif").write_bytes(b"made-up raster")
    assert inputs.check_raster(raster_row(tmp_path / "raster.tif")).startswith("Raster made_up_raster unchanged")
    with pytest.raises(InputError, match="changed"):
        inputs.check_raster(raster_row(tmp_path / "raster.tif", size=99))


def test_check_raster_failure(tmp_path):
    """A raster address that does not answer must stop with an input error."""
    row = {"name": "made_up_raster", "address": (tmp_path / "missing.tif").as_uri()}
    with pytest.raises(InputError, match="identity request"):
        inputs.check_raster(row)


def spatial_index_parts(path):
    """Return the spatial index tables and the spatial index extension rows that a GeoPackage holds."""
    with sqlite3.connect(path) as connection:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        rows = connection.execute("SELECT extension_name FROM gpkg_extensions").fetchall() if "gpkg_extensions" in tables else []
    return [table for table in tables if table.startswith("rtree_")] + [row[0] for row in rows if "rtree" in row[0]]


def test_write_gpkg_spatial_index(tmp_path):
    """A GeoPackage from the writer holds a spatial index, as the same rows from the default writer do."""
    frame = geopandas.GeoDataFrame({"name": ["a", "b"]}, geometry=[shapely.LineString([(0, 0), (1, 1)])] * 2, crs="EPSG:3347")
    assert inputs.write_gpkg(frame, tmp_path / "small.gpkg", "small", "made-up note")
    assert spatial_index_parts(tmp_path / "small.gpkg")
    pyogrio.write_dataframe(frame, tmp_path / "default.gpkg", layer="small", driver="GPKG")
    assert spatial_index_parts(tmp_path / "default.gpkg")


def test_vector_source(tmp_path):
    """A zipped file geodatabase opens through its inner folder, and any other file opens by its path."""
    with zipfile.ZipFile(tmp_path / "lines.zip", "w") as archive:
        archive.writestr("made_up.gdb/a00000001.gdbtable", b"made-up table")
    with zipfile.ZipFile(tmp_path / "plain.zip", "w") as archive:
        archive.writestr("plain.txt", b"text")
    assert inputs.vector_source(tmp_path / "lines.zip") == f"/vsizip/{(tmp_path / 'lines.zip').as_posix()}/made_up.gdb"
    assert inputs.vector_source(tmp_path / "plain.zip") == str(tmp_path / "plain.zip")
    assert inputs.vector_source(SAMPLE / "census_divisions.gpkg") == str(SAMPLE / "census_divisions.gpkg")


def test_run_download_missing_map(tmp_path, capsys):
    """A project without an input map fails with exit code 1 and a plain message."""
    assert inputs.run_download(tmp_path) == 1
    assert "Download failed: the input map" in capsys.readouterr().out

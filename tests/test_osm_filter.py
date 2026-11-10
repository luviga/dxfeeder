"""Tests of the OpenStreetMap filter.

Each test runs on small made-up OpenStreetMap files in a temporary folder, with no network.
The voltage tests cover tags with one value, with several values, and with a value that does not read as a number.
The filter test checks the kept ways, the copy that a tie keeps, and the location tag.
"""

import pytest

from dxfeeder import osm_filter


def osm_text(nodes, ways):
    """Return the text of a small made-up OpenStreetMap file near Halifax with the given nodes and ways."""
    parts = ['<?xml version="1.0" encoding="UTF-8"?><osm version="0.6">']
    parts += [f'<node id="{node}" version="1" lat="{44.6 + node / 1000:.4f}" lon="{-63.6 + node / 1000:.4f}"/>'
              for node in nodes]
    for way, refs, tags in ways:
        parts.append(f'<way id="{way}" version="1">' + "".join(f'<nd ref="{ref}"/>' for ref in refs)
                     + "".join(f'<tag k="{key}" v="{value}"/>' for key, value in tags.items()) + "</way>")
    return "".join(parts) + "</osm>"


FIRST = osm_text(range(1, 9), [
    (10, [1, 2], {"power": "line", "voltage": "138000;69000"}),
    (11, [3, 4], {"power": "cable", "location": "underwater"}),
    (12, [5, 6], {"construction:power": "line"}),
    (13, [6, 7], {"power": "minor_line", "voltage": "69000"}),
    (14, [7, 8], {"power": "minor_line", "voltage": "25000"}),
    (15, [8, 99], {"power": "line"}),
])
SECOND = osm_text([1, 2, 3, 4, 9], [(10, [1, 9, 2], {"power": "line"}), (11, [4, 3], {"power": "cable"})])


def osm_files(root):
    """Write the two made-up OpenStreetMap files in alphabetical order of their names and return their paths."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "first.osm").write_text(FIRST, encoding="ascii")
    (root / "second.osm").write_text(SECOND, encoding="ascii")
    return [root / "first.osm", root / "second.osm"]


@pytest.mark.parametrize("tag, expected", [
    ("138000", [138.0]), ("138000;69000", [138.0, 69.0]), ("69000; 345000", [69.0, 345.0]), ("12500", [12.5]),
    ("medium", None), ("138 kV", None), ("138000;", None), ("", None), (None, None),
])
def test_voltage_values(tag, expected):
    """The function returns every value of a voltage tag in kV in tag order, and None for a tag that does not read as numbers."""
    assert osm_filter.voltage_values(tag) == expected


@pytest.mark.parametrize("tag, expected", [
    ("138000", 138.0), ("138000;69000", 138.0), ("69000; 345000", 345.0), ("13800.5", 13.8005),
    ("medium", None), ("138 kV", None), ("138000;", None), ("", None), (None, None),
])
def test_voltage_kv(tag, expected):
    """The function returns the highest value of a voltage tag in kV, and None for a tag that does not read as numbers."""
    assert osm_filter.voltage_kv(tag) == expected


@pytest.mark.parametrize("tags, kept", [
    ({"power": "line"}, True), ({"power": "cable"}, True), ({"power": "minor_line", "voltage": "50000"}, True),
    ({"power": "minor_line", "voltage": "34500"}, False), ({"power": "minor_line"}, False),
    ({"construction:power": "line"}, False), ({"disused:power": "cable"}, False), ({"power": "tower"}, False),
    ({"power": "minor_underground_cable"}, False),
])
def test_keeps_way(tags, kept):
    """The grid definition keeps lines, cables, and minor lines at 50 kV or more, with the plain power key only."""
    assert osm_filter.keeps_way(tags) is kept


def test_filter_osm(tmp_path):
    """The filter keeps three ways, takes the copy with more nodes, and breaks a tie by file order."""
    frame = osm_filter.filter_osm(osm_files(tmp_path / "osm"))
    assert frame["way_id"].tolist() == [10, 11, 13]
    assert frame.crs == "EPSG:4326"
    assert [len(shape.coords) for shape in frame.geometry] == [3, 2, 2]
    assert frame.geometry[1].coords[0] == pytest.approx((-63.597, 44.603))
    assert frame["voltage"].isna()[0] and frame["voltage"][2] == "69000"
    assert frame["location"].isna()[0] and frame["location"][1] == "underwater" and frame["location"].isna()[2]

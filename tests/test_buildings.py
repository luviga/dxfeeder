"""Tests of the building load points.

Each test runs on the synthetic site of synthetic_site.py, written to a temporary folder, with no network.
A closed building way and a building relation each give one load point inside the site; a building tagged building=no and a
building outside the site give none. Each load carries the demand and power factor of the settings file.
"""

from pathlib import Path

import pytest
import shapely

import synthetic_site as site
from dxfeeder import buildings, inputs

PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = inputs.read_settings(PROJECT)


@pytest.fixture(scope="module")
def points(tmp_path_factory):
    """Return the load points of the synthetic site."""
    extract, _ = site.write_site(tmp_path_factory.mktemp("buildings"))
    return buildings.load_points(extract, site.site_shape(), SETTINGS["building_skipped_values"])


def test_load_points_ids(points):
    """The five building ways inside the site and the building relation give load points, in id order."""
    assert [point[0] for point in points] == ["r301", "w201", "w202", "w203", "w204", "w207"]


def test_load_points_on_footprint(points):
    """Each load point lies on its square footprint."""
    corners = {f"w{way_id}": (east, north) for way_id, (east, north, _) in site.BUILDINGS.items()}
    corners["r301"] = site.RELATION_CORNER
    for load_id, lon, lat in points:
        assert shapely.Polygon(site.square(*corners[load_id])).buffer(1e-9).contains(shapely.Point(lon, lat))


def test_is_building():
    """A building tag counts unless its value is skipped, and a feature without the tag is no building."""
    assert buildings.is_building({"building": "house"}, ["no"]) and not buildings.is_building({"building": "no"}, ["no"])
    assert not buildings.is_building({"amenity": "school"}, ["no"])


def test_footprint_open_ring():
    """A list of places that closes no ring gives no footprint."""
    assert buildings.footprint([[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]) is None
    assert buildings.footprint([]) is None


def test_loads_table(points):
    """Every load takes the demand and the power factor of the settings file."""
    table = buildings.loads_table(points, SETTINGS)
    assert [row["id"] for row in table] == [point[0] for point in points]
    assert {(row["kw"], row["pf"]) for row in table} == {(SETTINGS["load_kw_default"], SETTINGS["load_pf"])}

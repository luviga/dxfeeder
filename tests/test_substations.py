"""Tests of the substation readers.

Each test runs on the synthetic site of synthetic_site.py, written to a temporary folder, with no network.
The reference layer and the OpenStreetMap file each give the one substation of the site, and a point outside the site is left out.
"""

import pytest
import shapely

import synthetic_site as site
from dxfeeder import substations


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """Return the extract and the reference GeoPackage of the synthetic site."""
    return site.write_site(tmp_path_factory.mktemp("substations"))


def test_from_layer(files):
    """The reference layer gives the substation with its OpenStreetMap type letter and id."""
    found = substations.from_layer(files[1], "substations", site.site_shape())
    assert [row[0] for row in found] == [f"n{site.SUBSTATION[0]}"]
    assert found[0][1:] == pytest.approx(site.place(*site.SUBSTATION[1]))


def test_from_osm(files):
    """The extract gives the node tagged power=substation."""
    found = substations.from_osm(files[0], site.site_shape(), "power", "substation")
    assert [row[0] for row in found] == [f"n{site.SUBSTATION[0]}"]


def test_keep_inside():
    """Rows outside the site shape are left out and the rest come sorted by id."""
    shape = shapely.box(0, 0, 1, 1)
    assert substations.keep_inside(shape, [("b", 0.5, 0.5), ("c", 2.0, 2.0), ("a", 1.0, 0.5)]) == [("a", 1.0, 0.5), ("b", 0.5, 0.5)]

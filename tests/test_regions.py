"""Tests of the region shapes module.

The sample test reads the census test file.
The made-up tests use squares and circles on the map, so each way two shapes can meet has a small case.
"""

from pathlib import Path

import pandas
import pytest
import shapely

from dxfeeder import inputs, regions

SAMPLE = Path(__file__).resolve().parent / "sample_data"
PATHS = {inputs.CENSUS: SAMPLE / "census_divisions.gpkg"}


def test_read_census():
    """The census test file gives 44 divisions in EPSG:3347, with 4-digit codes in sorted order."""
    census = regions.read_census(PATHS[inputs.CENSUS])
    assert len(census) == 44 and census.crs == regions.MAP_CRS
    assert census["CDUID"].tolist() == sorted(census["CDUID"]) and set(census["CDUID"].str.len()) == {4}


@pytest.mark.parametrize("second, length, touches", [
    (shapely.box(10, 0, 20, 10), 10.0, True), (shapely.box(10, 10, 20, 20), 0.0, True), (shapely.box(30, 0, 40, 10), 0.0, False),
    (shapely.Polygon([(12, 0), (20, 0), (20, 20), (0, 20), (0, 12), (12, 12)]), 0.0, False),
])
def test_shared_boundary(second, length, touches):
    """A shared edge gives its length, a shared corner gives a touch, and a gap gives no touch, also when the boxes overlap."""
    seam, found = regions.shared_boundary(shapely.box(0, 0, 10, 10), second)
    assert seam.length == pytest.approx(length) and found is touches


@pytest.mark.parametrize("order", [0, 1])
def test_gap_line(order):
    """The indexed gap line equals the shortest line between the whole shapes, in either order of the two regions."""
    shapes = [shapely.Point(0, 0).buffer(50, quad_segs=64), shapely.box(80, 30, 90, 200)]
    first, second = shapes[order], shapes[1 - order]
    assert len(regions.boundary_pieces(shapes[0])) > 1
    assert shapely.equals_exact(regions.gap_line(first, second), shapely.shortest_line(first, second), 0.0)


def test_csv_text():
    """The table text gives each length with 6 decimals and leaves a missing gap length blank."""
    frame = pandas.DataFrame({"corridor_id": ["X__Y"], "seam_km": [1.5], "gap_km": [None]})
    assert regions.csv_text(frame) == "corridor_id,seam_km,gap_km\nX__Y,1.500000,\n"

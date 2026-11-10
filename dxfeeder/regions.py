"""Region shapes and their boundaries.

This module reads census divisions and measures how two region shapes meet.
It finds the shared boundary of two shapes, or the shortest gap line between them when they do not touch.
It measures each length on the map in EPSG:3347.
It writes no file.
"""

import io

import numpy
import pyogrio
import shapely

from dxfeeder.inputs import vector_source

MAP_CRS = "EPSG:3347"
# The boundary clip box reaches 1 km past the overlap of the two region boxes, so no shared boundary lies on its edge.
CLIP_MARGIN_M = 1000.0
# Short boundary pieces keep each exact distance in the gap line search cheap.
PIECE_SEGMENTS = 64


def read_census(path):
    """Return the census divisions in EPSG:3347, with each code padded to 4 digits and the rows sorted by code."""
    census = pyogrio.read_dataframe(vector_source(path)).to_crs(MAP_CRS)
    census["CDUID"] = census["CDUID"].str.zfill(4)
    return census.sort_values("CDUID").reset_index(drop=True)


def shared_boundary(first, second):
    """Return the shared boundary of two regions as lines, and True if the regions touch at all."""
    overlap = shapely.box(*first.bounds).intersection(shapely.box(*second.bounds))
    if overlap.is_empty:
        return shapely.MultiLineString(), False
    low_x, low_y, high_x, high_y = overlap.bounds
    box = (low_x - CLIP_MARGIN_M, low_y - CLIP_MARGIN_M, high_x + CLIP_MARGIN_M, high_y + CLIP_MARGIN_M)
    meeting = shapely.clip_by_rect(first.boundary, *box).intersection(shapely.clip_by_rect(second.boundary, *box))
    parts = shapely.get_parts(shapely.get_parts(meeting))
    lines = [part for part in parts if part.geom_type == "LineString" and not part.is_empty]
    return shapely.MultiLineString(lines), not meeting.is_empty


def boundary_pieces(shape):
    """Return the boundary of a shape cut into short lines of at most 64 segments each."""
    pieces = []
    for ring in shapely.get_parts(shape.boundary):
        coords = shapely.get_coordinates(ring)
        pieces += [shapely.LineString(coords[start:start + PIECE_SEGMENTS + 1]) for start in range(0, len(coords) - 1, PIECE_SEGMENTS)]
    return pieces


def gap_line(first, second):
    """Return the shortest line from the first region to the second, found through an index of boundary pieces.

    The line has the length of the shortest line between the whole shapes, at a small part of the cost.
    """
    pieces = [boundary_pieces(first), boundary_pieces(second)]
    # The index holds the longer list of pieces, because a query from the shorter list then costs least.
    indexed = 0 if len(pieces[0]) >= len(pieces[1]) else 1
    (asked, found), distances = shapely.STRtree(pieces[indexed]).query_nearest(pieces[1 - indexed], return_distance=True)
    best = int(numpy.argmin(distances))
    pair = {indexed: pieces[indexed][found[best]], 1 - indexed: pieces[1 - indexed][asked[best]]}
    return shapely.shortest_line(pair[0], pair[1])


def csv_text(frame):
    """Return a table as CSV text with LF line endings and each number in km with 6 decimals."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.6f")
    return buffer.getvalue()

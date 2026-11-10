"""Distribution transformers of the feeders.

This module walks each feeder tree from the leaves toward the substation and groups the served loads into transformers.
Each load enters the walk at its attached node, with a reach equal to its service drop length.
The reach of a load grows by the edge length each time its group moves one node up.
A group closes at the current node when one more step up would take a load beyond the transformer radius, when the group holds
the largest number of loads per transformer, or at the substation; the transformer sits at the node where its group closes.
A node that gathers more loads than one transformer takes closes full groups of the farthest loads first.
Each transformer is sized from its apparent load over the loading factor, rounded up to the next standard size.
"""

from dxfeeder.network import post_order


def standard_size(kva, sizes):
    """Return the smallest standard size at or above the given kVA, or the largest size and False when none is large enough."""
    for size in sorted(sizes):
        if size >= kva:
            return size, True
    return max(sizes), False


def group_loads(forest, loads, settings):
    """Return the transformers of the forest as dictionaries with node, loads and kVA, in node and group order.

    Each entry of "loads" is a pair (load id, reach in metres from the transformer node along the tree plus the drop).
    """
    limit = int(settings["loads_per_transformer_max"])
    radius = float(settings["transformer_radius_m"])
    demand = {load["id"]: load["kw"] / load["pf"] for load in loads}
    own = {}
    for load_id in sorted(forest["attached"]):
        node, drop = forest["attached"][load_id]
        own.setdefault(node, []).append((drop, load_id))
    groups = []
    for root in forest["roots"]:
        pending = {}
        for node in post_order(forest, root):
            items = list(own.get(node, []))
            for child in forest["children"].get(node, []):
                step = forest["length"][child]
                items += [(reach + step, load_id) for reach, load_id in pending.pop(child, [])]
            items.sort(key=lambda item: (-item[0], item[1]))
            closed = []
            while len(items) > limit:
                closed.append(items[:limit])
                items = items[limit:]
            if items and (node == root or len(items) == limit or max(item[0] for item in items) + forest["length"][node] > radius):
                closed.append(items)
                items = []
            pending[node] = items
            groups += [(node, group) for group in closed]
    transformers = []
    counts = {}
    for node, group in groups:
        counts[node] = counts.get(node, 0) + 1
        kva = sum(demand[load_id] for _, load_id in group) / float(settings["transformer_loading_factor"])
        size, fits = standard_size(kva, settings["transformer_sizes_kva"])
        transformers.append({"node": node, "number": counts[node], "kva": size, "fits": fits,
                             "loads": sorted((load_id, reach) for reach, load_id in group)})
    return sorted(transformers, key=lambda item: (item["node"], item["number"]))

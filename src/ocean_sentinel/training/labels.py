"""Taxonomy -> binary label resolution.

The CNN's `label` field is flat binary for the hackathon. This module is the
ONE place that decides which taxonomy entries map to "ship" vs "not_ship".
Future heads (species, threat, vessel-type) read `taxonomy.*` directly and
bypass this module.
"""
from __future__ import annotations

from ocean_sentinel.training.schema import Label, Taxonomy


def binary_label_for(taxonomy: Taxonomy) -> Label:
    """Collapse rich taxonomy to the binary CNN label.

    Rule: only `category == "vessel"` is ship. Biological, ambient, and
    non-vessel anthropogenic sounds are all not_ship.
    """
    return "ship" if taxonomy.category == "vessel" else "not_ship"

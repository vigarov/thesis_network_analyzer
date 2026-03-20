"""Canonical encoding of per-unit identifiers for checkpoints and visualization.

``unit_node_ids`` must be self-describing: layer, unit type, and index are embedded
so consumers do not need a parallel ``units_meta`` array.

Format (current): ``{scope}|{layer_name}|{unit_type}|{unit_index}``

Legacy format (still parsed for older results): ``{scope}:{layer_name}:{unit_type}_{index}``
"""

from __future__ import annotations

import re
from typing import TypedDict

_LEGACY_RE = re.compile(r"^([^:]+):([^:]+):(neuron|channel)_(\d+)$")


class ParsedUnitNodeId(TypedDict):
    node_id: str
    layer_name: str
    unit_index: int
    unit_type: str


def format_unit_node_id(
    scope: str,
    layer_name: str,
    unit_type: str,
    unit_index: int,
) -> str:
    """Build a canonical unit node id (used as dict keys and in .npz files)."""
    if "|" in scope or "|" in layer_name:
        raise ValueError("scope and layer_name must not contain '|'")
    if unit_type not in ("neuron", "channel"):
        raise ValueError(f"unit_type must be 'neuron' or 'channel', got {unit_type!r}")
    return f"{scope}|{layer_name}|{unit_type}|{int(unit_index)}"


def parse_unit_node_id(node_id: str) -> ParsedUnitNodeId | None:
    """Parse a unit node id into structured fields, or ``None`` if unrecognized."""
    s = str(node_id).strip()
    if not s:
        return None

    parts = s.split("|")
    if len(parts) == 4:
        _scope, layer_name, unit_type, idx_s = parts
        if unit_type not in ("neuron", "channel"):
            return None
        try:
            idx = int(idx_s)
        except ValueError:
            return None
        return ParsedUnitNodeId(
            node_id=s,
            layer_name=layer_name,
            unit_index=idx,
            unit_type=unit_type,
        )

    m = _LEGACY_RE.match(s)
    if m:
        _scope, layer_name, unit_type, idx_s = m.groups()
        return ParsedUnitNodeId(
            node_id=s,
            layer_name=layer_name,
            unit_index=int(idx_s),
            unit_type=unit_type,
        )

    return None

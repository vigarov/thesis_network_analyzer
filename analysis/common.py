"""Shared helpers used by the webapp and analysis notebooks."""

import numpy as np
import pandas as pd

from models.unit_node_id import parse_unit_node_id

# Optimizer palette 
_OPT_DEFAULT_COLOR = "#7c3aed"
_OPT_PAL = {
	"sgd": "#2563eb",
	"adam": "#b45309",
	"adagrad": "#16a34a",
	"pure_shampoo": "#0891b2",
	"grafted_shampoo": "#c026d3",
}

#  Plot/dropdown optimizer order (unknown families sort last, then by id).
_OPT_ORDER_PREFIXES = (
	"sgd",
	"adagrad",
	"adam",
	"pure_shampoo",
	"grafted_shampoo",
)
def _sort_optimizer_ids(oids: list[str]) -> list[str]:
	def _key(oid: str) -> tuple[int, str]:
		for i, prefix in enumerate(_OPT_ORDER_PREFIXES):
			if oid.startswith(prefix):
				return (i, oid)
		return (len(_OPT_ORDER_PREFIXES), oid)

	return sorted(oids, key=_key)


def _opt_color(oid: str) -> str:
	for prefix, color in _OPT_PAL.items():
		if oid.startswith(prefix):
			return color
	return _OPT_DEFAULT_COLOR


def _opt_label(oid: str) -> str:
	return oid.split("_lr")[0].upper() if "_lr" in oid else oid


def _opt_display(oid: str) -> str:
	"""Human-readable optimizer name for plot ticks / printouts (e.g. `Grafted shampoo`)."""
	no_lr = oid.rsplit("_lr", 1)[0] if "_lr" in oid else oid
	return no_lr.replace("_", " ").capitalize()


def _opt_type(oid: str) -> str:
	if oid.startswith("pure_shampoo"):
		return "pure_shampoo"
	if oid.startswith("grafted_shampoo"):
		return "grafted_shampoo"
	for t in ("sgd", "adam", "adagrad"):
		if oid.startswith(t):
			return t
	return "unknown"


def _layer_name_short(layer_name: str) -> str:
	"""Compact layer token for neuron labels (e.g. `hidden.0` → `h0`)."""
	ln = layer_name.strip()
	if ln.startswith("hidden."):
		return "h" + ln.split(".", 1)[1]
	if ln == "head":
		return "head"
	if "." in ln:
		a, b = ln.split(".", 1)
		return (a[0].lower() if a else "?") + b
	return ln


def _compact_neuron_label(nid: str) -> str:
	"""Display form `<layer short>:<unit index>` (e.g. `h0:16`)."""
	parsed = parse_unit_node_id(nid)
	if parsed is not None:
		return f"{_layer_name_short(parsed['layer_name'])}:{parsed['unit_index']}"
	parts = str(nid).strip().split("|")
	if len(parts) == 4:
		try:
			idx = int(parts[3])
		except ValueError:
			return str(nid)
		return f"{_layer_name_short(parts[1])}:{idx}"
	return str(nid)


def _is_dead_column_to_bool(s: pd.Series) -> np.ndarray:
	"""Parse `is_dead` values that may be bool-like strings."""
	dm = pd.Series(s).fillna(False)
	if dm.dtype == object:
		return dm.astype(str).str.lower().isin(("true", "1", "t")).to_numpy(dtype=bool)
	return dm.astype(bool).to_numpy()

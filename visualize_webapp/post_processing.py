import numpy as np
import pandas as pd
from typing import Iterable

from visualize_webapp.common import _is_dead_column_to_bool


# `pp` stands for post-processing
# See compute_results/post_processing.py


def _pp_train_act(
	e: str, m: str, o: str, *, rid: str, w_cache: bool = True
) -> dict[str, np.ndarray]:
	raise NotImplementedError("Not using the post-train full activations anymore.")
	# return _npz(e, m, o, "post_processing_train_act.npz", rid=rid, w_cache=w_cache)


def _ever_dead_and_ppd_for_diagram(
	df_dead: pd.DataFrame | None,
) -> tuple[set[str] | None, set[str] | None]:
	"""Cumulative dead/PPD sets for architecture coloring."""
	if df_dead is None or df_dead.empty:
		return None, None
	dead_mask = _is_dead_column_to_bool(pd.Series(df_dead["is_dead"]))
	sub = df_dead.loc[dead_mask]
	if sub.empty:
		return None, None
	ever_dead = {str(x) for x in sub["neuron_id"].tolist()}
	if "is_ppd" in df_dead.columns:
		pr = pd.Series(sub["is_ppd"]).fillna(False)
		if pr.dtype == object:
			ppd_mask = pr.astype(str).str.lower().isin(("true", "1", "t"))
		else:
			ppd_mask = pr.fillna(False).astype(bool)
		ever_ppd = {str(x) for x in sub.loc[ppd_mask, "neuron_id"].tolist()}
	else:
		ever_ppd = set(ever_dead)
	gray_dead = ever_dead - ever_ppd
	return gray_dead, ever_ppd


def _layer_recovery_events_cumsum(
	layer_df: pd.DataFrame,
	all_cp_idxs: list[int],
) -> np.ndarray:
	"""Per checkpoint cumulative dead→alive transitions for one layer."""
	n_cp = len(all_cp_idxs)
	if n_cp == 0:
		return np.array([], dtype=np.int64)
	all_neurons = layer_df["neuron_id"].unique()
	if len(all_neurons) == 0:
		return np.zeros(n_cp, dtype=np.int64)

	sub = layer_df[["neuron_id", "checkpoint_idx", "is_dead"]]
	full_idx = pd.DataFrame(
		[(n, c) for n in all_neurons for c in all_cp_idxs],
		columns=["neuron_id", "checkpoint_idx"],
	)
	merged = full_idx.merge(sub, on=["neuron_id", "checkpoint_idx"], how="left")
	dead = _is_dead_column_to_bool(merged["is_dead"].fillna(False))

	events = np.zeros(n_cp, dtype=np.int64)
	n_per = n_cp
	for i, _nid in enumerate(all_neurons):
		d = dead[i * n_per : (i + 1) * n_per]
		for j in range(n_cp - 1):
			if d[j] and not d[j + 1]:
				events[j + 1] += 1
	return np.cumsum(events)


def compute_recovery_cumsum_by_layer(
	df_dead: pd.DataFrame,
	layer_order: list[str],
) -> tuple[list[int], dict[str, np.ndarray]]:
	"""Cumulative recovery counts per hidden layer (excludes `head`)."""
	hidden = [ln for ln in layer_order if ln != "head"]
	if not hidden or df_dead is None or df_dead.empty:
		return [], {}

	sub = df_dead[df_dead["layer_name"].isin(hidden)]
	if sub.empty:
		return [], {}

	all_cp_idxs = sorted(x for x in sub["checkpoint_idx"].unique().tolist())
	out: dict[str, np.ndarray] = {}
	for ln in hidden:
		layer_df = sub[sub["layer_name"] == ln]
		out[ln] = _layer_recovery_events_cumsum(layer_df, all_cp_idxs)
	return all_cp_idxs, out


def per_neuron_recovery_counts(
	df_dead: pd.DataFrame,
	layer_order: list[str],
) -> tuple[dict[str, int], dict[str, list[int]]]:
	"""Per-neuron dead→alive recovery counts and destination checkpoints."""
	hidden = [ln for ln in layer_order if ln != "head"]
	if not hidden or df_dead is None or df_dead.empty:
		return {}, {}

	sub = df_dead[df_dead["layer_name"].isin(hidden)]
	if sub.empty:
		return {}, {}

	nids = [str(x) for x in sub["neuron_id"].unique()]
	counts: dict[str, int] = {nid: 0 for nid in nids}
	recovery_checkpoint_idx: dict[str, list[int]] = {nid: [] for nid in nids}
	for ln in hidden:
		layer_df = sub[sub["layer_name"] == ln]
		all_cp_idxs = sorted(x for x in layer_df["checkpoint_idx"].unique().tolist())
		n_cp = len(all_cp_idxs)
		if n_cp < 2:
			continue
		all_neurons = layer_df["neuron_id"].unique()
		cols = layer_df[["neuron_id", "checkpoint_idx", "is_dead"]]
		full_idx = pd.DataFrame(
			[(n, c) for n in all_neurons for c in all_cp_idxs],
			columns=["neuron_id", "checkpoint_idx"],
		)
		merged = full_idx.merge(cols, on=["neuron_id", "checkpoint_idx"], how="left")
		dead = _is_dead_column_to_bool(merged["is_dead"].fillna(False))
		n_per = n_cp
		for i, nid in enumerate(all_neurons):
			d = dead[i * n_per : (i + 1) * n_per]
			key = str(nid)
			for j in range(n_cp - 1):
				if d[j] and not d[j + 1]:
					counts[key] += 1
					recovery_checkpoint_idx[key].append(int(all_cp_idxs[j + 1]))
	return counts, recovery_checkpoint_idx


def split_ever_dead_by_final_checkpoint(
	df_dead: pd.DataFrame,
) -> tuple[set[str], set[str]]:
	# Unused
	"""Return `(dead_at_last_checkpoint, recovered_by_last_checkpoint)`."""
	if df_dead.empty:
		return set(), set()
	last_cp = int(df_dead["checkpoint_idx"].max())
	final = df_dead[df_dead["checkpoint_idx"] == last_cp]
	dead_mask = _is_dead_column_to_bool(pd.Series(final["is_dead"]))
	dead_at_end = {str(x) for x in final.loc[dead_mask, "neuron_id"].tolist()}
	ever_mask = _is_dead_column_to_bool(pd.Series(df_dead["is_dead"]))
	ever_dead = {str(x) for x in df_dead.loc[ever_mask, "neuron_id"].tolist()}
	recovered = ever_dead - dead_at_end
	return dead_at_end, recovered

# ok mpl
def neuron_assigned_count_series(
	df_nd: pd.DataFrame,
	neuron_id: str,
) -> tuple[str | None, list[int], np.ndarray]:
	"""Assigned-digit count over checkpoints for one neuron. Returns (layer_name, all_cp_idxs (for that layer), y)"""
	nid_key = str(neuron_id)
	mask_nid = df_nd["neuron_id"].astype(str) == nid_key
	if not mask_nid.any():
		return None, [], np.array([], dtype=int)

	layer_name = str(df_nd.loc[mask_nid, "layer_name"].iloc[0])
	layer_df = df_nd[df_nd["layer_name"] == layer_name]
	all_cp_idxs = sorted(layer_df["checkpoint_idx"].unique())
	assigned = df_nd[mask_nid & (df_nd["status"] == "assigned")]
	per_cp = assigned.groupby("checkpoint_idx", sort=False).size()
	y = np.array([int(per_cp.get(ci, 0)) for ci in all_cp_idxs], dtype=int)
	return layer_name, all_cp_idxs, y


# ok mpl
def layer_inactive_count_lookup(
	df_nd: pd.DataFrame,
	layer_name: str,
) -> tuple[list[int], pd.Series]:
	"""Inactive counts indexed by `(checkpoint_idx, digit)` for a layer."""
	layer_df = df_nd[df_nd["layer_name"] == layer_name]
	all_cp_idxs = sorted(layer_df["checkpoint_idx"].unique())
	inactive = layer_df[layer_df["status"] == "inactive"]
	counts = inactive.groupby(["checkpoint_idx", "digit"]).size().reset_index(name="count")
	lookup = counts.set_index(["checkpoint_idx", "digit"])["count"]
	return all_cp_idxs, lookup


def layer_status_count_matrix(
	df_nd: pd.DataFrame,
	layer_name: str,
	status: str,
) -> tuple[list[int], np.ndarray, np.ndarray]:
	"""Return `(checkpoint_idxs, neuron_ids, counts_matrix)` for one layer/status.

	`counts_matrix` has shape `(n_neurons, n_checkpoints)` and stores, per
	neuron/checkpoint, how many digits have the requested status.
	"""
	layer_df = df_nd[df_nd["layer_name"] == layer_name]
	all_neurons = layer_df["neuron_id"].unique()
	all_cp_idxs = sorted(layer_df["checkpoint_idx"].unique())
	if len(all_neurons) == 0 or len(all_cp_idxs) == 0:
		return all_cp_idxs, all_neurons, np.zeros((0, len(all_cp_idxs)), dtype=np.float64)

	matched = layer_df[layer_df["status"] == status]
	per_nc = matched.groupby(["neuron_id", "checkpoint_idx"]).size().reset_index(name="n")
	full_idx = pd.DataFrame(
		[(n, c) for n in all_neurons for c in all_cp_idxs],
		columns=["neuron_id", "checkpoint_idx"],
	)
	per_nc = full_idx.merge(per_nc, on=["neuron_id", "checkpoint_idx"], how="left")
	per_nc["n"] = per_nc["n"].fillna(0).astype(int)

	mat = (
		per_nc.pivot(index="neuron_id", columns="checkpoint_idx", values="n")
		.reindex(index=all_neurons, columns=all_cp_idxs, fill_value=0)
		.to_numpy(dtype=np.float64)
	)
	return all_cp_idxs, all_neurons, mat


def dead_layer_inactive_trace_matrix(
	layer_name: str,
	df_nd: pd.DataFrame,
	df_dead: pd.DataFrame,
	*,
	neuron_ids: Iterable[str] | None = None,
) -> tuple[list[int], list[str], np.ndarray]:
	"""Inactive-digit count traces for dead neurons in one layer.

	Returns `(checkpoint_idxs, selected_neuron_ids, traces)` where `traces` has
	shape `(n_selected_neurons, n_checkpoints)`.
	"""
	layer_dead = df_dead[df_dead["layer_name"] == layer_name]
	dead_mask = _is_dead_column_to_bool(pd.Series(layer_dead["is_dead"]))
	raw = layer_dead.loc[dead_mask, "neuron_id"].unique()
	if neuron_ids is not None:
		allow = {str(x) for x in neuron_ids}
		selected = [str(n) for n in raw if str(n) in allow]
	else:
		selected = [str(n) for n in raw]

	layer_nd = df_nd[df_nd["layer_name"] == layer_name]
	all_cp_idxs = sorted(layer_nd["checkpoint_idx"].unique())
	if not selected or not all_cp_idxs:
		return all_cp_idxs, selected, np.zeros((0, len(all_cp_idxs)), dtype=np.float64)

	inactive_counts = (
		layer_nd[layer_nd["status"] == "inactive"]
		.groupby(["neuron_id", "checkpoint_idx"])
		.size()
		.reset_index(name="n_inactive")
	)
	mat = (
		inactive_counts.pivot(index="neuron_id", columns="checkpoint_idx", values="n_inactive")
		.reindex(index=selected, columns=all_cp_idxs, fill_value=0)
		.to_numpy(dtype=np.float64)
	)
	return all_cp_idxs, selected, mat


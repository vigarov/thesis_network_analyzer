import re
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from tqdm.auto import tqdm

from models.unit_node_id import parse_unit_node_id
from analysis.constants import _NETWORK_N_DIGITS, _NETWORK_SAMPLES_PER_DIGIT
from analysis.IQA_optimization.SteerPyrComplex import SteerablePyramid
from analysis.rdx import compute_G, distance_to_rank
from analysis.scoring_constants import (
	MNIST_IMAGE_SHAPE,
	N_SAMPLES_PER_TRIAL,
	OPTIMIZER_PREFIX_TO_REGISTRY_CLASS,
	PYRAMID_LEVELS,
	USE_CW_SSIM,
	WAVELET_ORIENTATIONS,
)

index_grp = ["neuron_id", "period_id"]

#_PRETRAIN_SHUFFLE_MISLABEL_ID = re.compile(r"^pretrain_shuffle_mislabel_K\d+_tr\d+$")
_PRETRAIN_SHUFFLE_MISLABEL_ID = re.compile(
	r"^cat1_sample_shuffle_control_tr\d+$"
	r"|^cat1_sample_shuffle_finetune_K\d+_tr\d+$"
	r"|^cat1_sample_shuffle_constrained_digits[\d-]+_tr\d+$"
	r"|^cat1_sample_shuffle_interleaved_digits[\d-]+_tr\d+$"
)


def is_pretrain_shuffle_mislabel_experiment(eid: str) -> bool:
	"""True for Category-1 sample-shuffle v2 experiment ids (`cat1_sample_shuffle_*`)."""
	return bool(_PRETRAIN_SHUFFLE_MISLABEL_ID.match(str(eid)))


def _unit_key(prefix: str, neuron_id: str) -> str:
	return f"{prefix}__{str(neuron_id).replace(':', '__')}"


def _dnn_hidden_layer_names_from_nts(nts) -> list[str]:
	layer_names = set()
	for nid in nts["unit_node_ids"]:
		parsed = parse_unit_node_id(str(nid))
		if parsed is not None and parsed["layer_name"].startswith("hidden."):
			layer_names.add(parsed["layer_name"])
	return sorted(layer_names, key=lambda name: int(name.split(".")[-1]))


def _dnn_hidden_layer_names_from_model(model) -> list[str]:
	layer_names = [name for name in model.hookable_layers() if name.startswith("hidden.")]
	return sorted(layer_names, key=lambda name: int(name.split(".")[-1]))


def _layer_index_from_neuron_id(neuron_id: str) -> int:
	neuron_info = parse_unit_node_id(neuron_id)
	assert neuron_info is not None
	layer_name = neuron_info["layer_name"]
	if not layer_name.startswith("hidden."):
		raise ValueError(f"Custom saliency maps currently support DNN hidden layers, got {layer_name!r}")
	return int(layer_name.split(".")[-1]) // 2


def custom_saliency_map(weights, activations, convolve_: bool = False):
	"""Compute dz_l/dx for each DNN hidden layer using ReLU activation masks."""
	L = len(weights)
	input_dim = weights[0].shape[1]

	F_layers = []
	for l in range(L):
		W = weights[l]
		if l > 0:
			F_prev = F_layers[-1]
			mask = (activations[l - 1] > 0).float()
			F_prev_use = F_prev.clone() * mask.unsqueeze(1)
		else:
			F_prev_use = torch.eye(input_dim, device=weights[0].device)

		F_l = W @ F_prev_use
		if convolve_:
			F_l = torch.tensor(
				convolve(F_l.clone().cpu().numpy(), gaussian_kernel_5x5(sigma=0.2, size=3), mode="reflect")
			).to(F_l.device)
		F_layers.append(F_l.clone())
	return torch.stack(F_layers)


def _all_custom_saliency_maps_from_weights_and_activations(weights, per_layer_activations, convolve_: bool = False):
	"""Return saliency maps with shape (n_samples, n_layers, n_units, 28, 28)."""
	all_neurons_saliency_maps = []
	for image_i in range(per_layer_activations.shape[-1]):
		digit_activations = per_layer_activations[:, :, image_i]
		full_jacobian = custom_saliency_map(weights, digit_activations, convolve_=convolve_)
		all_neurons_saliency_maps.append(full_jacobian.reshape(*full_jacobian.shape[:-1], *MNIST_IMAGE_SHAPE))
	return torch.stack(all_neurons_saliency_maps)


def _neuron_maps_from_all_saliency(all_saliency_maps, neuron_id: str):
	neuron_info = parse_unit_node_id(neuron_id)
	assert neuron_info is not None
	layer_idx = _layer_index_from_neuron_id(neuron_id)
	return [
		all_saliency_maps[sample_idx, layer_idx, neuron_info["unit_index"]].detach().cpu()
		for sample_idx in range(all_saliency_maps.shape[0])
	]


def get_neuron_saliency_maps_from_all(all_saliency_maps, neuron_id: str):
	"""Slice one neuron's flat sample maps from a precomputed all-neuron saliency tensor."""
	return _neuron_maps_from_all_saliency(all_saliency_maps, neuron_id)


def get_all_saliency_maps(nts, checkpoint_idx: int = 0, *, device=None, convolve_: bool = False):
	"""Compute custom saliency maps for all DNN hidden neurons at one checkpoint."""
	device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
	layer_names = _dnn_hidden_layer_names_from_nts(nts)
	if not layer_names:
		raise ValueError("No DNN hidden layers found in neuron timeseries")

	weights = []
	activations = []
	for layer_name in layer_names:
		layer_nids_with_indices = []
		for nid in nts["unit_node_ids"]:
			nid = str(nid)
			parsed = parse_unit_node_id(nid)
			if parsed is not None and parsed["layer_name"] == layer_name:
				layer_nids_with_indices.append((parsed["unit_index"], nid))
		layer_nids = [nid for _, nid in sorted(layer_nids_with_indices)]
		weights.append(torch.tensor(np.stack([nts[_unit_key("weights", nid)][checkpoint_idx] for nid in layer_nids])).float().to(device))
		activations.append(torch.tensor(np.stack([nts[_unit_key("act", nid)][checkpoint_idx] for nid in layer_nids])).float().to(device))

	with torch.no_grad():
		all_saliency_maps = _all_custom_saliency_maps_from_weights_and_activations(
			weights,
			torch.stack(activations),
			convolve_=convolve_,
		).cpu()
	del weights, activations
	if device.type == "cuda":
		torch.cuda.empty_cache()
	return all_saliency_maps


def _stack_depth_from_layer_name(layer_name) -> int:
	"""DNN hidden stack index: `hidden.{2k}` → `k` (matches `_layer_index_from_neuron_id`)."""
	return int(str(layer_name).split(".")[-1]) // 2


def get_all_saliency_maps_up_to_layer(
	nts,
	checkpoint_idx: int,
	max_stack_depth: int,
	*,
	device=None,
	convolve_: bool = False,
):
	"""Like `get_all_saliency_maps` but only Jacobians through hidden layers with stack depth ≤ `max_stack_depth`."""
	device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
	layer_names_full = _dnn_hidden_layer_names_from_nts(nts)
	if not layer_names_full:
		raise ValueError("No DNN hidden layers found in neuron timeseries")
	last_depth = _stack_depth_from_layer_name(layer_names_full[-1])
	max_stack_depth = int(min(max(max_stack_depth, 0), last_depth))
	layer_names = [ln for ln in layer_names_full if _stack_depth_from_layer_name(ln) <= max_stack_depth]

	weights = []
	activations = []
	for layer_name in layer_names:
		layer_nids_with_indices = []
		for nid in nts["unit_node_ids"]:
			nid = str(nid)
			parsed = parse_unit_node_id(nid)
			if parsed is not None and parsed["layer_name"] == layer_name:
				layer_nids_with_indices.append((parsed["unit_index"], nid))
		layer_nids = [nid for _, nid in sorted(layer_nids_with_indices)]
		weights.append(torch.tensor(np.stack([nts[_unit_key("weights", nid)][checkpoint_idx] for nid in layer_nids])).float().to(device))
		activations.append(torch.tensor(np.stack([nts[_unit_key("act", nid)][checkpoint_idx] for nid in layer_nids])).float().to(device))

	with torch.no_grad():
		all_saliency_maps = _all_custom_saliency_maps_from_weights_and_activations(
			weights,
			torch.stack(activations),
			convolve_=convolve_,
		).cpu()
	del weights, activations
	if device.type == "cuda":
		torch.cuda.empty_cache()
	return all_saliency_maps


def get_neuron_effective_receptive_field(nts, neuron_id: str, iteration: int = 0, *, device=None, convolve_: bool = False):
	"""Return custom saliency maps for one neuron at a checkpoint as a flat sample list."""
	all_saliency_maps = get_all_saliency_maps(nts, checkpoint_idx=iteration, device=device, convolve_=convolve_)
	return _neuron_maps_from_all_saliency(all_saliency_maps, neuron_id)


def get_all_model_saliency_maps(model, eval_digits, device, *, convolve_: bool = False):
	"""Capture model activations on eval digits and compute matching custom saliency maps."""
	layer_names = _dnn_hidden_layer_names_from_model(model)
	if not layer_names:
		raise ValueError("Custom saliency maps currently require DNN hidden layers")

	captured = {}
	hooks = []
	for layer_name in layer_names:
		def _hook(_module, _args, output, *, name=layer_name):
			captured[name] = output.detach()
		hooks.append(model.hookable_layers()[layer_name].register_forward_hook(_hook))

	was_training = model.training
	try:
		model.eval()
		with torch.no_grad():
			x = eval_digits.detach().to(device) if isinstance(eval_digits, torch.Tensor) else torch.as_tensor(eval_digits, device=device)
			model(x)
		if set(captured) != set(layer_names):
			missing = sorted(set(layer_names) - set(captured))
			raise RuntimeError(f"Missing hooked activations for layers: {missing}")

		weights = [model.hookable_layers()[layer_name].weight.detach().float().to(device) for layer_name in layer_names]
		activations = torch.stack([captured[layer_name].detach().float().to(device) for layer_name in layer_names])
		return _all_custom_saliency_maps_from_weights_and_activations(weights, activations.transpose(1, 2), convolve_=convolve_)
	finally:
		for hook in hooks:
			hook.remove()
		model.train(was_training)


def get_neuron_model_effective_receptive_field(model, neuron_id: str, eval_digits, device, *, convolve_: bool = False):
	all_saliency_maps = get_all_model_saliency_maps(model, eval_digits, device, convolve_=convolve_)
	return _neuron_maps_from_all_saliency(all_saliency_maps, neuron_id)


def nan_to_len(metrics, checkpoint_idx):
	if np.isnan(checkpoint_idx):
		return len(metrics["checkpoint_iterations"])
	else:
		assert isinstance(checkpoint_idx, (int, float))
		return int(checkpoint_idx)


def get_iteration(metrics, checkpoint_idx):
	cp_idx = nan_to_len(metrics, checkpoint_idx)
	return metrics["checkpoint_iterations"][cp_idx + 1] if cp_idx < len(metrics["checkpoint_iterations"]) - 1 else (metrics["checkpoint_iterations"][-1] + 1)



def get_n_assigned_digits(df_nd):
	"""
	Get the number of digits assigned to each neuron at each checkpoint,
	considering only rows where status == "assigned".
	Other neurons/checkpoints receive a value of 0.
	"""
	grouped = df_nd[df_nd["status"] == "assigned"].groupby(["neuron_id", "checkpoint_idx"])["digit"].count().rename("n_assigned_digits")
	full_index = pd.MultiIndex.from_frame(df_nd[["neuron_id", "checkpoint_idx"]].drop_duplicates())
	return grouped.reindex(full_index, fill_value=0)


def get_recovery_periods(metrics, df_nd, n_assigned_digits, df_dead, mode="assigned_any"):
	"""
	Get recovery periods for each neuron as [start, end) checkpoint intervals.

	A recovery period starts when the neuron becomes assigned after having been
	dead (is_dead at least once since the last period ended or since the start).

	mode="assigned_any"  (default)
		Period: dead -> assigned (to any digit) -> not assigned to any digit.
		While in a period the neuron may cycle through different digit
		assignments; the period ends only when n_assigned_digits drops to 0.

	mode="assigned_same"
		Period: dead -> assigned (to a set of digits D) -> not assigned to any
		digit in D anymore.  If the neuron picks up new digits during the period
		those new digits do NOT keep the period alive once all of D are gone.
		After the period ends, the neuron must become dead again before the next
		period can start.

	Returns a DataFrame with columns:
	  neuron_id, layer_name, period_id, start, end
	where start is the first checkpoint (inclusive) and end is the first
	checkpoint after the period (exclusive). end is NaN if the neuron is still
	active at the last checkpoint.
	"""
	nd = (
		n_assigned_digits
		.reset_index()						  # -> neuron_id, checkpoint_idx, n_assigned_digits
		.sort_values(["neuron_id", "checkpoint_idx"])
		.reset_index(drop=True)
	)

	# Carry layer_name alongside so callers can filter by layer.
	layer_map = df_nd[["neuron_id", "layer_name"]].drop_duplicates("neuron_id").set_index("neuron_id")["layer_name"]
	nd["layer_name"] = nd["neuron_id"].map(layer_map)

	# --- Join is_dead from df_dead -------------------------------------------
	dead_status = df_dead[["neuron_id", "checkpoint_idx", "is_dead"]]
	nd = nd.merge(dead_status, on=["neuron_id", "checkpoint_idx"], how="left")
	nd["is_dead"] = nd["is_dead"].fillna(False)

	nd["is_assigned"] = nd["n_assigned_digits"] > 0

	# -------------------------------------------------------------------------
	# For assigned_any:
	# For each neuron, we qualify only periods in training where 
	# * the neuron is originally dead
	# * the neuron becomes assigned to a (set of) digit(s) 
	# The period stop whenever the neuron is not assigned to any digit anymore (but it can still be partially active)
	# This "resets" whenever the neuron dies again (aka not _active_ for any sample of any digits)
	# -------------------------------------------------------------------------
	def _process_neuron_any(g: pd.DataFrame) -> pd.DataFrame:
		g = g.sort_values("checkpoint_idx")
		ia = g["is_assigned"].to_numpy()
		id_ = g["is_dead"].to_numpy()
		qualified = False  # True once we've seen is_dead since anchor
		prev_assigned = False
		out = np.zeros(len(g), dtype=bool)
		for i in range(len(g)):
			if prev_assigned and not ia[i]:
				qualified = False
			if id_[i]:
				qualified = True
			out[i] = (not prev_assigned) and ia[i] and qualified
			prev_assigned = ia[i]
		g = g.copy()
		g["is_start"] = out
		return g

	# -------------------------------------------------------------------------
	# For assigned_same:
	# For each neuron, we qualify only periods in training where 
	# * the neuron is originally dead
	# * the neuron becomes assigned to a (set of) digit(s) D
	# The period stop whenever the neuron is not assigned to the any of the digits in D anymore 
	# This "resets" whenever the neuron dies again (aka not _active_ for any sample of any digits)
	# -------------------------------------------------------------------------
	def _process_neuron_same(g: pd.DataFrame, digit_sets: dict) -> pd.DataFrame:
		g = g.sort_values("checkpoint_idx")
		ia = g["is_assigned"].to_numpy()
		id_ = g["is_dead"].to_numpy()
		ckpts = g["checkpoint_idx"].to_numpy()
		neuron = g["neuron_id"].iloc[0]

		qualified = False
		in_period = False
		original_digits: frozenset = frozenset()
		out_start = np.zeros(len(g), dtype=bool)
		out_active = np.zeros(len(g), dtype=bool)

		for i in range(len(g)):
			current_digits = digit_sets.get((neuron, ckpts[i]), frozenset())

			# Period ends when none of the original digits are assigned anymore.
			if in_period and not (original_digits & current_digits):
				in_period = False
				original_digits = frozenset()
				qualified = False  # must become dead again before next period

			if id_[i]:
				qualified = True

			if not in_period and ia[i] and qualified:
				in_period = True
				original_digits = current_digits
				out_start[i] = True

			out_active[i] = in_period

		g = g.copy()
		g["is_start"] = out_start
		g["is_active_period"] = out_active
		return g

	# --- Apply per-neuron processing -----------------------------------------
	if mode == "assigned_any":
		nd = pd.concat(
			[_process_neuron_any(g) for _, g in nd.groupby("neuron_id", sort=False)],
			ignore_index=True,
		).sort_values(["neuron_id", "checkpoint_idx"])

		nd["prev_active"] = nd.groupby("neuron_id")["is_assigned"].shift(1, fill_value=False)
		nd["is_end"] = nd["prev_active"] & ~nd["is_assigned"]  # >0 -> 0
		nd["period_id"] = nd.groupby("neuron_id")["is_start"].cumsum() - 1

		active = nd[nd["is_assigned"] & (nd["period_id"] >= 0)]
		ends   = nd[nd["is_end"]	  & (nd["period_id"] >= 0)]

	elif mode == "assigned_same":
		# Precompute per-(neuron, checkpoint) frozenset of assigned digits.
		assigned_rows = df_nd[df_nd["status"] == "assigned"]
		digit_sets = (
			assigned_rows.groupby(["neuron_id", "checkpoint_idx"])["digit"]
			.apply(frozenset)
			.to_dict()
		)

		nd = pd.concat(
			[_process_neuron_same(g, digit_sets) for _, g in nd.groupby("neuron_id", sort=False)],
			ignore_index=True,
		).sort_values(["neuron_id", "checkpoint_idx"])

		nd["prev_active_period"] = nd.groupby("neuron_id")["is_active_period"].shift(1, fill_value=False)
		nd["is_end"] = nd["prev_active_period"] & ~nd["is_active_period"]
		nd["period_id"] = nd.groupby("neuron_id")["is_start"].cumsum() - 1

		active = nd[nd["is_active_period"] & (nd["period_id"] >= 0)]
		ends   = nd[nd["is_end"]		   & (nd["period_id"] >= 0)]

	# --- Aggregate [start, end) per (neuron_id, period_id) --------------------
	starts_agg = (
		active
		.groupby(["neuron_id", "layer_name", "period_id"])["checkpoint_idx"]
		.min()
		.rename("start")
	)
	ends_agg = (
		ends
		.groupby(["neuron_id", "layer_name", "period_id"])["checkpoint_idx"]
		.min()
		.rename("end")
	)

	periods = starts_agg.to_frame().join(ends_agg, how="left").reset_index().assign(
		start_iter=lambda df: df["start"].map(lambda c: get_iteration(metrics, c)),
		end_iter=lambda df: df["end"].map(lambda c: get_iteration(metrics, c)),
	).assign(total_length_iter=lambda df: df["end_iter"] - df["start_iter"])
	return periods


def find_impacted_periods(periods_mode_any: pd.DataFrame, periods_mode_same: pd.DataFrame, return_PRINT = False):
	"""
	Find periods that are impacted by the mode change.
	"""
	a,s = periods_mode_any.set_index(["neuron_id", "period_id", "start_iter"]), periods_mode_same.set_index(["neuron_id", "period_id", "start_iter"])
	# a and s should have the same periods - though their end might differ (since the start is only defined by the dead -> assigned event)
	assert a.index.equals(s.index)
	impacted_mask = a["end_iter"] != s["end_iter"]
	a_impacted = a[impacted_mask]
	s_impacted = s[impacted_mask]
	if return_PRINT:
		print(f"The impacted periods are:")
		print(f"  neuron_id, period_id, start_iter")
		for i, row in a_impacted.iterrows():
			print(f"  {i}")
			print(f"	Any: {row['end_iter']} vs Same: {s_impacted.loc[i]['end_iter']}")
		return None
	else:
		joined = a_impacted.join(s_impacted, lsuffix="_any", rsuffix="_same")
		impacted = joined.drop(columns=[c for c in joined.columns if not (c.startswith("end_iter"))])
		# print(f"Impacted")
		# display(impacted.head(20))
		return impacted


def get_period_data(recovery_periods,neuron_id:str,period_id: int| None = None):
	# Access the recpvery_period data. It is possible that the neuron_id has only one period. In that case, allow period_id to be None and return it
	neuron_data = recovery_periods.set_index(["neuron_id","period_id"]).loc[neuron_id]
	if period_id is None:
		if len(neuron_data) == 1:
			return neuron_data.iloc[0] # iloc because it is possible that filtering removes the period with id 0 -> loc 0 would be incorrect
		else:
			raise ValueError(f"Neuron {neuron_id} has {len(neuron_data)} periods. Please specify the period_id.")
	else:
		return neuron_data.loc[period_id]


def get_trial_start_end_iter(start_iter,start: bool = True):
	return ((start_iter // N_SAMPLES_PER_TRIAL)+(0 if start else 1))*N_SAMPLES_PER_TRIAL

def remove_inelligible_periods(recovery_periods, *, require_until_trial_end: bool = True):
	"""
	Remove periods that do not last until the end of their trial.

	require_until_trial_end: when False, return `recovery_periods` unchanged (no filtering).
	"""
	if not require_until_trial_end:
		return recovery_periods
	to_remove = recovery_periods[recovery_periods["end_iter"] < get_trial_start_end_iter(recovery_periods["start_iter"],start=False)]
	print(f"Removing {len(to_remove)} period(s) that do not last until the end of their trial.")
	if len(to_remove) > 0:
		for i, row in to_remove.iterrows():
			print(f"Removing: {row}\n")
	return recovery_periods.drop(to_remove.index)


def get_trial_starts(recovery_periods, neuron_id:str, period_id: int|None = None):
	period_data = get_period_data(recovery_periods,neuron_id,period_id)
	trial_start_iter = get_trial_start_end_iter(period_data["start_iter"],start=True)
	delta = period_data["start_iter"] - trial_start_iter
	start_cp = period_data["start"] - delta
	return trial_start_iter, delta, start_cp

def get_trial_neuron_activations(nts, recovery_periods, neuron_id:str, period_id: int|None = None, surrounding_n_trials: tuple[int,int] | None = None):
	# nts data is indexed by checkpoint -> find the start checkpoint first
	_,_,start_cp = get_trial_starts(recovery_periods,neuron_id,period_id)
	if surrounding_n_trials is None:
		surrounding_n_trials = (0,0)
	activations = nts["act__"+neuron_id][start_cp-surrounding_n_trials[0]*N_SAMPLES_PER_TRIAL:start_cp + N_SAMPLES_PER_TRIAL + surrounding_n_trials[1]*N_SAMPLES_PER_TRIAL] # Shape (100,50)
	return activations


def get_assigned_digits_for_trial_activations(nts,recovery_periods,neuron_id,period_id):
	trial_activations = get_trial_neuron_activations(nts,recovery_periods,neuron_id,period_id,surrounding_n_trials=(0,0))
	_, delta, _ = get_trial_starts(recovery_periods,neuron_id,period_id)
	active_samples_at_threshold = (trial_activations[delta] > 0).reshape((_NETWORK_N_DIGITS,_NETWORK_SAMPLES_PER_DIGIT)) # shape (10,5)
	assigned_digits = active_samples_at_threshold.all(axis=1) # shape (10,)
	assert assigned_digits.sum() >= 1
	assigned_digits = np.argwhere(assigned_digits).flatten()
	assert all(maybe_assigned == (d in assigned_digits) for d,maybe_assigned in enumerate(np.all(active_samples_at_threshold,axis=1)))
	return assigned_digits


def sample_idcs_for_digit(digit):
	return np.array(range(_NETWORK_SAMPLES_PER_DIGIT)) + digit * _NETWORK_SAMPLES_PER_DIGIT
def sample_ics_for_digits(digits):
	return np.sort(np.concatenate([sample_idcs_for_digit(d) for d in digits]))


def get_last_inactive_samples_indices(nts,recovery_periods,neuron_id,period_id):
	trial_activations = get_trial_neuron_activations(nts,recovery_periods,neuron_id,period_id,surrounding_n_trials=(0,0))
	_, delta, _ = get_trial_starts(recovery_periods,neuron_id,period_id)
	assigned_digits = get_assigned_digits_for_trial_activations(nts,recovery_periods,neuron_id,period_id)
	
	active_samples_before_threshold = (trial_activations[delta-1] > 0).reshape((_NETWORK_N_DIGITS,_NETWORK_SAMPLES_PER_DIGIT))
	samples_for_active_digits = active_samples_before_threshold[assigned_digits,:] # shape (k,5)
	assert not all(maybe_assigned for maybe_assigned in np.all(samples_for_active_digits,axis=1))
	last_inactive_samples = np.argmax(~samples_for_active_digits,axis=1) # shape (k,)
	indices = [d * _NETWORK_SAMPLES_PER_DIGIT + sample for d,sample in zip(assigned_digits,last_inactive_samples)]
	if len(indices) > 1:
		print(f"more than 1 for {neuron_id} {period_id}: {indices=}")
	return indices


def point_of_max_acceleration(
	nts,
	recovery_periods,
	neuron_id,
	period_id,
	*,
	peak_time_aggregate: Literal["min", "max"] = "min",
):
	trial_activations = get_trial_neuron_activations(nts,recovery_periods,neuron_id,period_id,surrounding_n_trials=(0,0))
	_, delta, _ = get_trial_starts(recovery_periods,neuron_id,period_id)
	if peak_time_aggregate not in ("min", "max"):
		raise ValueError(f"peak_time_aggregate must be 'min' or 'max', got {peak_time_aggregate!r}")
	if delta < 2:
		# Then we don't have enough data to get accelerations peaks before the event 
		# (at delta == 2, we have 3 data points: [iter_start, iter_start + 1, iter_start + {delta=2}])
		# --> return the trial start
		return 0
	derivative = np.diff(trial_activations,axis=0) # shape (99,50)
	second_derivative = np.diff(derivative,axis=0) # shape (98,50)
	
	assigned_digits = get_assigned_digits_for_trial_activations(nts,recovery_periods,neuron_id,period_id)
	indices_to_keep = np.concatenate([d*_NETWORK_SAMPLES_PER_DIGIT + np.arange(_NETWORK_SAMPLES_PER_DIGIT) for d in assigned_digits])
	# Keep all the accelerations since the trial start until the threshold (`-2` because we want the second derivative at the sample *just BEFORE* the threshold)
	# and only those for the samples of the digits activated at the threshold
	accelerations_to_consider = second_derivative[0:delta-1,indices_to_keep] # shape (time,k)
	acceleration_peak = np.argmax(accelerations_to_consider,axis=0) # shape (k,)
	# Per-digit argmax time; min = earliest peak across channels, max = latest (closest to assignment); +2 maps to activation index space
	agg = np.max if peak_time_aggregate == "max" else np.min
	return int(agg(acceleration_peak) + 2)


def get_all_points_of_max_acceleration(
	nts,
	recovery_periods,
	*,
	peak_time_aggregate: Literal["min", "max"] = "min",
):
	columns_to_keep = index_grp + ["start_iter"]
	ret = (recovery_periods.drop(columns={c for c in recovery_periods.columns if c not in columns_to_keep})\
		.assign(point_of_max_acceleration=lambda df: df\
			.apply(lambda row: point_of_max_acceleration(nts,recovery_periods,row["neuron_id"],row["period_id"], peak_time_aggregate=peak_time_aggregate),axis=1))\
		.set_index(index_grp))
	ret["delay_from_trial_start"] = ret["start_iter"].apply(lambda x: x%N_SAMPLES_PER_TRIAL)
	ret = ret.drop(columns=["start_iter"])
	return ret


def filter_periods(recovery_periods, df_dead, strict = False):
	"""
	Filter periods for cps that are not dead at some point in the trial before the period starts.
	"""
	trial_starts = recovery_periods[["start_iter","neuron_id","period_id"]].assign(trial_start=lambda df: df["start_iter"].map(get_trial_start_end_iter))
	mrgd = df_dead.merge(trial_starts,left_on=["neuron_id","iteration"],right_on=["neuron_id","trial_start"],how="inner")[["neuron_id","period_id","is_dead"]]
	if strict:
		return recovery_periods.loc[mrgd[mrgd["is_dead"]].index]
	else:
		print(f"Keeping recovery periods that are not dead at some point during that trial where the recovery occurs. \n Would've removed {(~mrgd['is_dead']).sum()}/{len(mrgd)} period(s) otherwise.")
		return recovery_periods


def get_period_angles(
	nts,
	recovery_periods,
	from_point:str = "acceleration",
	use_real_progression = True,
	*,
	peak_time_aggregate: Literal["min", "max"] = "min",
):
	"""
	from_point: "trial_start" or "acceleration" : decides whether to use the start of the trial, or the point of highest acceleration as reference for the angle calculation

	peak_time_aggregate: when from_point is "acceleration", how to combine per-digit argmax peak times (earliest vs latest before assignment).
	"""
	if from_point not in {"trial_start","acceleration"}:
		raise ValueError(f"from_point must be one of {{'trial_start','acceleration'}}, got {from_point}")
	def get_activation_angle(neuron_id,period_id):
		activations = get_trial_neuron_activations(nts,recovery_periods,neuron_id,period_id)
		assert len(activations) == N_SAMPLES_PER_TRIAL, f"{len(activations)=}"
		_, delta, _ = get_trial_starts(recovery_periods,neuron_id,period_id)
		if from_point == "trial_start" :
			activation_at_start_of_trial = activations[0]
			reference_activation = activation_at_start_of_trial
			reference_delta = delta
		else:
			assert from_point == "acceleration"
			trial_index_of_max_acceleration = point_of_max_acceleration(
				nts, recovery_periods, neuron_id, period_id, peak_time_aggregate=peak_time_aggregate
			)
			assert trial_index_of_max_acceleration <= delta
			reference_activation = activations[trial_index_of_max_acceleration]
			reference_delta = delta - trial_index_of_max_acceleration
		activation_at_assignment = activations[delta]
		angles = np.degrees(np.arctan2(activation_at_assignment-reference_activation,reference_delta if use_real_progression else 1)) # shape (50,)
		last_inactive_sample_indices = get_last_inactive_samples_indices(nts,recovery_periods,neuron_id,period_id)
		return np.max(angles[last_inactive_sample_indices])


	period_angles = recovery_periods.drop(columns={c for c in recovery_periods.columns if c not in index_grp}).assign(activation_angle=lambda df: df.apply(lambda row: get_activation_angle(row["neuron_id"],row["period_id"]),axis=1)).set_index(index_grp)
	return period_angles


def angle_score_function(x):
	# f(x) = s * e^(-k/x) such that f(10) = 0.5 and f(90) = 1
	angle_at_0_5 = 10
	angle_at_1 = 90
	s = 2**((angle_at_0_5)/(angle_at_1-angle_at_0_5))
	k = angle_at_0_5*np.log(2*s)
	eps = 1e-12
	return s * np.exp(-k/(np.clip(x,0,None)+1e-12))


def length_score_function(x):
	# Piecewise-linear at threshold T = 3*N_SAMPLES_PER_TRIAL = 300
	# linear before threshold, 1 after
	threshold = 3 * N_SAMPLES_PER_TRIAL
	return np.clip(x / threshold, 0, 1)


def get_assignments(df_nd,recovery_periods):
	# For each recovery period, count:
	#   n_assigned_digits  - distinct digits ever "assigned"   to the neuron during [start, end)
	#   n_partial_digits   - distinct digits ever "partial_*"  for the neuron during [start, end)

	# --- 1. Attach period info to every df_nd row (fan-out on neuron_id) ----------
	df_in_period = df_nd.merge(
		recovery_periods[["neuron_id", "period_id", "start", "end"]],
		on="neuron_id",
		how="inner",
	)

	# Keep only rows that belong to [start, end).
	# For open periods (end is NaN) every row from start onward qualifies.
	mask = (df_in_period["checkpoint_idx"] >= df_in_period["start"]) & (
		df_in_period["end"].isna() | (df_in_period["checkpoint_idx"] < df_in_period["end"])
	)
	df_in_period = df_in_period[mask]

	# --- 2. Count distinct digits per status type --------------------------------

	df_at_start = df_in_period[df_in_period["checkpoint_idx"] == df_in_period["start"]]

	n_assigned = (
		df_in_period[df_in_period["status"] == "assigned"]
		.groupby(index_grp)["digit"]
		.nunique()
		.rename("n_assigned_digits_unique")
	)

	n_partial = (
		df_in_period[df_in_period["status"].str.startswith("partial")]
		.groupby(index_grp)["digit"]
		.nunique()
		.rename("n_partial_digits_unique")
	)

	n_assigned_at_start = (
		df_at_start[df_at_start["status"] == "assigned"]
		.groupby(index_grp)["digit"]
		.nunique()
		.rename("n_assigned_at_start")
	)

	n_partial_at_start = (
		df_at_start[df_at_start["status"].str.startswith("partial")]
		.groupby(index_grp)["digit"]
		.nunique()
		.rename("n_partial_at_start")
	)
	

	# --- 3. Join back onto recovery_periods --------------------------------------
	int_cols = ["n_assigned_digits_unique", "n_partial_digits_unique", "n_assigned_at_start", "n_partial_at_start"]
	assignments = (
		n_assigned_at_start.to_frame()
		.join(n_partial_at_start,  on=index_grp)
		.join(n_assigned,		 on=index_grp)
		.join(n_partial,		  on=index_grp)
		.fillna({c: 0 for c in int_cols})
		.astype({c: int for c in int_cols})
	)
	return assignments


def n_active_decay(x):
	assert x>= 0
	if x == 0:
		raise ValueError("x cannot be 0")
		return 0
	elif x <= 1:
		return 1
	else:
		#											^ 5 == N_DIGITS //2 
		# Exponential decay such that n_active_decay(5) = 0.5 and limit in ifty is `lim`
		lim = 0.3
		scale = np.log(7/2) / 4 # for decay(5) = 0.5
		return lim + np.exp((-(scale*(x-1)-np.log(1-lim))))

def n_partial_decay(x):
	assert x>= 0
	if x == 0:
		return 1
	else:
		# Exponential decay such that n_partial_decay(1) = 0.5
		return np.exp(-np.log(1/0.5)*x)


def registry_class_from_optimizer_id(optimizer_id: str) -> str:
	if "_lr" not in optimizer_id:
		raise ValueError(f"unexpected optimizer_id shape: {optimizer_id!r}")
	prefix = optimizer_id.rsplit("_lr", 1)[0]
	try:
		return OPTIMIZER_PREFIX_TO_REGISTRY_CLASS[prefix]
	except KeyError as e:
		raise KeyError(f"unknown optimizer_id prefix {prefix!r}") from e


def find_target_checkpoint(neuron_id, period_id, nts, recovery_periods, mode):
	"""Return the checkpoint index to use for GBP saliency map computation.

	mode="assignment_point" -> the period start checkpoint.
	mode="max_activity"	 -> the checkpoint within the period where the sum of
							   activations for the digits assigned at the assignment
							   point is maximal.
	"""
	_, delta, start_cp = get_trial_starts(recovery_periods, neuron_id, period_id)
	period_row = recovery_periods[
		(recovery_periods["neuron_id"] == neuron_id) &
		(recovery_periods["period_id"] == period_id)
	].iloc[0]

	if mode == "assignment_point":
		return int(period_row["start"])

	trial_activations = get_trial_neuron_activations(
		nts, recovery_periods, neuron_id, period_id, surrounding_n_trials=(0, 0))
	# shape: (N_SAMPLES_PER_TRIAL, N_DIGITS * N_SAMPLES_PER_EVALUATION_DIGIT)

	assigned_digits = get_assigned_digits_for_trial_activations(
		nts, recovery_periods, neuron_id, period_id)

	sample_cols = np.concatenate([
		d * _NETWORK_SAMPLES_PER_DIGIT + np.arange(_NETWORK_SAMPLES_PER_DIGIT)
		for d in assigned_digits
	])

	end_cp = period_row["end"]
	period_end_row = int(end_cp - start_cp) if not np.isnan(end_cp) else N_SAMPLES_PER_TRIAL
	period_end_row = min(period_end_row, N_SAMPLES_PER_TRIAL)

	activity = trial_activations[delta:period_end_row, sample_cols].sum(axis=1)
	return int(period_row["start"]) + int(np.argmax(activity))


def _mssim(A, B, full = False):
	"""MSSIM per Nilsson & Akenine-Moller 2020 eq.5: spatial mean of per-pixel SSIM.

	Equivalent to the scalar returned by skimage.metrics.structural_similarity.
	data_range is set to the range of the combined pair so that signed
	floating-point saliency maps are handled correctly.
	"""
	A = np.asarray(A, dtype=float)
	B = np.asarray(B, dtype=float)
	data_range = float(np.ptp(np.stack([A, B])))
	assert data_range > 0
	# skimage source when gaussian_weights=True, win_size=None: see https://github.com/scikit-image/scikit-image/blob/main/src/skimage/metrics/_structural_similarity.py#L180
	# truncate = 3.5
	# r = int(truncate * sigma + 0.5)  # same convention as scipy.ndimage
	# win_size = 2 * r + 1
	sigma = 0.5
	# --> 5x5 kernel
	if full:
		mssim, ssim = structural_similarity(A, B, data_range=data_range, gaussian_weights = True, sigma = sigma, use_sample_covariance = False, gradient = False,full = full)
	else:
		mssim, ssim = structural_similarity(A, B, data_range=data_range, gaussian_weights = True, sigma = sigma, use_sample_covariance = False, gradient = False,full = full), None
	return float(mssim), ssim


def _stack_saliency_maps_for_mssim(flat_maps, keep_torch = False):
	def _to_np(m):
		return m.numpy() if isinstance(m, torch.Tensor) else np.asarray(m)

	if keep_torch:
		arr = torch.stack([flat_maps[i] for i in range(_NETWORK_N_DIGITS * _NETWORK_SAMPLES_PER_DIGIT)]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # (50, H, W)
	else:
		arr = np.stack([_to_np(flat_maps[i]) for i in range(_NETWORK_N_DIGITS * _NETWORK_SAMPLES_PER_DIGIT)]) # (50, H, W)
	return arr.reshape(_NETWORK_N_DIGITS,_NETWORK_SAMPLES_PER_DIGIT, *arr.shape[1:]) # (D,S,H,W) = (10,5,28,28)


from scipy.ndimage import convolve

def gaussian_kernel_5x5(sigma: float = 1.0,size:int = 5) -> np.ndarray:
	"""Generate a 5x5 Gaussian kernel."""
	center = size // 2
	x, y = np.mgrid[-center:center+1, -center:center+1]
	kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
	return kernel / kernel.sum() # (size,size)

def ssim_luminance(
	img_a: np.ndarray | torch.Tensor,
	img_b: np.ndarray | torch.Tensor,
	sigma: float = 0.5,
	data_range: float = 1.0,
	K1: float = 0.01,
) -> tuple[np.ndarray, float]:
	"""
	Compute the luminance component of SSIM between two images.

	l(x, y) = (2·μA·μB + C1) / (μA² + μB² + C1)

	Args:
		img_a: First image as a 2D NumPy array (grayscale).
		img_b: Second image as a 2D NumPy array (grayscale).
		sigma: Std dev for the 5x5 Gaussian kernel (default 0.5).
		L:	 Dynamic range of pixel values (255 for uint8).
		K1:	Stability constant (default 0.01).

	Returns:
		luminance_map: Per-pixel luminance similarity (2D array).
		mean_luminance: Mean luminance similarity across the image (scalar).
	"""
	assert img_a.shape == img_b.shape, "Images must have the same shape."
	assert img_a.ndim == 2, "Images must be 2D (grayscale)."

	if isinstance(img_a, torch.Tensor):
		img_a = img_a.numpy()
	if isinstance(img_b, torch.Tensor):
		img_b = img_b.numpy()

	L = data_range
	C1 = (K1 * L) ** 2
	kernel = gaussian_kernel_5x5(sigma=sigma)

	img_a = img_a.astype(np.float64)
	img_b = img_b.astype(np.float64)

	mu_a = convolve(img_a, kernel, mode="reflect")
	mu_b = convolve(img_b, kernel, mode="reflect")

	luminance_map = (2 * mu_a * mu_b + C1) / (mu_a**2 + mu_b**2 + C1)
	mean_luminance = float(np.mean(luminance_map))

	return luminance_map, mean_luminance


def _get_cw_ssim_steerable_pyramid(
	H: int,
	W: int,
	level: int,
	orientations: int,
	device: torch.device,
	prebuilt_pyramids: dict,
):
	"""Return a cached SteerablePyramid; build on first use per (H, W, level, orientations, device).

	`prebuilt_pyramids` is caller-owned mutable cache (same dict across CW-SSIM calls).
	"""
	key = (H, W, level, orientations, str(device))
	sp = prebuilt_pyramids.get(key)
	if sp is None:
		sp = SteerablePyramid(
			imgSize=[H, W],
			K=orientations,
			N=level,
			hilb=True,
			includeHF=True,
			device=device,
		)
		sp.eval()
		prebuilt_pyramids[key] = sp
	return sp, prebuilt_pyramids


def _gauss_kernel_2d(size: int, sigma: float, device) -> torch.Tensor:
	"""Returns a (1, 1, size, size) normalized Gaussian kernel."""
	coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
	g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
	g = torch.outer(g, g)
	g /= g.sum()
	return g.unsqueeze(0).unsqueeze(0)


def cw_ssim_with_gauss(
	img1: torch.Tensor,
	img2: torch.Tensor,
	data_range: float,
	level: int = 3,
	orientations: int = 8,
	win_size: int = 5,
	sigma: float = 1.0,
	epsilon: float = 1e-12,
	prebuilt_pyramids: dict | None = None,
) -> float:
	"""
	CW-SSIM with a local Gaussian window applied at each pyramid band.
	Produces a per-pixel cssim map per band/orientation, then averages.

	Parameters
	----------
	img1, img2   : 2-D float tensors (H, W)
	data_range   : normalisation range
	level		: pyramid levels
	orientations : oriented sub-bands
	win_size	 : Gaussian window size (default 5 to match skimage SSIM)
	sigma		: Gaussian sigma (default 1.5, skimage's default)
	epsilon	  : numerical stability constant
	"""
	assert img1.shape == img2.shape, "Images must have the same shape."
	assert img1.dim() == 2, "Expected 2-D tensors (H, W)."
	assert win_size % 2 == 1, "Window size must be odd"

	if prebuilt_pyramids is None:
		prebuilt_pyramids = {}

	device = img1.device
	H, W = img1.shape

	img1 = img1 / data_range
	img2 = img2 / data_range

	x = img1.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
	y = img2.unsqueeze(0).unsqueeze(0)

	sp, prebuilt_pyramids = _get_cw_ssim_steerable_pyramid(H, W, level, orientations, device, prebuilt_pyramids)

	with torch.no_grad():
		cw_x = sp(x)
		cw_y = sp(y)

	all_scores = []
	for band_ind in range(1, level + 1):
		b1 = cw_x[band_ind][:, :, 0, :, :, :]  # (1, 2, ori, H', W')
		b2 = cw_y[band_ind][:, :, 0, :, :, :]

		# The height and width of the corresponding bands divide by 2 for every level -> also decrease the focus kernel size by 2 (to keep it odd x odd)
		H_b, W_b = b1.shape[-2], b1.shape[-1]
		ws = max(1,min(win_size - (2 * (band_ind-1)), H_b, W_b))
		pad = ws // 2
		gauss = _gauss_kernel_2d(ws, sigma, device)  # (1, 1, ws, ws)
		# print(f"{ws=},{pad=}")

		# corr: (1, 2, ori, H', W')
		a, b = b1[:, 0], b1[:, 1]   # (1, ori, H', W')
		c, d = b2[:, 0], -b2[:, 1]
		corr = torch.stack((a*c - b*d, b*c + a*d), dim=1)  # (1, 2, ori, H', W')

		# apply Gaussian per orientation, per real/imag channel
		# reshape to (ori, 1, H', W') to use groups-style conv
		ori = orientations
		corr_r = corr[:, 0].reshape(ori, 1, H_b, W_b)  # real part
		corr_i = corr[:, 1].reshape(ori, 1, H_b, W_b)  # imag part
		# print(f"{corr_r.shape=},{corr_i.shape=}")

		# convolve each orientation independently: (ori, 1, H'', W'')
		corr_r_smooth = F.conv2d(corr_r, gauss, padding=pad)
		corr_i_smooth = F.conv2d(corr_i, gauss, padding=pad)

		# |smoothed corr| -> (ori, H'', W'')
		num = torch.sqrt(corr_r_smooth[:, 0]**2 + corr_i_smooth[:, 0]**2 + epsilon)

		# denominator: smooth |b1|^2 + |b2|^2
		abs1_sq = (b1[:, 0]**2 + b1[:, 1]**2)  # (1, ori, H', W')
		abs2_sq = (b2[:, 0]**2 + b2[:, 1]**2)
		varr = (abs1_sq + abs2_sq).reshape(ori, 1, H_b, W_b)
		denom = F.conv2d(varr, gauss, padding=pad)[:, 0]   # (ori, H'', W'')

		# cssim map: (ori, H'', W'') -> scalar
		cssim_map = (2 * num + epsilon) / (denom + epsilon)
		# print(f"{cssim_map.shape=},{cssim_map=}")
		all_scores.append(cssim_map.mean())

	return torch.stack(all_scores).mean().item()


def get_data_range(stacked_saliency_maps):
	if isinstance(stacked_saliency_maps, torch.Tensor):
		out = stacked_saliency_maps.abs().max().item()
	else:
		out = np.abs(stacked_saliency_maps).max().item()
	return float(out)


def saliency_digit_pair_mssim(model_maps_d, exp_maps_d, *, store_full_grids=False, prebuilt_pyramids: dict | None = None):
	"""MSSIM between paired saliency maps for one digit (arrays of shape (S, H, W)).

	`mean_of_mssim` is the mean of diagonal entries of the SxS MSSIM grid
	(same sample index for model vs expert).
	"""
	if exp_maps_d.shape[0] != model_maps_d.shape[0]:
		raise ValueError("model and expert must have the same number of samples per digit")
	if model_maps_d.shape[0] != _NETWORK_SAMPLES_PER_DIGIT:
		raise ValueError("model and expert must have the same number of samples per digit")
	if store_full_grids:
		if tuple(model_maps_d.shape[1:]) != MNIST_IMAGE_SHAPE:
			raise ValueError(f"model saliency maps must be {MNIST_IMAGE_SHAPE} but are {model_maps_d.shape[1:]}")

	full_ssims_list = []

	if store_full_grids:
		assert not USE_CW_SSIM
		mssim_grid = np.empty((_NETWORK_SAMPLES_PER_DIGIT, _NETWORK_SAMPLES_PER_DIGIT), dtype=np.float64)
		for sample_idx in range(_NETWORK_SAMPLES_PER_DIGIT):
			for expert_sample_idx in range(_NETWORK_SAMPLES_PER_DIGIT):
				mssim, ssim = _mssim(
					model_maps_d[sample_idx],
					exp_maps_d[expert_sample_idx],
					full=True,
				)
				full_ssims_list.append(ssim)
				mssim_grid[sample_idx, expert_sample_idx] = mssim
	else:
		if prebuilt_pyramids is None:
			prebuilt_pyramids = {}
		# Only fill the diagonal (i,i)
		mssim_grid = np.empty((_NETWORK_SAMPLES_PER_DIGIT,), dtype=np.float64)
		for i in range(_NETWORK_SAMPLES_PER_DIGIT):
			if USE_CW_SSIM:
				similarity = cw_ssim_with_gauss(
					model_maps_d[i],
					exp_maps_d[i],
					data_range = get_data_range(torch.cat([model_maps_d[i],exp_maps_d[i]],dim=0)),
					prebuilt_pyramids=prebuilt_pyramids,
				)
			else:
				mssim, _ = _mssim(
					model_maps_d[i],
					exp_maps_d[i],
					full=False,
				)
				similarity = mssim
			mssim_grid[i] = similarity

	mean_of_mssim = float(np.mean(np.diag(mssim_grid) if store_full_grids else mssim_grid))
	mean_model = model_maps_d.mean(axis=0)
	mean_expert = exp_maps_d.mean(axis=0)
	if USE_CW_SSIM:
		mssim_of_mean = cw_ssim_with_gauss(
			mean_model,
			mean_expert,
			data_range = get_data_range(torch.cat([mean_model,mean_expert],dim=0)),
			prebuilt_pyramids=prebuilt_pyramids,
		)
	else:
		mssim_of_mean = _mssim(mean_model, mean_expert, full=False)[0]
	out_records = {
		"mean_of_mssim": mean_of_mssim,
		"MSSIM_of_mean": mssim_of_mean,
	}
	if store_full_grids:
		out_records["full_mssim_grid"] = mssim_grid
		out_records["full_ssim_grid"] = np.array(full_ssims_list).reshape(_NETWORK_SAMPLES_PER_DIGIT, _NETWORK_SAMPLES_PER_DIGIT, *MNIST_IMAGE_SHAPE)
	return out_records


def get_period_mssims(recovery_periods, period_saliency_maps, expert_saliency, *, prebuilt_pyramids: dict | None = None):
	pyramids_cache = {} if prebuilt_pyramids is None else prebuilt_pyramids
	records = []
	for row in tqdm(list(recovery_periods.itertuples()), desc="Computing SSIM"):
		nid, pid = row.neuron_id, row.period_id
		layer_name = row.layer_name

		model_maps = _stack_saliency_maps_for_mssim(
			period_saliency_maps[(nid, pid)], keep_torch = USE_CW_SSIM
		)
		exp_maps = _stack_saliency_maps_for_mssim(
			expert_saliency[nid], keep_torch = USE_CW_SSIM
		)

		for d in range(_NETWORK_N_DIGITS):
			# mean_of_mssim:
			#   For each model sample s, compute MSSIM against every expert sample e -> (5,) array (diag of the pairwise distances) 
			#   Mean across e gives MSSIM[d,s]; mean across s gives the digit-level score.

			m = saliency_digit_pair_mssim(
				model_maps[d],
				exp_maps[d],
				store_full_grids=False,
				prebuilt_pyramids=pyramids_cache,
			)
			records.append({
				"neuron_id": nid,
				"period_id": pid,
				"digit": d,
				"layer_name": layer_name,
				"mean_of_mssim": m["mean_of_mssim"],
				"MSSIM_of_mean": m["MSSIM_of_mean"]
			})

	df_ssim = pd.DataFrame(records).set_index(["neuron_id", "period_id", "digit"])
	return df_ssim


# Per-layer MSSIM baseline: control neurons (non-dead) in each layer that
# appears in recovery_periods, evaluated at the last checkpoint.
def get_layer_baselines(nts,recovery_periods,eval_digits,device,expert_model,
	*,
	metrics,
	expert_saliency,
	prebuilt_pyramids: dict | None = None,
):
	# `metrics` needed only for last chekpoint 

	pyramid_cache = {} if prebuilt_pyramids is None else prebuilt_pyramids

	# Dead neurons grouped by layer (only layers that actually have dead neurons)
	dead_neurons_per_layer = (
		recovery_periods.groupby("layer_name")["neuron_id"]
		.apply(set)
		.to_dict()
	)

	# All neurons grouped by layer (from nts)
	_all_nids = [str(nid) for nid in nts["unit_node_ids"]]
	_neurons_by_layer = defaultdict(list)
	for _nid in _all_nids:
		_neurons_by_layer[parse_unit_node_id(_nid)["layer_name"]].append(_nid)

	# Control neurons per layer: non-dead neurons in the relevant layers only
	control_by_layer = {
		layer: [nid for nid in _neurons_by_layer[layer] if nid not in dead_set]
		for layer, dead_set in dead_neurons_per_layer.items()
	}

	last_cp = len(metrics["checkpoint_iterations"]) - 1
	control_saliency = {}  # nid -> list[50 tensors]
	all_expert_saliency_maps = get_all_model_saliency_maps(expert_model, eval_digits, device)

	# __ Iterate by layer: saliency for model and expert _________________________
	for layer, nids in control_by_layer.items():
		for nid in tqdm(nids, desc=f"Control saliency layer={layer}"):
			control_saliency[nid] = get_neuron_effective_receptive_field(
				nts,
				nid,
				iteration=last_cp,
				device=device,
			)

			if nid not in expert_saliency:
				expert_saliency[nid] = _neuron_maps_from_all_saliency(all_expert_saliency_maps, nid)

	# __ MSSIM per control neuron x digit _________________________________________
	control_records = []
	for layer, nids in control_by_layer.items():
		for nid in nids:
			model_maps = _stack_saliency_maps_for_mssim(
				control_saliency[nid], keep_torch = USE_CW_SSIM
			)
			exp_maps = _stack_saliency_maps_for_mssim(
				expert_saliency[nid], keep_torch = USE_CW_SSIM
			)

			for d in range(_NETWORK_N_DIGITS):
				m = saliency_digit_pair_mssim(
					model_maps[d],
					exp_maps[d],
					store_full_grids=False,
					prebuilt_pyramids=pyramid_cache,
				)
				control_records.append({
					"layer_name": layer,
					"neuron_id": nid,
					"digit": d,
					"mean_of_mssim": m["mean_of_mssim"],
					"MSSIM_of_mean": m["MSSIM_of_mean"],
				})

	df_control_ssim = pd.DataFrame(control_records)
	df_layer_baseline = (
		df_control_ssim
		.groupby(["layer_name","digit"])[["mean_of_mssim", "MSSIM_of_mean"]]
		.mean()
	)
	return df_layer_baseline


def get_normalize_ssims(period_ssims, layer_ssims):
	merged_df = period_ssims.reset_index().merge(layer_ssims,on=["layer_name","digit"],how="left",suffixes=("","_layer")).set_index(index_grp + ["digit"])
	
	normalized = merged_df.assign(
		normalized_mean_of_mssim=lambda df: df["mean_of_mssim"] / df["mean_of_mssim_layer"], 
		normalized_MSSIM_of_mean=lambda df: df["MSSIM_of_mean"] / df["MSSIM_of_mean_layer"], 
		).drop(columns = {"mean_of_mssim","mean_of_mssim_layer","MSSIM_of_mean","MSSIM_of_mean_layer","layer_name"})

	return normalized


def map_pairwise_mse(map_a, map_b) -> float:
	a = np.asarray(map_a, dtype=np.float64)
	b = np.asarray(map_b, dtype=np.float64)
	return float(np.mean((a - b) ** 2))


def get_cross_sample_distances_from_maps(
	model_saliency_maps,
	*,
	data_range,
	compute_full_grid=False,
	prebuilt_pyramids: dict | None = None,
	map_metric: str = "cw_ssim",
):
	"""
	`MSSIM` for a model, cross samples for:
		* a given digit when compute_full_grid = False
		* for cross-sample-cross-digit when compute_full_grids = True
	
	When compute_full_grid=False, you can control whether to store the ssim of the sample-to-self (diagonal of the grid).
	"""
	if prebuilt_pyramids is None:
		prebuilt_pyramids = {}
	if map_metric not in {"cw_ssim", "mse"}:
		raise ValueError(f"map_metric must be 'cw_ssim' or 'mse', got {map_metric!r}")
	out_records = {}
	if compute_full_grid:
		if model_saliency_maps.shape[:2] != (_NETWORK_N_DIGITS, _NETWORK_SAMPLES_PER_DIGIT):
			raise ValueError(f"When computing full grids model saliency maps must be {_NETWORK_N_DIGITS}x{_NETWORK_SAMPLES_PER_DIGIT} but are {model_saliency_maps.shape[:2]}")
	else:
		if model_saliency_maps.shape[0] != _NETWORK_SAMPLES_PER_DIGIT:
			raise ValueError("model and expert must have the same number of samples per digit")
	
	if compute_full_grid:
		similarity_grid = np.empty((_NETWORK_N_DIGITS * _NETWORK_SAMPLES_PER_DIGIT, _NETWORK_N_DIGITS * _NETWORK_SAMPLES_PER_DIGIT), dtype=np.float64)
		for d1 in range(_NETWORK_N_DIGITS):
			for d2 in range(d1,_NETWORK_N_DIGITS):
				for i in range(_NETWORK_SAMPLES_PER_DIGIT):
					j_start = i if d1 == d2 else 0
					for j in range(j_start, _NETWORK_SAMPLES_PER_DIGIT):
						if map_metric == "mse":
							value = map_pairwise_mse(
								model_saliency_maps[d1, i],
								model_saliency_maps[d2, j],
							)
						elif USE_CW_SSIM:
							value = cw_ssim_with_gauss(
								model_saliency_maps[d1, i],
								model_saliency_maps[d2, j],
								data_range=data_range,
								prebuilt_pyramids=prebuilt_pyramids,
							)
						else:
							mssim, _ = _mssim(
								model_saliency_maps[d1, i],
								model_saliency_maps[d2, j],
								full=False,
							)
							value = float(mssim)
						out_index = (d1 * _NETWORK_SAMPLES_PER_DIGIT + i, d2 * _NETWORK_SAMPLES_PER_DIGIT + j)
						similarity_grid[*out_index] = value
						if out_index[0] != out_index[1]:
							similarity_grid[*out_index[::-1]] = value

	else:
		similarity_grid = np.empty((_NETWORK_SAMPLES_PER_DIGIT, _NETWORK_SAMPLES_PER_DIGIT), dtype=np.float64)
		for i in range(_NETWORK_SAMPLES_PER_DIGIT):
			for j in range(i, _NETWORK_SAMPLES_PER_DIGIT):
				if map_metric == "mse":
					value = map_pairwise_mse(model_saliency_maps[i], model_saliency_maps[j])
				elif USE_CW_SSIM:
					value = cw_ssim_with_gauss(
						model_saliency_maps[i],
						model_saliency_maps[j],
						data_range=data_range,
						prebuilt_pyramids=prebuilt_pyramids,
					)
				else:
					mssim, _ = _mssim(
						model_saliency_maps[i],
						model_saliency_maps[j],
						full=False,
					)
					value = float(mssim)
				similarity_grid[i, j] = value,
				if i != j:
					similarity_grid[j, i] = value
	
	out_records["cross_sample_distances"] = similarity_grid # (S,S) or (D,S,S)
	return out_records


def get_all_cross_sample_distances(recovery_periods, period_saliency_maps, expert_saliency, *, cross_digit_grid = False, prebuilt_pyramids: dict | None = None):
	pyramids_cache = {} if prebuilt_pyramids is None else prebuilt_pyramids
	records_model = []
	records_expert = []
	for row in tqdm(list(recovery_periods.itertuples()), desc=f"Computing {'SSIM' if not USE_CW_SSIM else 'CW-SSIM'}"):
		nid, pid = row.neuron_id, row.period_id
		model_maps = _stack_saliency_maps_for_mssim(
			period_saliency_maps[(nid, pid)], keep_torch = USE_CW_SSIM
		)
		exp_maps = _stack_saliency_maps_for_mssim(
			expert_saliency[nid], keep_torch = USE_CW_SSIM
		)
		data_range_model = get_data_range(model_maps)
		data_range_expert = get_data_range(exp_maps)

		if cross_digit_grid:
			model_distances_dict = get_cross_sample_distances_from_maps(
				model_maps, data_range = data_range_model, compute_full_grid=True, prebuilt_pyramids=pyramids_cache) # (D*S,D*S)
			expert_distances_dict = get_cross_sample_distances_from_maps(
				exp_maps, data_range = data_range_expert, compute_full_grid=True, prebuilt_pyramids=pyramids_cache) # (D*S,D*S)

			records_model.append({
				"neuron_id": nid,
				"period_id": pid,
				"cross_sample_distances": model_distances_dict["cross_sample_distances"], # (D,S,S)
			})
			records_expert.append({
				"neuron_id": nid,
				"period_id": pid,
				"cross_sample_distances": expert_distances_dict["cross_sample_distances"], # (D,S,S)
			})
		else:
			for d in range(_NETWORK_N_DIGITS):
				model_distances_dict = get_cross_sample_distances_from_maps(
					model_maps[d], data_range = data_range_model, compute_full_grid=False, prebuilt_pyramids=pyramids_cache)
				expert_distances_dict = get_cross_sample_distances_from_maps(
					exp_maps[d], data_range = data_range_expert, compute_full_grid=False, prebuilt_pyramids=pyramids_cache)
				
				records_model.append({
					"neuron_id": nid,
					"period_id": pid,
					"digit": d,
					"cross_sample_distances": model_distances_dict["cross_sample_distances"], # (S,S)
				})
				records_expert.append({
					"neuron_id": nid,
					"period_id": pid,
					"digit": d,
					"cross_sample_distances": expert_distances_dict["cross_sample_distances"], # (S,S)
				})

	df_distances_model = pd.DataFrame(records_model).set_index(index_grp + (["digit"] if not cross_digit_grid else []))
	df_distances_expert = pd.DataFrame(records_expert).set_index(index_grp + (["digit"] if not cross_digit_grid else []))
	
	return df_distances_model, df_distances_expert


def compute_repr_sim_factor(
	nts,
	neuron_id,
	period_id,
	model_similarities,
	expert_similarities,
	recovery_periods,
	*,
	invert_grids: bool = True,
):
	if invert_grids:
		d1, d2 = 1 / model_similarities, 1 / expert_similarities
	else:
		d1, d2 = model_similarities, expert_similarities
	G_se = compute_G(distance_to_rank(d1), distance_to_rank(d2))
	assigned_digits = get_assigned_digits_for_trial_activations(nts,recovery_periods,neuron_id,period_id)
	sampled = sample_ics_for_digits(assigned_digits)
	idcs_of_interest = np.ix_(sampled,sampled)
	# rest_mask = np.ones(G_se.shape, dtype=bool)
	# rest_mask[idcs_of_interest] = False
	return np.nan_to_num(G_se[idcs_of_interest].mean())#/np.abs(G_se[rest_mask]).mean())


def repr_similarity_score_function(x):
	# Piece-wise sigmoid: > 0 = 0; < 0 = sigmoid-like such that s(-1.5) = 0.9
	k = 2 * np.log(9)
	return 0.0 if x >= 0 else 1 / (1 + np.exp(k * (x + 1)))


def get_repr_similiraity_scores(df_repr_similarity_factors):
	return df_repr_similarity_factors.assign(
		repr_similarity_score = lambda df: df.apply(lambda row: repr_similarity_score_function(row["representation_similarity_factor"]),axis=1)
	).drop(columns={"representation_similarity_factor"})


def _infer_last_checkpoint_idx(nts=None, metrics=None) -> int:
	if metrics is not None:
		return len(metrics["checkpoint_iterations"]) - 1
	if nts is None:
		raise ValueError("Either nts or metrics is required to infer the last checkpoint index.")

	for key, value in nts.items():
		if str(key).startswith(("weights__", "act__")) and hasattr(value, "shape") and len(value.shape) > 0:
			return int(value.shape[0]) - 1
	raise ValueError("Could not infer checkpoint count from nts.")


def _uniform_checkpoint_indices(first: int, last_exclusive: int, n_points: int) -> np.ndarray:
	"""Up to `n_points` checkpoint indices in `[first, last_exclusive)`, spaced uniformly."""
	if last_exclusive <= first or n_points <= 0:
		return np.array([], dtype=np.int64)
	span = last_exclusive - first
	if span <= n_points:
		return np.arange(first, last_exclusive, dtype=np.int64)
	xs = np.linspace(first, last_exclusive - 1, num=n_points)
	idx = np.unique(np.round(xs).astype(np.int64))
	return idx[(idx >= first) & (idx < last_exclusive)]


def _legacy_stride_checkpoint_indices(first: int, last_exclusive: int, stride: int) -> np.ndarray:
	"""Deprecated: dense samples at the start then `stride` spacing (old behaviour)."""
	if last_exclusive <= first or stride < 1:
		return np.array([], dtype=np.int64)
	dense_end = min(first + 5, last_exclusive)
	seg_dense = np.arange(first, dense_end, dtype=np.int64)
	if last_exclusive <= dense_end:
		return seg_dense
	rest = np.arange(dense_end, last_exclusive, stride, dtype=np.int64)
	return np.unique(np.r_[seg_dense, rest])


def get_period_checkpoint_indices(
	recovery_periods,
	*,
	n_points: int | None = 20,
	stride: int | None = None,
	nts=None,
	metrics=None,
	btsp_start_strategy: str = "trial_start",
	peak_time_aggregate: Literal["min", "max"] = "min",
):
	"""Return one row per sampled checkpoint in each recovery period's effective window.

	Effective window `[s, e)` uses the same `s` as the through-time repr/SSIM code
	(`btsp_start_strategy` / `peak_time_aggregate`), and `e` is the period end
	capped by the last checkpoint.

	**Sampling modes**

	- **Default (`stride is None`):** uniform grid with at most `n_points` indices
	  in `[s, e)`. If `n_points` is `None`, `20` is used.
	- **Deprecated (`stride` set):** legacy dense prefix (up to five checkpoints)
	  then every `stride`-th checkpoint; pass `stride` only for old notebooks.

	btsp_start_strategy: "trial_start" or "acceleration" - determines the checkpoint
	where the period's interval starts. "acceleration" uses the point of max acceleration
	as the start, while "trial_start" uses the period's assigned start checkpoint.

	peak_time_aggregate: when using acceleration, earliest ("min") vs latest ("max") peak across assigned-digit channels.
	"""
	last_checkpoint_idx = _infer_last_checkpoint_idx(nts=nts, metrics=metrics)
	cap = last_checkpoint_idx + 1

	effective_starts = []
	for _, row in recovery_periods.iterrows():
		_, _, trial_start_cp = get_trial_starts(recovery_periods, row["neuron_id"], row["period_id"])
		if btsp_start_strategy == "acceleration":
			assert nts is not None
			accel_point = point_of_max_acceleration(
				nts, recovery_periods, row["neuron_id"], row["period_id"], peak_time_aggregate=peak_time_aggregate
			)
			effective_start = trial_start_cp + accel_point
			effective_starts.append(int(effective_start))
		else:
			assert btsp_start_strategy == "trial_start"
			effective_starts.append(int(trial_start_cp))

	starts = np.array(effective_starts, dtype=np.int64)

	ends = np.minimum(recovery_periods["end"].fillna(cap).astype(np.int64).to_numpy(), cap)
	if stride is not None:
		checkpoint_lists = [_legacy_stride_checkpoint_indices(int(s), int(e), int(stride)) for s, e in zip(starts, ends, strict=True)]
	else:
		npt = 20 if n_points is None else int(n_points)
		checkpoint_lists = [_uniform_checkpoint_indices(int(s), int(e), npt) for s, e in zip(starts, ends, strict=True)]

	out = recovery_periods.assign(checkpoint_idx=checkpoint_lists).explode(
		"checkpoint_idx",
		ignore_index=True,
	)
	# Empty intervals become [] -> explode yields NaN; original loop emitted no rows.
	out = out.dropna(subset=["checkpoint_idx"])
	out["checkpoint_idx"] = out["checkpoint_idx"].astype(np.int64)

	if metrics is not None:
		out["iteration"] = out["checkpoint_idx"].map(lambda c: get_iteration(metrics, int(c)))

	cols = ["neuron_id", "period_id", "layer_name", "checkpoint_idx"]
	if metrics is not None:
		cols.append("iteration")
	return out[cols]


def get_period_scores_through_time(
	nts,
	recovery_periods,
	expert_saliency,
	*,
	device=None,
	convolve_: bool = False,
	metrics=None,
	prebuilt_pyramids: dict | None = None,
	checkpoint_df=None,
	ssim_all_digits: bool = True,
	btsp_start_strategy: str = "trial_start",
	peak_time_aggregate: Literal["min", "max"] = "min",
	compute_agg_df: bool = True,
):
	"""Compute through-time MSSIM and representation scores without caching saliency maps.

	The older through-time flow first materialized `period_saliency_maps_through_time`
	for every `(neuron, period, checkpoint)`. Long periods can make that dictionary
	too large. This streams by checkpoint instead: compute all-neuron saliency once
	for the checkpoint, score every period row that needs it, then immediately drop
	the checkpoint saliency tensor.
	"""
	pyramids_cache = {} if prebuilt_pyramids is None else prebuilt_pyramids
	if checkpoint_df is None:
		checkpoint_df = get_period_checkpoint_indices(
			recovery_periods,
			n_points=20,
			nts=nts,
			metrics=metrics,
			btsp_start_strategy=btsp_start_strategy,
			peak_time_aggregate=peak_time_aggregate,
		)
	if checkpoint_df.empty:
		df_ssim = pd.DataFrame(
			columns=index_grp + ["checkpoint_idx", "digit", "layer_name", "mean_of_mssim", "mssim_of_mean"]
		).set_index(index_grp + ["checkpoint_idx", "digit"])
		df_repr = pd.DataFrame(
			columns=index_grp + ["checkpoint_idx", "representation_similarity_factor"]
		).set_index(index_grp + ["checkpoint_idx"])
		return df_ssim, df_repr, combine_through_time_metrics(df_ssim, df_repr)

	expert_maps_cache = {}
	expert_grid_cache = {}
	ssim_records = []
	repr_records = []
	period_digits_cache: dict[tuple[int, int], set] = {}

	for checkpoint_idx, checkpoint_rows in tqdm(
		checkpoint_df.groupby("checkpoint_idx", sort=True),
		desc="Computing through-time metrics",
	):
		checkpoint_idx = int(checkpoint_idx)
		max_stack_depth = int(checkpoint_rows["layer_name"].map(_stack_depth_from_layer_name).max())
		all_saliency_maps = get_all_saliency_maps_up_to_layer(
			nts,
			checkpoint_idx,
			max_stack_depth,
			device=device,
			convolve_=convolve_,
		)

		for row in checkpoint_rows.to_dict("records"):
			nid, pid = row["neuron_id"], row["period_id"]
			model_saliency = get_neuron_saliency_maps_from_all(all_saliency_maps, nid)
			model_maps = _stack_saliency_maps_for_mssim(model_saliency, keep_torch=USE_CW_SSIM)

			if nid not in expert_maps_cache:
				expert_maps_cache[nid] = _stack_saliency_maps_for_mssim(
					expert_saliency[nid],
					keep_torch=USE_CW_SSIM,
				)
			exp_maps = expert_maps_cache[nid]

			period_digits = []
			if not ssim_all_digits:
				key = (nid, pid)
				if key not in period_digits_cache:
					period_digits_cache[key] = set(
						get_assigned_digits_for_trial_activations(
							nts, recovery_periods, nid, pid
						).flatten().tolist()
					)
				period_digits = period_digits_cache[key]

			for d in range(_NETWORK_N_DIGITS):
				if not ssim_all_digits and d not in period_digits:
					continue
				m = saliency_digit_pair_mssim(
					model_maps[d],
					exp_maps[d],
					store_full_grids=False,
					prebuilt_pyramids=pyramids_cache,
				)
				ssim_records.append({
					"neuron_id": nid,
					"period_id": pid,
					"checkpoint_idx": checkpoint_idx,
					"digit": d,
					"layer_name": row["layer_name"],
					"mean_of_mssim": m["mean_of_mssim"],
					"mssim_of_mean": m["MSSIM_of_mean"],
				})

			model_distances = get_cross_sample_distances_from_maps(
				model_maps,
				data_range=get_data_range(model_maps),
				compute_full_grid=True,
				prebuilt_pyramids=pyramids_cache,
			)["cross_sample_distances"]

			if nid not in expert_grid_cache:
				expert_grid_cache[nid] = get_cross_sample_distances_from_maps(
					exp_maps,
					data_range=get_data_range(exp_maps),
					compute_full_grid=True,
					prebuilt_pyramids=pyramids_cache,
				)["cross_sample_distances"]

			repr_records.append({
				"neuron_id": nid,
				"period_id": pid,
				"checkpoint_idx": checkpoint_idx,
				"representation_similarity_factor": compute_repr_sim_factor(
					nts,
					nid,
					pid,
					model_distances,
					expert_grid_cache[nid],
					recovery_periods,
				),
			})

			del model_saliency, model_maps, model_distances

		del all_saliency_maps
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	df_ssim = pd.DataFrame(ssim_records).set_index(index_grp + ["checkpoint_idx", "digit"])
	df_repr = pd.DataFrame(repr_records).set_index(index_grp + ["checkpoint_idx"])
	if (
		metrics is not None
		and checkpoint_df is not None
		and not checkpoint_df.empty
		and "iteration" in checkpoint_df.columns
	):
		iter_merge = checkpoint_df[index_grp + ["checkpoint_idx", "iteration"]].drop_duplicates(
			subset=index_grp + ["checkpoint_idx"]
		)
		df_repr = (
			df_repr.reset_index()
			.merge(iter_merge, on=index_grp + ["checkpoint_idx"], how="left")
			.set_index(index_grp + ["checkpoint_idx"])
		)
	df_robustness = None
	if compute_agg_df:
		df_robustness = combine_through_time_metrics(df_ssim, df_repr)
		checkpoint_index = checkpoint_df.set_index(index_grp + ["checkpoint_idx"])
		if "iteration" in checkpoint_index.columns:
			df_robustness = df_robustness.join(checkpoint_index[["iteration"]], how="left")

	return df_ssim, df_repr, df_robustness


def get_period_saliency_maps_through_time(
	nts,
	recovery_periods,
	*,
	device=None,
	convolve_: bool = False,
	metrics=None,
	checkpoint_df=None,
):
	"""Compute neuron saliency maps at every checkpoint in each recovery period."""
	if checkpoint_df is None:
		checkpoint_df = get_period_checkpoint_indices(
			recovery_periods,
			n_points=20,
			nts=nts,
			metrics=metrics,
		)

	period_saliency_maps_through_time = {}

	for checkpoint_idx, checkpoint_rows in tqdm(
		checkpoint_df.groupby("checkpoint_idx", sort=True),
		desc="Computing saliency maps through time",
	):
		checkpoint_idx = int(str(checkpoint_idx))
		all_saliency_maps = get_all_saliency_maps(
			nts,
			checkpoint_idx=checkpoint_idx,
			device=device,
			convolve_=convolve_,
		)
		for row in checkpoint_rows.to_dict("records"):
			period_saliency_maps_through_time[(row["neuron_id"], row["period_id"], checkpoint_idx)] = get_neuron_saliency_maps_from_all(
				all_saliency_maps,
				row["neuron_id"],
			)
		del all_saliency_maps
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	return period_saliency_maps_through_time


def get_period_mssims_through_time(
	recovery_periods,
	period_saliency_maps_through_time,
	expert_saliency,
	*,
	prebuilt_pyramids: dict | None = None,
	checkpoint_df=None,
):
	"""Compute model-vs-expert saliency similarity per period checkpoint and digit."""
	pyramids_cache = {} if prebuilt_pyramids is None else prebuilt_pyramids
	if checkpoint_df is None:
		checkpoint_df = pd.DataFrame(
			[
				{"neuron_id": nid, "period_id": pid, "checkpoint_idx": checkpoint_idx}
				for nid, pid, checkpoint_idx in period_saliency_maps_through_time
			]
		).merge(recovery_periods[index_grp + ["layer_name"]], on=index_grp, how="left")

	records = []
	for row in tqdm(checkpoint_df.to_dict("records"), desc="Computing SSIM through time"):
		nid, pid = row["neuron_id"], row["period_id"]
		checkpoint_idx = int(row["checkpoint_idx"])
		model_maps = _stack_saliency_maps_for_mssim(
			period_saliency_maps_through_time[(nid, pid, checkpoint_idx)], keep_torch=USE_CW_SSIM
		)
		exp_maps = _stack_saliency_maps_for_mssim(
			expert_saliency[nid], keep_torch=USE_CW_SSIM
		)

		for d in range(_NETWORK_N_DIGITS):
			m = saliency_digit_pair_mssim(
				model_maps[d],
				exp_maps[d],
				store_full_grids=False,
				prebuilt_pyramids=pyramids_cache,
			)
			records.append({
				"neuron_id": nid,
				"period_id": pid,
				"checkpoint_idx": checkpoint_idx,
				"digit": d,
				"layer_name": row["layer_name"],
				"mean_of_mssim": m["mean_of_mssim"],
				"mssim_of_mean": m["MSSIM_of_mean"],
			})

	return pd.DataFrame(records).set_index(index_grp + ["checkpoint_idx", "digit"])


def get_repr_similarity_factors_through_time(
	nts,
	recovery_periods,
	period_saliency_maps_through_time,
	expert_saliency,
	*,
	prebuilt_pyramids: dict | None = None,
	checkpoint_df=None,
):
	"""Compute representation similarity factor at every checkpoint in each period."""
	pyramids_cache = {} if prebuilt_pyramids is None else prebuilt_pyramids
	if checkpoint_df is None:
		checkpoint_df = pd.DataFrame(
			[
				{"neuron_id": nid, "period_id": pid, "checkpoint_idx": checkpoint_idx}
				for nid, pid, checkpoint_idx in period_saliency_maps_through_time
			]
		)

	expert_grid_cache = {}
	records = []
	for row in tqdm(checkpoint_df.to_dict("records"), desc="Computing representation similarity through time"):
		nid, pid = row["neuron_id"], row["period_id"]
		checkpoint_idx = int(row["checkpoint_idx"])
		model_maps = _stack_saliency_maps_for_mssim(
			period_saliency_maps_through_time[(nid, pid, checkpoint_idx)], keep_torch=USE_CW_SSIM
		)

		model_distances_dict = get_cross_sample_distances_from_maps(
			model_maps,
			data_range=get_data_range(model_maps),
			compute_full_grid=True,
			prebuilt_pyramids=pyramids_cache,
		)

		if nid not in expert_grid_cache:
			exp_maps = _stack_saliency_maps_for_mssim(
				expert_saliency[nid], keep_torch=USE_CW_SSIM
			)
			expert_grid_cache[nid] = get_cross_sample_distances_from_maps(
				exp_maps,
				data_range=get_data_range(exp_maps),
				compute_full_grid=True,
				prebuilt_pyramids=pyramids_cache,
			)["cross_sample_distances"]

		records.append({
			"neuron_id": nid,
			"period_id": pid,
			"checkpoint_idx": checkpoint_idx,
			"representation_similarity_factor": compute_repr_sim_factor(
				nts,
				nid,
				pid,
				model_distances_dict["cross_sample_distances"],
				expert_grid_cache[nid],
				recovery_periods,
			),
		})

	return pd.DataFrame(records).set_index(index_grp + ["checkpoint_idx"])


def combine_through_time_metrics(df_ssim_through_time, df_repr_similarity_factors_through_time):
	"""Merge per-digit MSSIM rows with per-checkpoint representation factors."""
	ssim_reset = df_ssim_through_time.reset_index()
	group_columns = index_grp + ["checkpoint_idx"]
	aggregation = {
		"mean_of_mssim": "mean",
		"mssim_of_mean": "mean",
	}
	if "layer_name" in ssim_reset.columns:
		aggregation["layer_name"] = "first"

	ssim_by_checkpoint = ssim_reset.groupby(group_columns).agg(aggregation)
	return ssim_by_checkpoint.join(df_repr_similarity_factors_through_time, how="left")


def summarize_through_time_metrics(df_robustness_through_time):
	"""Aggregate through-time metric trajectories into mean and SEM per period."""
	metric_columns = ["mean_of_mssim", "mssim_of_mean", "representation_similarity_factor"]
	df = df_robustness_through_time.reset_index()
	summary = df.groupby(index_grp)[metric_columns].agg(["mean", "sem"])
	summary.columns = [f"TT_{metric}_{stat}" for metric, stat in summary.columns]
	return summary


def compute_period_robustness_trajectory(
	df_repr_through_time: pd.DataFrame,
	recovery_periods: pd.DataFrame | None = None,
) -> pd.DataFrame:
	"""Per BTSP period statistics from `representation_similarity_factor` (= `d(t_k)`).

	For each period, `\\bar{d} = \\frac{1}{N}\\sum_k d(t_k)` and feature robustness
	`r(b_i) = \\sqrt{\\frac{1}{N}\\sum_k (d(t_k)-\\bar{d})^2}` (population RMSE, `ddof=0`).

	Returns one row per period with `d_tk` (length `N`, typically ~20) plus scalars
	`d_bar`, `robustness`, and `n_checkpoints`, and aligned lists `sampled_checkpoint_idx`
	(and `sampled_iteration` when an `iteration` column is present on the input).
	"""
	empty_cols = [
		*index_grp,
		"layer_name",
		"n_checkpoints",
		"d_bar",
		"robustness",
		"d_tk",
		"sampled_checkpoint_idx",
		"sampled_iteration",
	]
	empty_out = pd.DataFrame(columns=empty_cols)
	cols_needed = ["neuron_id", "period_id", "checkpoint_idx", "representation_similarity_factor"]
	if df_repr_through_time.empty:
		return empty_out
	dfr0 = df_repr_through_time.reset_index()
	missing = [c for c in cols_needed if c not in dfr0.columns]
	if missing:
		return empty_out

	has_iteration = "iteration" in dfr0.columns
	use_cols = cols_needed + (["iteration"] if has_iteration else [])
	df = dfr0[use_cols].copy()
	df = df.replace([np.inf, -np.inf], np.nan)
	vals = np.asarray(pd.to_numeric(df["representation_similarity_factor"], errors="coerce"), dtype=np.float64)
	df = df.loc[np.isfinite(vals)].copy()
	rows: list[dict] = []
	for _, g in df.groupby(["neuron_id", "period_id"], sort=False):
		nid = g["neuron_id"].iloc[0]
		pid = g["period_id"].iloc[0]
		g = g.sort_values("checkpoint_idx")
		d_tk = g["representation_similarity_factor"].to_numpy(dtype=np.float64)
		cp_tk = g["checkpoint_idx"].astype(np.int64).to_numpy()
		n = int(d_tk.size)
		if n == 0:
			continue
		d_bar = float(np.mean(d_tk))
		robustness = float(np.sqrt(np.mean((d_tk - d_bar) ** 2)))
		row_dict: dict = {
			"neuron_id": nid,
			"period_id": pid,
			"n_checkpoints": n,
			"d_bar": d_bar,
			"robustness": robustness,
			"d_tk": d_tk.tolist(),
			"sampled_checkpoint_idx": cp_tk.astype(int).tolist(),
		}
		if has_iteration:
			it_arr = pd.to_numeric(g["iteration"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
			row_dict["sampled_iteration"] = [float(x) if np.isfinite(x) else float("nan") for x in it_arr]
		else:
			row_dict["sampled_iteration"] = []
		rows.append(row_dict)

	out = pd.DataFrame(rows)
	if out.empty:
		return empty_out.copy()

	if recovery_periods is not None and "layer_name" in recovery_periods.columns:
		layer_df = (
			recovery_periods[["neuron_id", "period_id", "layer_name"]]
			.groupby(["neuron_id", "period_id"], sort=False, as_index=False)
			.first()
		)
		out = out.merge(layer_df, on=["neuron_id", "period_id"], how="left")
	else:
		out["layer_name"] = ""

	return out


def final_stream_repr_distance_through_time(
	nts,
	recovery_periods,
	expert_saliency,
	checkpoint_df: pd.DataFrame,
	*,
	device=None,
	convolve_: bool = False,
	prebuilt_pyramids: dict | None = None,
) -> pd.DataFrame:
	"""Stream saliency Jacobians and record `repr_distance` only (no MSSIM).

	`repr_distance` is the same scalar previously stored as
	`representation_similarity_factor` in `get_period_scores_through_time`.
	Callers must pass a non-empty `checkpoint_df` (e.g. from
	`get_period_checkpoint_indices` with `n_points=20`).
	"""
	pyramids_cache = {} if prebuilt_pyramids is None else prebuilt_pyramids
	if checkpoint_df.empty:
		return pd.DataFrame(
			columns=index_grp + ["checkpoint_idx", "repr_distance"]
		).set_index(index_grp + ["checkpoint_idx"])

	expert_maps_cache: dict = {}
	expert_grid_cache: dict = {}
	repr_records: list[dict] = []

	for checkpoint_idx, checkpoint_rows in tqdm(
		checkpoint_df.groupby("checkpoint_idx", sort=True),
		desc="Computing repr_distance through time",
	):
		checkpoint_idx = int(checkpoint_idx)
		max_stack_depth = int(checkpoint_rows["layer_name"].map(_stack_depth_from_layer_name).max())
		all_saliency_maps = get_all_saliency_maps_up_to_layer(
			nts,
			checkpoint_idx,
			max_stack_depth,
			device=device,
			convolve_=convolve_,
		)

		for row in checkpoint_rows.to_dict("records"):
			nid, pid = row["neuron_id"], row["period_id"]
			model_saliency = get_neuron_saliency_maps_from_all(all_saliency_maps, nid)
			model_maps = _stack_saliency_maps_for_mssim(model_saliency, keep_torch=USE_CW_SSIM)

			if nid not in expert_maps_cache:
				expert_maps_cache[nid] = _stack_saliency_maps_for_mssim(
					expert_saliency[nid],
					keep_torch=USE_CW_SSIM,
				)
			exp_maps = expert_maps_cache[nid]

			model_distances = get_cross_sample_distances_from_maps(
				model_maps,
				data_range=get_data_range(model_maps),
				compute_full_grid=True,
				prebuilt_pyramids=pyramids_cache,
			)["cross_sample_distances"]

			if nid not in expert_grid_cache:
				expert_grid_cache[nid] = get_cross_sample_distances_from_maps(
					exp_maps,
					data_range=get_data_range(exp_maps),
					compute_full_grid=True,
					prebuilt_pyramids=pyramids_cache,
				)["cross_sample_distances"]

			repr_records.append({
				"neuron_id": nid,
				"period_id": pid,
				"checkpoint_idx": checkpoint_idx,
				"repr_distance": compute_repr_sim_factor(
					nts,
					nid,
					pid,
					model_distances,
					expert_grid_cache[nid],
					recovery_periods,
				),
			})

			del model_saliency, model_maps, model_distances

		del all_saliency_maps
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	df_repr = pd.DataFrame(repr_records).set_index(index_grp + ["checkpoint_idx"])
	if not checkpoint_df.empty and "iteration" in checkpoint_df.columns:
		iter_merge = checkpoint_df[index_grp + ["checkpoint_idx", "iteration"]].drop_duplicates(
			subset=index_grp + ["checkpoint_idx"]
		)
		df_repr = (
			df_repr.reset_index()
			.merge(iter_merge, on=index_grp + ["checkpoint_idx"], how="left")
			.set_index(index_grp + ["checkpoint_idx"])
		)
	return df_repr


def final_stream_grad_nam_repr_distance_through_time(
	nts,
	recovery_periods,
	expert_saliency,
	checkpoint_df: pd.DataFrame,
	conv_acts: np.ndarray,
	*,
	target_size: tuple[int, int],
	device=None,
	map_metric: str = "mse",
	invert_grids: bool = False,
) -> pd.DataFrame:
	"""CIFAR Grad-NAM path: cached pre-GAP conv acts + NTS DNN Jacobians at each checkpoint."""
	# imported here to avoid circular import
	from analysis.grad_nam import grad_nam_maps_for_neuron_from_nts

	assert map_metric == "mse", "CW-SSIM is unstable with Grad-NAM maps (see local grad_cam_tests.ipynb)"
	assert not invert_grids, "Actually, we don't want to invert the grids for the CIFAR Grad-NAM maps since we use MSE which is already well ordered"
	device = torch.device(device) if device is not None else torch.device(
		"cuda" if torch.cuda.is_available() else "cpu"
	)
	if checkpoint_df.empty:
		return pd.DataFrame(
			columns=index_grp + ["checkpoint_idx", "repr_distance"]
		).set_index(index_grp + ["checkpoint_idx"])

	expert_grid_cache: dict = {}
	repr_records: list[dict] = []

	for checkpoint_idx, checkpoint_rows in tqdm(
		checkpoint_df.groupby("checkpoint_idx", sort=True),
		desc="Computing Grad-NAM repr_distance through time",
	):
		checkpoint_idx = int(checkpoint_idx)
		for row in checkpoint_rows.to_dict("records"):
			nid, pid = row["neuron_id"], row["period_id"]
			model_maps = _stack_saliency_maps_for_mssim(
				grad_nam_maps_for_neuron_from_nts(
					conv_acts,
					nts,
					checkpoint_idx,
					nid,
					target_size=target_size,
					device=device,
				),
				keep_torch=False,
			)

			if nid not in expert_grid_cache:
				exp_maps = _stack_saliency_maps_for_mssim(
					expert_saliency[nid],
					keep_torch=False,
				)
				expert_grid_cache[nid] = get_cross_sample_distances_from_maps(
					exp_maps,
					data_range=0.0,
					compute_full_grid=True,
					map_metric=map_metric,
				)["cross_sample_distances"]

			model_distances = get_cross_sample_distances_from_maps(
				model_maps,
				data_range=0.0,
				compute_full_grid=True,
				map_metric=map_metric,
			)["cross_sample_distances"]

			repr_records.append(
				{
					"neuron_id": nid,
					"period_id": pid,
					"checkpoint_idx": checkpoint_idx,
					"repr_distance": compute_repr_sim_factor(
						nts,
						nid,
						pid,
						model_distances,
						expert_grid_cache[nid],
						recovery_periods,
						invert_grids=invert_grids,
					),
				}
			)
	
	# Format final df
	df_repr = pd.DataFrame(repr_records).set_index(index_grp + ["checkpoint_idx"])
	if not checkpoint_df.empty and "iteration" in checkpoint_df.columns:
		iter_merge = checkpoint_df[index_grp + ["checkpoint_idx", "iteration"]].drop_duplicates(
			subset=index_grp + ["checkpoint_idx"]
		)
		df_repr = (
			df_repr.reset_index()
			.merge(iter_merge, on=index_grp + ["checkpoint_idx"], how="left")
			.set_index(index_grp + ["checkpoint_idx"])
		)
	return df_repr


def final_compute_period_repr_trajectory_summary(
	df_repr_through_time: pd.DataFrame,
	recovery_periods: pd.DataFrame | None = None,
) -> pd.DataFrame:
	"""Per-period trajectory stats from `repr_distance` (= :math:`d(t_k)`).

	`repr_trajectory_dispersion` is the population RMSE of :math:`d(t_k)` around
	its mean (same formula as `robustness` in `compute_period_robustness_trajectory`).

	`robustness_score` is `clip(1 - repr_trajectory_dispersion, 0, 1)` with NaNs
	propagated when dispersion is non-finite.
	"""
	empty_cols = [
		*index_grp,
		"layer_name",
		"n_checkpoints",
		"d_bar",
		"repr_trajectory_dispersion",
		"robustness_score",
		"d_tk",
		"sampled_checkpoint_idx",
		"sampled_iteration",
	]
	empty_out = pd.DataFrame(columns=empty_cols)
	cols_needed = ["neuron_id", "period_id", "checkpoint_idx", "repr_distance"]
	if df_repr_through_time.empty:
		return empty_out
	dfr0 = df_repr_through_time.reset_index()
	missing = [c for c in cols_needed if c not in dfr0.columns]
	if missing:
		return empty_out

	has_iteration = "iteration" in dfr0.columns
	use_cols = cols_needed + (["iteration"] if has_iteration else [])
	df = dfr0[use_cols].copy()
	df = df.replace([np.inf, -np.inf], np.nan)
	vals = np.asarray(pd.to_numeric(df["repr_distance"], errors="coerce"), dtype=np.float64)
	df = df.loc[np.isfinite(vals)].copy()
	rows: list[dict] = []
	for _, g in df.groupby(["neuron_id", "period_id"], sort=False):
		nid = g["neuron_id"].iloc[0]
		pid = g["period_id"].iloc[0]
		g = g.sort_values("checkpoint_idx")
		d_tk = g["repr_distance"].to_numpy(dtype=np.float64)
		cp_tk = g["checkpoint_idx"].astype(np.int64).to_numpy()
		n = int(d_tk.size)
		if n == 0:
			continue
		d_bar = float(np.mean(d_tk))
		disp = float(np.sqrt(np.mean((d_tk - d_bar) ** 2)))
		if np.isfinite(disp):
			r_score = float(np.clip(1.0 - disp, 0.0, 1.0))
		else:
			r_score = float("nan")
		row_dict: dict = {
			"neuron_id": nid,
			"period_id": pid,
			"n_checkpoints": n,
			"d_bar": d_bar,
			"repr_trajectory_dispersion": disp,
			"robustness_score": r_score,
			"d_tk": d_tk.tolist(),
			"sampled_checkpoint_idx": cp_tk.astype(int).tolist(),
		}
		if has_iteration:
			it_arr = pd.to_numeric(g["iteration"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
			row_dict["sampled_iteration"] = [float(x) if np.isfinite(x) else float("nan") for x in it_arr]
		else:
			row_dict["sampled_iteration"] = []
		rows.append(row_dict)

	out = pd.DataFrame(rows)
	if out.empty:
		return empty_out.copy()

	if recovery_periods is not None and "layer_name" in recovery_periods.columns:
		layer_df = (
			recovery_periods[["neuron_id", "period_id", "layer_name"]]
			.groupby(["neuron_id", "period_id"], sort=False, as_index=False)
			.first()
		)
		out = out.merge(layer_df, on=["neuron_id", "period_id"], how="left")
	else:
		out["layer_name"] = ""

	return out


def init_cw_ssim_pyramid(device: torch.device, prebuilt_pyramids: dict):
	"""Pre-build steerable pyramid when USE_CW_SSIM is true (same as notebook cell)."""
	sp = None
	if USE_CW_SSIM:
		_H, _W = MNIST_IMAGE_SHAPE
		_lvl, _ori = PYRAMID_LEVELS, WAVELET_ORIENTATIONS
		sp, prebuilt_pyramids = _get_cw_ssim_steerable_pyramid(_H, _W, _lvl, _ori, device, prebuilt_pyramids)
	return sp, prebuilt_pyramids
"""Generic training loop that iterates through experiment trials."""
from pathlib import Path
from typing import Any, cast

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from compute_results.checkpoint import (
	NeuronTimeseriesCollector,
	PersistentActivationCapture,
	extract_unit_weights,
	extract_unit_activations,
	save_model_checkpoint,
)
from compute_results.post_processing import compute_post_processing
from compute_results.config_guard import (
	load_initial_optimizer_state_dict,
	parse_checkpoint_cadence,
)
from experiments.base import Experiment, TrialSpec
from models.base import AnalyzableModel
from optimizers.base import OptimizerSignalExtractor


NUM_SWITCH_FINEGRAIN_ITS = 20


class SoftmaxMSELoss(nn.Module):
	"""MSE between softmax(logits) and one-hot class indices."""

	def __init__(self, num_classes: int):
		super().__init__()
		self.num_classes = num_classes

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		probs = torch.softmax(logits, dim=1)
		oh = F.one_hot(targets.long(), self.num_classes).float()
		return F.mse_loss(probs, oh)


def _num_classes_for_loss(model: AnalyzableModel) -> int:
	fields = model.config_fields()
	nc = fields.get("num_classes")
	if nc is not None:
		return int(nc)
	for m in reversed(list(model.modules())):
		if isinstance(m, nn.Linear):
			return int(m.out_features)
	raise ValueError(
		"Cannot infer num_classes for MSE loss: set num_classes in model.config_fields() "
		"or use a model whose last layer is nn.Linear."
	)


def _criterion_from_config(config: dict[str, Any], model: AnalyzableModel) -> nn.Module:
	loss_key = str(config.get("loss", "ce"))
	if loss_key == "ce":
		return nn.CrossEntropyLoss()
	if loss_key == "mse":
		return SoftmaxMSELoss(_num_classes_for_loss(model))
	raise ValueError(
		f"Unknown loss in config: {loss_key!r}. Expected 'ce' or 'mse'."
	)


@torch.no_grad()
def evaluate(
	model: nn.Module,
	loader: torch.utils.data.DataLoader,
	device: torch.device,
	criterion: nn.Module,
) -> tuple[float, float]:
	"""Return (loss, accuracy) on *loader*."""
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0
	for x, y in loader:
		if x.device != device or y.device != device:
			x, y = x.to(device), y.to(device)
		logits = model(x)
		total_loss += criterion(logits, y).item() * y.size(0)
		correct += (logits.argmax(1) == y).sum().item()
		total += y.size(0)
	if total == 0:
		return 0.0, 0.0
	return total_loss / total, correct / total


def _drop_leading_once_only(trial_list: list[TrialSpec]) -> list[TrialSpec]:
	"""Remove a prefix of trials marked `once_only` (for pretrain checkpoint mode)."""
	out = list(trial_list)
	while out and out[0].once_only:
		out.pop(0)
	if not out:
		raise ValueError(
			"initial_model_mode 'pretrain' skipped all trials (only leading once_only trials)."
		)
	return out


def resolve_save_model_path(template: str, optimizer_dir: Path) -> Path:
	"""Expand `!OPT` to *optimizer_dir*; otherwise resolve under *optimizer_dir*."""
	t = template.strip()
	if not t:
		raise ValueError("save_model path is empty.")
	if "!OPT" in t:
		return Path(t.replace("!OPT", str(optimizer_dir)))
	return optimizer_dir / Path(t)


def _run_checkpoint(
	*,
	tag: str,
	iteration: int,
	model: AnalyzableModel,
	experiment: Experiment,
	device: torch.device,
	checkpoint_dir: Path,
	save_model_cp: bool,
	keep_tensors: bool,
	persistent_capture: PersistentActivationCapture,
	neuron_collector: NeuronTimeseriesCollector,
	eval_loaders: dict[str, torch.utils.data.DataLoader],
	criterion: nn.Module,
	metrics: dict[str, list],
	train_loss_accum: list[float],
	train_logger: SummaryWriter | None = None,
) -> None:
	"""Optionally save .pt weights; always capture activations, metrics, evaluate."""
	model.eval()
	
	if save_model_cp:
		save_model_checkpoint(model, checkpoint_dir, tag)

	eval_inputs, _ = experiment.evaluation_inputs(device)
	activations = persistent_capture.capture(eval_inputs, device)
	act_values = extract_unit_activations(activations, model.clickable_units(), keep_on_gpu=keep_tensors)
	weight_values = extract_unit_weights(model, keep_on_gpu=keep_tensors)
	neuron_collector.record(tag, act_values, weight_values)

	for loader_name, loader in eval_loaders.items():
		loss, acc = evaluate(model, loader, device, criterion)
		metrics.setdefault(f"loss_{loader_name}", []).append(loss)
		metrics.setdefault(f"acc_{loader_name}", []).append(acc)
		if train_logger is not None:
			train_logger.add_scalar(f"eval/{loader_name}_loss", loss, iteration)
			train_logger.add_scalar(f"eval/{loader_name}_acc", acc, iteration)

	n_train = len(train_loss_accum)
	if n_train > 0:
		arr = np.array(train_loss_accum, dtype=np.float64)
		train_mean = float(np.mean(arr))
		train_se = (
			float(np.std(arr, ddof=1) / np.sqrt(n_train)) if n_train > 1 else float("nan")
		)
	else:
		train_mean = float("nan")
		train_se = float("nan")
	metrics.setdefault("loss_all_train_mean", []).append(train_mean)
	metrics.setdefault("loss_all_train_se", []).append(train_se)
	if train_logger is not None:
		train_logger.add_scalar("train/loss_mean", train_mean, iteration)
		if not np.isnan(train_se):
			train_logger.add_scalar("train/loss_se", train_se, iteration)
	train_loss_accum.clear()

	metrics.setdefault("checkpoint_tags", []).append(tag)
	metrics.setdefault("checkpoint_iterations", []).append(iteration)
	model.train()

def train_with_config(
	*,
	experiment: Experiment,
	model: AnalyzableModel,
	extractor: OptimizerSignalExtractor,
	config: dict[str, Any],
	device: torch.device,
	results_dir: Path,
	train_logger: SummaryWriter | None = None,
) -> None:
	"""Run the full multi-trial training and save all outputs."""
	model = model.to(device)
	model.train()

	units = model.clickable_units()
	node_ids = [u["node_id"] for u in units]
	extractor.set_units(units)

	optimizer = extractor.create_optimizer(model.parameters())
	criterion = _criterion_from_config(config, model)

	trial_epochs = int(
		config.get("trial_epochs", config.get("stage_epochs", 1))
	)
	batch_size = config["batch_size"]
	seed = config["seed"]
	experiment_runs: int = int(
		config.get("experiment_runs", config.get("trials", 1))
	)
	experiment_variability: str = str(
		config.get("experiment_variability", config.get("trial_variability", ""))
	)
	cadence_mode, cadence_k = parse_checkpoint_cadence(
		config.get("checkpoint_cadence", "every_epoch")
	)
	save_model_cp: bool = bool(config.get("save_model_cp", False))
	keep_tensors: bool = bool(config.get("internal_keep_tensors", False))
	save_model_tmpl = str(config.get("save_model", "") or "").strip()
	initial_mode = str(config.get("initial_model_mode", "init") or "init")
	use_initial_model = config.get("use_initial_model", None)

	_init_path = str(use_initial_model or "").strip()
	if _init_path and Path(_init_path).is_dir():
		opt_state = load_initial_optimizer_state_dict(
			_init_path,
			registry_class_name=type(extractor).__name__,
			map_location=device,
		)
		optimizer.load_state_dict(opt_state)

	checkpoint_dir = results_dir / "checkpoints"
	optimizer_dir = results_dir / "optimizers" / extractor.optimizer_id()

	if keep_tensors:
		experiment.to_device(device)
	experiment.set_experiment_variability(experiment_variability)
	returned_trials = experiment.build_trials(
		batch_size=batch_size, seed=seed, num_experiment_runs=experiment_runs
	)

	assert returned_trials and len(returned_trials) > 0, "build_trials returned an empty list"
	is_nested = isinstance(returned_trials[0], list)

	neuron_collector = NeuronTimeseriesCollector(units, keep_tensors=keep_tensors)
	persistent_capture = PersistentActivationCapture(model, keep_on_gpu=keep_tensors)
	persistent_capture.install()
	metrics: dict[str, list] = {}
	train_loss_accum: list[float] = []

	signal_log: dict[str, dict[str, list[float]]] = {
		s: {nid: [] for nid in node_ids}
		for s in extractor.signal_names()
	}
	shampoo_layer_block_signals: dict[str, list[np.ndarray]] = {}
	iterations: list[int] = []

	all_eval_loaders: dict[str, torch.utils.data.DataLoader] = {}
	for trial_or_trial_list in returned_trials:
		if is_nested:
			assert isinstance(trial_or_trial_list, list)
			for trial in trial_or_trial_list:
				all_eval_loaders.update(trial.eval_loaders)
		else:
			assert isinstance(trial_or_trial_list, TrialSpec)
			all_eval_loaders.update(trial_or_trial_list.eval_loaders)

	_run_cp_kwargs: dict[str, Any] = {
		"model": model,
		"experiment": experiment,
		"device": device,
		"checkpoint_dir": checkpoint_dir,
		"save_model_cp": save_model_cp,
		"keep_tensors": keep_tensors,
		"persistent_capture": persistent_capture,
		"neuron_collector": neuron_collector,
		"eval_loaders": all_eval_loaders,
		"criterion": criterion,
		"metrics": metrics,
		"train_loss_accum": train_loss_accum,
		"train_logger": train_logger,
	}

	_run_checkpoint(tag="init", iteration=0, **_run_cp_kwargs)

	global_iter = 0
	trial_names_list: list[str] = []
	trial_end_checkpoint_idxs: list[int] = []
	trial_end_iterations_list: list[int] = []

	for run_idx in range(experiment_runs):
		if is_nested:
			trials_this_run = list(cast(list[TrialSpec], returned_trials[run_idx]))
		elif run_idx > 0:
			assert isinstance(returned_trials, list) and all(isinstance(s, TrialSpec) for s in returned_trials)
			trials_this_run = [s for s in returned_trials if not s.once_only] # type: ignore[attr-defined]
		else:
			trials_this_run = list(cast(list[TrialSpec], returned_trials))

		if use_initial_model and run_idx == 0 and initial_mode == "pretrain":
			trials_this_run = _drop_leading_once_only(cast(list[TrialSpec], trials_this_run))

		tag_prefix = f"run{run_idx}_" if experiment_runs > 1 else ""

		running_once_only = trials_this_run[0].once_only if trials_this_run else False
		for trial in trials_this_run: # type: ignore[attr-defined]
			assert isinstance(trial, TrialSpec)
			if running_once_only and not trial.once_only:
				running_once_only = False
				if run_idx == 0 and save_model_tmpl and initial_mode == "pretrain":
					dest = resolve_save_model_path(save_model_tmpl, optimizer_dir)
					dest.parent.mkdir(parents=True, exist_ok=True)
					torch.save(model.state_dict(), dest)
					print(f"Saved model to {dest}; initial_mode == 'pretrain'")
			# Trial start
			_run_checkpoint(
				tag=f"tsw_{tag_prefix}{trial.name}_entry",
				iteration=max(0, global_iter - 1),
				**_run_cp_kwargs,
			)
			sw_iters_remaining = NUM_SWITCH_FINEGRAIN_ITS

			for epoch in range(trial_epochs):
				for x, y in tqdm(
					trial.train_loader,
					desc=f"{tag_prefix}{trial.name} epoch{epoch}",
				):
					x, y = x.to(device), y.to(device)
					optimizer.zero_grad()
					logits = model(x)
					loss = criterion(logits, y)
					loss.backward()
					train_loss_accum.append(loss.item())
					if train_logger is not None:
						train_logger.add_scalar("train/loss", loss.item(), global_iter)

					before_signals = extractor.on_before_step(model, optimizer)
					optimizer.step()
					after_signals = extractor.on_after_step(model, optimizer)

					all_signals = {**before_signals, **after_signals}
					for s_name in extractor.signal_names():
						unit_vals = all_signals.get(s_name, {})
						for nid in node_ids:
							signal_log[s_name][nid].append(
								unit_vals.get(nid, float("nan"))
							)
					for bk, mat in extractor.on_after_step_shampoo_blocks(
						model, optimizer,
					).items():
						shampoo_layer_block_signals.setdefault(bk, []).append(mat)
					iterations.append(global_iter)
					global_iter += 1

					if sw_iters_remaining > 0:
						sw_iter_idx = (NUM_SWITCH_FINEGRAIN_ITS + 1) - sw_iters_remaining
						_run_checkpoint(
							tag=f"tsw_{tag_prefix}{trial.name}_iter{sw_iter_idx}",
							iteration=max(0, global_iter - 1),
							**_run_cp_kwargs,
						)
						sw_iters_remaining -= 1
					elif cadence_mode == "every_k_its" and cadence_k is not None and global_iter % cadence_k == 0:
						_run_checkpoint(
							tag=f"iter{global_iter}",
							iteration=max(0, global_iter - 1),
							**_run_cp_kwargs,
						)

				if cadence_mode == "every_epoch":
					tag = f"{tag_prefix}{trial.name}_epoch{epoch}"
					_run_checkpoint(
						tag=tag,
						iteration=max(0, global_iter - 1),
						**_run_cp_kwargs,
					)

			if trial.post_trial_callback is not None:
				trial.post_trial_callback(model)
				_run_checkpoint(
					tag=f"{tag_prefix}{trial.name}_post_callback",
					iteration=max(0, global_iter - 1),
					**_run_cp_kwargs,
				)

			# Record trial boundary (last checkpoint index and last iteration of this trial)
			trial_names_list.append(f"{tag_prefix}{trial.name}")
			trial_end_checkpoint_idxs.append(len(metrics["checkpoint_tags"]) - 1)
			trial_end_iterations_list.append(global_iter - 1 if global_iter > 0 else 0)

	if save_model_tmpl and initial_mode == "init":
		dest = resolve_save_model_path(save_model_tmpl, optimizer_dir)
		dest.parent.mkdir(parents=True, exist_ok=True)
		torch.save(model.state_dict(), dest)
		print(f"Saved model to {dest}; initial_mode == 'init'")

	# --- Cleanup hooks and save outputs ---
	persistent_capture.remove()
	optimizer_dir.mkdir(parents=True, exist_ok=True)

	signal_arrays: dict[str, Any] = {
		"iteration": np.array(iterations),
		"unit_node_ids": np.array(node_ids),
	}
	for s_name, unit_data in signal_log.items():
		if not node_ids or not iterations:
			signal_arrays[s_name] = np.array([])
			continue
		# Check if this signal stores arrays (e.g. effective_lr) vs scalars
		first_val = unit_data[node_ids[0]][0] if unit_data[node_ids[0]] else None
		if isinstance(first_val, np.ndarray):
			# Array-valued: save per-unit as s_name__{safe} with shape (n_iters, n_vals)
			for nid in node_ids:
				safe = nid.replace(":", "__")
				try:
					signal_arrays[f"{s_name}__{safe}"] = np.stack(unit_data[nid])
				except (ValueError, TypeError):
					signal_arrays[f"{s_name}__{safe}"] = np.array(
						unit_data[nid], dtype=object
					)
		else:
			# Scalar-valued: keep matrix format (n_iters, n_units)
			matrix = np.column_stack(
				[np.array(unit_data[nid]) for nid in node_ids]
			)
			signal_arrays[s_name] = matrix
	for bk, series in shampoo_layer_block_signals.items():
		if not series:
			signal_arrays[bk] = np.array([])
		else:
			signal_arrays[bk] = np.stack(series)
	np.savez_compressed(str(optimizer_dir / "signals.npz"), **signal_arrays)

	neuron_collector.save(optimizer_dir / "neuron_timeseries.npz")

	metric_arrays = {}
	for k, v in metrics.items():
		try:
			metric_arrays[k] = np.array(v)
		except (ValueError, TypeError):
			metric_arrays[k] = np.array(v, dtype=object)
	metric_arrays["trial_names"] = np.array(trial_names_list, dtype=object)
	metric_arrays["trial_end_checkpoint_idxs"] = np.array(trial_end_checkpoint_idxs, dtype=np.int64)
	metric_arrays["trial_end_iterations"] = np.array(trial_end_iterations_list, dtype=np.int64)
	np.savez_compressed(str(optimizer_dir / "training_metrics.npz"), **metric_arrays)

	compute_post_processing(units, optimizer_dir)

"""Threshold-based MNIST pretraining (Phase A epochs + Phase B binary search)."""
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from compute_results.config_guard import build_pretrain_report, save_pretrain_report
from compute_results.constants import BS_LINEAR_BRACKET_THRESHOLD
from experiments.mnist.base import digit_indices, make_loader
from models import apply_model_weight_init, get_model
from optimizers import get_extractor


@dataclass
class PretrainRunConfig:
	"""Runtime parameters for one pretrain session (shared across seeds/optimizers)."""

	model_class: str
	model_config: dict[str, Any]
	activation: str
	he_init: int | str | float
	init_epsilon: float
	base_lr: float
	train_k_samples: int
	threshold_acc: float
	batch_size: int
	max_epochs: int
	dataset_seed: int
	save: bool
	out_dir: str
	opt_extra_kwargs: dict[str, Any]


def build_mnist_train_and_eval(
	train_k_samples: int,
	batch_size: int,
	*,
	dataset_seed: int,
	data_root: str = "./data",
) -> tuple[Subset, DataLoader, DataLoader]:
	"""Mirror `MNISTWrapper._ensure_datasets` (train stats, eval holdout, uniform K).

	Per-digit train pools are shuffled with *dataset_seed* before taking the first
	`K // 10` indices (after eval holdout).
	"""
	n_digits = 10
	raw_train = datasets.MNIST(
		root=data_root, train=True, download=True, transform=transforms.ToTensor(),
	)
	pixels = raw_train.data.float() / 255.0
	mean, std = pixels.mean().item(), pixels.std().item()
	tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])

	full_train = datasets.MNIST(
		root=data_root, train=True, download=True, transform=tfm,
	)
	test_ds = datasets.MNIST(
		root=data_root, train=False, download=True, transform=tfm,
	)

	all_digits = tuple(range(n_digits))
	train_by_digit_indices = digit_indices(full_train, all_digits)
	eval_indices_list: list[int] = []
	for d in all_digits:
		eval_indices_list.extend(train_by_digit_indices[d][-5:])
	eval_indices_set = frozenset(eval_indices_list)
	if train_k_samples < 1:
		raise ValueError(f"train_k_samples must be >= 1, got {train_k_samples}")
	if train_k_samples % n_digits != 0:
		raise ValueError(
			f"train_k_samples={train_k_samples} must be divisible by n_digits={n_digits} "
			"for a uniform per-digit train split"
		)
	k_per_digit = train_k_samples // n_digits
	gen = torch.Generator()
	gen.manual_seed(dataset_seed)
	train_indices: list[int] = []
	for d in all_digits:
		pool = [i for i in train_by_digit_indices[d] if i not in eval_indices_set]
		if len(pool) < k_per_digit:
			raise ValueError(
				f"Digit {d}: need {k_per_digit} train samples after eval holdout, "
				f"only {len(pool)} available"
			)
		perm = torch.randperm(len(pool), generator=gen).tolist()
		train_indices.extend(pool[i] for i in perm[:k_per_digit])
	assert len(train_indices) == train_k_samples
	train_ds = Subset(full_train, train_indices)
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

	test_by_digit = digit_indices(test_ds, all_digits)
	all_test_idx = sum(test_by_digit.values(), [])
	all_test = make_loader(test_ds, all_test_idx, batch_size=len(test_ds), shuffle=False)
	return train_ds, train_loader, all_test


def materialize_epoch_batches(
	train_ds: Subset,
	batch_size: int,
	epoch_idx: int,
	materialize_seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
	"""Fixed batch order for one epoch (replayed identically in Phase B)."""
	g = torch.Generator()
	g.manual_seed(materialize_seed + epoch_idx * 100_003)
	loader = DataLoader(
		train_ds, batch_size=batch_size, shuffle=True, generator=g,
	)
	return [(xb.clone(), yb.clone()) for xb, yb in loader]


def take_snapshot(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
	return {
		"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
		"optimizer": copy.deepcopy(optimizer.state_dict()),
		"torch_rng": torch.get_rng_state(),
		"cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
	}


def load_snapshot(
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	snap: dict,
	device: torch.device,
) -> None:
	model.load_state_dict({k: v.to(device) for k, v in snap["model"].items()})
	optimizer.load_state_dict(copy.deepcopy(snap["optimizer"]))

	torch.set_rng_state(snap["torch_rng"])
	if snap["cuda_rng"] is not None:
		torch.cuda.set_rng_state_all(snap["cuda_rng"])


@torch.no_grad()
def full_loader_loss_acc(
	model: torch.nn.Module,
	loader: DataLoader,
	device: torch.device,
) -> tuple[float, float]:
	model.eval()
	total_loss = 0.0
	correct = 0
	n = 0
	for xb, yb in loader:
		xb = xb.to(device)
		yb = yb.to(device)
		logits = model(xb)
		total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
		correct += (logits.argmax(dim=-1) == yb).sum().item()
		n += yb.numel()
	return total_loss / n, correct / n


def train_batches_range(
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
	batches: list[tuple[torch.Tensor, torch.Tensor]],
	start: int,
	end: int,
	device: torch.device,
) -> tuple[float, int]:
	"""Train `batches[start:end]` (end exclusive).

	Assumes *model* / *optimizer* / RNG already reflect `start` completed steps.
	"""
	model.train()
	running = 0.0
	seen = 0
	end = min(end, len(batches))
	for i in range(start, end):
		xb, yb = batches[i]
		xb = xb.to(device)
		yb = yb.to(device)
		optimizer.zero_grad(set_to_none=True)
		logits = model(xb)
		loss = criterion(logits, yb)
		loss.backward()
		optimizer.step()
		running += loss.item() * yb.size(0)
		seen += yb.size(0)
	return running / max(seen, 1), end - start


def train_first_n_batches(
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
	batches: list[tuple[torch.Tensor, torch.Tensor]],
	n_steps: int,
	device: torch.device,
	*,
	mode: str = "train",
	bracket_lo: int | None = None,
	threshold: float | None = None,
	eval_loader: DataLoader | None = None,
	log_prefix: str | None = None,
	warmup_done: bool = False,
) -> tuple[float, int] | tuple[float, int, float]:
	"""Train on the first *n_steps* batches."""
	model.train()
	running = 0.0
	seen = 0
	m = min(n_steps, len(batches))

	if mode == "train":
		for i in range(m):
			xb, yb = batches[i]
			xb = xb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad(set_to_none=True)
			logits = model(xb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()
			running += loss.item() * yb.size(0)
			seen += yb.size(0)
		return running / max(seen, 1), m

	if bracket_lo is None or threshold is None or eval_loader is None:
		raise ValueError("bracket_lo, threshold, and eval_loader are required for mode='linear_search'")

	warmup = 0 if warmup_done else min(bracket_lo, m)
	for i in range(warmup):
		xb, yb = batches[i]
		xb = xb.to(device)
		yb = yb.to(device)
		optimizer.zero_grad(set_to_none=True)
		logits = model(xb)
		loss = criterion(logits, yb)
		loss.backward()
		optimizer.step()
		running += loss.item() * yb.size(0)
		seen += yb.size(0)

	crossing_k = m
	crossing_acc = float("nan")
	for i in range(warmup, m):
		model.train()
		xb, yb = batches[i]
		xb = xb.to(device)
		yb = yb.to(device)
		optimizer.zero_grad(set_to_none=True)
		logits = model(xb)
		loss = criterion(logits, yb)
		loss.backward()
		optimizer.step()
		running += loss.item() * yb.size(0)
		seen += yb.size(0)
		_, acc = full_loader_loss_acc(model, eval_loader, device)
		if log_prefix is not None:
			print(f"{log_prefix} step {i + 1}, all_test_acc={acc:.6f}")
		if acc >= threshold:
			crossing_k = i + 1
			crossing_acc = acc
			break

	return running / max(seen, 1), crossing_k, crossing_acc


def binary_search_first_threshold_step(
	*,
	slug: str,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
	batches: list[tuple[torch.Tensor, torch.Tensor]],
	snapshot_epoch_start: dict,
	threshold: float,
	device: torch.device,
	eval_loader: DataLoader,
	verify_first: bool = False,
) -> int:
	"""Smallest k in 1..N with all_test acc >= threshold after k steps from snapshot."""
	n = len(batches)
	print(f"[{slug}] --- Phase B: binary search on crossing epoch ({n} steps) ---")

	if verify_first:
		load_snapshot(model, optimizer, snapshot_epoch_start, device)
		_, acc0 = full_loader_loss_acc(model, eval_loader, device)
		print(
			f"[{slug}] BS boundary g(0): eval after 0 steps, all_test_acc={acc0:.6f} "
			f"(expect < {threshold:.6f})"
		)
		if acc0 >= threshold:
			raise RuntimeError(
				f"[{slug}] g(0) already >= threshold; cannot bracket crossing inside this epoch."
			)

		load_snapshot(model, optimizer, snapshot_epoch_start, device)
		train_first_n_batches(model, optimizer, criterion, batches, n, device)
		_, acc_full = full_loader_loss_acc(model, eval_loader, device)
		print(
			f"[{slug}] BS boundary g({n}): eval after full epoch, all_test_acc={acc_full:.6f} "
			f"(expect >= {threshold:.6f})"
		)
		if acc_full < threshold:
			raise RuntimeError(
				f"[{slug}] g(N) below threshold after replay; batch order may not match Phase A."
			)

	lo, hi = 0, n
	probe = 0
	cached_step: int | None = None  # model state matches this many completed steps
	while lo + 1 < hi:
		if hi - lo < BS_LINEAR_BRACKET_THRESHOLD:
			print(
				f"[{slug}] BS bracket ({lo}, {hi}) has {hi - lo} steps "
				f"(< {BS_LINEAR_BRACKET_THRESHOLD}); switching to in-place linear scan from step {lo}"
			)
			if cached_step != lo:
				load_snapshot(model, optimizer, snapshot_epoch_start, device)
			_, crossing_k, crossing_acc = train_first_n_batches(
				model, optimizer, criterion, batches, hi, device,
				mode="linear_search",
				bracket_lo=lo,
				threshold=threshold,
				eval_loader=eval_loader,
				log_prefix=f"[{slug}] BS linear:",
				warmup_done=(cached_step == lo),
			)
			print(
				f"[{slug}] BS linear scan done: first crossing at step {crossing_k}, "
				f"all_test_acc={crossing_acc:.6f}"
			)
			lo, hi = crossing_k - 1, crossing_k
			break

		mid = (lo + hi) // 2
		if mid == lo:
			mid = lo + 1
		probe += 1
		print(
			f"[{slug}] BS probe #{probe}: bracket step indices ({lo}, {hi}) "
			f"≈ ({100 * lo / n:.1f}%, {100 * hi / n:.1f}%] of epoch → train/eval at mid={mid}"
		)
		if cached_step == lo:
			print(
				f"[{slug}] BS probe #{probe}: resume from step {lo} "
				f"(train {mid - lo} new steps, skip {lo} replayed steps)"
			)
			train_batches_range(model, optimizer, criterion, batches, lo, mid, device)
		else:
			load_snapshot(model, optimizer, snapshot_epoch_start, device)
			train_first_n_batches(model, optimizer, criterion, batches, mid, device)
		_, acc_mid = full_loader_loss_acc(model, eval_loader, device)
		print(f"[{slug}] BS probe #{probe}: after {mid} steps, all_test_acc={acc_mid:.6f}")
		if acc_mid >= threshold:
			hi = mid
			cached_step = None
			print(f"[{slug}] BS probe #{probe}: acc >= threshold → hi ← {hi} (search left / earlier steps)")
		else:
			lo = mid
			cached_step = mid
			print(f"[{slug}] BS probe #{probe}: acc < threshold → lo ← {lo} (search right / later steps)")

	print(
		f"[{slug}] BS finished: minimal k = {hi} with g({lo}) < {threshold:.4f} and g({hi}) >= {threshold:.4f}"
	)
	return hi


def run_training_until_threshold_with_refinement(
	*,
	slug: str,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
	train_ds: Subset,
	batch_size: int,
	materialize_seed: int,
	eval_loader: DataLoader,
	threshold_acc: float,
	max_epochs: int,
	device: torch.device,
) -> dict:
	"""Phase A: full epochs (fixed shuffle per epoch). Phase B: exact crossing step in last epoch."""
	train_loss_hist: list[float] = []
	test_loss_hist: list[float] = []
	iter_snapshots: list[int] = []
	global_step = 0
	reached = False
	crossing_epoch_idx: int | None = None
	crossing_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None
	snapshot_at_crossing_start: dict | None = None

	epoch_start_snapshots: list[dict] = [take_snapshot(model, optimizer)]

	print(f"[{slug}] --- Phase A: full epochs until all_test >= {threshold_acc} ---")

	for epoch_idx in range(max_epochs):
		snap_before = epoch_start_snapshots[epoch_idx]
		assert snap_before is not None
		batches = materialize_epoch_batches(
			train_ds, batch_size, epoch_idx, materialize_seed,
		)
		load_snapshot(model, optimizer, snap_before, device)

		running = 0.0
		seen = 0
		desc = f"[{slug}] Phase A epoch {epoch_idx + 1}/{max_epochs}"
		for xb, yb in tqdm(batches, desc=desc, total=len(batches)):
			xb = xb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad(set_to_none=True)
			logits = model(xb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()
			global_step += 1
			running += loss.item() * yb.size(0)
			seen += yb.size(0)

		train_loss = running / max(seen, 1)
		test_loss, test_acc = full_loader_loss_acc(model, eval_loader, device)
		train_loss_hist.append(train_loss)
		test_loss_hist.append(test_loss)
		iter_snapshots.append(global_step)

		epoch_start_snapshots.append(take_snapshot(model, optimizer))

		print(
			f"[{slug}] Phase A epoch {epoch_idx + 1:3d}  iters={global_step:5d}  "
			f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  all_test_acc={test_acc:.4f}"
		)

		if test_acc >= threshold_acc:
			reached = True
			crossing_epoch_idx = epoch_idx
			crossing_batches = batches
			snapshot_at_crossing_start = snap_before
			break

	exact_step = global_step
	final_train_loss = train_loss_hist[-1] if train_loss_hist else float("nan")
	final_test_loss = test_loss_hist[-1] if test_loss_hist else float("nan")
	final_test_acc = float("nan")
	step_in_crossing_epoch: int | None = None

	if reached and crossing_batches is not None and snapshot_at_crossing_start is not None:
		steps_before = global_step - len(crossing_batches)
		verify_first = any(ok in slug.lower() for ok in {"sgd", "adam", "adagrad"})
		print(f"verify_first: {verify_first} for {slug}")
		k = binary_search_first_threshold_step(
			slug=slug,
			model=model,
			optimizer=optimizer,
			criterion=criterion,
			batches=crossing_batches,
			snapshot_epoch_start=snapshot_at_crossing_start,
			threshold=threshold_acc,
			device=device,
			eval_loader=eval_loader,
			verify_first=verify_first,
		)
		load_snapshot(model, optimizer, snapshot_at_crossing_start, device)
		final_train_loss, _ = train_first_n_batches(
			model, optimizer, criterion, crossing_batches, k, device,
		)
		final_test_loss, final_test_acc = full_loader_loss_acc(model, eval_loader, device)
		exact_step = steps_before + k
		step_in_crossing_epoch = k
		iter_snapshots[-1] = exact_step
		train_loss_hist[-1] = final_train_loss
		test_loss_hist[-1] = final_test_loss
		print(
			f"[{slug}] Exact crossing: global optimizer step {exact_step} "
			f"(epoch {crossing_epoch_idx + 1} step {k}/{len(crossing_batches)}), "
			f"all_test_acc={final_test_acc:.6f}"
		)
	elif not reached:
		print(
			f"[{slug}] Phase A stopped at max_epochs={max_epochs} without reaching "
			f"threshold_acc={threshold_acc}."
		)

	print(f"[{slug}] Total optimizer iterations (reported): {exact_step}")

	return {
		"train": list(train_loss_hist),
		"test": list(test_loss_hist),
		"iters": list(iter_snapshots),
		"steps": exact_step,
		"reached": reached,
		"final_test_acc": final_test_acc,
		"final_train_loss": final_train_loss,
		"final_test_loss": final_test_loss,
		"phase_a_epochs": len(train_loss_hist),
		"crossing_epoch": (crossing_epoch_idx + 1) if reached and crossing_epoch_idx is not None else None,
		"step_in_crossing_epoch": step_in_crossing_epoch,
	}


def resolve_save_root(out_dir_template: str, *, slug: str, seed: int) -> Path:
	"""Expand `!OPT` and `!SD` placeholders in *out_dir_template*."""
	if "!OPT" not in out_dir_template:
		raise ValueError("out_dir must contain the placeholder '!OPT'.")
	if "!SD" in out_dir_template:
		path = out_dir_template.replace("!SD", str(seed))
	else:
		path = out_dir_template
	path = path.replace("!OPT", slug).replace("grafted_shampoo", "graft_shampoo")
	return Path(path).expanduser()


def save_pretrained_checkpoint(
	*,
	save_root: Path,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	run_cfg: PretrainRunConfig,
	slug: str,
	opt_name: str,
	opt_class: str,
	training_result: dict,
	seed: int,
	pretrain_config: dict[str, Any],
) -> None:
	save_root.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), save_root / "model.pt")
	torch.save(optimizer.state_dict(), save_root / "optimizer.pt")
	report = build_pretrain_report(
		pretrain_config=pretrain_config,
		training_result=training_result,
		model_seed=seed,
		optimizer_name=opt_name,
		optimizer_class=opt_class,
		optimizer_id=slug,
	)
	save_pretrain_report(save_root, report)
	print(
		f"[seed={seed}][{slug}] Saved model + optimizer state at exact crossing "
		f"under {save_root.resolve()}"
	)


def train_optimizer_for_seed(
	seed: int,
	*,
	run_cfg: PretrainRunConfig,
	dataset_seed: int,
	train_ds: Subset,
	eval_loader: DataLoader,
	opt_name: str,
	opt_class: str,
	opt_kwargs: dict[str, Any],
	device: torch.device,
) -> dict:
	"""Train one (optimizer, model_seed) pair; optionally save checkpoint."""
	criterion = torch.nn.CrossEntropyLoss()
	extractor = get_extractor(opt_class, **opt_kwargs)
	slug = extractor.optimizer_id()
	materialize_seed = dataset_seed + sum(ord(c) for c in slug) % 1_000_000

	torch.manual_seed(seed)
	model = get_model(run_cfg.model_class, **run_cfg.model_config).to(device)
	apply_model_weight_init(
		model,
		he_init=run_cfg.he_init,
		init_epsilon=run_cfg.init_epsilon,
		activation=run_cfg.activation,
	)
	optimizer = extractor.create_optimizer(model.parameters())

	print(
		f"\n=== model_seed={seed}  dataset_seed={dataset_seed}  "
		f"optimizer {opt_name!r} -> {opt_class} (optimizer_id={slug}) ==="
	)

	training_result = run_training_until_threshold_with_refinement(
		slug=slug,
		model=model,
		optimizer=optimizer,
		criterion=criterion,
		train_ds=train_ds,
		batch_size=run_cfg.batch_size,
		materialize_seed=materialize_seed,
		eval_loader=eval_loader,
		threshold_acc=run_cfg.threshold_acc,
		max_epochs=run_cfg.max_epochs,
		device=device,
	)

	result = {
		"label": opt_name.strip(),
		"extractor": opt_class,
		"slug": slug,
		"iters": training_result["iters"],
		"train": training_result["train"],
		"test": training_result["test"],
		"steps": training_result["steps"],
		"reached": training_result["reached"],
		"final_test_acc": training_result["final_test_acc"],
		"final_train_loss": training_result["final_train_loss"],
		"final_test_loss": training_result["final_test_loss"],
		"phase_a_epochs": training_result["phase_a_epochs"],
		"crossing_epoch": training_result["crossing_epoch"],
		"step_in_crossing_epoch": training_result["step_in_crossing_epoch"],
		"model": model,
		"optimizer": optimizer,
	}

	if run_cfg.save:
		if not training_result["reached"]:
			print(f"[seed={seed}][{slug}] SAVE skipped: threshold not reached.")
		else:
			result["save_root"] = resolve_save_root(
				run_cfg.out_dir, slug=slug, seed=seed,
			)

	return result

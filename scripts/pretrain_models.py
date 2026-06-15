#!/usr/bin/env python3
"""Pretrain registered MNIST models until full-test accuracy crosses a threshold.

Results are written under a user-specified directory template (`!OPT` / `!SD`)
as `model.pt`, `optimizer.pt`, `report.json`, and `pretrain_config.json`.

Cluster-oriented paths:
	--output-dir  Checkpoint output template with !OPT / !SD (alias: --out-dir)
	--env-file    Dotenv with WANDB_* keys; enables TensorBoard logging synced to W&B

Usage examples
--------------
From a simulation or pretrain config file:

	uv run pretrain-models --config input_configs/cat2_sequence_pretrain_control_multi.json \\
		--output-dir '/scratch/pretrain/!OPT/!SD/'

With explicit arguments:

	uv run pretrain-models --model DNN5Hidden64 --optimizer adam \\
		--output-dir '/scratch/pretrain/!OPT/!SD/' \\
		--train-k-samples 1000 --multi_seeds 6 --env-file /path/to/.env

Full notebook seed list (bare `--multi_seeds`):

	uv run pretrain-models --config path/to/config.json --multi_seeds
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

import optimizers  # noqa: F401 - register extractors

from compute_results.config_guard import (
	ConfigConflictError,
	build_pretrain_config,
	expand_registry_selection,
	guard_pretrain_output,
	parse_pretrain_inputs,
	save_pretrain_config,
	validate_pretrain_config_constraints,
)
from compute_results.constants import (
	INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS,
	MULTI_SEEDS_USE_DEFAULT,
)
from compute_results.pretrain import (
	PretrainRunConfig,
	build_mnist_train_and_eval,
	resolve_save_root,
	save_pretrained_checkpoint,
	train_optimizer_for_seed,
)
from experiments.base import get_registered_experiment_class
from models import list_models, parse_he_init
from optimizers import get_extractor, list_extractors
from scripts.utils.cluster_utils import (
	add_io_arguments,
	train_logger,
)
from scripts.run_simulation import _resolve_optimizer


def _pretrain_skip_reason(raw: dict) -> str | None:
	"""Return a human-readable skip reason, or None if pretrain should run."""
	experiment_class = raw.get("experiment_class")
	if not experiment_class:
		return None
	cls = get_registered_experiment_class(str(experiment_class))
	if getattr(cls, "has_pretrain", False):
		return None
	return f"{experiment_class} does not require pretrain checkpoints (has_pretrain=False)"


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Pretrain models until MNIST test accuracy crosses a threshold."
	)
	p.add_argument("--config", type=Path, help="Path to a JSON config file (sim or pretrain).")
	p.add_argument("--model", help="Model class name (required without --config).")
	p.add_argument(
		"--optimizer",
		help=(
			"Optimizer shorthand(s) or extractor class name(s), comma-separated. "
			"Use 'all' to include every registered extractor."
		),
	)
	p.add_argument(
		"--expert",
		action="store_true",
		help="Expert preset: 20000 samples, threshold 0.95, expert_ out-dir prefix.",
	)
	p.add_argument("--train-k-samples", type=int, default=None, metavar="K")
	p.add_argument("--threshold-acc", type=float, default=None)
	p.add_argument("--max-epochs", type=int, default=None)
	p.add_argument(
		"--dataset-seed",
		type=int,
		default=None,
		help="Train-index selection and epoch batch-order seed.",
	)
	p.add_argument(
		"--multi_seeds",
		nargs="?",
		const=MULTI_SEEDS_USE_DEFAULT,
		default=None,
		metavar="SEEDS",
		help=(
			"Model weight-init seeds. Bare flag uses the notebook default list; "
			"with CSV overrides config seed(s)."
		),
	)
	p.add_argument("--base-lr", type=float, default=None)
	p.add_argument("--batch-size", type=int, default=None)
	p.add_argument("--activation", default=None)
	p.add_argument(
		"--he-init",
		type=str,
		default=None,
		metavar="K",
		help="He init layer count / 'all' (same as run-simulation).",
	)
	p.add_argument("--init-epsilon", type=float, default=None, metavar="VAR")
	p.add_argument(
		"--out-dir",
		default=None,
		help="Output template with !OPT and optionally !SD placeholders (alias: --output-dir).",
	)
	add_io_arguments(
		p,
		output_help=(
			"Output template with !OPT and optionally !SD placeholders (overrides --out-dir)."
		),
	)
	p.add_argument("--save", action="store_true", help="Force saving checkpoints.")
	p.add_argument("--no-save", action="store_true", help="Skip writing checkpoints.")
	p.add_argument(
		"--force",
		action="store_true",
		help="Re-run even when matching outputs exist (config mismatch still errors).",
	)
	p.add_argument(
		"--device",
		default=None,
		help="Training device (default: cuda when available, else cpu). Overrides JSON config.",
	)
	p.add_argument(
		"--shampoo-preconditioner-epsilon",
		type=float,
		default=None,
		metavar="EPS",
	)
	return p.parse_args()


def _validate_names(model_class: str, optimizers_list: list[str]) -> None:
	valid_mod = set(list_models())
	valid_opt = set(INITIAL_MODEL_OPTIMIZER_SHORTHAND_TO_CLASS) | set(list_extractors())
	if model_class not in valid_mod:
		print(
			f"ERROR: Unknown model {model_class!r} (valid: {sorted(valid_mod)})",
			file=sys.stderr,
		)
		sys.exit(1)
	invalid_opt = [o for o in optimizers_list if o not in valid_opt]
	if invalid_opt:
		print(
			f"ERROR: Unknown optimizer(s) {invalid_opt} (valid: {sorted(valid_opt)})",
			file=sys.stderr,
		)
		sys.exit(1)


def main() -> int:
	print("Starting pretrain_models...")
	args = _parse_args()
	print(f"Parsed args")

	if args.config is not None:
		raw = json.loads(args.config.read_text())
		skip_reason = _pretrain_skip_reason(raw)
		if skip_reason is not None:
			print(f"Skipping pretrain: {skip_reason}")
			return 0
	else:
		if not (args.model and args.optimizer):
			print(
				"ERROR: Provide --config or both --model and --optimizer.",
				file=sys.stderr,
			)
			sys.exit(1)
		raw = {}

	try:
		params = parse_pretrain_inputs(raw, cli=args)
	except ValueError as e:
		print(f"ERROR: {e}", file=sys.stderr)
		sys.exit(1)

	if args.output_dir is not None:
		params["out_dir"] = str(args.output_dir.expanduser())

	try:
		optimizer_names = expand_registry_selection(
			params["optimizer_names"],
			available=list_extractors(),
		)
	except ValueError as e:
		print(f"ERROR: {e}", file=sys.stderr)
		sys.exit(1)

	params["optimizer_names"] = optimizer_names
	_validate_names(params["model_class"], optimizer_names)

	pretrain_config = build_pretrain_config(
		model_class=params["model_class"],
		model_config=params["model_config"],
		activation=params["activation"],
		optimizer_names=optimizer_names,
		base_lr=params["base_lr"],
		batch_size=params["batch_size"],
		he_init=params["he_init"],
		init_epsilon=params["init_epsilon"],
		dataset_seed=params["dataset_seed"],
		model_seeds=params["model_seeds"],
		train_k_samples=params["train_k_samples"],
		threshold_acc=params["threshold_acc"],
		max_epochs=params["max_epochs"],
		data_root=params["data_root"],
		expert=params["expert"],
		opt_extra_kwargs=params["opt_extra_kwargs"],
	)

	try:
		validate_pretrain_config_constraints(
			pretrain_config,
			out_dir=params["out_dir"],
			save=params["save"],
			model_seeds=params["model_seeds"],
		)
	except (ValueError, TypeError) as e:
		print(f"ERROR: {e}", file=sys.stderr)
		sys.exit(1)

	try:
		parse_he_init(params["he_init"])
	except (TypeError, ValueError) as e:
		print(f"ERROR: Invalid he_init: {e}", file=sys.stderr)
		sys.exit(1)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)

	device = torch.device(params["device"])
	print(f"device={device}")
	run_cfg = PretrainRunConfig(
		model_class=params["model_class"],
		model_config=params["model_config"],
		activation=params["activation"],
		he_init=params["he_init"],
		init_epsilon=params["init_epsilon"],
		base_lr=params["base_lr"],
		train_k_samples=params["train_k_samples"],
		threshold_acc=params["threshold_acc"],
		batch_size=params["batch_size"],
		max_epochs=params["max_epochs"],
		dataset_seed=params["dataset_seed"],
		save=params["save"],
		out_dir=params["out_dir"],
		opt_extra_kwargs=params["opt_extra_kwargs"],
	)

	train_ds, _, all_test = build_mnist_train_and_eval(
		params["train_k_samples"],
		params["batch_size"],
		dataset_seed=params["dataset_seed"],
	)
	print(
		f"dataset_seed={params['dataset_seed']}  "
		f"Train size K={params['train_k_samples']} "
		f"({params['train_k_samples'] // 10} per digit, uniform), "
		f"batches/epoch: {len(train_ds) // params['batch_size']}, "
		f"model_seeds={params['model_seeds']}"
	)

	for seed_idx, seed in enumerate(params["model_seeds"]):
		print(
			f"\n########## model_seed {seed} "
			f"({seed_idx + 1}/{len(params['model_seeds'])}) ##########"
		)
		for opt_name in optimizer_names:
			opt_class, opt_kwargs = _resolve_optimizer(
				opt_name, params["base_lr"], **params["opt_extra_kwargs"]
			)
			slug = get_extractor(opt_class, **opt_kwargs).optimizer_id()
			save_root = resolve_save_root(params["out_dir"], slug=slug, seed=seed)

			if params["save"]:
				try:
					if guard_pretrain_output(
						save_root,
						pretrain_config,
						force=params["force"],
					):
						print(
							f"Skipping existing output (use --force to re-run): {save_root}",
							file=sys.stderr,
						)
						continue
				except ConfigConflictError as e:
					print(f"CONFIG CONFLICT:\n{e}", file=sys.stderr)
					sys.exit(1)

			try:
				log_root = resolve_save_root(params["out_dir"], slug=slug, seed=seed)
				run_name = f"pretrain/{params['model_class']}/{slug}/seed_{seed}"
				with train_logger(
					env_file=args.env_file,
					run_name=run_name,
					log_dir=log_root / "tensorboard",
					config={
						"model": params["model_class"],
						"optimizer": opt_name,
						"optimizer_id": slug,
						"model_seed": seed,
						"dataset_seed": params["dataset_seed"],
						"train_k_samples": params["train_k_samples"],
						"threshold_acc": params["threshold_acc"],
					},
				) as writer:
					result = train_optimizer_for_seed(
						seed,
						run_cfg=run_cfg,
						dataset_seed=params["dataset_seed"],
						train_ds=train_ds,
						eval_loader=all_test,
						opt_name=opt_name,
						opt_class=opt_class,
						opt_kwargs=opt_kwargs,
						device=device,
						train_logger=writer,
					)
			except Exception as e:
				print(f"ERROR: seed={seed} optimizer={opt_name}: {e}", file=sys.stderr)
				raise

			if run_cfg.save and "save_root" in result:
				save_pretrained_checkpoint(
					save_root=save_root,
					model=result["model"],
					optimizer=result["optimizer"],
					run_cfg=run_cfg,
					slug=result["slug"],
					opt_name=opt_name,
					opt_class=opt_class,
					training_result=result,
					seed=seed,
					pretrain_config=pretrain_config,
				)
				save_pretrain_config(save_root, pretrain_config)

	print("Done.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

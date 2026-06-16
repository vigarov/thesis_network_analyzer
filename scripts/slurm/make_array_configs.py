#!/usr/bin/env python3
"""Expand base configs into one config.json per (seed, base_lr) for Slurm array jobs.

`run_simulation` sweeps seeds x learning rates internally (via --multi_seed /
--multi_lr) by building an `itertools.product`. On a cluster it is easier to map
each point of that product to its own array task. This script materializes that
product up-front: for every `*.json` under `--input-dir` and every
`(seed, base_lr)` combination, it writes a config with `seed` and `base_lr`
baked in to `--output-dir`.

Each output filename is prefixed with the requested `lr`/`seed` so the
combinations never collide, e.g. `lr0.01_seed6_<original-name>.json`. A sweep
dimension without its flag falls back to each config's own value (so passing only
--multi_lr keeps each config's seed, and vice versa).

Usage
-----
	uv run python scripts/slurm/make_array_configs.py \\
		--input-dir input_configs \\
		--output-dir input_configs_array \\
		--multi_seed 6,7,14 \\
		--multi_lr 1e-3,1e-2,1e-1

Bare flags use the shared default sweep lists:

	uv run python scripts/slurm/make_array_configs.py \\
		--input-dir input_configs --output-dir input_configs_array \\
		--multi_seed --multi_lr

The number of generated configs (printed at the end) is the array size: submit the
simulation with `sbatch --array=0-(N-1)` pointed at `--output-dir`.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

from compute_results.constants import (
	DEFAULT_MULTI_LRS,
	DEFAULT_MULTI_SEEDS,
	MULTI_LRS_USE_DEFAULT,
	MULTI_SEEDS_USE_DEFAULT,
)

# Match run_simulation's fallbacks when a config omits seed / base_lr.
DEFAULT_SEED = 3003
DEFAULT_BASE_LR = 1e-3

# Sweep keys are CLI-only for run_simulation; strip them from generated configs.
_SWEEP_KEYS = ("multi_seed", "multi_seeds", "multi_lr", "multi_lrs")


def _parse_csv(value: str) -> list[str]:
	return [p.strip() for p in value.split(",") if p.strip()]


def _resolve_seeds(cli_multi_seed: str | None, default_seed: int) -> list[int]:
	"""Seeds to sweep, mirroring run_simulation's --multi_seed semantics."""
	if cli_multi_seed is None:
		return [int(default_seed)]
	if cli_multi_seed == MULTI_SEEDS_USE_DEFAULT:
		return list(DEFAULT_MULTI_SEEDS)
	return [int(p) for p in _parse_csv(cli_multi_seed)]


def _resolve_lrs(cli_multi_lr: str | None, default_lr: float) -> list[float]:
	"""Base learning rates to sweep, mirroring run_simulation's --multi_lr semantics."""
	if cli_multi_lr is None:
		return [float(default_lr)]
	if cli_multi_lr == MULTI_LRS_USE_DEFAULT:
		return list(DEFAULT_MULTI_LRS)
	return [float(p) for p in _parse_csv(cli_multi_lr)]


def _fmt_lr(lr: float) -> str:
	"""Filename-friendly lr token matching the optimizer slug (e.g. 0.01, 1e-05)."""
	return str(float(lr))


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description=(
			"Expand base configs into one config.json per (seed, base_lr) "
			"combination for Slurm array jobs."
		)
	)
	p.add_argument(
		"--input-dir",
		type=Path,
		required=True,
		help="Directory of base *.json configs to expand.",
	)
	p.add_argument(
		"--output-dir",
		type=Path,
		required=True,
		help="Directory to write the expanded per-(seed, lr) configs into.",
	)
	p.add_argument(
		"--multi_seed",
		"--multi_seeds",
		dest="multi_seed",
		nargs="?",
		const=MULTI_SEEDS_USE_DEFAULT,
		default=None,
		metavar="SEEDS",
		help=(
			"Seeds to sweep. Bare flag uses the notebook default list; with a CSV "
			"(e.g. 6,7,14) overrides. Omit to keep each config's own seed."
		),
	)
	p.add_argument(
		"--multi_lr",
		"--multi_lrs",
		dest="multi_lr",
		nargs="?",
		const=MULTI_LRS_USE_DEFAULT,
		default=None,
		metavar="LRS",
		help=(
			"Base learning rates to sweep. Bare flag uses the default sweep list; "
			"with a CSV (e.g. 1e-3,1e-2,1e-1) overrides. Omit to keep each config's "
			"own base_lr."
		),
	)
	p.add_argument(
		"--glob",
		default="*.json",
		help="Glob (within --input-dir) selecting base configs. Default: *.json.",
	)
	p.add_argument(
		"--force",
		action="store_true",
		help="Overwrite existing output configs instead of erroring on collision.",
	)
	return p.parse_args()


def main() -> int:
	args = _parse_args()

	input_dir = args.input_dir.expanduser()
	output_dir = args.output_dir.expanduser()

	if not input_dir.is_dir():
		print(f"ERROR: --input-dir is not a directory: {input_dir}", file=sys.stderr)
		return 1

	base_configs = sorted(input_dir.glob(args.glob))
	if not base_configs:
		print(
			f"ERROR: no configs matching {args.glob!r} under {input_dir}",
			file=sys.stderr,
		)
		return 1

	output_dir.mkdir(parents=True, exist_ok=True)

	written: list[Path] = []
	for cfg_path in base_configs:
		try:
			raw = json.loads(cfg_path.read_text())
		except json.JSONDecodeError as e:
			print(f"ERROR: {cfg_path}: invalid JSON: {e}", file=sys.stderr)
			return 1

		seeds = _resolve_seeds(args.multi_seed, raw.get("seed", DEFAULT_SEED))
		lrs = _resolve_lrs(args.multi_lr, raw.get("base_lr", DEFAULT_BASE_LR))

		for seed, lr in itertools.product(seeds, lrs):
			new_config = {k: v for k, v in raw.items() if k not in _SWEEP_KEYS}
			new_config["seed"] = int(seed)
			new_config["base_lr"] = float(lr)

			name = f"lr{_fmt_lr(lr)}_seed{int(seed)}_{cfg_path.stem}.json"
			out_path = output_dir / name
			if out_path.exists() and not args.force:
				print(
					f"ERROR: output already exists (use --force to overwrite): {out_path}",
					file=sys.stderr,
				)
				return 1
			out_path.write_text(json.dumps(new_config, indent=2) + "\n")
			written.append(out_path)

	print(
		f"Wrote {len(written)} config(s) from {len(base_configs)} base config(s) "
		f"to {output_dir}"
	)
	print(
		f"Submit the array with: sbatch --array=0-{len(written) - 1} "
		f"(point the simulation at {output_dir})."
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

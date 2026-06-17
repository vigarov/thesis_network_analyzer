"""Periodically sync wandb offline runs.

Started from the login node via `uv run python scripts/slurm/wandb_syncher.py`.
Uses `.synch` in the repo root as a lock file and for communicatioj when array
end (`done:<task_id>` lines appended by worker scripts).

Usage
-----
	uv run python scripts/slurm/wandb_syncher.py \\
		--synch-file /path/to/repo/.synch \\
		--interval 10 \\
		--array-count 8 \\
		--root /path/to/results
"""
import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_SHUTDOWN = False


def _handle_signal(signum: int, _frame: object) -> None:
	global _SHUTDOWN
	_SHUTDOWN = True


def _read_done_tasks(synch_file: Path) -> set[int]:
	done: set[int] = set()
	try:
		text = synch_file.read_text()
	except OSError:
		return done
	for line in text.splitlines():
		line = line.strip()
		if line.startswith("done:"):
			try:
				done.add(int(line.split(":", 1)[1]))
			except ValueError:
				continue
	return done


def _find_offline_runs(root: Path) -> list[Path]:
	if not root.is_dir():
		return []
	return [path for path in root.rglob("offline-run-*") if path.is_dir()]


def _parent_experiment_dir(path: Path) -> Path | None:
	for parent in path.parents:
		name = parent.name
		if name.startswith("cat1") or name.startswith("cat2"):
			return parent
	return None


def _optimizer_count_for_experiment(exp_dir: Path) -> int | None:
	counts: list[int] = []
	for optimizers_dir in exp_dir.glob("*/dnn_5x64/optimizers"):
		if not optimizers_dir.is_dir():
			continue
		counts.append(
			sum(1 for child in optimizers_dir.iterdir() if child.is_dir())
		)
	return max(counts) if counts else None


def _init_finished_experiments(root: Path) -> set[Path]:
	"""Mark experiment dirs that already have the full optimizer set.

	When offline runs exist at startup, infer the expected optimizer count as
	the maximum across cat1/cat2 experiment dirs, then treat any experiment
	that already has that many ``dnn_5x64/optimizers`` subdirs as finished
	(sync skipped; no ``done:`` line is written).
	"""
	offline_runs = _find_offline_runs(root)
	if not offline_runs:
		return set()

	exp_dirs: set[Path] = set()
	for run_dir in offline_runs:
		parent = _parent_experiment_dir(run_dir)
		if parent is not None:
			exp_dirs.add(parent.resolve())

	if not exp_dirs:
		return set()

	counts: dict[Path, int] = {}
	for exp_dir in exp_dirs:
		n = _optimizer_count_for_experiment(exp_dir)
		if n is not None:
			counts[exp_dir] = n

	if not counts:
		return set()

	ground_truth = max(counts.values())
	finished = {exp for exp, n in counts.items() if n == ground_truth}
	if finished:
		print(
			f"Init: expected {ground_truth} optimizer(s); "
			f"treating {len(finished)} experiment(s) as already finished",
			flush=True,
		)
		for exp in sorted(finished, key=lambda p: p.name):
			print(f"  skip finished experiment: {exp.name}", flush=True)
	return finished


def _should_sync_run(run_dir: Path, finished_experiments: set[Path]) -> bool:
	if not finished_experiments:
		return True
	parent = _parent_experiment_dir(run_dir)
	if parent is None:
		return True
	return parent.resolve() not in finished_experiments


def _wandb_sync(run_dir: Path) -> None:
	print(f"wandb sync: {run_dir}", flush=True)
	subprocess.run(
		["wandb", "sync", str(run_dir)],
		check=False,
	)


def _acquire_synch_file(synch_file: Path) -> None:
	synch_file.parent.mkdir(parents=True, exist_ok=True)
	try:
		fd = os.open(synch_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
	except FileExistsError:
		print(
			f"Syncher already running (lock file exists): {synch_file}",
			file=sys.stderr,
		)
		sys.exit(1)
	try:
		os.write(fd, f"pid:{os.getpid()}\n".encode())
	finally:
		os.close(fd)


def _release_synch_file(synch_file: Path) -> None:
	try:
		synch_file.unlink()
	except OSError:
		pass


def main() -> int:
	parser = argparse.ArgumentParser(description="Periodic wandb offline-run sync")
	parser.add_argument(
		"--synch-file",
		type=Path,
		required=True,
		help="Lock / status file in the repo root",
	)
	parser.add_argument(
		"--interval",
		type=float,
		default=10.0,
		help="Seconds between sync passes",
	)
	parser.add_argument(
		"--array-count",
		type=int,
		default=0,
		help="Exit once this many unique array tasks report done (0 = run until SIGTERM)",
	)
	parser.add_argument(
		"--root",
		type=Path,
		required=True,
		help="Directory tree to search for offline-run-*",
	)
	args = parser.parse_args()

	root = args.root.resolve()
	_acquire_synch_file(args.synch_file.resolve())
	atexit.register(_release_synch_file, args.synch_file.resolve())

	signal.signal(signal.SIGTERM, _handle_signal)
	signal.signal(signal.SIGINT, _handle_signal)

	print(
		f"Started wandb syncher (interval={args.interval}s, "
		f"array_count={args.array_count}, pid={os.getpid()})",
		flush=True,
	)

	finished_experiments = _init_finished_experiments(root)

	while not _SHUTDOWN:
		for run_dir in _find_offline_runs(root):
			if _SHUTDOWN:
				break
			if not _should_sync_run(run_dir, finished_experiments):
				continue
			_wandb_sync(run_dir)

		if args.array_count > 0:
			done = _read_done_tasks(args.synch_file.resolve())
			if len(done) >= args.array_count:
				print(
					f"All {args.array_count} array tasks reported done; exiting",
					flush=True,
				)
				break

		end = time.time() + args.interval
		while not _SHUTDOWN and time.time() < end:
			time.sleep(0.5)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

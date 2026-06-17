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

	while not _SHUTDOWN:
		for run_dir in _find_offline_runs(root):
			if _SHUTDOWN:
				break
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

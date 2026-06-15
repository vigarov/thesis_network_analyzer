import os
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter


def add_output_dir_argument(
	parser: ArgumentParser,
	*,
	help: str,
	required: bool = False,
) -> None:
	"""Register `--output-dir` on *parser*."""
	parser.add_argument(
		"--output-dir",
		type=Path,
		required=required,
		default=None,
		help=help,
	)


def add_io_arguments(
	parser: ArgumentParser,
	*,
	output_help: str,
	require_output: bool = False,
) -> None:
	"""Register `--output-dir` and `--env-file` for cluster training runs."""
	add_output_dir_argument(parser, help=output_help, required=require_output)
	parser.add_argument(
		"--env-file",
		type=Path,
		default=None,
		help=(
			"Dotenv file with WANDB_* keys. When set, training logs scalars to "
			"TensorBoard and W&B syncs them for remote monitoring."
		),
	)


@contextmanager
def train_logger(
	*,
	env_file: Path | None,
	run_name: str,
	log_dir: Path,
	config: dict[str, Any] | None = None,
) -> Iterator[SummaryWriter | None]:
	"""Context manager: optional W&B + TensorBoard scalar logging."""
	if env_file is None:
		yield None
		return

	env_path = env_file.expanduser()
	if not env_path.is_file():
		raise FileNotFoundError(f"env file not found: {env_path}")
	load_dotenv(env_path, override=False)
	log_dir.mkdir(parents=True, exist_ok=True)

	import wandb

	wandb.init(
		project=os.environ.get("WANDB_PROJECT", "network-analyzer"),
		entity=os.environ.get("WANDB_ENTITY") or None,
		name=run_name,
		config=config or {},
		sync_tensorboard=True,
		dir=str(log_dir.parent),
	)
	writer = SummaryWriter(log_dir=str(log_dir))
	try:
		yield writer
	finally:
		writer.close()
		wandb.finish()

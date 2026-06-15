#!/usr/bin/env bash
# Usage: bash scripts/slurm/pretrain.sh [--auto-sync [SECONDS]]

#SBATCH --job-name=net-pretrain-all
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --partition=gpu_p100_16gb,gpu_rtx8000_48gb,gpu_v100_32gb,gpu_a100_40gb,gpu_a100_80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

if [[ -z "${SLURM_JOB_ID:-}" && "${BASH_SOURCE[0]}" == "${0}" ]]; then
	set -euo pipefail
	# shellcheck source=common.sh
	source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" # Load the common functions
	_run_submit_wrapper "${BASH_SOURCE[0]}" "$@" # calls `sbatch` and exits
fi
# else (coming from _run_submit_wrapper):
set -euo pipefail
# Slurm runs a copy under /var/spool/slurmd/...; source via --chdir=PROJECT_ROOT from common.sh.
source scripts/slurm/common.sh
require_array_task_id

CONFIG="$(config_for_array_task "${SLURM_ARRAY_TASK_ID}")"
echo "pretrain config=${CONFIG}"

run_uv pretrain-models \
	--config "${CONFIG}" \
	--output-dir "${PRETRAIN_OUTPUT_DIR}" \
	--env-file "${ENV_FILE}"

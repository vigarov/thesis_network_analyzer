#!/usr/bin/env bash
# Submit: sbatch scripts/slurm/simulation.sh
# Or (login node, optional wandb sync loop): bash scripts/slurm/simulation.sh [--auto-sync [SECONDS]]
#
# One array task per file in input_configs/*.json
# Adjust #SBATCH --array to 0-(N-1) where N = number of JSON configs (currently 8 → 0-7).

#SBATCH --job-name=net-simulation
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G


# when run directly (not submitted through sbatch)
if [[ -z "${SLURM_JOB_ID:-}" && "${BASH_SOURCE[0]}" == "${0}" ]]; then
	set -euo pipefail
	# shellcheck source=common.sh
	source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" # Load the common functions
	_run_submit_wrapper "${BASH_SOURCE[0]}" "$@" # calls `sbatch` and exits
fi
# else (coming from _run_submit_wrapper):
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" # Load the common functions
require_array_task_id

CONFIG="$(config_for_array_task "${SLURM_ARRAY_TASK_ID}")"
echo "simulation config=${CONFIG}"

run_uv compute \
	--config "${CONFIG}" \
	--output-dir "${RESULTS_DIR}" \
	--env-file "${ENV_FILE}"

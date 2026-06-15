#!/usr/bin/env bash
# Submit: sbatch scripts/slurm/analyse.sh
# Or (login node, optional wandb sync loop): bash scripts/slurm/analyse.sh [--auto-sync [SECONDS]]
#
# Scans all simulation results under RESULTS_DIR (single job, not an array).

#SBATCH --job-name=net-analyse
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G

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

ANALYSE_ARGS=(
	--input-dir "${RESULTS_DIR}"
	--output-dir "${ANALYSIS_OUTPUT_DIR}"
)
if [[ -n "${EXPERT_MODEL_DIR:-}" ]]; then
	ANALYSE_ARGS+=(--expert-model-dir "${EXPERT_MODEL_DIR}")
fi

echo "analyse input=${RESULTS_DIR} output=${ANALYSIS_OUTPUT_DIR}"

run_uv analyse "${ANALYSE_ARGS[@]}"

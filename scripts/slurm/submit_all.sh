#!/usr/bin/env bash
# Submit the full pipeline with job dependencies:
#   pretrain (array) → simulation (array) → analyse
#
# Usage (from repo root, after filling .env):
#   bash scripts/slurm/submit_all.sh [--auto-sync [SECONDS]]
#
# Slurm log paths (--output / --error) and --array are passed at submit time from common.sh
# using SLURM_LOG_DIR and the number of input_configs/*.json files.
#
# With --auto-sync, a background loop on the login node periodically runs
# `wandb sync` on offline runs under RESULTS_DIR / pretrain output / PROJECT_ROOT.
# The loop stops (with a final sync) when the pipeline succeeds or fails.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" # Load the common functions

parse_submit_args "$@"

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_rc=0

_submit_pipeline() {
	local pretrain_id sim_id analyse_id

	pretrain_id="$(_sbatch_with_logs "${_SLURM_DIR}/pretrain.sh")"
	echo "Submitted pretrain array: ${pretrain_id} (logs under ${SLURM_LOG_DIR})"
	_wait_for_slurm_job "${pretrain_id}" || return 1

	sim_id="$(_sbatch_with_logs "${_SLURM_DIR}/simulation.sh" --dependency="afterok:${pretrain_id}")"
	echo "Submitted simulation array: ${sim_id} (after pretrain, logs under ${SLURM_LOG_DIR})"
	_wait_for_slurm_job "${sim_id}" || return 1

	analyse_id="$(_sbatch_with_logs "${_SLURM_DIR}/analyse.sh" --dependency="afterok:${sim_id}")"
	echo "Submitted analyse: ${analyse_id} (after simulation, logs under ${SLURM_LOG_DIR})"
	_wait_for_slurm_job "${analyse_id}" || return 1

	echo "Pipeline completed successfully."
	return 0
}

_with_optional_auto_sync _submit_pipeline || _rc=$?
exit "${_rc}"

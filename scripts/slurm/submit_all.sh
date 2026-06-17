#!/usr/bin/env bash
# Submit the full pipeline with job dependencies:
#   pretrain (array) → simulation (array) → analyse
#
# Usage (from repo root, after filling .env):
#   bash scripts/slurm/submit_all.sh [--no-auto-sync] [--auto-sync [SECONDS]]
#
# Slurm log paths (--output / --error) and --array are passed at submit time from common.sh
# using SLURM_LOG_DIR and the number of input_configs/*.json files.
#
# By default, a background wandb syncher on the login node periodically runs
# `wandb sync` on offline runs for the active pipeline stage (pretrain or results).
# Pass --no-auto-sync to disable, or --auto-sync [SECONDS] to change the interval.
# The syncher restarts between stages and stops when each stage completes.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" # Load the common functions

parse_submit_args "$@"

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_rc=0

_submit_pipeline() {
	_run_slurm_stage_with_sync pretrain "${_SLURM_DIR}/pretrain.sh" || return 1

	_run_slurm_stage_with_sync simulation "${_SLURM_DIR}/simulation.sh" \
		--dependency="afterok:${LAST_SLURM_JOB_ID}" || return 1

	_run_slurm_stage_with_sync analyse "${_SLURM_DIR}/analyse.sh" \
		--dependency="afterok:${LAST_SLURM_JOB_ID}" || return 1

	echo "Pipeline completed successfully."
	return 0
}

_submit_pipeline || _rc=$?
exit "${_rc}"

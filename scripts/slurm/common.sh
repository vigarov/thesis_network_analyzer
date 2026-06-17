#!/usr/bin/env bash
# Shared setup for scripts/slurm/*.sh — source after #SBATCH lines, not executed directly.
set -euo pipefail

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SLURM_DIR}/../.." && pwd)"

DEFAULT_AUTO_SYNC_INTERVAL=10
AUTO_SYNC_INTERVAL=""
WANDB_SYNCHER_PID=""

# Load paths + W&B variables. Override before sourcing, or set ENV_FILE in the environment.
ENV_FILE="${ENV_FILE:-${_REPO_ROOT}/.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
	echo "ERROR: env file not found: ${ENV_FILE}" >&2
	echo "Copy .env.example to .env and fill in the placeholders." >&2
	exit 1
fi
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${PROJECT_ROOT:?Set PROJECT_ROOT in ${ENV_FILE}}"
: "${RESULTS_DIR:?Set RESULTS_DIR in ${ENV_FILE}}"
: "${PRETRAIN_OUTPUT_DIR:?Set PRETRAIN_OUTPUT_DIR in ${ENV_FILE}}"
: "${ANALYSIS_OUTPUT_DIR:?Set ANALYSIS_OUTPUT_DIR in ${ENV_FILE}}"
: "${SLURM_LOG_DIR:?Set SLURM_LOG_DIR in ${ENV_FILE}}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT in ${ENV_FILE}}"

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
mkdir -p "${RESULTS_DIR}" "${ANALYSIS_OUTPUT_DIR}" "${SLURM_LOG_DIR}"
SYNCH_FILE="${PROJECT_ROOT}/.synch"

mapfile -t CONFIG_FILES < <(find "${PROJECT_ROOT}/input_configs" -maxdepth 1 -name '*.json' | sort)
if ((${#CONFIG_FILES[@]} == 0)); then
	echo "ERROR: no JSON configs under ${PROJECT_ROOT}/input_configs" >&2
	exit 1
fi

CONFIG_COUNT="${#CONFIG_FILES[@]}"

cd "${PROJECT_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"

_UV_ENV_READY=false

_run_uv_setup() {
	module purge
	module load CUDA/12.6.0 || true
	export _REPO_ROOT
	"${_REPO_ROOT}/setup.sh"
}

run_uv() {
	if [[ "${_UV_ENV_READY}" != true ]]; then
		_run_uv_setup
		_UV_ENV_READY=true
	fi
	uv run "$@"
}

config_for_array_task() {
	local idx="${1:?}"
	if ((idx < 0 || idx >= CONFIG_COUNT)); then
		echo "ERROR: array task id ${idx} out of range (0..$((CONFIG_COUNT - 1)))" >&2
		exit 1
	fi
	echo "${CONFIG_FILES[idx]}"
}

require_array_task_id() {
	if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
		echo "ERROR: SLURM_ARRAY_TASK_ID is unset; submit with sbatch --array=0-$((CONFIG_COUNT - 1))" >&2
		exit 1
	fi
}

slurm_array_spec() {
	local spec="0-$((CONFIG_COUNT - 1))"
	if [[ -n "${SLURM_ARRAY_MAX_PARALLEL:-}" ]]; then
		spec="${spec}%${SLURM_ARRAY_MAX_PARALLEL}"
	fi
	printf '%s' "${spec}"
}

_slurm_stage_from_script() {
	basename "${1:?}" .sh
}

_slurm_is_array_stage() {
	case "${1:?}" in
	pretrain | simulation) return 0 ;;
	*) return 1 ;;
	esac
}

# Set SLURM_OUTPUT_PATH / SLURM_ERROR_PATH for sbatch --output / --error.
# Array jobs use %A (job id) and %a (array task id); single jobs use %j.
slurm_log_paths() {
	local stage="${1:?}"
	local mode="${2:-single}"

	if [[ "${mode}" == "array" ]]; then
		SLURM_OUTPUT_PATH="${SLURM_LOG_DIR}/${stage}_%A_%a.out"
		SLURM_ERROR_PATH="${SLURM_LOG_DIR}/${stage}_%A_%a.err"
	else
		SLURM_OUTPUT_PATH="${SLURM_LOG_DIR}/${stage}_%j.out"
		SLURM_ERROR_PATH="${SLURM_LOG_DIR}/${stage}_%j.err"
	fi
}

slurm_log_cli_args() {
	local stage="${1:?}"
	local mode="${2:-single}"
	slurm_log_paths "${stage}" "${mode}"
	printf '%s\n' "--output=${SLURM_OUTPUT_PATH}" "--error=${SLURM_ERROR_PATH}"
}

_sbatch_log_and_array_args() {
	local script_path="${1:?}"
	local stage="${2:-$(_slurm_stage_from_script "${script_path}")}"
	local -a args=()

	if _slurm_is_array_stage "${stage}"; then
		mapfile -t args < <(slurm_log_cli_args "${stage}" array)
		args+=("--array=$(slurm_array_spec)")
	else
		mapfile -t args < <(slurm_log_cli_args "${stage}" single)
	fi
	printf '%s\0' "${args[@]}"
}

parse_submit_args() {
	AUTO_SYNC_INTERVAL="${DEFAULT_AUTO_SYNC_INTERVAL}"
	while (($#)); do
		case "$1" in
		--no-auto-sync)
			AUTO_SYNC_INTERVAL=""
			shift
			;;
		--auto-sync)
			shift
			if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
				AUTO_SYNC_INTERVAL="$1"
				shift
			else
				AUTO_SYNC_INTERVAL="${DEFAULT_AUTO_SYNC_INTERVAL}"
			fi
			;;
		*)
			echo "ERROR: unknown submit argument: $1" >&2
			echo "Usage: bash $0 [--no-auto-sync] [--auto-sync [SECONDS]]" >&2
			exit 2
			;;
		esac
	done
}

_pretrain_sync_root() {
	local pretrain_base="${PRETRAIN_OUTPUT_DIR}"
	pretrain_base="${pretrain_base%%!OPT*}"
	pretrain_base="${pretrain_base%%!SD*}"
	pretrain_base="${pretrain_base%/}"
	printf '%s' "${pretrain_base}"
}

_wandb_sync_root_for_stage() {
	case "${1:?}" in
	pretrain) _pretrain_sync_root ;;
	*) printf '%s' "${RESULTS_DIR}" ;;
	esac
}

_start_wandb_auto_sync() {
	local interval="${1:-${DEFAULT_AUTO_SYNC_INTERVAL}}"
	local array_count="${2:-0}"
	local sync_root="${3:?}"

	local -a syncher_cmd=(
		python "${_SLURM_DIR}/wandb_syncher.py"
		--synch-file "${SYNCH_FILE}"
		--interval "${interval}"
		--array-count "${array_count}"
		--root "${sync_root}"
	)

	# not uv run, we don't want to load modules from the login node
	uv run "${syncher_cmd[@]}" &
	WANDB_SYNCHER_PID=$!

	if ! kill -0 "${WANDB_SYNCHER_PID}" 2>/dev/null; then
		wait "${WANDB_SYNCHER_PID}" 2>/dev/null || true
		echo "ERROR: failed to start wandb syncher (is another syncher running?)" >&2
		return 1
	fi

	# Syncher exits immediately if .synch already exists (another instance).
	sleep 0.2
	if ! kill -0 "${WANDB_SYNCHER_PID}" 2>/dev/null; then
		wait "${WANDB_SYNCHER_PID}" 2>/dev/null || true
		WANDB_SYNCHER_PID=""
		echo "ERROR: wandb syncher did not start (${SYNCH_FILE} may be locked)" >&2
		return 1
	fi

	echo "Started wandb syncher (every ${interval}s, array_count=${array_count}, pid=${WANDB_SYNCHER_PID})"
}

_stop_wandb_auto_sync() {
	if [[ -n "${WANDB_SYNCHER_PID:-}" ]]; then
		kill "${WANDB_SYNCHER_PID}" 2>/dev/null || true
		wait "${WANDB_SYNCHER_PID}" 2>/dev/null || true
		WANDB_SYNCHER_PID=""
	fi
}

notify_array_task_done() {
	if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
		return 0
	fi
	if [[ ! -f "${SYNCH_FILE}" ]]; then
		return 0
	fi
	echo "done:${SLURM_ARRAY_TASK_ID}" >>"${SYNCH_FILE}"
}

_wait_for_slurm_job() {
	local job_id="${1:?}"
	echo "Waiting for Slurm job ${job_id}..."
	while squeue -h -j "${job_id}" 2>/dev/null | grep -q .; do
		sleep 5
	done

	local failed
	failed="$(sacct -j "${job_id}" -X --format=State -n 2>/dev/null | grep -Ev 'COMPLETED|COMPLETING' | grep -Ev '^$' || true)"
	if [[ -n "${failed}" ]]; then
		echo "ERROR: job ${job_id} did not complete successfully:" >&2
		sacct -j "${job_id}" -X --format=JobID,State,ExitCode -n 2>/dev/null >&2 || true
		return 1
	fi
	echo "Job ${job_id} completed successfully."
	return 0
}

_run_slurm_stage_with_sync() {
	local stage="${1:?}"
	local script_path="${2:?}"
	shift 2
	local array_count=0 sync_root job_id rc=0

	if _slurm_is_array_stage "${stage}"; then
		array_count="${CONFIG_COUNT}"
	fi
	sync_root="$(_wandb_sync_root_for_stage "${stage}")"

	if [[ -n "${AUTO_SYNC_INTERVAL:-}" ]]; then
		_start_wandb_auto_sync "${AUTO_SYNC_INTERVAL}" "${array_count}" "${sync_root}" || return 1
	fi

	job_id="$(_sbatch_with_logs "${script_path}" "${stage}" "$@")"
	LAST_SLURM_JOB_ID="${job_id}"
	echo "Submitted ${job_id} (${script_path}, logs under ${SLURM_LOG_DIR})"

	_wait_for_slurm_job "${job_id}" || rc=$?

	if [[ -n "${AUTO_SYNC_INTERVAL:-}" ]]; then
		_stop_wandb_auto_sync
	fi
	return "${rc}"
}

_sbatch_with_logs() {
	local script_path="${1:?}"
	shift
	local stage="${1:-}"
	if [[ -n "${stage}" && "${stage}" != --* ]]; then
		shift
	else
		stage="$(_slurm_stage_from_script "${script_path}")"
	fi

	local -a sbatch_args=(--parsable --account="${SLURM_ACCOUNT}" --chdir="${PROJECT_ROOT}")
	mapfile -d '' -t _log_array_args < <(_sbatch_log_and_array_args "${script_path}" "${stage}")
	sbatch_args+=("${_log_array_args[@]}")
	if (($#)); then
		sbatch_args+=("$@")
	fi
	sbatch "${sbatch_args[@]}" "${script_path}"
}

_submit_single_job() {
	local script_path="${1:?}"
	local stage="${2:-$(_slurm_stage_from_script "${script_path}")}"
	local rc=0

	if [[ -n "${AUTO_SYNC_INTERVAL:-}" ]]; then
		trap '_stop_wandb_auto_sync' EXIT INT TERM
	fi

	_run_slurm_stage_with_sync "${stage}" "${script_path}" || rc=$?

	if [[ -n "${AUTO_SYNC_INTERVAL:-}" ]]; then
		trap - EXIT INT TERM
	fi
	return "${rc}"
}

_run_submit_wrapper() {
	local script_path="${1:?}"
	shift
	parse_submit_args "$@"
	_submit_single_job "${script_path}"
	exit $?
}

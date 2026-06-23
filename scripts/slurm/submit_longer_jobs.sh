#!/usr/bin/env bash
# Re-submit timed-out analysis shards with tiered wall times, wait, then min-max.
#
# Usage (from repo root):
#   bash scripts/slurm/submit_longer_jobs.sh
#   bash scripts/slurm/submit_longer_jobs.sh --dry-run
#   bash scripts/slurm/submit_longer_jobs.sh --no-minmax   # wait only, skip min-max
#
# Requires .env (PROJECT_ROOT, RESULTS_DIR, ANALYSIS_OUTPUT_DIR, SLURM_*).

set -euo pipefail

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/slurm/common.sh
source "${_SLURM_DIR}/common.sh"

SCRIPTS_DIR="${_SLURM_DIR}/generated/analyse_jobs"
DRY_RUN=0
RUN_MINMAX=1
POLL_INTERVAL=5

while (($# > 0)); do
	case "$1" in
	--dry-run)
		DRY_RUN=1
		shift
		;;
	--no-minmax)
		RUN_MINMAX=0
		shift
		;;
	-h | --help)
		sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
		exit 0
		;;
	*)
		echo "Unknown option: $1" >&2
		exit 2
		;;
	esac
done

submit_shard() {
	local time_limit="$1"
	local slug="$2"
	local job_name="net-a-${slug}"
	local script="${SCRIPTS_DIR}/analyse_${slug}.sh"

	if [[ ! -f "${script}" ]]; then
		echo "ERROR: script not found: ${script}" >&2
		return 1
	fi

	if ((DRY_RUN)); then
		echo "[dry-run] ${time_limit} ${slug}"
		return 0
	fi

	local job_id
	job_id="$(
		sbatch --parsable \
			--account="${SLURM_ACCOUNT}" \
			--chdir="${PROJECT_ROOT}" \
			--output="${SLURM_LOG_DIR}/${job_name}_%j.out" \
			--error="${SLURM_LOG_DIR}/${job_name}_%j.err" \
			--time="${time_limit}" \
			"${script}"
	)"
	job_id="${job_id%%;*}"
	echo "Submitted ${job_id} (${time_limit}) ${slug}"
	SUBMITTED_JOB_IDS+=("${job_id}")
}

wait_for_jobs() {
	local job_ids=("$@")
	if ((${#job_ids[@]} == 0)); then
		return 0
	fi

	local job_list
	job_list="$(IFS=,; echo "${job_ids[*]}")"
	echo "Waiting for ${#job_ids[@]} job(s): ${job_list}"

	while squeue -h -j "${job_list}" 2>/dev/null | grep -q .; do
		sleep "${POLL_INTERVAL}"
	done

	local job_id failed
	local -a failures=()
	for job_id in "${job_ids[@]}"; do
		failed="$(
			sacct -j "${job_id}" -X --format=State -n 2>/dev/null \
				| grep -Ev 'COMPLETED|COMPLETING' \
				| grep -Ev '^$' || true
		)"
		if [[ -n "${failed}" ]]; then
			failures+=("${job_id}")
			echo "ERROR: job ${job_id} did not complete successfully:" >&2
			sacct -j "${job_id}" -X --format=JobID,State,ExitCode,Elapsed,Timelimit -n 2>/dev/null >&2 || true
		fi
	done

	if ((${#failures[@]} > 0)); then
		echo "ERROR: ${#failures[@]} job(s) failed: ${failures[*]}" >&2
		return 1
	fi

	echo "All ${#job_ids[@]} job(s) completed successfully."
}

SUBMITTED_JOB_IDS=()

# 4h30m
for slug in \
	cat2_sequence_control_tr__adagrad_lr0.01__afc59cb956 \
	cat2_sequence_control_tr__adam_lr0.0005__2e209b0041 \
	cat2_sequence_control_tr__grafted_shampoo_lr0.01__f7e47a9cb9 \
	cat2_sequence_labelperm__pure_shampoo_lr0.01__406be2d8bc \
	cat2_sequence_recover_K1__grafted_shampoo_lr0.01__e01a09e3da \
	cat2_sequence_reinforce__adagrad_lr0.01__2b9d218acc
do
	submit_shard "04:30:00" "${slug}"
done

# 6h45m
for slug in \
	cat1_sample_shuffle_cont__adagrad_lr0.01__609c6326da \
	cat1_sample_shuffle_cont__sgd_lr0.01__7ad30f8b88 \
	cat2_sequence_recover_K1__adam_lr0.0005__9206ccb4f9 \
	cat2_sequence_reinforce__adam_lr0.0005__bab87a3f8c \
	cat2_sequence_reinforce__pure_shampoo_lr0.01__8cddccb29a \
	cat2_sequence_reinforce__sgd_lr0.01__16192fdf2a
do
	submit_shard "06:45:00" "${slug}"
done

# 10h
for slug in \
	cat2_sequence_recover_K1__pure_shampoo_lr0.01__b4e9eff68d \
	cat2_sequence_recover_K1__sgd_lr0.01__b4aa72b754 \
	cat2_sequence_reinforce__grafted_shampoo_lr0.01__c8e80fcb2a
do
	submit_shard "10:00:00" "${slug}"
done

if ((DRY_RUN)); then
	echo "Dry run complete (15 shard(s))."
	exit 0
fi

wait_for_jobs "${SUBMITTED_JOB_IDS[@]}"

if ((RUN_MINMAX)); then
	echo "Running global robustness min-max normalization..."
	cd "${PROJECT_ROOT}"
	run_uv python scripts/slurm/min_max_individual.py \
		--input-dir "${RESULTS_DIR}" \
		--output-dir "${ANALYSIS_OUTPUT_DIR}"
fi

echo "Done."

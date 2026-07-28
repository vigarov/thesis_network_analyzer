#!/usr/bin/env bash
# Pull from origin while preserving cluster-local skip-worktree files
# (pyproject.toml, uv.lock, egg-info, etc.).
#
# Usage (from anywhere):
#   bash scripts/slurm/cluster_pull.sh
#   bash scripts/slurm/cluster_pull.sh --rebase
#
# Requires skip-worktree flags on cluster-specific files; see .gitignore comments.

set -euo pipefail

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SLURM_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
	echo "ERROR: not inside a git repository: ${_REPO_ROOT}" >&2
	exit 1
fi

mapfile -t SKIP_WORKTREE_FILES < <(git ls-files -v | awk '/^S / {print $2}')

BACKUP_DIR=""
PULL_SUCCEEDED=0

restore_cluster_files() {
	local f
	for f in "${SKIP_WORKTREE_FILES[@]}"; do
		if [[ -f "${BACKUP_DIR}/${f}" ]]; then
			mkdir -p "$(dirname "${f}")"
			cp "${BACKUP_DIR}/${f}" "${f}"
		fi
	done
}

reflag_skip_worktree() {
	local f
	for f in "${SKIP_WORKTREE_FILES[@]}"; do
		git update-index --skip-worktree "${f}"
	done
}

cleanup() {
	local exit_code=$?
	if [[ -n "${BACKUP_DIR}" && -d "${BACKUP_DIR}" ]]; then
		if ((exit_code != 0 || PULL_SUCCEEDED == 0)); then
			restore_cluster_files
		fi
		rm -rf "${BACKUP_DIR}"
	fi
	if ((${#SKIP_WORKTREE_FILES[@]} > 0)); then
		reflag_skip_worktree || true
	fi
}
trap cleanup EXIT

if ((${#SKIP_WORKTREE_FILES[@]} == 0)); then
	echo "No skip-worktree files found; running plain git pull."
	git pull "$@"
	exit 0
fi

echo "Preserving skip-worktree files across pull:"
printf '  %s\n' "${SKIP_WORKTREE_FILES[@]}"

BACKUP_DIR="$(mktemp -d)"
for f in "${SKIP_WORKTREE_FILES[@]}"; do
	if [[ ! -f "${f}" ]]; then
		echo "ERROR: skip-worktree file missing: ${f}" >&2
		exit 1
	fi
	mkdir -p "${BACKUP_DIR}/$(dirname "${f}")"
	cp "${f}" "${BACKUP_DIR}/${f}"
	git update-index --no-skip-worktree "${f}"
done

git checkout -- "${SKIP_WORKTREE_FILES[@]}"

git pull "$@"
PULL_SUCCEEDED=1

restore_cluster_files
reflag_skip_worktree

echo "Pull complete; cluster-local files restored."


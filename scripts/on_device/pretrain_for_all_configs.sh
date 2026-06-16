#!/usr/bin/env bash
# Pretrain models for every input config (except test.json) on the local machine.
#
# Usage:
#   bash scripts/on_device/pretrain_for_all_configs.sh
#   bash scripts/on_device/pretrain_for_all_configs.sh --expert
#   bash scripts/on_device/pretrain_for_all_configs.sh --multi-seeds
#   bash scripts/on_device/pretrain_for_all_configs.sh --multi-seeds 6,7,14
#   bash scripts/on_device/pretrain_for_all_configs.sh --expert --multi-seeds
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

EXPERT=false
MULTI_SEEDS_ARGS=()

usage() {
	cat <<'EOF'
Usage: pretrain_for_all_configs.sh [--expert] [--multi-seeds [SEEDS]]

Run pretrain-models for each input_configs/*.json (skips test.json).

Options (forwarded to pretrain-models):
  --expert              Expert preset: 20000 samples, threshold 0.95
  --multi-seeds         Model weight-init seeds; bare flag uses notebook defaults
  --multi-seeds SEEDS   Comma-separated seed list (e.g. 6,7,14)
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--expert)
			EXPERT=true
			shift
			;;
		--multi-seeds)
			if [[ $# -ge 2 && "${2}" != --* ]]; then
				MULTI_SEEDS_ARGS=(--multi_seeds "${2}")
				shift 2
			else
				MULTI_SEEDS_ARGS=(--multi_seeds)
				shift
			fi
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "ERROR: unknown option: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

PRETRAIN_ARGS=()
OUTPUT_SUBDIR=""
if [[ "${EXPERT}" == true ]]; then
	PRETRAIN_ARGS+=(--expert)
	OUTPUT_SUBDIR="expert"
else
	OUTPUT_SUBDIR="base"
fi
if ((${#MULTI_SEEDS_ARGS[@]} > 0)); then
	PRETRAIN_ARGS+=("${MULTI_SEEDS_ARGS[@]}")
fi

OUTPUT_DIR="pretrained_models/${OUTPUT_SUBDIR}/!OPT/!SD/"

for config in input_configs/*.json; do
	[[ "$(basename "${config}")" == "test.json" ]] && continue
	echo "pretrain config=${config}"
	uv run pretrain-models \
		--config "${config}" \
		--output-dir "${OUTPUT_DIR}" \
		"${PRETRAIN_ARGS[@]}"
done

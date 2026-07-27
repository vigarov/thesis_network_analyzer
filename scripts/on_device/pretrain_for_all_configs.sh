#!/usr/bin/env bash
# Pretrain models for every input config (except test.json) on the local machine.
#
# Usage:
#   bash scripts/on_device/pretrain_for_all_configs.sh
#   bash scripts/on_device/pretrain_for_all_configs.sh input_configs/cifar10
#   bash scripts/on_device/pretrain_for_all_configs.sh --expert input_configs
#   bash scripts/on_device/pretrain_for_all_configs.sh --multi-seeds 6,7,14 input_configs/cifar10
#   bash scripts/on_device/pretrain_for_all_configs.sh --multi-lr 1e-3,1e-2,1e-1
#   bash scripts/on_device/pretrain_for_all_configs.sh --expert --multi-seeds --multi-lr
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_REPO_ROOT}"

EXPERT=false
MULTI_SEEDS_ARGS=()
MULTI_LR_ARGS=()
CONFIG_DIR="input_configs"

usage() {
	cat <<'EOF'
Usage: pretrain_for_all_configs.sh [OPTIONS] [CONFIG_DIR]

Run pretrain-models for each *.json in CONFIG_DIR (non-recursive; skips test.json).
Checkpoints are written under pretrained_models/<dataset>/<base|expert>/!OPT/!SD/.

Arguments:
  CONFIG_DIR            Parent directory to traverse (default: input_configs)

Options (forwarded to pretrain-models):
  --expert              Expert preset: 20000 samples (MNIST) / 40000 (CIFAR-10)
  --multi-seeds         Model weight-init seeds; bare flag uses notebook defaults
  --multi-seeds SEEDS   Comma-separated seed list (e.g. 6,7,14)
  --multi-lr            Base learning rates; bare flag uses the default sweep
  --multi-lr LRS        Comma-separated lr list (e.g. 1e-3,1e-2,1e-1)
EOF
}

_read_dataset() {
	local line
	line="$(grep '"dataset"' "$1" | head -n 1)"
	if echo "${line}" | grep -q 'cifar'; then
		echo cifar10
	else
		echo mnist
	fi
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
		--multi-lr)
			if [[ $# -ge 2 && "${2}" != --* ]]; then
				MULTI_LR_ARGS=(--multi_lr "${2}")
				shift 2
			else
				MULTI_LR_ARGS=(--multi_lr)
				shift
			fi
			;;
		-h|--help)
			usage
			exit 0
			;;
		--*)
			echo "ERROR: unknown option: $1" >&2
			usage >&2
			exit 1
			;;
		*)
			CONFIG_DIR="$1"
			shift
			;;
	esac
done

if [[ ! -d "${CONFIG_DIR}" ]]; then
	echo "ERROR: config directory not found: ${CONFIG_DIR}" >&2
	exit 1
fi

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
if ((${#MULTI_LR_ARGS[@]} > 0)); then
	PRETRAIN_ARGS+=("${MULTI_LR_ARGS[@]}")
fi

mapfile -t CONFIG_FILES < <(find "${CONFIG_DIR}" -maxdepth 1 -name '*.json' | sort)
if ((${#CONFIG_FILES[@]} == 0)); then
	echo "ERROR: no JSON configs under ${CONFIG_DIR}" >&2
	exit 1
fi

for config in "${CONFIG_FILES[@]}"; do
	[[ "$(basename "${config}")" == "test.json" ]] && continue
	dataset="$(_read_dataset "${config}")"
	OUTPUT_DIR="pretrained_models/${dataset}/${OUTPUT_SUBDIR}/!OPT/!SD/"
	echo "pretrain config=${config} dataset=${dataset} output=${OUTPUT_DIR}"
	uv run pretrain-models \
		--config "${config}" \
		--output-dir "${OUTPUT_DIR}" \
		"${PRETRAIN_ARGS[@]}"
done

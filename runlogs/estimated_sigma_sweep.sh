#!/usr/bin/env bash
# Does a blur sigma chosen from the free-space contradiction score recover the
# gain that the oracle gets from the true alignment error?
#
# Four conditions per noise level, everything else identical:
#   none      no blur, the plain CGRF baseline
#   fixed1    sigma 1.0 everywhere
#   fixed2    sigma 2.0 everywhere
#   oracle    sigma from the TRUE error  (upper bound, not deployable)
#   estimated sigma from the free-space score (deployable)
#
# The eval YAML filename does not encode the blur setting, so runs at the same
# noise level overwrite each other's summary. Results are read from these logs
# instead; the author's originals are recoverable from their .orig copies.
set -u

cd /nfs/home/charlie/InCoP
PY=/nfs/home/charlie/miniconda3/envs/InCoP/bin/python
MODEL=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
GPU=${GPU:-5}
OUT=runlogs/estimated_sigma
mkdir -p "$OUT"

# Thresholds come from quantile-matching the score against the oracle's own
# split of the samples, so the two policies assign the same proportions.
EST_BINS="0.429:0,0.600:1.0,inf:2.0"
ORA_BINS="0.16:0,0.27:1.0,inf:2.0"

run () {
  local tag=$1 pos=$2 rot=$3; shift 3
  local log="$OUT/${tag}_${pos}.log"
  echo "=== $tag  pos_std=$pos rot_std=$rot  -> $log"
  CUDA_VISIBLE_DEVICES=$GPU $PY opencood/tools/inference_isaac.py \
    --model_dir "$MODEL" \
    --fusion_method intermediate --eval_split test --num_workers 6 \
    --pose_noise_pos_std "$pos" --pose_noise_rot_std "$rot" \
    "$@" > "$log" 2>&1
  echo "    exit=$?  $(grep -h 'Isaac multi-class' "$log" | tail -1)"
}

for pos in 0.10 0.20 0.30 0.60 0.80; do
  rot=$(awk -v p="$pos" 'BEGIN{printf "%.1f", p*10}')
  run none      "$pos" "$rot"
  run fixed1    "$pos" "$rot" --partner_blur_sigma 1.0
  run fixed2    "$pos" "$rot" --partner_blur_sigma 2.0
  run oracle    "$pos" "$rot" --oracle_sigma_bins "$ORA_BINS"
  run estimated "$pos" "$rot" --estimated_sigma_bins "$EST_BINS"
done

echo "ALL DONE"

#!/usr/bin/env bash
# Can the estimated sigma selector be fixed by calibration alone?
#
# The zero-noise row showed where it loses: at sigma_pos = 0 it scores 0.3422
# against the oracle's 0.3984, because it blurs frames whose pose is already
# right. Measured on the null (contradiction.csv, sigma = 0 draws), the
# deployed first threshold of 0.429 fires on 57.5% of perfectly aligned
# samples, and 13.0% of draws have no evidence at all and take the fallback.
#
# Two single changes attack that, plus their combination:
#   fb0   fallback 0 instead of 2   - removes the 13% hard floor of false alarms
#   hi    thresholds 0.650 / 0.800  - trades detection for specificity
#                                     (28.7% false alarms, 81.6% of >0.5 m draws)
#   both  the two together
#
# If none of these beats a single fixed sigma across the whole noise range, the
# limit is the score's discriminability at low error, not its calibration, and
# this route needs a different signal rather than better thresholds.
set -u
cd /nfs/home/charlie/InCoP
PY=/nfs/home/charlie/miniconda3/envs/InCoP/bin/python
MODEL=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
GPU=${GPU:-2}
OUT=runlogs/estimated_sigma
BASE_BINS="0.429:0,0.600:1.0,inf:2.0"
HI_BINS="0.650:0,0.800:1.0,inf:2.0"

run () {
  local tag=$1 pos=$2 rot=$3; shift 3
  local log="$OUT/${tag}_${pos}.log"
  echo "=== $tag pos_std=$pos -> $log"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u opencood/tools/inference_isaac.py \
    --model_dir "$MODEL" \
    --fusion_method intermediate --eval_split test --num_workers 6 \
    --pose_noise_pos_std "$pos" --pose_noise_rot_std "$rot" \
    "$@" > "$log" 2>&1
  echo "    exit=$?  $(grep -h 'Isaac multi-class' "$log" | tail -1)"
}

for pos in 0.00 0.10 0.20 0.30 0.60 0.80; do
  rot=$(awk -v p="$pos" 'BEGIN{printf "%.1f", p*10}')
  run est_fb0  "$pos" "$rot" --estimated_sigma_bins "$BASE_BINS" --estimated_sigma_fallback 0.0
  run est_hi   "$pos" "$rot" --estimated_sigma_bins "$HI_BINS"
  run est_both "$pos" "$rot" --estimated_sigma_bins "$HI_BINS"  --estimated_sigma_fallback 0.0
done
echo "ALL DONE"

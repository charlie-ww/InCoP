#!/usr/bin/env bash
# The row the first sweep left out: no pose noise at all.
#
# It matters more than it looks. Blurring the partner costs accuracy when the
# pose is already right, so this is the one condition where blurring everything
# by a fixed sigma is punished and per-frame selection can pay for itself.
# Averaging the sweep over 0.1-0.8 without it implicitly assumes the robots are
# never well localised, which biases the comparison against any per-frame
# method.
set -u
cd /nfs/home/charlie/InCoP
PY=/nfs/home/charlie/miniconda3/envs/InCoP/bin/python
MODEL=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
GPU=${GPU:-2}
OUT=runlogs/estimated_sigma
EST_BINS="0.429:0,0.600:1.0,inf:2.0"
ORA_BINS="0.16:0,0.27:1.0,inf:2.0"

run () {
  local tag=$1; shift
  local log="$OUT/${tag}_0.00.log"
  echo "=== $tag  pos_std=0.00 -> $log"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u opencood/tools/inference_isaac.py \
    --model_dir "$MODEL" \
    --fusion_method intermediate --eval_split test --num_workers 6 \
    --pose_noise_pos_std 0.0 --pose_noise_rot_std 0.0 \
    "$@" > "$log" 2>&1
  echo "    exit=$?  $(grep -h 'Isaac multi-class' "$log" | tail -1)"
}

run none
run fixed1    --partner_blur_sigma 1.0
run fixed2    --partner_blur_sigma 2.0
run oracle    --oracle_sigma_bins "$ORA_BINS"
run estimated --estimated_sigma_bins "$EST_BINS"
echo "ALL DONE"

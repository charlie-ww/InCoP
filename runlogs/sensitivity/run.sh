#!/usr/bin/env bash
# Does jiggling the partner pose predict how much this frame was damaged?
#
# Per sample the probe runs the model K times with the partner pose drawn from
# the localisation prior, plus once with the TRUE pose to get that frame's own
# ceiling. The spread of the detections is the estimate; the gap to the ceiling
# is the damage it is meant to predict. Neither the probe nor the free-space
# score reads ground truth; the true pose is used only to score them.
#
# --estimated_sigma_bins "inf:0" keeps the free-space estimator running (so its
# score lands in the same CSV) while choosing sigma 0, i.e. no blur, so both
# estimators are measured against the unmitigated condition.
set -u
cd /nfs/home/charlie/InCoP
PY=/nfs/home/charlie/miniconda3/envs/InCoP/bin/python
MODEL=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
GPU=${GPU:-2}
OUT=runlogs/sensitivity
K=${K:-6}
N=${N:-400}

for pos in 0.10 0.20 0.40 0.80; do
  rot=$(awk -v p="$pos" 'BEGIN{printf "%.1f", p*10}')
  log="$OUT/probe_${pos}.log"
  echo "=== pos_std=$pos rot_std=$rot -> $log"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u opencood/tools/inference_isaac.py \
    --model_dir "$MODEL" \
    --fusion_method intermediate --eval_split test --num_workers 6 \
    --pose_noise_pos_std "$pos" --pose_noise_rot_std "$rot" \
    --max_samples "$N" --sensitivity_probe "$K" \
    --estimated_sigma_bins "inf:0" \
    --sensitivity_csv "$OUT/probe_${pos}.csv" > "$log" 2>&1
  echo "    exit=$?"
  grep -A16 "Pose-sensitivity probe:" "$log" | sed 's/^/    /'
done
echo "ALL DONE"

#!/usr/bin/env bash
# Upper bound for "just look at the other robot".
#
# The other robot is in the ego's sensing wedge in 73.9% of test frames and the
# LiDAR returns points from it in 98% of those, so the ego could in principle
# measure the partner's position directly. That gives position, not heading.
# This sweep asks whether position is enough:
#
#   pos_visible  position exact wherever the LiDAR actually hits the partner
#                (>= 3 returns), heading still noisy; other frames untouched
#   pos_always   position exact in every frame - the ceiling of fixing position
#   yaw_always   the complement: heading exact, position noisy
#
# These are upper bounds: the position comes from the true transform, not from
# a detector. If even this does not recover the loss, a real detector cannot.
#
# It starts with a check that the hook is sound: under heavy noise, substituting
# the true transform must reproduce the zero-noise score on the same samples.
set -u
cd /nfs/home/charlie/InCoP
PY=/nfs/home/charlie/miniconda3/envs/InCoP/bin/python
MODEL=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
OUT=runlogs/direct_obs
HERE=$(dirname "$0")
mkdir -p "$OUT"

# Wait for a GPU unused for 5 minutes ONCE, then keep it for the whole batch.
# Between runs the card is only checked for someone else having moved in during
# the few seconds it sat empty; only then is a new card looked for.
GPU=${USE_GPU:-}   # a card the user has already watched sit idle; skips the 5-minute wait
gpu_for_next_run() {
    if [ -n "$GPU" ]; then
        # our own run may take a few seconds to release the card, so look a
        # few times before concluding someone else took it
        for _ in 1 2 3 4; do
            sleep 5
            if "$HERE/wait_for_gpu.sh" free "$GPU"; then return; fi
        done
        echo "    GPU $GPU is now in use by someone else; looking for another"
    fi
    echo "=== waiting for a GPU unused for 5 minutes..."
    GPU=$("$HERE/wait_for_gpu.sh" wait 15 300)
    echo "    acquired GPU $GPU; the remaining runs stay on it"
}

run() {   # run <log> <pos_std> <extra args...>
    local log=$1 pos=$2; shift 2
    local rot; rot=$(awk -v p="$pos" 'BEGIN{printf "%.1f", p*10}')
    gpu_for_next_run
    echo "--- $(basename "$log")  on GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" $PY -u opencood/tools/inference_isaac.py \
        --model_dir "$MODEL" --fusion_method intermediate --eval_split test \
        --num_workers 6 --pose_noise_pos_std "$pos" --pose_noise_rot_std "$rot" \
        "$@" > "$log" 2>&1
    echo "    exit=$?  $(grep -h 'Isaac multi-class' "$log" | tail -1)"
    grep -h 'direct_obs\[' "$log" | sed 's/^/    /'
}
map03() { grep -h 'Isaac multi-class' "$1" | tail -1 | sed -n 's/.*mAP@0\.3=\([0-9.]*\).*/\1/p'; }

# ---- sanity: true transform under heavy noise == zero noise, same 200 samples
run "$OUT/sanity_zero.log"  0.00 --max_samples 200
run "$OUT/sanity_clean.log" 0.60 --max_samples 200 --direct_obs clean
A=$(map03 "$OUT/sanity_zero.log"); B=$(map03 "$OUT/sanity_clean.log")
echo "SANITY zero-noise=$A  clean-under-0.60=$B"
if ! awk -v a="$A" -v b="$B" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(a!="" && b!="" && d<=0.01)}'; then
    echo "SANITY FAILED - the hook does not reproduce zero noise; sweep not run"
    exit 3
fi
echo "SANITY OK"

for pos in 0.10 0.20 0.30 0.60 0.80; do
    run "$OUT/pos_visible_${pos}.log" "$pos" --direct_obs pos_visible
    run "$OUT/pos_always_${pos}.log"  "$pos" --direct_obs pos_always
done
for pos in 0.20 0.60; do
    run "$OUT/yaw_always_${pos}.log"  "$pos" --direct_obs yaw_always
done
echo "ALL DONE"

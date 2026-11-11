#!/usr/bin/env bash
# Correct the partner's position where the ego sees it, and blur the rest.
#
# The direct-observation upper bound (runlogs/direct_obs) beat every blur at
# sigma_pos 0.10 but fell below blur from 0.20 up, because the heading error
# left after fixing position was not hedged. This adds the hedge:
#
#   seen frames    position corrected (oracle), blur sigma S for the heading
#   unseen frames  nothing corrected, blur sigma 2 (the best constant blur)
#
# Main row S = 1 at every level; S = 0 and S = 2 at 0.10 and 0.60 show how much
# the choice of S matters. S is graded on the test split, so the best S is a
# hindsight pick: read the best of the three as an upper bound.
#
# Sanity first: the per-sample blur hook must reproduce the plain fixed blur.
# Under sigma_pos 0.60 with the true transform and blur 2, the score has to
# match zero noise with --partner_blur_sigma 2.0 on the same 200 samples.
set -u
cd /nfs/home/charlie/InCoP
PY=/nfs/home/charlie/miniconda3/envs/InCoP/bin/python
MODEL=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
OUT=runlogs/direct_obs
HERE=$(dirname "$0")

# Wait for a GPU unused for 5 minutes ONCE, then keep it for the whole batch.
GPU=${USE_GPU:-}
gpu_for_next_run() {
    if [ -n "$GPU" ]; then
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

run "$OUT/sanity_blur_ref.log"  0.00 --max_samples 200 --partner_blur_sigma 2.0
run "$OUT/sanity_blur_hook.log" 0.60 --max_samples 200 --direct_obs clean \
    --direct_obs_blur_seen 2.0 --direct_obs_blur_unseen 2.0
A=$(map03 "$OUT/sanity_blur_ref.log"); B=$(map03 "$OUT/sanity_blur_hook.log")
echo "SANITY fixed-blur-at-zero=$A  hook-blur-clean-under-0.60=$B"
if ! awk -v a="$A" -v b="$B" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(a!="" && b!="" && d<=0.01)}'; then
    echo "SANITY FAILED - the per-sample blur does not match the fixed blur; sweep not run"
    exit 3
fi
echo "SANITY OK"

for pos in 0.10 0.20 0.30 0.60 0.80; do
    run "$OUT/seen1_unseen2_${pos}.log" "$pos" --direct_obs pos_visible \
        --direct_obs_blur_seen 1.0 --direct_obs_blur_unseen 2.0
done
for pos in 0.10 0.60; do
    run "$OUT/seen0_unseen2_${pos}.log" "$pos" --direct_obs pos_visible \
        --direct_obs_blur_seen 0.0 --direct_obs_blur_unseen 2.0
    run "$OUT/seen2_unseen2_${pos}.log" "$pos" --direct_obs pos_visible \
        --direct_obs_blur_seen 2.0 --direct_obs_blur_unseen 2.0
done
echo "ALL DONE"

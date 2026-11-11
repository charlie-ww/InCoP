#!/usr/bin/env bash
# Does choosing the calibration from a window of recent scores beat a single one?
#
# A single frame's score cannot separate a correct pose from a wrong one well
# enough to act on, and no threshold fixes that - the 18-run recalibration grid
# settled it: all three recalibrated variants came in below the original single
# calibration (0.2514 / 0.2542 / 0.2491 against 0.2575 mean mAP@0.3).
#
# The median of a 20-frame window is a different signal. It is monotone in the
# true error across all nine levels measured (Spearman rho +1.000), and the
# calibration that wins depends on the regime rather than on the frame. So the
# window picks the calibration and the calibration still picks each frame's
# sigma. The regime is decided from earlier frames only, so a frame never
# influences its own calibration.
#
# Threshold 0.55 on the window median recognises a well-localised stretch about
# 91% of the time and mistakes a misaligned one for it about 5% of the time,
# bootstrapped from runlogs/freespace/contradiction.csv.
#
# For reference, mean mAP@0.3 over the six noise levels:
#   no blur 0.2186 | best single fixed sigma 0.2500 | one calibration 0.2575
#   hindsight-best calibration per level 0.2645 | oracle 0.2735
# That 0.2645 is picked with hindsight from the grid, so it is a ceiling, not a
# result. This run is the honest version of it.
set -u
cd /nfs/home/charlie/InCoP
PY=/nfs/home/charlie/miniconda3/envs/InCoP/bin/python
MODEL=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
OUT=runlogs/estimated_sigma
HERE=$(dirname "$0")

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

for pos in 0.00 0.10 0.20 0.30 0.60 0.80; do
    rot=$(awk -v p="$pos" 'BEGIN{printf "%.1f", p*10}')
    log="$OUT/est_regime_${pos}.log"

    gpu_for_next_run
    echo "--- est_regime pos_std=$pos on GPU $GPU -> $log"

    CUDA_VISIBLE_DEVICES="$GPU" $PY -u opencood/tools/inference_isaac.py \
        --model_dir "$MODEL" \
        --fusion_method intermediate --eval_split test --num_workers 6 \
        --pose_noise_pos_std "$pos" --pose_noise_rot_std "$rot" \
        --estimated_sigma_bins "0.429:0,0.600:1.0,inf:2.0" \
        --estimated_sigma_fallback 2.0 \
        --regime_window 20 --regime_threshold 0.55 \
        > "$log" 2>&1
    echo "    exit=$?  $(grep -h 'Isaac multi-class' "$log" | tail -1)"
    grep -h 'regime:' "$log" | sed 's/^/    /'
done
echo "ALL DONE"

#!/usr/bin/env bash
# Print the index of a GPU that has NO compute process at all, or nothing.
#
# Idle utilisation is not enough: a job can sit at 0% between batches and still
# own the card. The only reliable test is whether the card's UUID appears in
# nvidia-smi's compute-apps list, so that is what this checks.
#
#   free_gpu            -> prints an index, or nothing
#   wait_for_gpu [secs] -> blocks until one is free, then prints its index
set -u

free_gpu() {
    local busy idx uuid
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)
    while IFS=, read -r idx uuid; do
        idx=${idx// /}
        uuid=${uuid// /}
        [ -n "$uuid" ] || continue
        case "$busy" in
            *"$uuid"*) ;;
            *) printf '%s\n' "$idx"; return 0 ;;
        esac
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)
    return 1
}

wait_for_gpu() {
    local interval=${1:-60} gpu
    while true; do
        if gpu=$(free_gpu); then
            printf '%s\n' "$gpu"
            return 0
        fi
        sleep "$interval"
    done
}

case "${1:-check}" in
    check) free_gpu || { echo "no free GPU" >&2; exit 1; } ;;
    wait)  wait_for_gpu "${2:-60}" ;;
    *)     echo "usage: $0 {check|wait [interval]}" >&2; exit 2 ;;
esac

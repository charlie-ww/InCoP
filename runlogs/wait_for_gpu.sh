#!/usr/bin/env bash
# Pick a GPU nobody is using.
#
# Rule (the user's): if some card is unused right now, take it immediately. Only
# when every card is busy, wait - and then a card counts as free once it has
# stayed unused for IDLE seconds without interruption (default 300), sampled
# every INTERVAL seconds (default 15), because a card that has just emptied may
# belong to a job that is between steps. Any sign of use resets that card's
# clock.
#
# "Unused" means all three:
#   - no compute process listed on the card
#   - almost no memory in use  (catches jobs nvidia-smi cannot list, e.g. in a
#     container)
#   - 0% utilisation
#
#   wait_for_gpu.sh check                    -> cards unused right now
#   wait_for_gpu.sh wait [interval] [idle]   -> an unused card now, or wait as above
#   wait_for_gpu.sh free <idx>               -> exit 0 if that card is unused now
#                                               (used between runs of a batch)
set -u
MEM_LIMIT_MIB=${GPU_FREE_MEM_MIB:-100}

unused_now() {
    # indices of cards with no process, negligible memory and zero utilisation
    local busy idx uuid mem util
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)
    while IFS=, read -r idx uuid mem util; do
        idx=${idx// /}; uuid=${uuid// /}; mem=${mem// /}; util=${util// /}
        [ -n "$uuid" ] || continue
        case "$busy" in *"$uuid"*) continue ;; esac
        [ "${mem:-99999}" -le "$MEM_LIMIT_MIB" ] 2>/dev/null || continue
        [ "${util:-100}" -eq 0 ] 2>/dev/null || continue
        printf '%s\n' "$idx"
    done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
                        --format=csv,noheader,nounits 2>/dev/null)
}

wait_for_gpu() {
    local interval=${1:-15} idle=${2:-300} now idx first
    # something is free right now: use it, no waiting
    first=$(unused_now | head -1)
    if [ -n "$first" ]; then
        printf '%s\n' "$first"
        return 0
    fi
    # every card was busy: a card must now stay unused for the full idle period
    echo "all GPUs busy; waiting for one to stay unused for ${idle}s" >&2
    declare -A since=()
    while true; do
        now=$(date +%s)
        declare -A seen=()
        for idx in $(unused_now); do
            seen[$idx]=1
            [ -n "${since[$idx]:-}" ] || since[$idx]=$now
            if [ $(( now - since[$idx] )) -ge "$idle" ]; then
                printf '%s\n' "$idx"
                return 0
            fi
        done
        for idx in "${!since[@]}"; do          # anything in use loses its clock
            [ -n "${seen[$idx]:-}" ] || unset "since[$idx]"
        done
        unset seen
        sleep "$interval"
    done
}

case "${1:-check}" in
    check) out=$(unused_now); [ -n "$out" ] && printf '%s\n' "$out" || { echo "no unused GPU right now" >&2; exit 1; } ;;
    wait)  wait_for_gpu "${2:-15}" "${3:-300}" ;;
    free)  [ -n "${2:-}" ] && unused_now | grep -qx "$2" ;;
    *)     echo "usage: $0 {check|wait [interval] [idle_seconds]|free <idx>}" >&2; exit 2 ;;
esac

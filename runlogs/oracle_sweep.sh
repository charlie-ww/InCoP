source /nfs/home/charlie/miniconda3/etc/profile.d/conda.sh
conda activate InCoP
cd /nfs/home/charlie/InCoP
RUN=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
run_one () {   # $1=pos_std  $2=threshold(或 none)
  p=$1; t=$2
  r=$(python -c "print($p*10)")
  if [ "$t" = "none" ]; then
    echo "=====ORACLE pos_std=$p thr=none====="
    extra=""
  else
    echo "=====ORACLE pos_std=$p thr=$t====="
    extra="--oracle_gate_threshold_m $t"
  fi
  CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_isaac.py \
    --model_dir $RUN --fusion_method intermediate --eval_split test \
    --pose_noise_pos_std $p --pose_noise_rot_std $r $extra --num_workers 6 2>&1 | tail -30
}

# A. 一致性檢查：τ=0 應與噪聲無關
run_one 0.0 0.0
run_one 0.4 0.0

# B. 主網格
for t in 0.05 0.10 0.20; do
  for p in 0.05 0.10 0.15 0.20 0.30 0.40 0.60 0.80; do
    run_one $p $t
  done
done
echo "=====ORACLE SWEEP DONE====="

source /nfs/home/charlie/miniconda3/etc/profile.d/conda.sh
conda activate InCoP
cd /nfs/home/charlie/InCoP
RUN=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
for p in 0.0 0.05 0.10 0.15 0.20 0.30 0.40 0.60 0.80; do
  r=$(python -c "print($p*10)")
  echo "=====SWEEP pos_std=$p rot_std=$r====="
  CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference_isaac.py \
    --model_dir $RUN --fusion_method intermediate --eval_split test \
    --pose_noise_pos_std $p --pose_noise_rot_std $r --num_workers 6 2>&1 \
    | tail -40
done
echo "=====SWEEP ALL DONE====="

source /nfs/home/charlie/miniconda3/etc/profile.d/conda.sh
conda activate InCoP
cd /nfs/home/charlie/InCoP
RUN=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
for sg in 0.5 1.0 2.0; do
  for p in 0.0 0.05 0.10 0.15 0.20; do
    r=$(python -c "print($p*10)")
    echo "=====BLUR sigma=$sg pos_std=$p====="
    CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_isaac.py \
      --model_dir $RUN --fusion_method intermediate --eval_split test \
      --pose_noise_pos_std $p --pose_noise_rot_std $r \
      --partner_blur_sigma $sg --num_workers 6 2>&1 | tail -30
  done
done
echo "=====BLUR SWEEP DONE====="

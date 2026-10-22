source /nfs/home/charlie/miniconda3/etc/profile.d/conda.sh
conda activate InCoP
cd /nfs/home/charlie/InCoP
RUN=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
HO=/nfs/home/charlie/InCoP/dataset/IsaacSimOPV2V_hospital_highoverlap/test
MAXJOBS=3

run_one () {
  p=$1; r=$(python -c "print($p*10)")
  CUDA_VISIBLE_DEVICES=6 python opencood/tools/inference_isaac.py \
    --model_dir $RUN --fusion_method intermediate --eval_split test \
    --eval_dataset_dir $HO \
    --pose_noise_pos_std $p --pose_noise_rot_std $r --num_workers 6 \
    > runlogs/highoverlap/ho_$p.log 2>&1
  echo "done pos_std=$p"
}

for p in 0.0 0.05 0.10 0.15 0.20 0.30 0.40 0.60 0.80; do
  while [ "$(jobs -rp | wc -l)" -ge $MAXJOBS ]; do sleep 5; done
  run_one $p &
done
wait
echo "=====HO SWEEP DONE====="

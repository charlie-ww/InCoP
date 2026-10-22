source /nfs/home/charlie/miniconda3/etc/profile.d/conda.sh
conda activate InCoP
cd /nfs/home/charlie/InCoP
RUN=opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42
LO=/nfs/home/charlie/InCoP/dataset/IsaacSimOPV2V_hospital/test
HO=/nfs/home/charlie/InCoP/dataset/IsaacSimOPV2V_hospital_highoverlap/test
MAXJOBS=3

run_one () {   # $1=split tag  $2=dir  $3=pos_std
  tag=$1; dir=$2; p=$3
  CUDA_VISIBLE_DEVICES=7 python opencood/tools/inference_isaac.py \
    --model_dir $RUN --fusion_method intermediate --eval_split test \
    --eval_dataset_dir $dir \
    --pose_noise_pos_std $p --pose_noise_rot_std $p --num_workers 6 \
    > runlogs/ratio11/${tag}_$p.log 2>&1
  echo "done $tag pos_std=$p (rot=$p, 1:1)"
}

for p in 0.0 0.05 0.10 0.15 0.20 0.30 0.40 0.60 0.80; do
  for pair in "lo:$LO" "ho:$HO"; do
    tag=${pair%%:*}; dir=${pair#*:}
    while [ "$(jobs -rp | wc -l)" -ge $MAXJOBS ]; do sleep 5; done
    run_one $tag $dir $p &
  done
done
wait
echo "=====RATIO11 SWEEP DONE====="

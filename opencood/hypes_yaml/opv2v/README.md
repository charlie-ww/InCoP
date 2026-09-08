# OPV2V LiDAR-only reproduction

This path isolates the existing Where2commFusion and ERMVPFusion modules
behind the same PointPillar encoder, anchor heads, dataset, and evaluation
protocol. It does not import the IsaacSim model or tools.

The Where2comm config enables round-0 supervision with
`supervise_single: true`. Every agent prediction (`psm_single` and
`rm_single`) is produced by the same classification/regression heads used
after fusion and is trained against `label_dict_single` with
`single_weight: 1.0`.

## Data

Extract OPV2V into dataset/OPV2V/{train,validate,test}, or edit the three
paths at the top of each YAML. Both configs use LiDAR only, a 70 m
communication range, 0.4 m XY voxels, and the OpenCOOD PointPillar backbone.
This path accepts dataset poses as-is and intentionally rejects pose-noise,
pose-error, and box-alignment experiment settings.

## Train

    python opencood/tools/train_opv2v.py \
      -y opencood/hypes_yaml/opv2v/point_pillar_where2comm_opv2v.yaml --half

    python opencood/tools/train_opv2v.py \
      -y opencood/hypes_yaml/opv2v/point_pillar_ermvp_opv2v.yaml --half

Do not pass an IsaacSim checkpoint. These models have a different encoder and
head contract and must be trained from scratch for a meaningful comparison.

## Caller replay regression

Run the deterministic same-tensor test before training:

    python -m unittest \
      opencood.tests.test_where2comm_replay \
      opencood.tests.test_point_pillar_single_loss

It sends identical feature, confidence, normalized transform, and physical
transform tensors through the OPV2V and Isaac standard-fusion callers and
requires bit-identical output and input gradients. The second test confirms
that fused and single-agent losses both backpropagate. These verify caller and
loss parity; they do not replace dataset training or AP evaluation.

## Evaluate

    python opencood/tools/inference_opv2v.py \
      --model_dir opencood/logs/<where2comm-run> --split test

    python opencood/tools/inference_opv2v.py \
      --model_dir opencood/logs/<ermvp-run> --split test

Use --dataset_dir /absolute/path/to/OPV2V/test to override the split path
stored in a run's config.yaml. Results are written as eval_opv2v.yaml in the
run directory.

This local configuration is LiDAR PointPillar, so Where2comm paper camera
numbers are not a valid target AP. A credible external comparison must use a
LiDAR OPV2V checkpoint/config with the same range, voxelization, split, and
evaluation protocol. Dataset files and a trained reference checkpoint are
required to produce that AP; forward/backward or replay success alone is not
reported as reproduction.

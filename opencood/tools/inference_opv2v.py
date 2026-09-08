# -*- coding: utf-8 -*-
"""Minimal OPV2V LiDAR-only evaluation for shared-backbone fusion models."""

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils
from opencood.visualization import simple_vis


IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def test_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate an OPV2V LiDAR-only fusion model."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument(
        "--dataset_dir",
        default="",
        help="override validate/test split directory from saved config",
    )
    parser.add_argument("--split", choices=("validate", "test"), default="test")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--tag", default="opv2v")
    return parser.parse_args()


def _validate_config(hypes):
    if hypes.get("fusion", {}).get("dataset") != "opv2v":
        raise ValueError("inference_opv2v.py accepts only OPV2V configs")
    if hypes.get("input_source") != ["lidar"]:
        raise ValueError("The reproduction path is LiDAR-only")
    method = hypes.get("model", {}).get("args", {}).get("fusion_method")
    supported = ("where2comm", "ermvp", "v2xvit", "cobevt", "ours")
    if method not in supported:
        raise ValueError(f"fusion_method must be one of {supported}")
    forbidden = ("noise_setting", "wild_setting", "box_align")
    present = [key for key in forbidden if key in hypes]
    if present:
        raise ValueError(
            "OPV2V is clean-pose only; remove these settings: "
            + ", ".join(present)
        )


def _as_numpy(tensor, empty_shape):
    if tensor is None:
        return np.empty(empty_shape, dtype=np.float32)
    return tensor.detach().cpu().numpy()


def _save_prediction_gt(pred_box, gt_box, lidar, sample_id, save_dir):
    np.save(
        os.path.join(save_dir, f"{sample_id:05d}_pred.npy"),
        _as_numpy(pred_box, (0, 8, 3)),
    )
    np.save(
        os.path.join(save_dir, f"{sample_id:05d}_gt.npy"),
        _as_numpy(gt_box, (0, 8, 3)),
    )
    np.save(
        os.path.join(save_dir, f"{sample_id:05d}_pcd.npy"),
        _as_numpy(lidar, (0, 4)),
    )


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)
    _validate_config(hypes)
    if opt.dataset_dir:
        hypes["validate_dir"] = opt.dataset_dir
    elif opt.split == "test":
        hypes["validate_dir"] = hypes.get(
            "test_dir", hypes["validate_dir"]
        )

    need_lidar = opt.save_vis or opt.save_npy
    dataset = build_dataset(hypes, visualize=need_lidar, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    result_stat = {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in IOU_THRESHOLDS
    }
    communication_megabytes = []
    npy_dir = os.path.join(opt.model_dir, f"npy_{opt.tag}")
    vis_dir = os.path.join(opt.model_dir, f"vis_{opt.tag}")
    if opt.save_npy:
        os.makedirs(npy_dir, exist_ok=True)
    if opt.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    evaluated = 0
    for sample_id, batch_data in tqdm(enumerate(loader), total=len(loader)):
        if opt.max_samples > 0 and evaluated >= opt.max_samples:
            break
        if batch_data is None:
            continue
        batch_data = train_utils.to_device(batch_data, device)
        with torch.no_grad():
            output = model(batch_data["ego"])
            pred_box, pred_score, gt_box = dataset.post_process(
                batch_data, {"ego": output}
            )
        for threshold in IOU_THRESHOLDS:
            eval_utils.caluclate_tp_fp(
                pred_box, pred_score, gt_box, result_stat, threshold
            )

        communication = output.get("communication")
        if isinstance(communication, dict) and "total_MB" in communication:
            communication_megabytes.append(float(communication["total_MB"]))
        if opt.save_npy:
            _save_prediction_gt(
                pred_box,
                gt_box,
                batch_data["ego"]["origin_lidar"][0],
                sample_id,
                npy_dir,
            )
        if opt.save_vis:
            simple_vis.visualize(
                {
                    "pred_box_tensor": pred_box,
                    "gt_box_tensor": gt_box,
                    "score_tensor": pred_score,
                },
                batch_data["ego"]["origin_lidar"][0],
                hypes["preprocess"]["cav_lidar_range"],
                os.path.join(vis_dir, f"{sample_id:05d}.png"),
                method="bev",
            )
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No valid OPV2V samples were evaluated")
    ap30, ap50, ap70 = eval_utils.eval_final_results(
        result_stat, opt.model_dir, infer_info=opt.tag
    )
    method = hypes["model"]["args"]["fusion_method"]
    print(
        f"{method}: samples={evaluated}, AP@0.3={ap30:.4f}, "
        f"AP@0.5={ap50:.4f}, AP@0.7={ap70:.4f}"
    )
    if communication_megabytes:
        average_mb = sum(communication_megabytes) / len(
            communication_megabytes
        )
        print(f"Average transmitted payload: {average_mb:.6f} MB/sample")


if __name__ == "__main__":
    main()

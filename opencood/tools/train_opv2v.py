# -*- coding: utf-8 -*-
"""Minimal OPV2V LiDAR-only training for shared-backbone fusion models."""

import argparse
import glob
import os
import statistics
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def train_parser():
    parser = argparse.ArgumentParser(
        description="Train a LiDAR-only PointPillar fusion model on OPV2V."
    )
    parser.add_argument("--hypes_yaml", "-y", required=True)
    parser.add_argument("--model_dir", default="", help="resume run directory")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    return parser.parse_args()


def _validate_config(hypes):
    if hypes.get("fusion", {}).get("dataset") != "opv2v":
        raise ValueError("train_opv2v.py accepts only fusion.dataset: opv2v")
    if hypes.get("input_source") != ["lidar"]:
        raise ValueError("The reproduction path is LiDAR-only: input_source: [lidar]")
    method = hypes.get("model", {}).get("args", {}).get("fusion_method")
    supported = ("where2comm", "ermvp", "v2xvit", "cobevt", "ours")
    if method not in supported:
        raise ValueError(f"fusion_method must be one of {supported}")
    model_args = hypes["model"]["args"]
    if (
        model_args.get("supervise_single", False)
        and not model_args.get("single_head_shared", True)
    ):
        raise ValueError(
            "OPV2V single-agent supervision must use the shared detection "
            "head so the encoder/decoder contract stays common across fusion "
            "methods."
        )
    forbidden = ("noise_setting", "wild_setting", "box_align")
    present = [key for key in forbidden if key in hypes]
    if present:
        raise ValueError(
            "OPV2V is clean-pose only; remove these settings: "
            + ", ".join(present)
        )


def _make_scheduler(hypes, optimizer, steps_per_epoch, init_epoch):
    config = hypes["lr_scheduler"]
    if config["core_method"] != "cosineannealwarm":
        return train_utils.setup_lr_schedular(
            hypes, optimizer, init_epoch=init_epoch
        )

    from timm.scheduler.cosine_lr import CosineLRScheduler

    total_steps = int(config["epoches"]) * steps_per_epoch
    warmup_steps = int(config["warmup_epoches"]) * steps_per_epoch
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_steps,
        lr_min=float(config["lr_min"]),
        warmup_lr_init=float(config["warmup_lr"]),
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    scheduler.step_update(init_epoch * steps_per_epoch)
    return scheduler


def _loader(dataset, batch_size, workers, prefetch, train):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": workers,
        "collate_fn": dataset.collate_batch_train,
        "shuffle": train,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": train,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
    return DataLoader(**kwargs)


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    _validate_config(hypes)

    print("Building OPV2V LiDAR-only datasets")
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    validate_dataset = build_dataset(hypes, visualize=False, train=False)
    train_params = hypes["train_params"]
    supervise_single = bool(
        hypes["model"]["args"].get("supervise_single", False)
    )
    single_weight = float(train_params.get("single_weight", 1.0))
    workers = (
        opt.num_workers
        if opt.num_workers is not None
        else int(train_params.get("num_workers", 4))
    )
    prefetch = (
        opt.prefetch_factor
        if opt.prefetch_factor is not None
        else int(train_params.get("prefetch_factor", 2))
    )
    train_loader = _loader(
        train_dataset,
        int(train_params["batch_size"]),
        workers,
        prefetch,
        train=True,
    )
    validate_loader = _loader(
        validate_dataset,
        int(train_params["batch_size"]),
        workers,
        prefetch,
        train=False,
    )

    model = train_utils.create_model(hypes)
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)
    scheduler = _make_scheduler(
        hypes, optimizer, max(len(train_loader), 1), init_epoch
    )
    writer = SummaryWriter(saved_path)
    use_amp = bool(opt.half and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_validation_loss = float("inf")
    best_checkpoint = None
    existing_best = glob.glob(
        os.path.join(saved_path, "net_epoch_bestval_at*.pth")
    )
    if len(existing_best) == 1:
        best_checkpoint = existing_best[0]

    print(
        f"Training {hypes['model']['args']['fusion_method']} on {device}; "
        f"AMP={use_amp}, workers={workers}"
    )
    epochs = int(train_params["epoches"])
    scheduler_is_cosine = (
        hypes["lr_scheduler"]["core_method"] == "cosineannealwarm"
    )
    for epoch in range(init_epoch, max(epochs, init_epoch)):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch_id, batch_data in enumerate(progress):
            if batch_data is None:
                continue
            ego_data = train_utils.to_device(batch_data["ego"], device)
            optimizer.zero_grad(set_to_none=True)
            ego_data["epoch"] = epoch
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(ego_data)
                fused_loss = criterion(output, ego_data["label_dict"])
                final_loss = fused_loss
                if supervise_single:
                    if "label_dict_single" not in ego_data:
                        raise KeyError(
                            "supervise_single requires label_dict_single "
                            "from the OPV2V dataloader."
                        )
                    single_loss = criterion(
                        output,
                        ego_data["label_dict_single"],
                        suffix="_single",
                    )
                    final_loss = fused_loss + single_weight * single_loss
            scaler.scale(final_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            criterion.logging(
                epoch, batch_id, len(train_loader), writer, pbar=progress
            )
            if supervise_single:
                criterion.logging(
                    epoch,
                    batch_id,
                    len(train_loader),
                    writer,
                    pbar=progress,
                    suffix="_single",
                )
                writer.add_scalar(
                    "Combined_loss", final_loss.item(),
                    epoch * len(train_loader) + batch_id,
                )
            if scheduler_is_cosine:
                scheduler.step_update(epoch * len(train_loader) + batch_id + 1)

        if not scheduler_is_cosine:
            scheduler.step(epoch)

        if (epoch + 1) % int(train_params.get("save_freq", 1)) == 0:
            torch.save(
                model.state_dict(),
                os.path.join(saved_path, f"net_epoch{epoch + 1}.pth"),
            )

        if (epoch + 1) % int(train_params.get("eval_freq", 1)) != 0:
            continue
        model.eval()
        validation_losses = []
        validation_single_losses = []
        with torch.no_grad():
            for batch_data in validate_loader:
                if batch_data is None:
                    continue
                ego_data = train_utils.to_device(batch_data["ego"], device)
                output = model(ego_data)
                fused_loss = criterion(output, ego_data["label_dict"])
                validation_losses.append(fused_loss.item())
                if supervise_single:
                    if "label_dict_single" not in ego_data:
                        raise KeyError(
                            "supervise_single requires label_dict_single "
                            "from the OPV2V dataloader."
                        )
                    single_loss = criterion(
                        output,
                        ego_data["label_dict_single"],
                        suffix="_single",
                    )
                    validation_single_losses.append(single_loss.item())
        if not validation_losses:
            raise RuntimeError("No valid OPV2V validation samples were loaded")
        validation_loss = statistics.mean(validation_losses)
        writer.add_scalar("Validate_Loss", validation_loss, epoch)
        validation_message = (
            f"epoch {epoch + 1}: fused validation loss="
            f"{validation_loss:.6f}"
        )
        if validation_single_losses:
            single_validation_loss = statistics.mean(
                validation_single_losses
            )
            writer.add_scalar(
                "Validate_Single_Loss", single_validation_loss, epoch
            )
            validation_message += (
                f", single validation loss={single_validation_loss:.6f}"
            )
        print(validation_message)
        if validation_loss < best_validation_loss:
            if best_checkpoint and os.path.isfile(best_checkpoint):
                os.remove(best_checkpoint)
            best_validation_loss = validation_loss
            best_checkpoint = os.path.join(
                saved_path, f"net_epoch_bestval_at{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), best_checkpoint)

    writer.close()
    print(f"Training finished. Checkpoints: {saved_path}")


if __name__ == "__main__":
    main()

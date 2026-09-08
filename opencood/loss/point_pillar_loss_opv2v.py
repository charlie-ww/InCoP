# -*- coding: utf-8 -*-
"""OpenCOOD PointPillar loss with optional shared-head auxiliary loss."""

import torch
import torch.nn as nn


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0 / 9.0, code_weights=None):
        super().__init__()
        self.beta = float(beta)
        if code_weights is None:
            self.code_weights = None
        else:
            self.register_buffer(
                "code_weights",
                torch.as_tensor(code_weights, dtype=torch.float32),
            )

    def forward(self, inputs, targets, weights=None):
        targets = torch.where(torch.isnan(targets), inputs, targets)
        difference = inputs - targets
        if self.code_weights is not None:
            difference = difference * self.code_weights.view(1, 1, -1)
        absolute = difference.abs()
        if self.beta < 1.0e-5:
            loss = absolute
        else:
            loss = torch.where(
                absolute < self.beta,
                0.5 * absolute.square() / self.beta,
                absolute - 0.5 * self.beta,
            )
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        return loss


class PointPillarLossOpv2v(nn.Module):
    """Anchor focal + Smooth-L1 loss used by OpenCOOD PointPillar."""

    def __init__(self, args):
        super().__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = float(args.get("alpha", 0.25))
        self.gamma = float(args.get("gamma", 2.0))
        self.cls_weight = float(args["cls_weight"])
        self.reg_coe = float(args["reg"])
        self.loss_dict = {}
        self.single_loss_dict = {}

    @staticmethod
    def _add_sin_difference(boxes1, boxes2, dim=6):
        pred_sin = (
            torch.sin(boxes1[..., dim:dim + 1])
            * torch.cos(boxes2[..., dim:dim + 1])
        )
        target_sin = (
            torch.cos(boxes1[..., dim:dim + 1])
            * torch.sin(boxes2[..., dim:dim + 1])
        )
        boxes1 = torch.cat(
            [boxes1[..., :dim], pred_sin, boxes1[..., dim + 1:]], dim=-1
        )
        boxes2 = torch.cat(
            [boxes2[..., :dim], target_sin, boxes2[..., dim + 1:]], dim=-1
        )
        return boxes1, boxes2

    @staticmethod
    def _sigmoid_cross_entropy(inputs, targets):
        return (
            torch.clamp(inputs, min=0)
            - inputs * targets
            + torch.log1p(torch.exp(-inputs.abs()))
        )

    def _classification_loss(self, inputs, targets, weights):
        prediction = torch.sigmoid(inputs)
        alpha_weight = (
            targets * self.alpha + (1 - targets) * (1 - self.alpha)
        )
        pt = targets * (1 - prediction) + (1 - targets) * prediction
        focal_weight = alpha_weight * pt.pow(self.gamma)
        loss = focal_weight * self._sigmoid_cross_entropy(inputs, targets)
        return loss * weights.unsqueeze(-1)

    def forward(self, output_dict, target_dict, suffix=""):
        psm = output_dict[f"psm{suffix}"]
        rm = output_dict[f"rm{suffix}"]
        targets = target_dict["targets"]

        labels = target_dict["pos_equal_one"].view(psm.shape[0], -1)
        positives = labels > 0
        negatives = labels == 0
        normalizer = positives.sum(1, keepdim=True).float().clamp_min(1.0)
        cls_weights = (negatives.float() + positives.float()) / normalizer
        reg_weights = positives.float() / normalizer

        cls_preds = psm.permute(0, 2, 3, 1).reshape(psm.shape[0], -1, 1)
        one_hot = torch.zeros(
            *labels.shape, 2, dtype=cls_preds.dtype, device=labels.device
        )
        one_hot.scatter_(-1, labels.unsqueeze(-1).long(), 1.0)
        conf_loss = (
            self._classification_loss(
                cls_preds, one_hot[..., 1:], cls_weights
            ).sum()
            / psm.shape[0]
        ) * self.cls_weight

        box_preds = rm.permute(0, 2, 3, 1).reshape(rm.shape[0], -1, 7)
        box_targets = targets.reshape(targets.shape[0], -1, 7)
        box_preds, box_targets = self._add_sin_difference(
            box_preds, box_targets
        )
        reg_loss = (
            self.reg_loss_func(
                box_preds, box_targets, weights=reg_weights
            ).sum()
            / rm.shape[0]
        ) * self.reg_coe
        total_loss = conf_loss + reg_loss
        losses = {
            "total_loss": total_loss,
            "reg_loss": reg_loss,
            "conf_loss": conf_loss,
        }
        if suffix == "_single":
            self.single_loss_dict = losses
        elif suffix:
            raise ValueError(f"Unsupported PointPillar loss suffix: {suffix}")
        else:
            self.loss_dict = losses
        return total_loss

    def logging(
        self, epoch, batch_id, batch_len, writer, pbar=None, suffix=""
    ):
        losses = self.single_loss_dict if suffix == "_single" else self.loss_dict
        label = "Single " if suffix == "_single" else ""
        description = (
            f"[epoch {epoch}][{batch_id + 1}/{batch_len}] "
            f"{label}Loss: {losses['total_loss'].item():.4f} "
            f"Conf: {losses['conf_loss'].item():.4f} "
            f"Loc: {losses['reg_loss'].item():.4f}"
        )
        if pbar is None:
            print(description)
        else:
            pbar.set_description(description)
        step = epoch * batch_len + batch_id
        writer.add_scalar(
            f"Regression_loss{suffix}", losses["reg_loss"].item(), step
        )
        writer.add_scalar(
            f"Confidence_loss{suffix}", losses["conf_loss"].item(), step
        )

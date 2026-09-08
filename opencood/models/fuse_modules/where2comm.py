# -*- coding: utf-8 -*-
"""OpenCOOD-style Where2comm communication and parameter-free fusion."""

import math
import random

import torch
from torch import nn
import torch.nn.functional as F

from opencood.models.fuse_modules.common import regroup
from opencood.models.fuse_modules.communication import CommunicationStatsMixin
from opencood.models.sub_modules.torch_transformation_utils import (
    warp_affine_simple,
)


def _gaussian_kernel(kernel_size, sigma, device, dtype):
    coordinates = torch.arange(kernel_size, device=device, dtype=dtype)
    coordinates = coordinates - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return (kernel_1d[:, None] * kernel_1d[None, :]).view(
        1, 1, kernel_size, kernel_size
    )


class Where2commAttention(nn.Module):
    """Parameter-free per-pixel scaled-dot attention used by OpenCOOD."""

    def forward(self, features):
        # features [N,C,H,W]. Agent 0 is the ego query.
        num_agents, channels, height, width = features.shape
        tokens = features.permute(2, 3, 0, 1).reshape(
            height * width, num_agents, channels
        )
        query = tokens[:, 0:1]
        logits = torch.bmm(query, tokens.transpose(1, 2)) / math.sqrt(channels)
        attention = torch.softmax(logits, dim=-1)
        output = torch.bmm(attention, tokens)
        return output[:, 0].reshape(
            height, width, channels
        ).permute(2, 0, 1)


class Where2commFusion(CommunicationStatsMixin, nn.Module):
    """One-round Where2comm shared by OPV2V and IsaacSim callers.

    OPV2V can use the official multi-scale path through ``forward_multiscale``.
    IsaacSim uses ``forward`` because its heterogeneous encoders already emit a
    common BEV message feature. Both paths use the same communication selection
    and parameter-free attention implementation.
    """

    requires_confidence_maps = True
    requires_fusion_context = False

    def __init__(self, args):
        super().__init__()
        if isinstance(args, int):
            args = {"input_dim": args}
        args = args or {}
        self.threshold = float(args.get("threshold", args.get("thre", 0.01)))
        self.communication_rounds = int(args.get("communication_rounds", 1))
        if self.communication_rounds != 1:
            raise NotImplementedError(
                "Where2comm supports the paper's one-round setting only."
            )
        self.use_confidence_map = bool(args.get("use_confidence_map", True))
        self.requires_confidence_maps = self.use_confidence_map
        self.training_selection = str(
            args.get("training_selection", "fixed_threshold")
        ).lower()
        if self.training_selection not in (
            "fixed_threshold", "official_random_topk"
        ):
            raise ValueError(
                "Where2comm training_selection must be fixed_threshold or "
                "official_random_topk."
            )
        self.gaussian_smooth = bool(args.get("gaussian_smooth", True))
        self.gaussian_kernel_size = int(args.get("gaussian_kernel_size", 5))
        self.gaussian_sigma = float(args.get("gaussian_sigma", 1.0))
        if self.gaussian_kernel_size <= 0 or self.gaussian_kernel_size % 2 != 1:
            raise ValueError(
                "Where2comm Gaussian kernel size must be a positive odd integer."
            )
        self.attention = Where2commAttention()
        self._init_communication_stats(args, method="Where2comm")

    @staticmethod
    def _length(value):
        return (
            int(value.detach().cpu().item())
            if torch.is_tensor(value) else int(value)
        )

    def _smooth(self, confidence):
        if not self.gaussian_smooth:
            return confidence
        kernel = _gaussian_kernel(
            self.gaussian_kernel_size,
            self.gaussian_sigma,
            confidence.device,
            confidence.dtype,
        )
        return F.conv2d(
            confidence, kernel, padding=self.gaussian_kernel_size // 2
        )

    def _selection_mask(self, score, topk_count=None):
        if topk_count is None:
            return score > self.threshold
        flat = score.flatten()
        count = min(max(int(topk_count), 0), flat.numel())
        if count == 0:
            return torch.zeros_like(score, dtype=torch.bool)
        if count == flat.numel():
            return torch.ones_like(score, dtype=torch.bool)
        indices = torch.topk(flat, count, sorted=False).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[indices] = True
        return mask.reshape_as(score)

    def _training_topk_count(self, num_cells):
        if (
            self.training
            and self.training_selection == "official_random_topk"
        ):
            return int(num_cells * random.uniform(0.0, 1.0))
        return None

    def _confidence_and_selection(self, confidence_maps, output_size):
        if not self.use_confidence_map:
            return None, None
        if confidence_maps is None:
            raise ValueError(
                "Where2comm requires shared pre-fusion detection logits as "
                "confidence_maps."
            )
        confidence = torch.sigmoid(confidence_maps).amax(dim=1, keepdim=True)
        if confidence.shape[-2:] != tuple(output_size):
            confidence = F.interpolate(
                confidence,
                size=tuple(output_size),
                mode="bilinear",
                align_corners=False,
            )
        return confidence, self._smooth(confidence)

    def _scene_mask(self, score, num_agents, height, width, device, dtype):
        mask = torch.ones(
            num_agents, 1, height, width, device=device, dtype=dtype
        )
        if not self.use_confidence_map:
            return mask
        topk_count = self._training_topk_count(height * width)
        for source_idx in range(1, num_agents):
            mask[source_idx, 0] = self._selection_mask(
                score[source_idx, 0], topk_count
            ).to(dtype)
        # Ego never communicates and must retain its complete local feature.
        mask[0] = 1
        return mask

    def _record_stats(
        self, masks, channels, dense_height, dense_width, mode
    ):
        feature_elements = index_elements = metadata_elements = 0
        dense_elements = message_count = 0
        per_sample = []
        for mask in masks:
            partner_mask = mask[1:]
            selected_count = int((partner_mask > 0).sum().item())
            sample_elements = selected_count * int(channels)
            feature_elements += sample_elements
            index_elements += selected_count * 2
            metadata_elements += (
                int(partner_mask.shape[0]) * dense_height * dense_width
            )
            dense_elements += (
                int(partner_mask.shape[0])
                * int(channels) * dense_height * dense_width
            )
            message_count += int(partner_mask.shape[0])
            per_sample.append(sample_elements)
        self._record_communication(
            feature_elements=feature_elements,
            index_elements=index_elements,
            metadata_elements=metadata_elements,
            dense_feature_elements=dense_elements,
            message_count=message_count,
            batch_size=len(masks),
            mode=mode,
            rounds=self.communication_rounds,
            per_sample_feature_elements=per_sample,
            notes=(
                "OpenCOOD-style one-round selection and parameter-free "
                f"scaled-dot attention; training selection: "
                f"{self.training_selection}; inference threshold: "
                f"{self.threshold}."
            ),
        )

    def forward(
        self, x, record_len, affine_matrix, confidence_maps=None, **kwargs
    ):
        """Single-scale shared path used by heterogeneous IsaacSim features."""
        del kwargs
        _, channels, height, width = x.shape
        _, selection_scores = self._confidence_and_selection(
            confidence_maps, (height, width)
        )
        split_features = regroup(x, record_len)
        split_scores = (
            regroup(selection_scores, record_len)
            if selection_scores is not None else None
        )
        outputs = []
        masks = []
        for batch_idx in range(affine_matrix.shape[0]):
            num_agents = self._length(record_len[batch_idx])
            source_features = split_features[batch_idx]
            score = split_scores[batch_idx] if split_scores is not None else None
            source_mask = self._scene_mask(
                score, num_agents, height, width, x.device, x.dtype
            )
            sparse_source = source_features * source_mask
            transforms = affine_matrix[batch_idx, 0, :num_agents]
            aligned = warp_affine_simple(
                sparse_source, transforms, (height, width)
            )
            outputs.append(self.attention(aligned))
            masks.append(source_mask)

        self._record_stats(
            masks,
            channels,
            height,
            width,
            mode=(
                "request_driven_spatially_sparse_bev"
                if self.use_confidence_map
                else "dense_bev_without_confidence_map"
            ),
        )
        return torch.stack(outputs)

    def forward_multiscale(
        self,
        spatial_features,
        confidence_maps,
        record_len,
        backbone,
        affine_matrix,
    ):
        """Official OpenCOOD-style fusion at every PointPillar backbone level."""
        _, selection_scores = self._confidence_and_selection(
            confidence_maps, confidence_maps.shape[-2:]
        )
        split_scores = (
            regroup(selection_scores, record_len)
            if selection_scores is not None else None
        )
        x = spatial_features
        fused_levels = []
        communication_masks = []
        stats_height = stats_width = stats_channels = None

        for level, block in enumerate(backbone.blocks):
            x = block(x)
            height, width = x.shape[-2:]
            split_features = regroup(x, record_len)
            level_outputs = []
            level_masks = []
            masked_level_features = []
            for batch_idx in range(affine_matrix.shape[0]):
                num_agents = self._length(record_len[batch_idx])
                if level == 0:
                    score = (
                        split_scores[batch_idx]
                        if split_scores is not None else None
                    )
                    source_mask = self._scene_mask(
                        score,
                        num_agents,
                        confidence_maps.shape[-2],
                        confidence_maps.shape[-1],
                        x.device,
                        x.dtype,
                    )
                    source_mask = F.interpolate(
                        source_mask,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    level_masks.append(source_mask)
                else:
                    # OpenCOOD applies the communication mask only at level 0.
                    source_mask = torch.ones(
                        num_agents, 1, height, width,
                        device=x.device, dtype=x.dtype,
                    )

                source_features = split_features[batch_idx]
                if level == 0:
                    source_features = source_features * source_mask
                    masked_level_features.append(source_features)
                transforms = affine_matrix[batch_idx, 0, :num_agents]
                aligned = warp_affine_simple(
                    source_features, transforms, (height, width)
                )
                level_outputs.append(self.attention(aligned))

            if level == 0:
                x = torch.cat(masked_level_features, dim=0)
                communication_masks = level_masks
                stats_height, stats_width = height, width
                stats_channels = x.shape[1]
            fused = torch.stack(level_outputs)
            fused_levels.append(
                backbone.deblocks[level](fused)
                if backbone.deblocks else fused
            )

        output = (
            torch.cat(fused_levels, dim=1)
            if len(fused_levels) > 1 else fused_levels[0]
        )
        if len(backbone.deblocks) > len(backbone.blocks):
            output = backbone.deblocks[-1](output)
        self._record_stats(
            communication_masks,
            stats_channels,
            stats_height,
            stats_width,
            mode="official_multiscale_spatially_sparse_bev",
        )
        return output

# -*- coding: utf-8 -*-
"""LiDAR-only PointPillar path with a shared OPV2V detector backbone."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.cobevt import CoBEVT
from opencood.models.fuse_modules.ermvp import ERMVPFusion
from opencood.models.fuse_modules.fusion_in_one import run_standard_fusion
from opencood.models.fuse_modules.ours import ComplementarityGuidedCLCFusion
from opencood.models.fuse_modules.v2xvit import V2XViTFusion
from opencood.models.fuse_modules.where2comm import Where2commFusion
from opencood.models.sub_modules.base_bev_backbone_opv2v import (
    BaseBEVBackboneOpv2v,
)
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.utils.transformation_utils import normalize_pairwise_tfm_range


class PointPillarFusionOpv2v(nn.Module):
    """OpenCOOD-style PointPillar with a selectable shared fusion module."""

    SUPPORTED_FUSIONS = (
        "where2comm", "ermvp", "v2xvit", "cobevt", "ours",
    )

    def __init__(self, args):
        super().__init__()
        self.lidar_range = args["lidar_range"]
        self.fusion_method = str(args["fusion_method"]).lower()
        self.supervise_single = bool(args.get("supervise_single", False))
        self.single_head_shared = bool(args.get("single_head_shared", True))
        if self.supervise_single and not self.single_head_shared:
            raise ValueError(
                "OPV2V single-agent supervision must use the shared "
                "detection head."
            )
        if self.fusion_method not in self.SUPPORTED_FUSIONS:
            raise ValueError(
                f"OPV2V path supports only {self.SUPPORTED_FUSIONS}, "
                f"got {self.fusion_method!r}."
            )

        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=self.lidar_range,
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackboneOpv2v(
            args["base_bev_backbone"], input_channels=64
        )
        self.shrink = (
            DownsampleConv(args["shrink_header"])
            if "shrink_header" in args else nn.Identity()
        )

        mask_args = args.get("lidar_support_mask", {}) or {}
        if isinstance(mask_args, bool):
            mask_args = {"enabled": mask_args}
        self.lidar_support_mask_enabled = bool(mask_args.get("enabled", False))
        self.lidar_support_mask_mode = str(
            mask_args.get("mode", "binary")
        ).lower()
        if self.lidar_support_mask_mode not in ("binary", "log_density"):
            raise ValueError(
                "lidar_support_mask.mode must be 'binary' or 'log_density'."
            )
        self.lidar_support_log_q95 = float(
            mask_args.get("log_density_q95", math.log1p(32.0))
        )
        if self.lidar_support_log_q95 <= 0.0:
            raise ValueError(
                "lidar_support_mask.log_density_q95 must be positive."
            )
        self.lidar_support_mask_dilation = int(
            mask_args.get("dilation_radius", 0)
        )
        if self.lidar_support_mask_dilation < 0:
            raise ValueError(
                "lidar_support_mask.dilation_radius must be non-negative."
            )
        self.lidar_support_apply_to_feature = bool(
            mask_args.get("apply_to_feature", True)
        )
        self.lidar_support_apply_to_ego_feature = bool(
            mask_args.get("apply_to_ego_feature", True)
        )
        voxel_size = args["voxel_size"]
        self.lidar_support_src_w = int(round(
            (self.lidar_range[3] - self.lidar_range[0]) / voxel_size[0]
        ))
        self.lidar_support_src_h = int(round(
            (self.lidar_range[4] - self.lidar_range[1]) / voxel_size[1]
        ))

        feature_dim = int(args.get("head_dim", 256))
        self.where2comm_multi_scale = False
        if self.fusion_method == "where2comm":
            fusion_args = dict(args["where2comm"])
            self.where2comm_multi_scale = bool(
                fusion_args.pop("multi_scale", False)
            )
            self.fusion_net = Where2commFusion(fusion_args)
        elif self.fusion_method == "ermvp":
            fusion_args = dict(args["ermvp"])
            fusion_args.setdefault("input_dim", feature_dim)
            fusion_args.setdefault("model_dim", feature_dim)
            self.fusion_net = ERMVPFusion(fusion_args)
        elif self.fusion_method == "v2xvit":
            fusion_args = dict(args["v2xvit"])
            fusion_args.setdefault("input_dim", feature_dim)
            fusion_args.setdefault("model_dim", feature_dim)
            self.fusion_net = V2XViTFusion(fusion_args)
        elif self.fusion_method == "cobevt":
            fusion_args = dict(args["cobevt"])
            fusion_args.setdefault("input_dim", feature_dim)
            fusion_args.setdefault("model_dim", feature_dim)
            self.fusion_net = CoBEVT(fusion_args)
        else:
            fusion_args = dict(args["ours"])
            fusion_args.setdefault("feat_dim", feature_dim)
            fusion_args.setdefault("cav_lidar_range", self.lidar_range)
            self.fusion_net = ComplementarityGuidedCLCFusion(
                feature_dim, fusion_args
            )
            if (
                self.fusion_net.require_support_cue
                and not self.lidar_support_mask_enabled
            ):
                raise ValueError(
                    "CGRF requires model.args.lidar_support_mask.enabled: true."
                )

        anchor_number = int(args["anchor_number"])
        self.cls_head = nn.Conv2d(feature_dim, anchor_number, kernel_size=1)
        self.reg_head = nn.Conv2d(
            feature_dim, 7 * anchor_number, kernel_size=1
        )

    def _run_fusion(
        self,
        feature,
        record_len,
        affine_matrix,
        support_mask=None,
        pairwise_t_matrix=None,
        confidence_maps=None,
        fusion_context=None,
    ):
        if isinstance(self.fusion_net, ComplementarityGuidedCLCFusion):
            return self.fusion_net(
                feature,
                record_len,
                affine_matrix,
                support_mask,
                pairwise_t_matrix,
            )
        return run_standard_fusion(
            self.fusion_net,
            feature,
            record_len,
            affine_matrix,
            confidence_maps=confidence_maps,
            fusion_context=fusion_context,
        )

    def _make_lidar_support_mask(self, processed_lidar, feature):
        """Map voxel point counts to the final fusion grid as Q_k."""
        if not self.lidar_support_mask_enabled:
            return None
        coords = processed_lidar.get("voxel_coords")
        if coords is None or coords.numel() == 0:
            return feature.new_zeros(
                (feature.shape[0], 1, feature.shape[-2], feature.shape[-1])
            )

        coords = coords.to(device=feature.device, dtype=torch.long)
        num_cav, _, height, width = feature.shape
        support = feature.new_zeros((num_cav, 1, height, width))
        batch_idx = coords[:, 0]
        y_idx = coords[:, 2]
        x_idx = coords[:, 3]
        valid = (
            (batch_idx >= 0)
            & (batch_idx < num_cav)
            & (y_idx >= 0)
            & (y_idx < self.lidar_support_src_h)
            & (x_idx >= 0)
            & (x_idx < self.lidar_support_src_w)
        )
        if not bool(valid.any().item()):
            return support

        point_counts = processed_lidar.get("voxel_num_points")
        if point_counts is not None:
            point_counts = point_counts.reshape(-1).to(
                device=feature.device, dtype=feature.dtype
            )
            if point_counts.numel() != coords.shape[0]:
                point_counts = None
        coords = coords[valid]
        if point_counts is not None:
            point_counts = point_counts[valid]
        batch_idx = coords[:, 0]
        y_idx = coords[:, 2]
        x_idx = coords[:, 3]
        y_feat = torch.div(
            y_idx * height, self.lidar_support_src_h, rounding_mode="floor"
        )
        x_feat = torch.div(
            x_idx * width, self.lidar_support_src_w, rounding_mode="floor"
        )
        flat_index = batch_idx * (height * width) + y_feat * width + x_feat
        flat_support = support.view(-1)
        if self.lidar_support_mask_mode == "log_density":
            if point_counts is None:
                point_counts = feature.new_ones((coords.shape[0],))
            flat_support.scatter_add_(
                0, flat_index, point_counts.clamp_min(0.0)
            )
            support = (
                torch.log1p(support) / self.lidar_support_log_q95
            ).clamp_(0.0, 1.0)
        else:
            flat_support[flat_index] = 1.0

        radius = self.lidar_support_mask_dilation
        if radius > 0:
            support = F.max_pool2d(
                support,
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )
        return support

    def _apply_lidar_support(self, feature, support_mask, record_len):
        if support_mask is None or not self.lidar_support_apply_to_feature:
            return feature
        hard_support = (support_mask > 0).to(dtype=feature.dtype)
        if not self.lidar_support_apply_to_ego_feature:
            hard_support = hard_support.clone()
            ego_indices = torch.cumsum(record_len, dim=0) - record_len
            hard_support[
                ego_indices.to(device=feature.device, dtype=torch.long)
            ] = 1
        return feature * hard_support

    def forward(self, data_dict):
        processed_lidar = data_dict["processed_lidar"]
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
            "record_len": data_dict["record_len"],
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        spatial_features = self.shrink(batch_dict["spatial_features_2d"])

        support_mask = self._make_lidar_support_mask(
            processed_lidar, spatial_features
        )
        spatial_features = self._apply_lidar_support(
            spatial_features, support_mask, data_dict["record_len"]
        )

        confidence_maps = None
        if getattr(self.fusion_net, "requires_confidence_maps", False):
            confidence_maps = self.cls_head(spatial_features)

        pairwise_t_matrix = data_dict["pairwise_t_matrix"]
        affine_matrix = normalize_pairwise_tfm_range(
            pairwise_t_matrix, self.lidar_range
        )
        fusion_context = {
            "pairwise_t_matrix": pairwise_t_matrix,
            "record_len": data_dict["record_len"],
        }
        for key in ("v2x_prior_encoding", "prior_encoding"):
            if key in data_dict:
                fusion_context[key] = data_dict[key]
        if self.where2comm_multi_scale:
            fused_feature = self.fusion_net.forward_multiscale(
                batch_dict["spatial_features"],
                confidence_maps,
                data_dict["record_len"],
                self.backbone,
                affine_matrix,
            )
            fused_feature = self.shrink(fused_feature)
        else:
            fused_feature = self._run_fusion(
                spatial_features,
                data_dict["record_len"],
                affine_matrix,
                support_mask=support_mask,
                pairwise_t_matrix=pairwise_t_matrix,
                confidence_maps=confidence_maps,
                fusion_context=fusion_context,
            )

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {"psm": psm, "rm": rm}
        if self.supervise_single:
            if confidence_maps is None:
                confidence_maps = self.cls_head(spatial_features)
            output_dict.update({
                "psm_single": confidence_maps,
                "rm_single": self.reg_head(spatial_features),
            })
        communication_getter = getattr(
            self.fusion_net, "get_communication_stats", None
        )
        if callable(communication_getter):
            output_dict["communication"] = communication_getter()
        return output_dict

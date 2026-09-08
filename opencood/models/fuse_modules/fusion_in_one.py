# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
"""Compatibility exports for cooperative BEV fusion modules.

The implementations live in focused modules in this package. Keep this file as
stable import surface for existing configs and models that import from
opencood.models.fuse_modules.fusion_in_one.
"""

from opencood.models.fuse_modules.common import (
    regroup,
    warp_feature,
)

from opencood.models.fuse_modules.ours import ComplementarityGuidedCLCFusion

from opencood.models.fuse_modules.ermvp import ERMVPFullBEVFusion, ERMVPFusion
from opencood.models.fuse_modules.cobevt import CoBEVT
from opencood.models.fuse_modules.v2xvit import V2XViTFusion
from opencood.models.fuse_modules.where2comm import Where2commFusion


def run_standard_fusion(
    fusion_net,
    feature,
    record_len,
    affine_matrix,
    confidence_maps=None,
    fusion_context=None,
):
    """Call a standard fusion module through one dataset-agnostic contract."""
    fusion_kwargs = {}
    if getattr(fusion_net, "requires_confidence_maps", False):
        fusion_kwargs["confidence_maps"] = confidence_maps
    if getattr(fusion_net, "requires_fusion_context", False):
        fusion_kwargs["fusion_context"] = fusion_context
    return fusion_net(feature, record_len, affine_matrix, **fusion_kwargs)

__all__ = [
    "regroup",
    "warp_feature",
    "ComplementarityGuidedCLCFusion",
    "ERMVPFusion",
    "ERMVPFullBEVFusion",
    "V2XViTFusion",
    "CoBEVT",
    "Where2commFusion",
    "run_standard_fusion",
]

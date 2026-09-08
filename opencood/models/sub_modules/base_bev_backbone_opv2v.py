# -*- coding: utf-8 -*-
"""OpenCOOD PointPillar BEV backbone used by the OPV2V reproduction path."""

import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackboneOpv2v(nn.Module):
    """The convolutional BEV backbone from the OpenCOOD PointPillar models."""

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        layer_nums = model_cfg.get("layer_nums", [])
        layer_strides = model_cfg.get("layer_strides", [])
        num_filters = model_cfg.get("num_filters", [])
        if not (len(layer_nums) == len(layer_strides) == len(num_filters)):
            raise ValueError(
                "layer_nums, layer_strides and num_filters must have equal length."
            )

        upsample_strides = model_cfg.get("upsample_strides", [])
        num_upsample_filters = model_cfg.get("num_upsample_filter", [])
        if len(upsample_strides) != len(num_upsample_filters):
            raise ValueError(
                "upsample_strides and num_upsample_filter must have equal length."
            )

        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for level, layer_num in enumerate(layer_nums):
            layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[level],
                    num_filters[level],
                    kernel_size=3,
                    stride=layer_strides[level],
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_filters[level], eps=1.0e-3, momentum=0.01
                ),
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_num):
                layers.extend([
                    nn.Conv2d(
                        num_filters[level],
                        num_filters[level],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        num_filters[level], eps=1.0e-3, momentum=0.01
                    ),
                    nn.ReLU(inplace=True),
                ])
            self.blocks.append(nn.Sequential(*layers))

            if upsample_strides:
                stride = upsample_strides[level]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[level],
                            num_upsample_filters[level],
                            int(stride),
                            stride=int(stride),
                            bias=False,
                        ),
                        nn.BatchNorm2d(
                            num_upsample_filters[level],
                            eps=1.0e-3,
                            momentum=0.01,
                        ),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    inverse_stride = int(np.round(1 / stride))
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[level],
                            num_upsample_filters[level],
                            inverse_stride,
                            stride=inverse_stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(
                            num_upsample_filters[level],
                            eps=1.0e-3,
                            momentum=0.01,
                        ),
                        nn.ReLU(inplace=True),
                    ))

        self.num_bev_features = sum(num_upsample_filters)
        if len(upsample_strides) > len(layer_nums):
            final_stride = int(upsample_strides[-1])
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    self.num_bev_features,
                    self.num_bev_features,
                    final_stride,
                    stride=final_stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    self.num_bev_features, eps=1.0e-3, momentum=0.01
                ),
                nn.ReLU(inplace=True),
            ))

    def forward(self, data_dict):
        spatial_features = data_dict["spatial_features"]
        x = spatial_features
        ups = []
        for level, block in enumerate(self.blocks):
            x = block(x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            data_dict[f"spatial_features_{stride}x"] = x
            ups.append(self.deblocks[level](x) if self.deblocks else x)

        x = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        data_dict["spatial_features_2d"] = x
        return data_dict

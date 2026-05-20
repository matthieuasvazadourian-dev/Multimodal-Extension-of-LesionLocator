"""
IntermediateFusionResEncUNet — shared-encoder intermediate feature-level fusion.

Subclasses ResidualEncoderUNet and overrides forward() to:
  1. Accept x: [B, 3, D, H, W]  (CT | PET | prompt)
  2. Split into (CT+prompt) and (PET+prompt) pairs, each [B, 2, D, H, W]
  3. Run self.encoder (shared weights) on each pair -> two skip lists
  4. Fuse each level's skip pair via a per-level fusion module
  5. Pass fused skips to self.decoder (unchanged)

Trainer passes input_channels=3 (num_image_channels + 1 for prompt).
We intercept that and build the backbone with input_channels=2 so the
CT-pretrained 2-channel stem loads with zero weight surgery.

Weight loading:
  - CT seed (TrainSeg800): strict=False — fusion keys will be in missing_keys
  - Trained intermediate ckpt: strict=True

Fusion variants: 'weighted' (WeightedSkipFusion, ~2C params/level) or
                 'mcsa' (MCSAFusionWrapper, windowed bidirectional cross-attention).
"""

from typing import List, Union, Tuple

import torch
import torch.nn as nn

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from lesionlocator.modules.fusion_modules import (
    WeightedSkipFusion,
    MCSAFusionWrapper,
)

# Minimum encoder level index at which MCSA is applied.
# Levels 0-1 (full/half res) have inter-window token counts of ~150K / ~19K
# — the O(L²) attention matrix would be ~90 GB / ~1.4 GB at batch_size=1.
# Level 2 (~2352 tokens, ~22 MB) is feasible and captures mid-level features
# where cross-modal fusion is most informative.
_MCSA_MIN_LEVEL = 2
_DEFAULT_WINDOW_SIZE = (4, 4, 4)


class IntermediateFusionResEncUNet(ResidualEncoderUNet):
    """
    Shared-encoder intermediate fusion U-Net for PET+CT segmentation.

    Parameters
    ----------
    input_channels : int
        Must be 3 (CT + PET + prompt), as passed by the trainer.
        The internal backbone is built with input_channels=2.
    num_classes : int
        Number of segmentation output classes.
    fusion_arch : str
        One of {'weighted', 'mcsa'}.
    **kwargs
        All other ResidualEncoderUNet constructor kwargs (conv_op, norm_op, etc.)
        forwarded unchanged to the backbone.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        fusion_arch: str = 'weighted',
        **kwargs,
    ):
        assert input_channels == 3, (
            f"IntermediateFusionResEncUNet expects input_channels=3 "
            f"(CT + PET + prompt), got {input_channels}"
        )
        assert fusion_arch in ('weighted', 'mcsa'), (
            f"fusion_arch must be one of weighted/mcsa, got {fusion_arch}"
        )

        # Build backbone with 2-channel stem (matches CT-pretrained checkpoint)
        super().__init__(input_channels=2, num_classes=num_classes, **kwargs)

        self.fusion_arch = fusion_arch

        # Determine per-level channel counts from encoder stages
        features = kwargs.get('features_per_stage', None)
        if features is None:
            raise ValueError("features_per_stage must be provided in architecture kwargs")
        if isinstance(features, int):
            n_stages = kwargs.get('n_stages', 7)
            features = [features] * n_stages
        features = list(features)
        n_levels = len(features)

        self.fusion_modules = nn.ModuleList(
            self._make_fusion_module(fusion_arch, features[k], k)
            for k in range(n_levels)
        )

    def _make_fusion_module(self, fusion_arch: str, channels: int, level: int) -> nn.Module:
        if fusion_arch == 'weighted':
            return WeightedSkipFusion(channels)
        elif fusion_arch == 'mcsa':
            if level < _MCSA_MIN_LEVEL:
                return WeightedSkipFusion(channels)
            return MCSAFusionWrapper(channels, _DEFAULT_WINDOW_SIZE)
        else:
            raise ValueError(f"Unknown fusion_arch: {fusion_arch}")

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        x : [B, 3, D, H, W]   channels = [CT, PET, prompt]
        """
        x_ct  = x[:, 0:1]    # [B, 1, D, H, W]
        x_pet = x[:, 1:2]    # [B, 1, D, H, W]
        p     = x[:, 2:3]    # [B, 1, D, H, W]

        inp_ct  = torch.cat([x_ct,  p], dim=1)   # [B, 2, D, H, W]
        inp_pet = torch.cat([x_pet, p], dim=1)   # [B, 2, D, H, W]

        skips_ct  = self.encoder(inp_ct)   # list of n_levels tensors
        skips_pet = self.encoder(inp_pet)

        skips_fused = [
            self.fusion_modules[k](skips_ct[k], skips_pet[k])
            for k in range(len(skips_ct))
        ]

        return self.decoder(skips_fused)

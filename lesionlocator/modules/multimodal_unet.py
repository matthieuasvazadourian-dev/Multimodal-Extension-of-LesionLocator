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
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from lesionlocator.modules.fusion_modules import (
    WeightedSkipFusion,
    MCSAFusionWrapper,
    SharedSpecificFusion,
    ShaSpecDomainClassifier,
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


class ShaSpecFusionResEncUNet(ResidualEncoderUNet):
    """
    ShaSpec (Wang et al., CVPR 2023) shared/specific fusion U-Net for PET+CT.

    Reuses the single pretrained ResEnc encoder as ShaSpec's SHARED encoder (run
    per modality), and adds lightweight per-level shared/specific projection heads
    (SharedSpecificFusion) plus a domain-classification head. This keeps params and
    activation memory close to the CT baseline — a full separate specific encoder
    per modality would ~3x the encoder cost, which the memory-constrained cluster
    can't absorb.

    Missing-modality robustness comes from training with random per-iteration
    modality dropout (`modality_dropout_p`): when a modality is hidden, its shared
    feature is substituted from the available modality and its specific feature is
    generated, so the substitution/generation path receives gradient. Two aux
    losses regularize the shared space:
      - distribution alignment: MSE(s_ct, s_pet) over levels (both-present iters)
      - domain classification:  CE on a modality classifier over the deepest
                                shared feature

    The combined aux loss for the current forward is stored on `self.last_aux_loss`
    (with `self.last_da` / `self.last_dc` for logging); the trainer adds it to the
    segmentation loss. It is only computed while training.

    CT-passthrough init: SharedSpecificFusion.fuse is zero-initialised, so at
    epoch 0 (both modalities present) the model is numerically identical to the
    CT-only pretrained backbone.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        modality_dropout_p: float = 0.5,
        lambda_da: float = 0.1,
        lambda_dc: float = 0.1,
        fusion_arch: str = 'shaspec',  # accepted and ignored (kept for kwarg uniformity)
        **kwargs,
    ):
        assert input_channels == 3, (
            f"ShaSpecFusionResEncUNet expects input_channels=3 (CT + PET + prompt), "
            f"got {input_channels}"
        )

        super().__init__(input_channels=2, num_classes=num_classes, **kwargs)

        self.fusion_arch = 'shaspec'
        self.modality_dropout_p = float(modality_dropout_p)
        self.lambda_da = float(lambda_da)
        self.lambda_dc = float(lambda_dc)

        features = kwargs.get('features_per_stage', None)
        if features is None:
            raise ValueError("features_per_stage must be provided in architecture kwargs")
        if isinstance(features, int):
            n_stages = kwargs.get('n_stages', 7)
            features = [features] * n_stages
        features = list(features)
        n_levels = len(features)

        # Named 'fusion_modules' so the trainer's existing intermediate-fusion
        # handling (force-trainable + higher LR + non-strict CT-seed load) applies.
        self.fusion_modules = nn.ModuleList(
            SharedSpecificFusion(features[k]) for k in range(n_levels)
        )
        # Domain classifier on the deepest level's shared feature.
        self.shaspec_domain_classifier = ShaSpecDomainClassifier(features[-1], n_domains=2)

        # Populated each forward; consumed by the trainer's loss.
        self.last_aux_loss = None
        self.last_da = None
        self.last_dc = None

        # Inference-time override: None -> both modalities present; 'ct'/'pet' ->
        # run single-modality (the absent one is substituted/generated). Set this on
        # the network to evaluate the missing-modality scenario. Ignored while training.
        self.inference_modality = None

    def _resolve_modality_mask(self) -> Tuple[bool, bool]:
        """Decide which modalities are present for this forward pass."""
        if self.training:
            if self.modality_dropout_p <= 0.0 or torch.rand(()).item() >= self.modality_dropout_p:
                return True, True
            drop_ct = torch.rand(()).item() < 0.5   # drop exactly one modality
            return (not drop_ct), drop_ct
        # Eval: honour an explicit single-modality request, else use both.
        if self.inference_modality == 'ct':
            return True, False
        if self.inference_modality == 'pet':
            return False, True
        return True, True

    def forward(self, x: torch.Tensor):
        """
        x : [B, 3, D, H, W]   channels = [CT, PET, prompt]
        """
        x_ct  = x[:, 0:1]
        x_pet = x[:, 1:2]
        p     = x[:, 2:3]

        has_ct, has_pet = self._resolve_modality_mask()

        skips_ct  = self.encoder(torch.cat([x_ct,  p], dim=1)) if has_ct  else None
        skips_pet = self.encoder(torch.cat([x_pet, p], dim=1)) if has_pet else None

        n_levels = len(skips_ct if has_ct else skips_pet)

        skips_fused = []
        shared_ct, shared_pet, both_flags = [], [], []
        for k in range(n_levels):
            f_ct  = skips_ct[k]  if has_ct  else None
            f_pet = skips_pet[k] if has_pet else None
            fused, s_ct, s_pet, both = self.fusion_modules[k](f_ct, f_pet, has_ct, has_pet)
            skips_fused.append(fused)
            shared_ct.append(s_ct)
            shared_pet.append(s_pet)
            both_flags.append(both)

        self._compute_aux_loss(shared_ct, shared_pet, both_flags, has_ct, has_pet)

        return self.decoder(skips_fused)

    def _compute_aux_loss(self, shared_ct, shared_pet, both_flags, has_ct, has_pet) -> None:
        """Compute ShaSpec distribution-alignment + domain-classification losses."""
        if not self.training:
            self.last_aux_loss = None
            self.last_da = None
            self.last_dc = None
            return

        ref = shared_ct[-1]
        zero = ref.new_zeros(())

        # Distribution alignment: only meaningful on iterations with both modalities.
        da = zero
        n_da = 0
        for s_ct, s_pet, both in zip(shared_ct, shared_pet, both_flags):
            if both:
                da = da + F.mse_loss(s_ct, s_pet)
                n_da += 1
        da = da / n_da if n_da > 0 else zero

        # Domain classification on the deepest shared feature(s) actually present.
        logits, labels = [], []
        if has_ct:
            logits.append(self.shaspec_domain_classifier(shared_ct[-1]))
            labels.append(torch.zeros(shared_ct[-1].shape[0], dtype=torch.long, device=ref.device))
        if has_pet:
            logits.append(self.shaspec_domain_classifier(shared_pet[-1]))
            labels.append(torch.ones(shared_pet[-1].shape[0], dtype=torch.long, device=ref.device))
        dc = F.cross_entropy(torch.cat(logits, 0), torch.cat(labels, 0)) if logits else zero

        self.last_da = da
        self.last_dc = dc
        self.last_aux_loss = self.lambda_da * da + self.lambda_dc * dc

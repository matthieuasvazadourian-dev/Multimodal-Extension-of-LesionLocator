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
    SpecificFeatureGenerator,
    SharedSpecificCombiner,
    DomainClassifier,
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
        One of {'weighted', 'mcsa'}. This governs how the two modalities'
        skip features are combined at each level and is unaffected by
        `missing_modality_robust` below.
    missing_modality_robust : bool
        Optional full-fidelity ShaSpec (Wang et al., CVPR 2023) add-on for
        robustness to a missing modality at inference. When False (default),
        forward() is exactly the original both-modalities-always-present path
        and no extra parameters are allocated. When True:
          - Two dedicated per-modality SPECIFIC encoders (`specific_encoder_ct`,
            `specific_encoder_pet`) are added, each a full independent-weight
            copy of the backbone encoder architecture (NOT shared with the
            main `self.encoder`, which already plays ShaSpec's SHARED-encoder
            role, weight-tied and run once per modality). This roughly
            doubles encoder-side compute/activation memory relative to
            weighted/mcsa alone — a deliberate quality-over-memory choice; see
            benchmark_fusion.md / re-benchmark before assuming this is
            unaffordable.
          - Training randomly drops one modality per iteration (probability
            `modality_dropout_p`). For the dropped modality: its SHARED
            feature is substituted by direct copy from the present modality
            (no learned adapter needed — the shared encoder is weight-tied
            and pulled together by the distribution-alignment loss below),
            and a `SpecificFeatureGenerator` fabricates its SPECIFIC feature
            from that substituted shared feature (its real specific encoder
            is not run, since there is nothing present to feed it).
          - Per modality, `SharedSpecificCombiner` merges (shared, specific)
            into one enriched feature; `fusion_modules[k]` (weighted or mcsa,
            completely unchanged) then fuses the two modalities' enriched
            features exactly as before.
          - Three auxiliary losses regularize this: distribution alignment
            (MSE between the real CT/PET *shared* features, both-present
            iterations only), domain classification (a classifier predicting
            which modality a shared feature came from), and generator
            reconstruction (MSE between each generator's output and the real
            specific feature it is trying to approximate, supervised only on
            both-present iterations where a real target exists). Their
            weighted sum is stashed on `self.last_aux_loss` each forward
            (only while training) for the trainer to add to the seg loss.
          - At inference, `self.inference_modality` ('ct'/'pet'/None) selects
            which modality to treat as missing; None (default) uses both.
        Does NOT change fusion_modules itself or its CT-passthrough init.
    **kwargs
        All other ResidualEncoderUNet constructor kwargs (conv_op, norm_op, etc.)
        forwarded unchanged to the backbone.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        fusion_arch: str = 'weighted',
        missing_modality_robust: bool = False,
        modality_dropout_p: float = 0.5,
        lambda_da: float = 0.1,
        lambda_dc: float = 0.1,
        lambda_gen: float = 0.1,
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

        self.missing_modality_robust = bool(missing_modality_robust)
        if self.missing_modality_robust:
            self.modality_dropout_p = float(modality_dropout_p)
            self.lambda_da = float(lambda_da)
            self.lambda_dc = float(lambda_dc)
            self.lambda_gen = float(lambda_gen)

            # Real per-modality specific encoders: independent weights, same
            # architecture as the shared backbone encoder. Built by harvesting
            # `.encoder` from a throwaway full ResidualEncoderUNet constructed
            # with the identical kwargs, so we never have to hand-replicate
            # ResidualEncoder's constructor signature.
            self.specific_encoder_ct = ResidualEncoderUNet(
                input_channels=2, num_classes=num_classes, **kwargs
            ).encoder
            self.specific_encoder_pet = ResidualEncoderUNet(
                input_channels=2, num_classes=num_classes, **kwargs
            ).encoder

            # gen_pet[k]: generates PET's specific feature from a shared feature (used when PET missing)
            # gen_ct[k]:  generates CT's specific feature from a shared feature (used when CT missing)
            self.gen_ct = nn.ModuleList(
                SpecificFeatureGenerator(features[k]) for k in range(n_levels)
            )
            self.gen_pet = nn.ModuleList(
                SpecificFeatureGenerator(features[k]) for k in range(n_levels)
            )
            self.combine_ct = nn.ModuleList(
                SharedSpecificCombiner(features[k]) for k in range(n_levels)
            )
            self.combine_pet = nn.ModuleList(
                SharedSpecificCombiner(features[k]) for k in range(n_levels)
            )
            self.domain_classifier = DomainClassifier(features[-1], n_domains=2)
            self.last_aux_loss = None
            self.last_da = None
            self.last_dc = None
            self.last_gen = None
            # Inference-time override: None -> both modalities present; 'ct'/'pet' ->
            # run single-modality (the absent one is substituted/generated). Ignored
            # while training.
            self.inference_modality = None

    def _make_fusion_module(self, fusion_arch: str, channels: int, level: int) -> nn.Module:
        if fusion_arch == 'weighted':
            return WeightedSkipFusion(channels)
        elif fusion_arch == 'mcsa':
            if level < _MCSA_MIN_LEVEL:
                return WeightedSkipFusion(channels)
            return MCSAFusionWrapper(channels, _DEFAULT_WINDOW_SIZE)
        else:
            raise ValueError(f"Unknown fusion_arch: {fusion_arch}")

    def _resolve_modality_mask(self) -> Tuple[bool, bool]:
        """Decide which modalities are present for this forward pass (robust mode only)."""
        if self.training:
            if self.modality_dropout_p <= 0.0 or torch.rand(()).item() >= self.modality_dropout_p:
                return True, True
            drop_ct = torch.rand(()).item() < 0.5   # drop exactly one modality
            return (not drop_ct), drop_ct
        if self.inference_modality == 'ct':
            return True, False
        if self.inference_modality == 'pet':
            return False, True
        return True, True

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        x : [B, 3, D, H, W]   channels = [CT, PET, prompt]
        """
        x_ct  = x[:, 0:1]    # [B, 1, D, H, W]
        x_pet = x[:, 1:2]    # [B, 1, D, H, W]
        p     = x[:, 2:3]    # [B, 1, D, H, W]

        if not self.missing_modality_robust:
            skips_ct  = self.encoder(torch.cat([x_ct,  p], dim=1))   # list of n_levels tensors
            skips_pet = self.encoder(torch.cat([x_pet, p], dim=1))
            skips_fused = [
                self.fusion_modules[k](skips_ct[k], skips_pet[k])
                for k in range(len(skips_ct))
            ]
            return self.decoder(skips_fused)

        has_ct, has_pet = self._resolve_modality_mask()
        skips_ct  = self.encoder(torch.cat([x_ct,  p], dim=1)) if has_ct  else None
        skips_pet = self.encoder(torch.cat([x_pet, p], dim=1)) if has_pet else None
        spec_ct   = self.specific_encoder_ct(torch.cat([x_ct,  p], dim=1))  if has_ct  else None
        spec_pet  = self.specific_encoder_pet(torch.cat([x_pet, p], dim=1)) if has_pet else None

        n_levels = len(self.fusion_modules)
        skips_fused = []
        for k in range(n_levels):
            shared_ct_k  = skips_ct[k]  if has_ct  else skips_pet[k]   # substitute = direct copy
            shared_pet_k = skips_pet[k] if has_pet else skips_ct[k]
            specific_ct_k  = spec_ct[k]  if has_ct  else self.gen_ct[k](shared_ct_k)
            specific_pet_k = spec_pet[k] if has_pet else self.gen_pet[k](shared_pet_k)
            comp_ct  = self.combine_ct[k](shared_ct_k,  specific_ct_k)
            comp_pet = self.combine_pet[k](shared_pet_k, specific_pet_k)
            skips_fused.append(self.fusion_modules[k](comp_ct, comp_pet))

        self._compute_aux_loss(skips_ct, skips_pet, spec_ct, spec_pet, has_ct, has_pet)

        return self.decoder(skips_fused)

    def _compute_aux_loss(self, skips_ct, skips_pet, spec_ct, spec_pet,
                           has_ct: bool, has_pet: bool) -> None:
        """ShaSpec distribution-alignment + domain-classification + generator-
        reconstruction aux losses. Operates on the RAW (pre-combine) shared and
        specific encoder features. Training only.
        """
        if not self.training:
            self.last_aux_loss = None
            self.last_da = None
            self.last_dc = None
            self.last_gen = None
            return

        n_levels = len(self.fusion_modules)
        ref = skips_ct[-1] if has_ct else skips_pet[-1]
        zero = ref.new_zeros(())

        # Distribution alignment: pull the two modalities' SHARED features
        # together so direct-copy substitution is a good approximation.
        if has_ct and has_pet:
            da = sum(F.mse_loss(skips_ct[k], skips_pet[k]) for k in range(n_levels)) / n_levels
        else:
            da = zero

        # Domain classification on the deepest shared feature(s) actually present.
        logits, labels = [], []
        if has_ct:
            logits.append(self.domain_classifier(skips_ct[-1]))
            labels.append(torch.zeros(skips_ct[-1].shape[0], dtype=torch.long, device=ref.device))
        if has_pet:
            logits.append(self.domain_classifier(skips_pet[-1]))
            labels.append(torch.ones(skips_pet[-1].shape[0], dtype=torch.long, device=ref.device))
        dc = F.cross_entropy(torch.cat(logits, 0), torch.cat(labels, 0))

        # Generator reconstruction: only supervisable on both-present iterations,
        # since that is the only time a real target specific feature exists.
        # Inputs/targets detached so this loss only trains the generators
        # themselves, not the shared/specific encoders that produced them.
        if has_ct and has_pet:
            gen = zero
            for k in range(n_levels):
                pred_pet_specific = self.gen_pet[k](skips_ct[k].detach())
                pred_ct_specific  = self.gen_ct[k](skips_pet[k].detach())
                gen = gen + F.mse_loss(pred_pet_specific, spec_pet[k].detach()) \
                          + F.mse_loss(pred_ct_specific,  spec_ct[k].detach())
            gen = gen / n_levels
        else:
            gen = zero

        self.last_da = da
        self.last_dc = dc
        self.last_gen = gen
        self.last_aux_loss = self.lambda_da * da + self.lambda_dc * dc + self.lambda_gen * gen


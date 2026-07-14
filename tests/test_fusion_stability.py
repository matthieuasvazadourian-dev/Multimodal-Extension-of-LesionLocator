"""
Multi-step training-stability test for the intermediate-fusion + ShaSpec
missing-modality-robustness combination.

Regression test for the mcsa+robust training divergence found on a real
50-epoch cluster run: validation loss climbed monotonically 0.4764->1.2172
over 18 epochs while the level-0 fusion gate collapsed to exactly 0, then
the run hung with no crash. Root cause (see lesionlocator/modules/
fusion_modules.py's BDSABlock and lesionlocator/training/train_segment.py's
_is_high_lr_fusion_param / per-group grad clipping): unbounded ShaSpec
combiner/generator outputs feeding MCSA's attention Q/K/V, driving softmax
saturation, amplified by a 10x LR on the ~33.5M-param attention block and a
single global gradient-clip norm that let the spike hide inside the diluted
combined norm.

Unlike tests/test_ct_passthrough.py (single forward pass, or one
backward() with no optimizer), this test builds the REAL training-time
optimizer configuration -- the same two param groups, LR multiplier, poly
LR schedule, and per-group gradient clipping used by train_segment.py's
train_cv_fold -- and runs enough real optimizer steps to catch a slow,
multi-step divergence that a single-step check cannot see. This is the one
test that would have failed on the shipped bug and passes after the fix.

Run on CPU (small patch, no AMP) so no GPU required.

Usage:
    python -m pytest tests/test_fusion_stability.py -v
"""

import pytest
import torch
import torch.optim as optim

from lesionlocator.training.train_segment import _is_high_lr_fusion_param

# Same real 7-stage nnUNet ResEnc config and CPU-safe patch size as
# tests/test_ct_passthrough.py -- see that file's comments for why a
# smaller patch collapses InstanceNorm3d at the deepest stage.
_ARCH_KWARGS = dict(
    n_stages=7,
    features_per_stage=[32, 64, 128, 256, 512, 512, 512],
    conv_op=torch.nn.Conv3d,
    kernel_sizes=[[3, 3, 3]] * 7,
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
    n_blocks_per_stage=[2, 2, 2, 2, 2, 2, 2],
    n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],
    conv_bias=True,
    norm_op=torch.nn.InstanceNorm3d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    dropout_op=None,
    dropout_op_kwargs=None,
    nonlin=torch.nn.LeakyReLU,
    nonlin_kwargs={'inplace': True},
    deep_supervision=False,
)
_PATCH_D, _PATCH_H, _PATCH_W = 64, 128, 128

# Same defaults train_segment.py's setup_training uses (learning_rate=1e-4,
# weight_decay=1e-5, fusion LR multiplier 10x, poly decay ^0.9, clip
# max_norm=12.0) so this test exercises the actual production configuration,
# not a toy substitute.
_LR = 1e-4
_WEIGHT_DECAY = 1e-5
_FUSION_LR_MULT = 10
_CLIP_MAX_NORM = 12.0
_N_STEPS = 40
_MAX_LOSS_GROWTH_FACTOR = 5.0


def _build_fusion(fusion_arch: str, missing_modality_robust: bool = True):
    from lesionlocator.modules.multimodal_unet import IntermediateFusionResEncUNet
    return IntermediateFusionResEncUNet(
        input_channels=3,
        num_classes=2,
        fusion_arch=fusion_arch,
        missing_modality_robust=missing_modality_robust,
        **_ARCH_KWARGS,
    )


def _build_real_optimizer(model: torch.nn.Module) -> optim.Optimizer:
    """Mirrors train_segment.py's setup_training optimizer construction for
    intermediate_fusion_mode exactly (same param-group split via the real
    _is_high_lr_fusion_param, same LR multiplier, same weight_decay)."""
    fusion_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and _is_high_lr_fusion_param(n)]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not _is_high_lr_fusion_param(n)]
    return optim.Adam(
        [{'params': backbone_params, 'lr': _LR},
         {'params': fusion_params,   'lr': _LR * _FUSION_LR_MULT}],
        weight_decay=_WEIGHT_DECAY,
    )


@pytest.mark.parametrize("fusion_arch", ["weighted", "mcsa"])
def test_multistep_training_stability(fusion_arch: str):
    """
    Run _N_STEPS real optimizer steps with the actual training-time
    configuration, cycling the modality mask every step (both-present /
    CT-only / PET-only) via the same _resolve_modality_mask monkeypatch
    pattern used in tests/test_ct_passthrough.py, so this deterministically
    exercises the shared-feature-aliasing path (multimodal_unet.py's
    forward, `shared_pet_k = skips_ct[k]` when PET is dropped) identified as
    part of the divergence mechanism.

    Asserts every step's loss and every parameter stay finite, and the loss
    doesn't blow up by more than _MAX_LOSS_GROWTH_FACTOR over the run. Not a
    convergence test (synthetic random data, no real segmentation target) --
    a numerical-stability test under the real optimizer/LR/clip config.
    """
    torch.manual_seed(0)
    model = _build_fusion(fusion_arch, missing_modality_robust=True)
    model.train()

    optimizer = _build_real_optimizer(model)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: (1.0 - step / max(_N_STEPS, 1)) ** 0.9
    )

    masks = [(True, True), (False, True), (True, False)]
    x = torch.randn(1, 3, _PATCH_D, _PATCH_H, _PATCH_W)

    losses = []
    for step in range(_N_STEPS):
        has_ct, has_pet = masks[step % len(masks)]
        model._resolve_modality_mask = lambda hc=has_ct, hp=has_pet: (hc, hp)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = out.pow(2).mean() + model.last_aux_loss

        assert torch.isfinite(loss), (
            f"fusion_arch='{fusion_arch}' step={step} has_ct={has_ct} has_pet={has_pet}: "
            f"non-finite loss {loss.item()}"
        )
        losses.append(loss.item())

        loss.backward()
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], max_norm=_CLIP_MAX_NORM)
        optimizer.step()
        scheduler.step()

        non_finite_params = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
        assert not non_finite_params, (
            f"fusion_arch='{fusion_arch}' step={step}: non-finite parameter(s) after step: "
            f"{non_finite_params}"
        )

    initial_loss = losses[0]
    final_loss = losses[-1]
    assert final_loss <= initial_loss * _MAX_LOSS_GROWTH_FACTOR, (
        f"fusion_arch='{fusion_arch}': loss grew from {initial_loss:.4f} to {final_loss:.4f} "
        f"over {_N_STEPS} steps (> {_MAX_LOSS_GROWTH_FACTOR}x) -- looks like divergence, "
        f"not noise. Full trajectory: {[f'{l:.4f}' for l in losses]}"
    )

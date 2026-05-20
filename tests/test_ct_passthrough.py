"""
CT-passthrough integration test.

Verifies that IntermediateFusionResEncUNet at initialization (all fusion
modules at CT-passthrough init) produces logits bit-identical (within fp32
tolerance) to a CT-only ResidualEncoderUNet on the same [CT, prompt] input.

Run on CPU (small patch) so no GPU required.

Usage:
    python -m pytest tests/test_ct_passthrough.py -v
"""

import pytest
import torch

# Minimal architecture kwargs that match the 7-stage nnUNet ResEnc config.
# These mirror plans.json for Dataset900/petct but at a tiny patch for fast CPU testing.
_ARCH_KWARGS = dict(
    n_stages=7,
    features_per_stage=[32, 64, 128, 256, 512, 512, 512],
    conv_op=torch.nn.Conv3d,
    kernel_sizes=[[3, 3, 3]] * 7,
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
    n_conv_per_stage=[2, 2, 2, 2, 2, 2, 2],
    n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],
    conv_bias=True,
    norm_op=torch.nn.InstanceNorm3d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    dropout_op=None,
    dropout_op_kwargs=None,
    nonlin=torch.nn.LeakyReLU,
    nonlin_kwargs={'inplace': True},
)


def _build_ct_only():
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    return ResidualEncoderUNet(
        input_channels=2,
        num_classes=2,
        **_ARCH_KWARGS,
    )


def _build_fusion(fusion_arch: str):
    from lesionlocator.modules.multimodal_unet import IntermediateFusionResEncUNet
    return IntermediateFusionResEncUNet(
        input_channels=3,
        num_classes=2,
        fusion_arch=fusion_arch,
        **_ARCH_KWARGS,
    )


@pytest.mark.parametrize("fusion_arch", ["weighted", "mcsa"])
def test_ct_passthrough_at_init(fusion_arch: str):
    """
    At init (CT-passthrough), fusion model on [CT, zeros_PET, prompt]
    must match CT-only model on [CT, prompt] within atol=1e-5.
    """
    torch.manual_seed(0)

    ct_model     = _build_ct_only().eval()
    fusion_model = _build_fusion(fusion_arch).eval()

    # Load CT-only weights into fusion model (strict=False, fusion keys are missing)
    missing, unexpected = fusion_model.load_state_dict(ct_model.state_dict(), strict=False)
    non_fusion_missing = [k for k in missing if 'fusion_modules' not in k]
    assert not non_fusion_missing, f"Non-fusion keys missing: {non_fusion_missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"

    # Tiny patch for CPU speed
    B, D, H, W = 1, 16, 16, 16
    torch.manual_seed(42)
    x_ct     = torch.randn(B, 1, D, H, W)
    x_prompt = torch.randn(B, 1, D, H, W)
    x_pet    = torch.randn(B, 1, D, H, W)  # content doesn't matter at CT-passthrough init

    with torch.inference_mode():
        out_ct    = ct_model(torch.cat([x_ct, x_prompt], dim=1))
        out_fusion = fusion_model(torch.cat([x_ct, x_pet, x_prompt], dim=1))

    # Deep supervision: both return lists; compare full-resolution head
    if isinstance(out_ct, (list, tuple)):
        out_ct = out_ct[0]
    if isinstance(out_fusion, (list, tuple)):
        out_fusion = out_fusion[0]

    max_diff = (out_fusion - out_ct).abs().max().item()
    assert torch.allclose(out_fusion, out_ct, atol=1e-5), (
        f"CT-passthrough failed for fusion_arch='{fusion_arch}'. "
        f"Max diff = {max_diff:.2e} (expected < 1e-5). "
        f"Check fusion module init — α_ct should be 1, α_pet should be 0, "
        f"all mixer weights should be 0."
    )

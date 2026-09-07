"""
Regression test for EmbeddingExtractor (lesionlocator_segment_and_track_embed.py)
under intermediate fusion.

Covers two bugs found in review of LesionLocator_track_embed:
  1. fusion_modules.N -- the actual CT+PET fused skip connection -- was never
     captured. Auto-detect only matched encoder.stages / decoder.stages, so
     the one tensor that genuinely combines both modalities was silently
     absent from every saved .npz.
  2. IntermediateFusionResEncUNet.forward calls the weight-tied shared
     encoder once per present modality (CT pass, then PET pass) within a
     single top-level forward. The old hook did a plain dict assignment, so
     the CT pass's encoder features were silently overwritten by the PET
     pass's -- every 'encoder_stages_N' key in a saved .npz was actually the
     PET encoding, mislabeled as if modality-agnostic.

Uses the same real 7-stage nnUNet ResEnc config and CPU-safe patch size as
tests/test_ct_passthrough.py (no GPU required).

Usage:
    python -m pytest tests/test_embedding_extraction.py -v
"""

import pytest
import torch

from lesionlocator.inference.lesionlocator_segment_and_track_embed import EmbeddingExtractor

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
# See test_ct_passthrough.py: smallest patch that survives all 7 stages
# without a 1-voxel bottleneck collapsing InstanceNorm3d.
_PATCH_D, _PATCH_H, _PATCH_W = 64, 128, 128


def _build_fusion(fusion_arch: str):
    from lesionlocator.modules.multimodal_unet import IntermediateFusionResEncUNet
    return IntermediateFusionResEncUNet(
        input_channels=3,
        num_classes=2,
        fusion_arch=fusion_arch,
        **_ARCH_KWARGS,
    )


def _build_ct_only():
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    return ResidualEncoderUNet(
        input_channels=2,
        num_classes=2,
        **_ARCH_KWARGS,
    )


def _random_input(channels):
    return torch.randn(1, channels, _PATCH_D, _PATCH_H, _PATCH_W)


@pytest.mark.parametrize("fusion_arch", ["weighted", "mcsa"])
def test_fusion_modules_captured_by_default(fusion_arch):
    """Auto-detect (mirrors the CLI default, no --embedding_layers) must
    always hook fusion_modules.N -- the actual fused skip -- regardless of
    which stage patterns were requested."""
    torch.manual_seed(0)
    model = _build_fusion(fusion_arch).eval()
    extractor = EmbeddingExtractor(model, layer_names=['decoder.stages.4'])  # the CLI's own default

    with torch.inference_mode():
        model(_random_input(3))
    emb = extractor.get_embeddings()

    fusion_keys = [k for k in emb if k.startswith('fusion_modules.')]
    assert len(fusion_keys) == len(model.fusion_modules), (
        f"Expected one fusion_modules.N key per fusion level, got {fusion_keys}"
    )


@pytest.mark.parametrize("fusion_arch", ["weighted", "mcsa"])
def test_encoder_passes_not_overwritten(fusion_arch):
    """Exact regression case: the shared encoder runs once for CT, once for
    PET, per forward. Both must survive, distinctly, under their own keys."""
    torch.manual_seed(0)
    model = _build_fusion(fusion_arch).eval()
    extractor = EmbeddingExtractor(model, layer_names=['encoder.stages.0'])
    extractor.pass_labels = ['ct', 'pet']  # what the predictor sets for every fusion case on this branch

    with torch.inference_mode():
        model(_random_input(3))
    emb = extractor.get_embeddings()

    assert 'encoder.stages.0_ct' in emb, emb.keys()
    assert 'encoder.stages.0_pet' in emb, emb.keys()
    assert 'encoder.stages.0' not in emb, "bare key must not survive a 2-invocation layer"
    assert not torch.allclose(emb['encoder.stages.0_ct'], emb['encoder.stages.0_pet']), (
        "CT and PET encoder passes must differ (different input channels); "
        "identical tensors would mean the two invocations collapsed back into one"
    )


def test_non_fusion_model_keeps_bare_keys():
    """A plain (non-fusion) model is unaffected: single invocation, and the
    predictor never sets pass_labels when intermediate_fusion_mode is False
    -> bare key, unchanged behavior."""
    torch.manual_seed(0)
    model = _build_ct_only().eval()
    extractor = EmbeddingExtractor(model, layer_names=['decoder.stages.4', 'encoder.stages.0'])

    with torch.inference_mode():
        model(_random_input(2))
    emb = extractor.get_embeddings()

    assert 'decoder.stages.4' in emb
    assert 'encoder.stages.0' in emb
    assert not any(k.endswith('_ct') or k.endswith('_pet') for k in emb)

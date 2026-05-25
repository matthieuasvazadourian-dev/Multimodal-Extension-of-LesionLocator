"""
Compute-cost benchmark for intermediate fusion variants.

Measures per variant: parameter count, FLOPs (fvcore), peak GPU memory,
and per-iteration wall-clock time (forward + backward). Reports both whole-model
and fusion-modules-only figures.

Usage:
  python -m lesionlocator.utilities.benchmark_fusion \
      --plans /scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec/plans.json \
      --dataset /scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec/dataset.json \
      --patch 192 224 224 \
      --arches weighted mcsa \
      --output /home/masva/benchmark_fusion.md
"""

import argparse
import gc
import json
import pydoc
import time
from typing import List, Tuple

import torch
import torch.nn as nn

from batchgenerators.utilities.file_and_folder_operations import load_json

from lesionlocator.modules.multimodal_unet import IntermediateFusionResEncUNet
from lesionlocator.utilities.plans_handling.plans_handler import PlansManager


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_model(plans_path: str, dataset_path: str, fusion_arch: str,
                 patch: Tuple[int, int, int]) -> IntermediateFusionResEncUNet:
    plans = load_json(plans_path)
    dataset_json = load_json(dataset_path)

    # Patch dataset to 2 channels (CT + PET) so PlansManager sees petct layout
    dataset_json = dict(dataset_json)
    dataset_json['channel_names'] = {'0': 'CT', '1': 'PET'}

    pm = PlansManager(plans)

    # Pick a 3D configuration matching the patch size.
    # petct modality overrides patch on ALL configs (including '2d'), so we must
    # explicitly skip '2d' to avoid building a Conv2d model on 5D input.
    config_name = None
    for name in pm.available_configurations:
        if name == '2d':
            continue
        cm = pm.get_configuration(name, modality='petct')
        if list(cm.patch_size) == list(patch):
            config_name = name
            break
    if config_name is None:
        three_d = [n for n in pm.available_configurations if n != '2d']
        config_name = three_d[0] if three_d else pm.available_configurations[0]
        print(f'[benchmark] No 3D config matched patch {patch}; using {config_name}.')

    cm = pm.get_configuration(config_name, modality='petct')
    arch_class_name = cm.network_arch_class_name
    arch_kwargs = dict(cm.network_arch_init_kwargs)
    arch_kwargs_req_import = cm.network_arch_init_kwargs_req_import

    # Resolve import strings
    for key in arch_kwargs_req_import:
        if arch_kwargs.get(key) is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

    # Inject fusion_arch — triggers IntermediateFusionResEncUNet construction
    arch_kwargs['fusion_arch'] = fusion_arch

    label_manager = pm.get_label_manager(dataset_json)
    num_classes = label_manager.num_segmentation_heads
    # input_channels=3: CT + PET + prompt (IntermediateFusionResEncUNet asserts this)
    model = IntermediateFusionResEncUNet(
        input_channels=3,
        num_classes=num_classes,
        **arch_kwargs,
    )
    return model


# ---------------------------------------------------------------------------
# FLOPs measurement via fvcore
# ---------------------------------------------------------------------------

def _count_flops(model: nn.Module, x: torch.Tensor) -> Tuple[float, float]:
    """Return (total_GFLOPs, fusion_GFLOPs). Returns (nan, nan) if fvcore unavailable."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print('[benchmark] fvcore not installed — skipping FLOPs measurement. '
              'Install with: pip install fvcore')
        return float('nan'), float('nan')

    model.eval()

    # Collect fusion skip shapes via hooks in the same forward pass used by FlopCountAnalysis.
    fusion_skip_shapes: List[Tuple[torch.Size, torch.Size]] = []

    def _hook(module, inputs, output):
        fusion_skip_shapes.append((inputs[0].shape, inputs[1].shape))

    handles = [m.register_forward_hook(_hook) for m in model.fusion_modules]
    try:
        with torch.no_grad():
            fca = FlopCountAnalysis(model, x)
            fca.unsupported_ops_warnings(True)
            total_flops = fca.total()
            unsupported = fca.unsupported_ops()
            if unsupported:
                import sys
                print(f'[benchmark] fvcore unsupported ops (may undercount): {unsupported}', file=sys.stderr)
            uncalled = fca.uncalled_modules()
            if uncalled:
                import sys
                print(f'[benchmark] fvcore uncalled modules: {uncalled}', file=sys.stderr)
    except Exception as e:
        for h in handles:
            h.remove()
        print(f'[benchmark] fvcore tracing failed ({e.__class__.__name__}: {e}) — skipping FLOPs.')
        return float('nan'), float('nan')
    for h in handles:
        h.remove()

    fusion_flops = 0
    for (module, (shape_ct, _shape_pet)) in zip(model.fusion_modules, fusion_skip_shapes):
        dummy = torch.zeros(shape_ct, device=x.device)
        try:
            f = FlopCountAnalysis(module, (dummy, dummy)).total()
        except Exception:
            f = 0
        fusion_flops += f

    return total_flops / 1e9, fusion_flops / 1e9


# ---------------------------------------------------------------------------
# Timing + memory
# ---------------------------------------------------------------------------

def _benchmark_timing(model: nn.Module, x: torch.Tensor,
                       n_warmup: int = 5, n_iters: int = 20) -> dict:
    """Run n_warmup forward+backward passes, then measure n_iters."""
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    def _step():
        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            # Deep supervision: out may be list; sum all outputs for a scalar loss
            if isinstance(out, (list, tuple)):
                loss = sum(o.mean() for o in out)
            else:
                loss = out.mean()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

    # Warm-up
    for _ in range(n_warmup):
        _step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_ms = 1000.0 * sum(times) / len(times)
    var_ms = sum((t * 1000.0 - mean_ms) ** 2 for t in times) / len(times)
    std_ms = var_ms ** 0.5
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        'iter_ms_mean': mean_ms,
        'iter_ms_std': std_ms,
        'peak_mem_gb': peak_mem_gb,
    }


def _benchmark_inference_timing(model: nn.Module, x: torch.Tensor,
                                 n_warmup: int = 5, n_iters: int = 20) -> dict:
    """Forward-only pass (no grad, no optimizer) — measures inference cost."""
    model.eval()

    def _step():
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            _ = model(x)

    for _ in range(n_warmup):
        _step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_ms = 1000.0 * sum(times) / len(times)
    var_ms  = sum((t * 1000.0 - mean_ms) ** 2 for t in times) / len(times)
    std_ms  = var_ms ** 0.5
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        'inf_ms_mean': mean_ms,
        'inf_ms_std': std_ms,
        'inf_peak_mem_gb': peak_mem_gb,
    }


# ---------------------------------------------------------------------------
# Full benchmark for one variant
# ---------------------------------------------------------------------------

def benchmark_variant(fusion_arch: str, plans_path: str, dataset_path: str,
                       patch: Tuple[int, int, int], device: str,
                       n_warmup: int, n_iters: int) -> dict:
    print(f'\n[benchmark] === {fusion_arch} ===')
    model = _build_model(plans_path, dataset_path, fusion_arch, patch).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    fusion_params = sum(p.numel() for n, p in model.named_parameters()
                        if 'fusion_modules' in n)
    print(f'  total_params={total_params:,}  fusion_params={fusion_params:,}')

    # Dummy input: [1, 3, D, H, W] — batch 1, channels CT+PET+prompt
    x = torch.randn(1, 3, *patch, device=device)

    # Timing before FLOPs: fvcore holds traced-graph tensors that survive gc.collect(),
    # contaminating peak-mem measurement. Run timing on a clean allocator, then FLOPs last.
    if device != 'cpu':
        timing = _benchmark_timing(model, x, n_warmup=n_warmup, n_iters=n_iters)
        print(f'  train: iter={timing["iter_ms_mean"]:.1f}±{timing["iter_ms_std"]:.1f} ms  '
              f'peak_mem={timing["peak_mem_gb"]:.2f} GB')
        inf_timing = _benchmark_inference_timing(model, x, n_warmup=n_warmup, n_iters=n_iters)
        print(f'  infer: iter={inf_timing["inf_ms_mean"]:.1f}±{inf_timing["inf_ms_std"]:.1f} ms  '
              f'peak_mem={inf_timing["inf_peak_mem_gb"]:.2f} GB')
    else:
        print('  [benchmark] Skipping timing on CPU.')
        timing     = {'iter_ms_mean': float('nan'), 'iter_ms_std': float('nan'),
                      'peak_mem_gb':  float('nan')}
        inf_timing = {'inf_ms_mean':  float('nan'), 'inf_ms_std':  float('nan'),
                      'inf_peak_mem_gb': float('nan')}

    total_gflops, fusion_gflops = _count_flops(model, x)
    print(f'  total_GFLOPs={total_gflops:.2f}  fusion_GFLOPs={fusion_gflops:.2f}')

    return {
        'fusion_arch': fusion_arch,
        'total_params_M': total_params / 1e6,
        'fusion_params_K': fusion_params / 1e3,
        'total_GFLOPs': total_gflops,
        'fusion_GFLOPs': fusion_gflops,
        **timing,
        **inf_timing,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _format_table(rows: List[dict]) -> str:
    keys = [
        ('fusion_arch',       'Variant',             '{:}'),
        ('total_params_M',    'Total params (M)',    '{:.3f}'),
        ('fusion_params_K',   'Fusion params (K)',   '{:.1f}'),
        ('total_GFLOPs',      'Total GFLOPs',        '{:.2f}'),
        ('fusion_GFLOPs',     'Fusion GFLOPs',       '{:.4f}'),
        ('peak_mem_gb',       'Train peak mem (GB)', '{:.2f}'),
        ('iter_ms_mean',      'Train iter (ms)',     '{:.1f}'),
        ('iter_ms_std',       'Train std (ms)',      '{:.1f}'),
        ('inf_peak_mem_gb',   'Infer peak mem (GB)', '{:.2f}'),
        ('inf_ms_mean',       'Infer iter (ms)',     '{:.1f}'),
        ('inf_ms_std',        'Infer std (ms)',      '{:.1f}'),
    ]
    header = '| ' + ' | '.join(label for _, label, _ in keys) + ' |'
    sep    = '| ' + ' | '.join('---' for _ in keys) + ' |'
    lines  = [header, sep]
    for row in rows:
        cells = []
        for key, _, fmt in keys:
            val = row.get(key, '')
            try:
                cells.append(fmt.format(val))
            except (ValueError, TypeError):
                cells.append(str(val))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _delta_row(r1: dict, r2: dict) -> dict:
    delta = {'fusion_arch': f'Δ ({r2["fusion_arch"]} − {r1["fusion_arch"]})'}
    for k in ('total_params_M', 'fusion_params_K', 'total_GFLOPs', 'fusion_GFLOPs',
              'peak_mem_gb', 'iter_ms_mean', 'iter_ms_std',
              'inf_peak_mem_gb', 'inf_ms_mean', 'inf_ms_std'):
        try:
            delta[k] = r2[k] - r1[k]
        except (TypeError, KeyError):
            delta[k] = float('nan')
    return delta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Benchmark fusion variants.')
    parser.add_argument('--plans',   required=True, help='Path to plans.json')
    parser.add_argument('--dataset', required=True, help='Path to dataset.json')
    parser.add_argument('--patch',   nargs=3, type=int, default=[192, 224, 224],
                        metavar=('D', 'H', 'W'), help='Patch size (default: 192 224 224)')
    parser.add_argument('--arches',  nargs='+', default=['weighted', 'mcsa'],
                        choices=['weighted', 'mcsa'],
                        help='Fusion architectures to benchmark (default: weighted mcsa)')
    parser.add_argument('--device',  default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--n_warmup', type=int, default=5)
    parser.add_argument('--n_iters',  type=int, default=20)
    parser.add_argument('--output',  default=None,
                        help='Write markdown report to this path (optional)')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[benchmark] CUDA not available, falling back to CPU.')
        args.device = 'cpu'

    patch = tuple(args.patch)
    results = []
    for arch in args.arches:
        r = benchmark_variant(
            fusion_arch=arch,
            plans_path=args.plans,
            dataset_path=args.dataset,
            patch=patch,
            device=args.device,
            n_warmup=args.n_warmup,
            n_iters=args.n_iters,
        )
        results.append(r)

    if len(results) == 2:
        results.append(_delta_row(results[0], results[1]))

    table = _format_table(results)
    print('\n## Benchmark results\n')
    print(table)

    import platform, socket
    try:
        import fvcore
        fvcore_ver = fvcore.__version__
    except Exception:
        fvcore_ver = 'n/a'
    env_lines = [
        f'- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}',
        f'- CUDA: {torch.version.cuda}',
        f'- PyTorch: {torch.__version__}',
        f'- fvcore: {fvcore_ver}',
        f'- Platform: {platform.platform()}',
        f'- Hostname: {socket.gethostname()}',
        f'- Patch: {patch}',
    ]
    env_block = '## Environment\n' + '\n'.join(env_lines) + '\n\n'

    if args.output:
        report = f'# Fusion benchmark — patch {patch}\n\n{env_block}{table}\n'
        with open(args.output, 'w') as f:
            f.write(report)
        print(f'\n[benchmark] Report written to {args.output}')
    else:
        print('\n## Environment')
        for line in env_lines:
            print(line)


if __name__ == '__main__':
    main()

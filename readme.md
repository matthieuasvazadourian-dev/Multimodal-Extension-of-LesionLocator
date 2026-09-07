# Multimodal Extension of LesionLocator — PET/CT Fusion

This repository builds upon and adapts the code from:

 [LesionLocator](https://github.com/MIC-DKFZ/LesionLocator) (Rokuss et al., CVPR 2025)
via the LETITIA fork (X. Chen, SDSC/EPFL).

> **Authors**: Maximilian Rokuss, Yannick Kirchhoff, Seval Akbal, Balint Kovacs, Saikat Roy, Constantin Ulrich, Tassilo Wald, Lukas T. Rotkopf, Heinz-Peter Schlemmer and Klaus Maier-Hein  
> **Paper**: [![CVPR](https://img.shields.io/badge/%20CVPR%202025%20-open%20access-blue.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Rokuss_LesionLocator_Zero-Shot_Universal_Tumor_Segmentation_and_Tracking_in_3D_Whole-Body_CVPR_2025_paper.html)

---

## Branches

| Branch | Strategy | How it works |
|---|---|---|
| `main` | **Early fusion** | PET added as a second input channel. The pretrained input convolution is widened 2→3 channels, PET filter initialised as a copy of the CT filter. Only that convolution is trained. |
| `intermediate-fusion` | **Feature-level fusion** | Input split back into (CT, prompt) and (PET, prompt). A weight-shared encoder runs on each: the two skip sets are fused per level, then decoded. Two modules: `weighted` (learned per-channel mixing) and `mcsa` (cross-modal attention, H2ASeg). |
| `shaspec-fusion` | **+ missing-modality robustness** | Adds ShaSpec shared/specific decomposition and modality dropout, so the model still runs when PET or CT is absent. |

---

## Installation

```bash
git clone https://github.com/matthieuasvazadourian-dev/Multimodal-Extension-of-LesionLocator
cd Multimodal-Extension-of-LesionLocator
git checkout shaspec-fusion

conda create -n lesionlocator python=3.12 -y
conda activate lesionlocator
pip install -e .
```

---

## Codebase

Folders in  `lesionlocator/`:

---

**`imageio/`** — reads and writes NIfTI volumes.

> **`simpleitk_reader_writer.py`** — where PET and CT are put onto a common voxel grid. Channel 0
> (CT) defines the reference geometry; PET and the label mask are resampled onto it at read time.
> This is why the two modalities need not be pre-registered.

**`preprocessing/`** — cropping, resampling and intensity normalisation before the network.

> **`normalization/default_normalization_schemes.py`** — the per-modality intensity schemes. CT is
> percentile-clipped and standardised; PET is Z-scored against a signal-derived body mask.
>
> **`preprocessors/default_preprocessor.py`** — the inference-time preprocessing chain, and where
> the PET body mask is built.

**`modules/`** — the network architectures.

> **`multimodal_unet.py`** — `IntermediateFusionResEncUNet`. Splits the input into per-modality
> streams, runs the weight-shared encoder on each, fuses the skip connections level by level, then
> decodes. Also hosts the ShaSpec robustness layer.
>
> **`fusion_modules.py`** — the fusion blocks themselves: `WeightedSkipFusion` (learned per-channel
> mixing), `MCSAFusion` (cross-modal attention), and the ShaSpec components.
>
> **`tracknet.py`** — the tracking network: registers consecutive timepoints and warps the lesion
> mask forward as a prompt.

**`training/`** — fine-tuning loops, data loading, cross-validation.

> **`train_segment.py`** — segmentation training entry point. Builds the model per `--fusion_arch`,
> handles the fine-tuning scope, runs patient-grouped cross-validation.
>
> **`train_track_c.py`** — TrackNet fine-tuning entry point.

**`inference/`** — prediction, sliding-window logic, evaluation.

> **`lesionlocator_segment_and_track.py`** — the main inference and evaluation entry point.
>
> **`lesionlocator_segment_and_track_embed.py`** — the same plus embedding export.

**`utilities/`** — configuration, metrics, prompts, and analysis tools.

> **`plans_handling/plans_handler.py`** — per-modality preprocessing configuration: target spacing,
> patch size, which normalisation applies to which channel.
>
> **`prompt_handling/`** — converts segmentation masks into point or box prompts.
>
> **`modality_grouping.py`** — detects which modalities each case actually has on disk, so one
> robust checkpoint can process a folder with missing scans.
>
> **`benchmark_fusion.py`** — parameters, FLOPs, peak memory and iteration time per fusion variant.
>
> **`display_fusion_weights.py`** — reads a trained checkpoint and reports how much each encoder
> level relied on CT versus PET.

---
Not in `lesionlocator/`:

`scripts/`

**Train**
```bash
./scripts/train_seg_petct.sh 0                                 # early fusion      (main)
./scripts/train_seg_petct_intermediate_weighted.sh 0           # weighted fusion
./scripts/train_seg_petct_intermediate_mcsa.sh 0               # attention fusion
./scripts/train_seg_petct_intermediate_weighted_robust.sh 0    # + ShaSpec         (shaspec)
./scripts/train_seg_petct_intermediate_mcsa_robust.sh 0
./scripts/train_track_petct.sh 0                               # TrackNet
```

**Evaluate**
```bash
./scripts/eval_seg_petct.sh 0
./scripts/eval_seg_petct_intermediate_{weighted,mcsa}.sh 0
./scripts/eval_track_petct.sh 0
```

**Missing modalities**
```bash
./scripts/eval_seg_missing_modality.sh 0         # route each subset to its own checkpoint
./scripts/eval_seg_missing_modality_robust.sh 0  # one robust ckpt: both / ct-only / pet-only
```

**Analysis**
```bash
./scripts/benchmark_fusion.sh        # params, FLOPs, peak memory, iteration time per variant
./scripts/display_fusion_weights.sh  # learned per-level CT vs PET split
```

---

## Flag reference

### `LesionLocator_create_prompt_json` 
A mask-to-prompt converter that takes only input and
output paths.

### `LesionLocator_segment` — single-timepoint segmentation

| Flag | Default | Meaning |
|---|---|---|
| `-i` | required | Input images. Folder with paired `_0000` (CT) / `_0001` (PET) files for `petct`. |
| `-p` | required | Prompts: instance masks (`.nii.gz`) or `.json` point/box files, named identically to their image. |
| `-o` | required | Output folder, created if absent. |
| `-m` | required | Model checkpoint folder. |
| `-t` | `box` | Prompt type: `point` or `box`. |
| `-f` | `0 1 2 3 4` | Checkpoint folds to load. |
| `--modality` | required | `ct`, `pet`, or `petct`. |
| `--fusion_arch` | auto | `weighted` or `mcsa`. `petct` only. Omit for early fusion. Auto-detected from the checkpoint. |
| `--missing_modality_robust` | auto | Set if the checkpoint was trained with ShaSpec. Auto-detected from the checkpoint. |
| `--force_inference_modality` | off | `ct` or `pet`. Measurement-only override forcing every case to drop the other modality even when both files are present. Requires `--missing_modality_robust`. |
| `-step_size` | `0.5` | Sliding-window step. Larger values are faster and less accurate. Maximum 1. |
| `-npp` | `3` | Number of preprocessing worker processes. |
| `-nps` | `3` | Number of export worker processes. |
| `-device` | `cuda` | `cuda`, `cpu`, or `mps`. |
| `--disable_tta` | off | Disable mirroring test-time augmentation. Faster, less accurate. |
| `--continue_prediction`, `--c` | off | Resume an aborted run; existing outputs are skipped. |
| `--visualize` | off | Save axial and coronal views of prediction vs ground truth. |
| `--verbose` | off | Verbose logging. |
| `--disable_progress_bar` | off | Disable the progress bar. |

### `LesionLocator_track` — segmentation + longitudinal tracking

| Flag | Default | Meaning |
|---|---|---|
| `-i` | required | Input images. Folder with paired `_0000` (CT) / `_0001` (PET) files for `petct`. |
| `-p` | required | Prompts: instance masks (`.nii.gz`) or `.json` point/box files, named identically to their image. |
| `-o` | required | Output folder, created if absent. |
| `-m` | required | Model checkpoint folder. |
| `-t` | `box` | Prompt type: `point` or `box`. |
| `-f` | `0 1 2 3 4` | Checkpoint folds to load. |
| `--modality` | required | `ct`, `pet`, or `petct`. |
| `--track` | off | Enable tracking via TrackNet. Without it this behaves as plain segmentation. |
| `--adaptive_mode` | off | Per lesion, pick tracking or fresh segmentation by Dice/NSD. Falls back to segmentation when tracking scores poorly. |
| `--empty_prompt` | off | Run the predictor with an empty prompt. |
| `--fusion_arch` | auto | `weighted` or `mcsa`. `petct` only. Omit for early fusion. Auto-detected from the checkpoint. |
| `--missing_modality_robust` | auto | Set if the checkpoint was trained with ShaSpec. Auto-detected from the checkpoint. |
| `--force_inference_modality` | off | `ct` or `pet`. Measurement-only override forcing every case to drop the other modality even when both files are present. Requires `--missing_modality_robust`. |
| `-step_size` | `0.5` | Sliding-window step. Larger values are faster and less accurate. Maximum 1. |
| `-npp` | `3` | Number of preprocessing worker processes. |
| `-nps` | `3` | Number of export worker processes. |
| `-device` | `cuda` | `cuda`, `cpu`, or `mps`. |
| `--disable_tta` | off | Disable mirroring test-time augmentation. Faster, less accurate. |
| `--continue_prediction`, `--c` | off | Resume an aborted run; existing outputs are skipped. |
| `--visualize` | off | Save axial and coronal views of prediction vs ground truth. |
| `--verbose` | off | Verbose logging. |
| `--disable_progress_bar` | off | Disable the progress bar. |

### `LesionLocator_track_embed` — tracking + embedding export


| Flag | Default | Meaning |
|---|---|---|
| `-i` | required | Input images. Folder with paired `_0000` (CT) / `_0001` (PET) files for `petct`. |
| `-p` | required | Prompts: instance masks (`.nii.gz`) or `.json` point/box files, named identically to their image. |
| `-o` | required | Output folder, created if absent. |
| `-m` | required | Model checkpoint folder. |
| `-t` | `box` | Prompt type: `point` or `box`. |
| `-f` | `0 1 2 3 4` | Checkpoint folds to load. |
| `--modality` | required | `ct`, `pet`, or `petct`. |
| `--track` | off | Enable tracking via TrackNet. |
| `--adaptive_mode` | off | Per lesion, pick tracking or fresh segmentation by Dice/NSD. |
| `--extract_embeddings` | off | Export intermediate features from the segmentation and tracking networks. |
| `--embedding_output_folder` | — | Where to write the `.npz` files. Required with `--extract_embeddings`. |
| `--embedding_layers` | all stages | Layers to hook by substring, e.g. `decoder.stages.2`. The default hooks every encoder and decoder stage. Fused skip connections are hooked independently of this flag. |
| `--lesion_focus` | off | Crop around the prompt centroid instead of running the full volume, for lesions exceeding the patch size. Applies to the segmentation branch; the tracking branch runs on the full volume. |
| `--crop_size` | `128` | Edge length of that crop. Scripts use `64`. Read only when `--lesion_focus` is set. |
| `--fusion_arch` | auto | `weighted` or `mcsa`. `petct` only. Omit for early fusion. Auto-detected from the checkpoint. Available on the `intermediate-fusion` and `shaspec-fusion` branches. |
| `--missing_modality_robust` | auto | Set if the checkpoint was trained with ShaSpec. Auto-detected from the checkpoint. Available on the `shaspec-fusion` branch. |
| `--force_inference_modality` | off | `ct` or `pet`. Measurement-only override forcing every case to drop the other modality even when both files are present. Requires `--missing_modality_robust`. Available on the `shaspec-fusion` branch. |
| `-step_size` | `0.5` | Sliding-window step. Larger values are faster and less accurate. Maximum 1. |
| `-npp` | `3` | Number of preprocessing worker processes. |
| `-nps` | `3` | Number of export worker processes. |
| `-device` | `cuda` | `cuda`, `cpu`, or `mps`. |
| `--disable_tta` | off | Disable mirroring test-time augmentation. Faster, less accurate. |
| `--continue_prediction`, `--c` | off | Resume an aborted run; existing outputs are skipped. |
| `--visualize` | off | Save axial and coronal views of prediction vs ground truth. |
| `--verbose` | off | Verbose logging. |
| `--disable_progress_bar` | off | Disable the progress bar. |

### `LesionLocator_train_segment` — segmentation fine-tuning

| Flag | Default | Meaning |
|---|---|---|
| `-i` | required | Input images. Folder with paired `_0000` (CT) / `_0001` (PET) files for `petct`. |
| `-p` | required | Prompts: instance masks (`.nii.gz`) or `.json` point/box files, named identically to their image. |
| `-o` | required | Output folder, created if absent. |
| `-m` | required | Model checkpoint folder. |
| `-t` | `box` | Prompt type: `point` or `box`. |
| `-f` | `0 1 2 3 4` | Checkpoint folds to load. |
| `--modality` | required | `ct`, `pet`, or `petct`. |
| `-iv` | — | Validation images, for per-epoch Dice and visualisation. |
| `-pv` | — | Validation prompts. |
| `--epochs` | `1` | Training epochs. Scripts use 50. |
| `--lr` | `1e-4` | Learning rate. Scripts use `5e-5`. |
| `--batch_size` | `3` | Batch size. Scripts use 1 for `petct`. |
| `--num_workers` | `4` | DataLoader workers. |
| `--train_fold` | `0` | Which cross-validation fold to train. Distinct from `-f`, which selects checkpoint folds to load. |
| `--ckpt_path` | none | Where to write inference-compatible checkpoints. Without it, none are produced. |
| `--finetune` | `all` | `encoder`, `decoder`, `all`, or `first_conv`. `first_conv` freezes everything except the widened input convolution. |
| `--eval_every` | `5` | Run test-set Dice every N epochs, and always on the last. `0` disables it. |
| `--cache` | off | Cache preprocessed samples in RAM after epoch 1. Removes repeated NIfTI reads. Roughly 200 MB per PET+CT case. |
| `--fusion_arch` | none | `weighted` or `mcsa`. Omit for early fusion. |
| `--missing_modality_robust` | off | Add the ShaSpec layer in front of the chosen `--fusion_arch`. |
| `--modality_dropout_p` | `0.5` | Probability of dropping one modality per iteration. Robust mode only. |
| `--lambda_da` | `0.1` | Distribution-alignment loss weight. Robust mode only. |
| `--lambda_dc` | `0.1` | Domain-classification loss weight. Robust mode only. |
| `--lambda_gen` | `0.1` | Generator-reconstruction loss weight. Robust mode only. |
| `--track` | off | Inherited from the shared parser; not used in training. |
| `--adaptive_mode` | off | Inherited from the shared parser; not used in training. |
| `-step_size` | `0.5` | Sliding-window step. Larger values are faster and less accurate. Maximum 1. |
| `-npp` | `3` | Number of preprocessing worker processes. |
| `-nps` | `3` | Number of export worker processes. |
| `-device` | `cuda` | `cuda`, `cpu`, or `mps`. |
| `--disable_tta` | off | Disable mirroring test-time augmentation. Faster, less accurate. |
| `--continue_prediction`, `--c` | off | Resume an aborted run; existing outputs are skipped. |
| `--visualize` | off | Save axial and coronal views of prediction vs ground truth. |
| `--verbose` | off | Verbose logging. |
| `--disable_progress_bar` | off | Disable the progress bar. |

### `LesionLocator_train_track` — TrackNet fine-tuning

| Flag | Default | Meaning |
|---|---|---|
| `-i` | required | Input images. Folder with paired `_0000` (CT) / `_0001` (PET) files for `petct`. |
| `-p` | required | Prompts: instance masks (`.nii.gz`) or `.json` point/box files, named identically to their image. |
| `-o` | required | Output folder, created if absent. |
| `-m` | required | Model checkpoint folder. |
| `-t` | `box` | Prompt type: `point` or `box`. |
| `-f` | `0 1 2 3 4` | Checkpoint folds to load. |
| `--modality` | required | `ct`, `pet`, or `petct`. |
| `-iv` | — | Validation images, for per-epoch Dice and visualisation. |
| `-pv` | — | Validation prompts. |
| `--epochs` | `1` | Training epochs. Scripts use 50. |
| `--lr` | `1e-4` | Learning rate. Scripts use `5e-5`. |
| `--batch_size` | `3` | Batch size. Scripts use 1 for `petct`. |
| `--num_workers` | `4` | DataLoader workers. |
| `--train_fold` | `0` | Which cross-validation fold to train. Distinct from `-f`, which selects checkpoint folds to load. |
| `--ckpt_path` | none | Where to write inference-compatible checkpoints. Without it, none are produced. |
| `--finetune` | `all` | `reg_net` (registration only), `unet` (segmentation only), or `all`. |
| `--reinit` | off | Reinitialise weights before training rather than starting from the checkpoint. |
| `--gradient_accumulation_steps` | `1` | Effective batch size = `batch_size` × this value. |
| `-step_size` | `0.5` | Sliding-window step. Larger values are faster and less accurate. Maximum 1. |
| `-npp` | `3` | Number of preprocessing worker processes. |
| `-nps` | `3` | Number of export worker processes. |
| `-device` | `cuda` | `cuda`, `cpu`, or `mps`. |
| `--disable_tta` | off | Disable mirroring test-time augmentation. Faster, less accurate. |
| `--continue_prediction`, `--c` | off | Resume an aborted run; existing outputs are skipped. |
| `--visualize` | off | Save axial and coronal views of prediction vs ground truth. |
| `--verbose` | off | Verbose logging. |
| `--disable_progress_bar` | off | Disable the progress bar. |


### Environment variables

| Variable | Meaning |
|---|---|
| `LesionLocator_compile` | `1` enables `torch.compile`. Disabled in code for missing-modality-robust variants, whose forward pass branches on which modality was dropped. |
| `LESIONLOCATOR_TRACKNET_VIS_DIR` | Where TrackNet writes debug visualisations. Defaults to `./tracknet_visualizations`. |
| `PYTORCH_CUDA_ALLOC_CONF` | Scripts use `expandable_segments:True` for training, `max_split_size_mb:64` for inference. |
| `MALLOC_ARENA_MAX`, `MALLOC_TRIM_THRESHOLD_`, `MALLOC_MMAP_THRESHOLD_` | Set by the training scripts to contain host-RAM growth in the preprocessing workers. |

---

## Extracting fused embeddings

```bash
LesionLocator_track_embed \
  -i .../Dataset901_USZMelanomaPETCT/imagesTr \
  -p .../Dataset901_USZMelanomaPETCT/labelsTr \
  -m /path/to/ckpt_root -o /path/to/output \
  -f 0 -t point -npp 1 -nps 1 \
  --modality petct --track --adaptive_mode \
  --lesion_focus --crop_size 64 \
  --extract_embeddings \
  --embedding_layers decoder.stages.2 \
  --embedding_output_folder /path/to/embeddings
```


### `.npz` contents

| Key | Contents |
|---|---|
| `fusion_modules_<N>` | Fused skip connection at level `N` — the tensor combining both modalities. Written on intermediate-fusion checkpoints, and hooked independently of `--embedding_layers`. |
| `decoder_stages_<N>` | Decoder stage feature tensor `[B, C, D, H, W]`, float32 |
| `encoder_stages_<N>` | Encoder stage feature tensor. On intermediate-fusion checkpoints the shared encoder runs once per modality, so these keys carry a `_ct` / `_pet` suffix, or `_pass0` / `_pass1` when modality labels are unavailable. |
| `dice` | Dice of this lesion's prediction vs ground truth |
| `bbox` | Crop box `[z0,z1,y0,y1,x0,x1]`, written when lesion-focused cropping was applied |
| `center`, `center_physical` | Lesion centroid, voxels and mm |
| `crop_size`, `data_physical_size` | Crop size used; full volume extent in mm |

---

## Citation

```bibtex
@InProceedings{Rokuss_2025_CVPR,
    author    = {Rokuss, Maximilian and Kirchhoff, Yannick and Akbal, Seval and Kovacs, Balint
                 and Roy, Saikat and Ulrich, Constantin and Wald, Tassilo and Rotkopf, Lukas T.
                 and Schlemmer, Heinz-Peter and Maier-Hein, Klaus},
    title     = {LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D
                 Whole-Body Imaging},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    year      = {2025}, pages = {30872-30885}
}

@article{lu2024h2aseg,
    title   = {H2ASeg: Hierarchical Adaptive Interaction and Weighting Network for Tumor
               Segmentation in PET/CT Images},
    author  = {Lu, Jinpeng and Chen, Jingyun and Cai, Linghan and Jiang, Songhan
               and Zhang, Yongbing},
    journal = {arXiv preprint arXiv:2403.18339}, year = {2024}
}

@InProceedings{Wang_2023_CVPR,
    author    = {Wang, Hu and Chen, Yuanhong and Ma, Congbo and Avery, Jodie and Hull, Louise
                 and Carneiro, Gustavo},
    title     = {Multi-Modal Learning With Missing Modality via Shared-Specific Feature
                 Modelling},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
                 Recognition (CVPR)},
    year      = {2023}, pages = {15878-15887}
}
```

"""
Per-case CT/PET file grouping with auto-detected single-modality fallback.

Approach B for missing-modality robustness (contrast `route_by_modality.py`'s
Approach A, which dispatches whole subsets to separate unimodal checkpoints):
here a single `missing_modality_robust` fusion checkpoint handles a mixed
folder directly, one case at a time. For each case, if a channel file is
genuinely absent, the missing slot in the returned file list is filled with
the PRESENT file's own path (a duplicate) rather than left missing or
zero-filled, and the case is tagged 'ct'/'pet' so the caller can set
`network.inference_modality` before that case's forward pass.

The duplicate is provably safe to feed through preprocessing: it is only
ever read by the shared/per-channel preprocessing steps (resampling,
per-channel normalization), which don't care about cross-modality content and
whose outcome for the *present* channel is unaffected by what sits in the
other slot. It is never read by the network for the absent modality —
`IntermediateFusionResEncUNet.forward` only encodes a channel when its
modality is flagged present (see `lesionlocator/modules/multimodal_unet.py`,
`skips_ct = ... if has_ct else None`). Zero-filling instead would risk NaNs
(e.g. `ZScoreNormalization`'s body-mask on an all-zero channel) for no benefit,
since the content is discarded either way.

Pure stdlib (no torch/SimpleITK) so this is unit-testable without the heavy
inference dependencies — see tests/test_modality_detection.py.
"""

import os
from typing import List, Optional, Tuple


def group_petct_cases_by_modality(all_files: List[str], source_folder: str,
                                  file_ending: str,
                                  missing_modality_robust: bool) -> List[Tuple[str, List[str], Optional[str]]]:
    """Group a flat file listing into per-case (case_id, file_group, inference_modality) triples.

    `all_files` is every image file in `source_folder` (as produced by
    `subfiles(source_folder, suffix=file_ending, join=True)`). CT is channel 0
    (`_0000`), PET is channel 1 (`_0001`) — the convention used throughout this
    project (dataset_json channel_names, `route_by_modality.py`).

    `case_id` is returned explicitly (rather than making callers reverse-engineer
    it from `file_group[0]`'s filename) so output naming/resume bookkeeping is
    invariant to *which* modality happened to be present for a case — deriving
    it from file_group[0] instead would make a case's output basename silently
    change if it gained/lost a modality between runs, breaking
    --continue_prediction (concretely: a case first run CT-only, whose PET file
    later appears, must be recognized as the *same case* needing a fusion
    rerun, not treated as already-done just because file_group[0] happened to
    stay the CT file both times).

    Returns one entry per case with both channels present or resolvable:
    - both present: (case_id, [ct_file, pet_file], None)
    - one missing, missing_modality_robust=True: (case_id, [x, x], 'ct'|'pet')
      where x is the present file's path duplicated into both slots
    - one missing, missing_modality_robust=False: raises ValueError — no
      inference code path exists for a dropped modality on that checkpoint
    - neither present for a detected case id (shouldn't normally happen since
      case ids are derived from existing files): skipped with a warning
    """
    case_ids = sorted({
        os.path.basename(f)[:-len('_0000' + file_ending)]
        for f in all_files if os.path.basename(f).endswith('_0000' + file_ending)
    } | {
        os.path.basename(f)[:-len('_0001' + file_ending)]
        for f in all_files if os.path.basename(f).endswith('_0001' + file_ending)
    })
    groups = []
    for case_id in case_ids:
        ct_file = os.path.join(source_folder, f'{case_id}_0000{file_ending}')
        pet_file = os.path.join(source_folder, f'{case_id}_0001{file_ending}')
        has_ct, has_pet = os.path.isfile(ct_file), os.path.isfile(pet_file)
        if has_ct and has_pet:
            groups.append((case_id, [ct_file, pet_file], None))
        elif not missing_modality_robust:
            missing = 'PET (_0001)' if has_ct else 'CT (_0000)'
            raise ValueError(
                f"Case '{case_id}' is missing {missing}, and the loaded "
                "checkpoint is not missing_modality_robust (no single-"
                "modality inference code path exists for it). Provide the "
                "missing file, or load a checkpoint trained with "
                "--missing_modality_robust."
            )
        elif has_ct:
            groups.append((case_id, [ct_file, ct_file], 'ct'))
        elif has_pet:
            groups.append((case_id, [pet_file, pet_file], 'pet'))
        else:
            print(f"[WARNING] Case '{case_id}' has neither CT nor PET file present — skipping.")
    return groups

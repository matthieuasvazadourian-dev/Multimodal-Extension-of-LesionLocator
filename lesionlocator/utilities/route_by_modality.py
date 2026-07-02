"""
Route cases in a mixed PET/CT input directory by which modalities are present.

Approach A ("on/off switch") for missing-modality robustness: instead of feeding
a single-modality case into the PET+CT fusion model (which structurally requires
both channels), detect per case which modalities exist and dispatch each subset
to the appropriate model:

    CT + PET present  -> PET+CT fusion model   (--modality petct)
    CT only           -> CT LesionLocator      (--modality ct)
    PET only          -> PET LesionLocator     (--modality pet)

Channel convention (matches the paired dataset layout and the inference code's
_group_input_files_by_case): `<case>_0000` = CT, `<case>_0001` = PET.

This helper only partitions the input into three symlink subdirectories and
reports the counts; the shell wrapper runs the matching inference command on
each non-empty subset. No case is copied — only symlinked.

For the PET-only subset the PET file (`_0001`) is symlinked as `_0000`, because
the unimodal PET model expects its single input on channel 0.

If `--prompts` is given, the matching per-case prompt file (named `<case_id>`
+ file ending, or `<case_id>.json`, with no channel suffix) is symlinked into
a parallel `labelsTr` subdirectory for each subset. This is required because
the inference entrypoint asserts that the number of prompt files equals the
number of image cases in the folder it's given — the full, unfiltered prompt
directory would fail that check against a routed subset.

Usage:
    python -m lesionlocator.utilities.route_by_modality \
        --images /path/to/imagesTr --outdir /tmp/route [--file-ending .nii.gz] \
        [--prompts /path/to/labelsTr]

Prints a JSON summary {"petct": n, "ct": n, "pet": n} to stdout.
"""

import argparse
import json
import os
from collections import defaultdict


def _case_id(basename: str, file_ending: str) -> str:
    """Strip the `_NNNN` channel suffix and file ending -> case id."""
    stem = basename[: -len(file_ending)] if basename.endswith(file_ending) else basename
    # channel suffix is the trailing _NNNN
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def partition_cases(images_dir: str, file_ending: str) -> dict:
    """
    Group files by case id and classify each case by modality presence.

    Returns dict: case_id -> {'ct': ct_path or None, 'pet': pet_path or None}.
    """
    cases = defaultdict(lambda: {"ct": None, "pet": None})
    for fname in sorted(os.listdir(images_dir)):
        if not fname.endswith(file_ending):
            continue
        stem = fname[: -len(file_ending)]
        if stem.endswith("_0000"):
            cases[_case_id(fname, file_ending)]["ct"] = os.path.join(images_dir, fname)
        elif stem.endswith("_0001"):
            cases[_case_id(fname, file_ending)]["pet"] = os.path.join(images_dir, fname)
    return dict(cases)


def _symlink(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def _route_prompt(prompts_dir: str, outdir: str, subset: str, cid: str, file_ending: str) -> None:
    """Symlink the per-case prompt file (mask or json) into <outdir>/<subset>/labelsTr/."""
    mask_src = os.path.join(prompts_dir, f"{cid}{file_ending}")
    json_src = os.path.join(prompts_dir, f"{cid}.json")
    if os.path.exists(mask_src):
        _symlink(mask_src, os.path.join(outdir, subset, "labelsTr", f"{cid}{file_ending}"))
    elif os.path.exists(json_src):
        _symlink(json_src, os.path.join(outdir, subset, "labelsTr", f"{cid}.json"))
    else:
        raise FileNotFoundError(f"No prompt file found for case {cid} in {prompts_dir}")


def route(images_dir: str, outdir: str, file_ending: str, prompts_dir: str = None) -> dict:
    """Build petct/ct/pet symlink subdirs under outdir. Returns per-subset counts."""
    cases = partition_cases(images_dir, file_ending)
    counts = {"petct": 0, "ct": 0, "pet": 0}

    for cid, mods in cases.items():
        has_ct, has_pet = mods["ct"] is not None, mods["pet"] is not None
        if has_ct and has_pet:
            subset = "petct"
            _symlink(mods["ct"],  os.path.join(outdir, subset, "imagesTr", f"{cid}_0000{file_ending}"))
            _symlink(mods["pet"], os.path.join(outdir, subset, "imagesTr", f"{cid}_0001{file_ending}"))
        elif has_ct:
            subset = "ct"
            _symlink(mods["ct"], os.path.join(outdir, subset, "imagesTr", f"{cid}_0000{file_ending}"))
        elif has_pet:
            subset = "pet"
            # normalize PET onto channel 0 for the unimodal PET model
            _symlink(mods["pet"], os.path.join(outdir, subset, "imagesTr", f"{cid}_0000{file_ending}"))
        else:
            continue
        counts[subset] += 1
        if prompts_dir is not None:
            _route_prompt(prompts_dir, outdir, subset, cid, file_ending)

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="Input imagesTr directory (mixed modalities)")
    parser.add_argument("--outdir", required=True, help="Output dir for petct/ct/pet symlink subsets")
    parser.add_argument("--file-ending", default=".nii.gz")
    parser.add_argument("--prompts", default=None,
                        help="Optional prompt/labels directory to route in parallel (per-case, no channel suffix)")
    args = parser.parse_args()

    counts = route(args.images, args.outdir, args.file_ending, args.prompts)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()

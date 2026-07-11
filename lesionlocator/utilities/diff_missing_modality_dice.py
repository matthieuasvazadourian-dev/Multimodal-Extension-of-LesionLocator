"""
Diff Dice/NSD/detection means across the three error_dict.json produced by
scripts/eval_seg_missing_modality_robust.sh (both/ct_only/pet_only) — Phase 6
"Job A" measurement of missing-modality robustness for one missing_modality_robust
checkpoint (contrast Phase 7's per-case auto-detected deployment).

Reads only the fold-level aggregate 'mean' fields already written by the
existing eval pipeline (lesionlocator_segment_and_track.py's error_all dict) —
no per-case identifiers are read or printed here.

Usage:
    python -m lesionlocator.utilities.diff_missing_modality_dice /path/to/fold_N
        (expects fold_N/{both,ct_only,pet_only}/error_dict.json)
"""
import argparse
import json
import os

_SCENARIOS = ('both', 'ct_only', 'pet_only')


def _load(fold_dir: str, subdir: str) -> dict:
    path = os.path.join(fold_dir, subdir, 'error_dict.json')
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fold_dir', type=str,
                        help="Folder containing both/ ct_only/ pet_only/ subdirs, "
                             "each with an error_dict.json (see "
                             "eval_seg_missing_modality_robust.sh).")
    args = parser.parse_args()

    results = {name: _load(args.fold_dir, name) for name in _SCENARIOS}

    print(f"{'scenario':<10} {'dice':>8} {'nsd':>8} {'lesion_found%':>14}")
    for name in _SCENARIOS:
        d = results[name]
        print(f"{name:<10} {d['dice']['mean']:>8.4f} {d['nsd']['mean']:>8.4f} {d['lesion_found']['mean']:>14.1f}")

    both_dice = results['both']['dice']['mean']
    print()
    for name in ('ct_only', 'pet_only'):
        delta = results[name]['dice']['mean'] - both_dice
        print(f"Delta vs both-present ({name}): {delta:+.4f} Dice")


if __name__ == '__main__':
    main()

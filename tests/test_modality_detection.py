"""
Per-case CT/PET modality auto-detection.

Pure stdlib (no torch/SimpleITK) — verifies
lesionlocator.utilities.modality_grouping.group_petct_cases_by_modality against
a synthetic on-disk file layout (real empty files, real os.path checks), so it
runs without any of the heavy inference dependencies.

Usage:
    python -m pytest tests/test_modality_detection.py -v
"""

import os

import pytest

from lesionlocator.utilities.modality_grouping import group_petct_cases_by_modality

_FILE_ENDING = '.nii.gz'


def _touch(folder: str, name: str) -> str:
    path = os.path.join(folder, name)
    open(path, 'w').close()
    return path


def _make_dataset(tmp_path, layout: dict) -> str:
    """layout: {case_id: 'both'|'ct'|'pet'}"""
    folder = str(tmp_path)
    for case_id, which in layout.items():
        if which in ('both', 'ct'):
            _touch(folder, f'{case_id}_0000{_FILE_ENDING}')
        if which in ('both', 'pet'):
            _touch(folder, f'{case_id}_0001{_FILE_ENDING}')
    return folder


def _all_files(folder: str) -> list:
    return sorted(os.path.join(folder, f) for f in os.listdir(folder))


def test_both_present_untagged(tmp_path):
    folder = _make_dataset(tmp_path, {'caseA': 'both'})
    groups = group_petct_cases_by_modality(_all_files(folder), folder, _FILE_ENDING, missing_modality_robust=True)
    assert len(groups) == 1
    (case_id, files, modality) = groups[0]
    assert case_id == 'caseA'
    assert modality is None
    assert files[0].endswith('caseA_0000' + _FILE_ENDING)
    assert files[1].endswith('caseA_0001' + _FILE_ENDING)


def test_ct_only_tags_ct_and_duplicates_path(tmp_path):
    folder = _make_dataset(tmp_path, {'caseB': 'ct'})
    groups = group_petct_cases_by_modality(_all_files(folder), folder, _FILE_ENDING, missing_modality_robust=True)
    assert len(groups) == 1
    (case_id, files, modality) = groups[0]
    assert case_id == 'caseB'
    assert modality == 'ct'
    assert len(files) == 2
    assert files[0] == files[1]
    assert files[0].endswith('caseB_0000' + _FILE_ENDING)


def test_pet_only_tags_pet_and_duplicates_path(tmp_path):
    folder = _make_dataset(tmp_path, {'caseC': 'pet'})
    groups = group_petct_cases_by_modality(_all_files(folder), folder, _FILE_ENDING, missing_modality_robust=True)
    assert len(groups) == 1
    (case_id, files, modality) = groups[0]
    assert case_id == 'caseC'
    assert modality == 'pet'
    assert len(files) == 2
    assert files[0] == files[1]
    assert files[0].endswith('caseC_0001' + _FILE_ENDING)


def test_mixed_folder_one_call(tmp_path):
    """The core Phase 7 scenario: one folder, three cases, three different
    modality compositions, detected correctly in a single grouping pass."""
    folder = _make_dataset(tmp_path, {'caseA': 'both', 'caseB': 'ct', 'caseC': 'pet'})
    groups = group_petct_cases_by_modality(_all_files(folder), folder, _FILE_ENDING, missing_modality_robust=True)
    by_case = {case_id: modality for case_id, _, modality in groups}
    assert by_case == {'caseA': None, 'caseB': 'ct', 'caseC': 'pet'}


def test_missing_channel_without_robust_checkpoint_raises(tmp_path):
    folder = _make_dataset(tmp_path, {'caseB': 'ct'})
    with pytest.raises(ValueError, match='missing_modality_robust'):
        group_petct_cases_by_modality(_all_files(folder), folder, _FILE_ENDING, missing_modality_robust=False)


def test_both_present_never_raises_regardless_of_robust_flag(tmp_path):
    """A fully-paired dataset must behave identically whether or not the
    checkpoint is robust — no regression for the existing, tested both-present
    eval scripts (early fusion, weighted, mcsa)."""
    folder = _make_dataset(tmp_path, {'caseA': 'both'})
    groups_strict = group_petct_cases_by_modality(_all_files(folder), folder, _FILE_ENDING, missing_modality_robust=False)
    groups_robust = group_petct_cases_by_modality(_all_files(folder), folder, _FILE_ENDING, missing_modality_robust=True)
    assert groups_strict == groups_robust


def test_neither_channel_skipped_with_warning(tmp_path, capsys):
    """Can't normally happen in a single-threaded call (case ids are derived
    from files that exist at listing time), but guards the codepath for a
    race between listing and the isfile() re-check: a case id whose files
    were never actually created on disk is skipped, not crashed."""
    folder = str(tmp_path)
    phantom_ct = os.path.join(folder, f'caseZ_0000{_FILE_ENDING}')
    groups = group_petct_cases_by_modality([phantom_ct], folder, _FILE_ENDING, missing_modality_robust=True)
    assert groups == []
    assert 'caseZ' in capsys.readouterr().out


def test_case_id_stable_across_modality_composition_changes(tmp_path):
    """Regression test for the resume/--continue_prediction bug this triple-
    return contract exists to prevent: case_id must be the SAME value whether
    a case is CT-only, PET-only, or both-present, so a caller deriving output
    filenames from case_id (not from file_group[0]'s filename) gets a stable
    identity for a case across runs, even if it gains/loses a modality in
    between. Before this fix, callers derived the basename from file_group[0],
    which silently changes ("case_0000" vs "case_0001") when the modality
    composition changes -- see lesionlocator_segment_and_track.py's
    output_basenames / finished_output_files derivation.
    """
    case_id = 'caseD'
    run1 = tmp_path / 'run1'
    run1.mkdir()
    folder_ct_only = _make_dataset(run1, {case_id: 'ct'})
    (id_ct_only, _, modality_ct_only) = group_petct_cases_by_modality(
        _all_files(folder_ct_only), folder_ct_only, _FILE_ENDING, missing_modality_robust=True)[0]

    run2 = tmp_path / 'run2'
    run2.mkdir()
    folder_both = _make_dataset(run2, {case_id: 'both'})
    (id_both, _, modality_both) = group_petct_cases_by_modality(
        _all_files(folder_both), folder_both, _FILE_ENDING, missing_modality_robust=True)[0]

    assert id_ct_only == id_both == case_id
    assert modality_ct_only == 'ct'
    assert modality_both is None

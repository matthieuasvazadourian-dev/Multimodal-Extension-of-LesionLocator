import itertools
import multiprocessing
import os
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List

import numpy as np
import torch
import json
from matplotlib import pyplot as plt
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json, subfiles
from torch._dynamo import OptimizedModule
from tqdm import tqdm

import lesionlocator
from lesionlocator.configuration import default_num_processes
from lesionlocator.inference.data_iterators import preprocessing_iterator_fromfiles
from lesionlocator.inference.export_prediction import export_prediction_from_logits
from lesionlocator.utilities.modality_grouping import group_petct_cases_by_modality
from lesionlocator.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from lesionlocator.utilities.file_path_utilities import check_workers_alive_and_busy
from lesionlocator.utilities.find_class_by_name import recursive_find_python_class
from lesionlocator.utilities.helpers import empty_cache, dummy_context
from lesionlocator.utilities.label_handling.label_handling import determine_num_input_channels
from lesionlocator.utilities.plans_handling.plans_handler import PlansManager
from lesionlocator.utilities.prompt_handling.prompt_handler import sparse_to_dense_prompt
from lesionlocator.utilities.surface_distance_based_measures import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient, compute_robust_hausdorff


def _safe_mean(values) -> float:
    return float(np.mean(values)) if len(values) > 0 else 0.0


def _safe_detection_percent(found: int, total: int) -> float:
    return (found / total) * 100 if total else 0.0


class LesionLocatorSegmenter(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 visualize: bool = False):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device
        self.visualize = visualize
        self.petct_mode = False
        self.first_conv_key = None

    @staticmethod
    def _find_first_conv_key(state_dict: dict) -> str:
        # The input conv is the 5-D weight with the smallest in_channels.
        # Alphabetical sort picks a decoder weight first in ResidualEncoderUNet
        # ("decoder..." < "encoder...") and misidentifies the input layer.
        candidates = [(k, tuple(state_dict[k].shape)) for k in state_dict
                      if 'weight' in k
                      and hasattr(state_dict[k], 'ndim')
                      and state_dict[k].ndim == 5]
        if not candidates:
            raise RuntimeError('Could not find input conv weight in state_dict.')
        min_in_ch = min(shape[1] for _, shape in candidates)
        for k, shape in sorted(candidates):
            if shape[1] == min_in_ch:
                return k
        raise RuntimeError('Could not find input conv weight in state_dict.')

    @staticmethod
    def _extend_first_conv_weights(state_dict: dict, first_conv_key: str, num_new_channels: int = 1) -> dict:
        # Extend every state-dict entry whose shape matches the input conv so
        # that shared-parameter aliases (encoder.*, decoder.encoder.*, .../all_modules.*)
        # stay consistent — otherwise load_state_dict raises a size mismatch.
        ref_shape = tuple(state_dict[first_conv_key].shape)
        state_dict = dict(state_dict)
        extended = []
        for key in list(state_dict.keys()):
            t = state_dict[key]
            if not (hasattr(t, 'ndim') and t.ndim == 5 and tuple(t.shape) == ref_shape):
                continue
            old_w = t
            out_ch, old_in_ch, *spatial = old_w.shape
            new_in_ch = old_in_ch + num_new_channels
            new_w = torch.zeros(out_ch, new_in_ch, *spatial, dtype=old_w.dtype)
            new_w[:, :old_in_ch - 1, ...] = old_w[:, :old_in_ch - 1, ...]
            for i in range(num_new_channels):
                new_w[:, old_in_ch - 1 + i, ...] = old_w[:, 0, ...]
            new_w[:, -1, ...] = old_w[:, -1, ...]
            state_dict[key] = new_w
            extended.append(key)
        print(f'[petct] Extended {len(extended)} input-conv alias(es): '
              f'{list(ref_shape)} -> {list(state_dict[extended[0]].shape)}')
        return state_dict

    def _resolve_network_module(self):
        """The real nn.Module, unwrapping torch.compile's OptimizedModule
        wrapper. Setting a plain instance attribute (e.g. inference_modality)
        on the OptimizedModule wrapper would NOT reach the wrapped module."""
        return self.network._orig_mod if isinstance(self.network, OptimizedModule) else self.network

    def _group_input_files_by_case(self, source_folder: str, file_ending: str, num_modalities: int) -> list:
        """Return a list of (case_id, file_group, inference_modality) triples.

        For single-modality datasets each file_group is a one-element list
        [file], inference_modality is always None, and case_id is the
        filename stem (unchanged naming behaviour from before this existed).

        For petct (num_modalities == 2, CT=channel 0, PET=channel 1): when
        both _0000/_0001 exist for a case, file_group is [ct_file, pet_file]
        and inference_modality is None (unchanged from prior behaviour). When
        exactly one is missing and the loaded checkpoint is
        missing_modality_robust, the missing slot is filled with the PRESENT
        file's own path (a duplicate) and inference_modality is set to 'ct' or
        'pet'. The duplicate is never read by the network for the absent
        modality — IntermediateFusionResEncUNet.forward only encodes a
        channel when its modality is flagged present (multimodal_unet.py,
        `skips_ct = ... if has_ct else None`) — it exists solely to keep the
        preprocessor's fixed 2-channel geometry/shape checks satisfied
        (SimpleITKIO.read_images requires all channels to share shape/spacing).
        A checkpoint that is NOT missing_modality_robust has no code path for
        a dropped modality, so a missing file is a hard error in that case.

        case_id is returned explicitly rather than left for the caller to
        derive from file_group[0]'s filename, so output naming/resume
        bookkeeping stays stable even if a case's modality composition
        changes between runs — see group_petct_cases_by_modality's docstring.
        """
        all_files = subfiles(source_folder, suffix=file_ending, join=True, sort=True)
        if num_modalities == 1:
            return [(os.path.basename(f)[:-len(file_ending)], [f], None) for f in all_files]
        # LesionLocator only ships 1- or 2-channel (CT+PET) configs today;
        # a genuine 3+-channel non-petct dataset would need this function
        # extended rather than silently mis-grouped by _0000/_0001 alone.
        assert num_modalities == 2, (
            f"Per-case modality auto-detection only supports CT+PET (2 channels), got num_modalities={num_modalities}."
        )
        return group_petct_cases_by_modality(all_files, source_folder, file_ending, self.missing_modality_robust)

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             modality: str = 'ct',
                                             checkpoint_name: str = 'checkpoint_final.pth',
                                             fusion_arch: str = None,
                                             missing_modality_robust: bool = False,
                                             force_inference_modality: str = None):
        """
        This is used when making predictions with a trained model
        """
        print("Loading segmentation model.")
        if use_folds is None:
            use_folds = LesionLocatorSegmenter.auto_detect_available_folds(model_training_output_dir, checkpoint_name)
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            # Prefer best_model.pth when available, otherwise fall back to checkpoint_name
            if isfile(join(model_training_output_dir, f'fold_{f}', 'best_model.pth')):
                ckpt_path = join(model_training_output_dir, f'fold_{f}', 'best_model.pth')
            else:
                ckpt_path = join(model_training_output_dir, f'fold_{f}', checkpoint_name)
            print(f"Loading segmentation model checkpoint from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path,
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None
                # Auto-detect fusion_arch from checkpoint when not provided via CLI
                fusion_arch_from_ckpt = checkpoint.get('fusion_arch', None)
                if fusion_arch_from_ckpt not in (None, 'weighted', 'mcsa'):
                    raise ValueError(
                        f"Checkpoint uses removed fusion_arch='{fusion_arch_from_ckpt}'. "
                        f"Old TAMW/Combined/ShaSpec checkpoints are incompatible with current code. Retrain with --fusion_arch weighted or mcsa."
                    )
                if fusion_arch is None and fusion_arch_from_ckpt is not None:
                    fusion_arch = fusion_arch_from_ckpt
                    print(f'[intermediate-fusion] Auto-detected fusion_arch={fusion_arch} from checkpoint.')
                elif fusion_arch is not None and fusion_arch_from_ckpt is not None and fusion_arch != fusion_arch_from_ckpt:
                    raise ValueError(
                        f'CLI --fusion_arch={fusion_arch} disagrees with checkpoint fusion_arch={fusion_arch_from_ckpt}'
                    )
                # Auto-detect missing_modality_robust from checkpoint when not requested via CLI
                robust_from_ckpt = checkpoint.get('missing_modality_robust', False)
                if not missing_modality_robust and robust_from_ckpt:
                    missing_modality_robust = True
                    print('[intermediate-fusion] Auto-detected missing_modality_robust=True from checkpoint.')

            parameters.append(checkpoint['network_weights'])

        # PET+CT: patch dataset_json so that num_input_channels = 2 (both fusion modes).
        self.petct_mode = (modality == 'petct')
        self.intermediate_fusion_mode = (fusion_arch is not None) and (modality == 'petct')
        self.missing_modality_robust = bool(missing_modality_robust) and self.intermediate_fusion_mode
        self.fusion_arch = fusion_arch
        self.first_conv_key = None
        # Measurement-only override: force every case's forward to drop a
        # modality (Phase 6 "Job A"). Distinct from Phase 7's per-case
        # auto-detected inference_modality (set from real missing files, see
        # _group_input_files_by_case and predict_from_files below).
        if force_inference_modality is not None and not self.missing_modality_robust:
            raise ValueError(
                "--force_inference_modality requires a missing_modality_robust "
                "checkpoint (got missing_modality_robust="
                f"{self.missing_modality_robust}). This checkpoint has no "
                "single-modality inference code path."
            )
        self.force_inference_modality = force_inference_modality
        if self.petct_mode:
            dataset_json = dict(dataset_json)
            dataset_json['channel_names'] = {'0': 'CT', '1': 'PET'}
            print('[petct] Patched dataset_json channel_names for PET+CT.')

        configuration_manager = plans_manager.get_configuration(configuration_name, modality=modality)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(lesionlocator.__path__[0], "training", "LesionLocatorTrainer"),
                                                    trainer_name, 'lesionlocator.training.LesionLocatorTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in lesionlocator.training.LesionLocatorTrainer. '
                               f'Please place it there (in any .py file)!')

        arch_class_name = configuration_manager.network_arch_class_name
        arch_init_kwargs = configuration_manager.network_arch_init_kwargs
        arch_init_kwargs_req_import = configuration_manager.network_arch_init_kwargs_req_import
        if self.intermediate_fusion_mode:
            arch_init_kwargs = dict(arch_init_kwargs)
            arch_class_name = 'lesionlocator.modules.multimodal_unet.IntermediateFusionResEncUNet'
            arch_init_kwargs['fusion_arch'] = self.fusion_arch
            if self.missing_modality_robust:
                arch_init_kwargs['missing_modality_robust'] = True
            print(f'[intermediate-fusion] Using IntermediateFusionResEncUNet with '
                  f'fusion_arch={self.fusion_arch}, missing_modality_robust={self.missing_modality_robust}')

        network = trainer_class.build_network_architecture(
            arch_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=True
        )
        # Checkpoints are trained with DS=True; disable DS at inference so forward returns
        # a single tensor (not a list). Must happen AFTER build and BEFORE load_state_dict.
        if hasattr(network, 'decoder') and hasattr(network.decoder, 'deep_supervision'):
            network.decoder.deep_supervision = False

        # PET+CT early fusion: extend first conv from 2 -> 3 input channels.
        # Not needed for intermediate fusion: shared encoder uses 2-channel input.
        if self.petct_mode and not self.intermediate_fusion_mode:
            self.first_conv_key = self._find_first_conv_key(parameters[0])
            expected_in_ch = num_input_channels + 1  # +1 for prompt channel
            current_in_ch = parameters[0][self.first_conv_key].shape[1]
            if current_in_ch < expected_in_ch:
                parameters = [self._extend_first_conv_weights(
                                  p, self.first_conv_key,
                                  num_new_channels=expected_in_ch - current_in_ch)
                              for p in parameters]
            else:
                print(f'[petct] Checkpoint first conv already has {current_in_ch} input '
                      f'channels (expected {expected_in_ch}); skipping extension.')

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters

        if self.intermediate_fusion_mode:
            has_fusion_keys = any(k.startswith('fusion_modules') for k in parameters[0])
            if has_fusion_keys:
                network.load_state_dict(parameters[0])
                print('[intermediate-fusion] Loaded trained intermediate checkpoint (strict).')
            else:
                missing, unexpected = network.load_state_dict(parameters[0], strict=False)
                non_fusion_missing = [
                    k for k in missing
                    if not k.startswith('fusion_modules')
                    and not k.startswith('specific_encoder_ct')
                    and not k.startswith('specific_encoder_pet')
                    and not k.startswith('gen_ct')
                    and not k.startswith('gen_pet')
                    and not k.startswith('combine_ct')
                    and not k.startswith('combine_pet')
                    and not k.startswith('domain_classifier')
                    and not (k.startswith('decoder.seg_layers.')
                             and not k.startswith('decoder.seg_layers.0.'))
                ]
                if non_fusion_missing:
                    raise RuntimeError(
                        f'[intermediate-fusion] Non-fusion keys missing from CT seed: {non_fusion_missing}'
                    )
                if unexpected:
                    raise RuntimeError(
                        f'[intermediate-fusion] Unexpected keys in CT seed checkpoint: {unexpected}'
                    )
                n_fusion = sum(1 for k in missing if k.startswith('fusion_modules'))
                n_aux_heads = sum(1 for k in missing
                                  if k.startswith('decoder.seg_layers.')
                                  and not k.startswith('decoder.seg_layers.0.'))
                print(f'[intermediate-fusion] CT seed loaded. '
                      f'Fusion modules ({n_fusion} keys) and DS aux heads ({n_aux_heads} keys) '
                      f'retain random init.')
        else:
            network.load_state_dict(parameters[0])

        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('LesionLocator_compile' in os.environ.keys()) and (os.environ['LesionLocator_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)


    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds


    def predict_from_files(self,
                           source_folder_or_file: str,
                           output_folder_or_file: str,
                           prompt_folder_or_file: str,
                           prompt_type: str,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is the default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")

        _channel_dict = self.dataset_json.get('channel_names', self.dataset_json.get('modality', {'0': 'CT'}))
        _num_modalities = len(_channel_dict)
        _file_ending = self.dataset_json['file_ending']

        if os.path.isdir(source_folder_or_file):
            assert os.path.isdir(prompt_folder_or_file), \
                "If '-i' is a folder then '-p' (prompt) must also be a folder."
            # Group input files by case. Each element is (case_id, file_group,
            # inference_modality): inference_modality is None (both present /
            # non-petct) or 'ct'/'pet' for a per-case auto-detected dropped
            # modality (Phase 7; only possible against a missing_modality_robust
            # checkpoint — see _group_input_files_by_case). case_id is used
            # directly for output naming/resume matching below (NOT derived
            # from file_group[0]'s filename) so a case's identity in the
            # output folder stays stable even if its modality composition
            # changes between runs -- see group_petct_cases_by_modality's
            # docstring for the concrete failure this avoids.
            input_files_grouped = self._group_input_files_by_case(
                source_folder_or_file, _file_ending, _num_modalities)
            case_ids = [c for c, _, _ in input_files_grouped]
            input_files = [g for _, g, _ in input_files_grouped]
            inference_modalities = [m for _, _, m in input_files_grouped]
            prompt_files_json = subfiles(prompt_folder_or_file, suffix='.json', join=True, sort=True)
            prompt_files_mask = subfiles(prompt_folder_or_file, suffix=_file_ending, join=True, sort=True)
            output_files = [join(output_folder_or_file, f'{c}{_file_ending}') for c in case_ids]

            # Assertions
            if len(input_files) == 0:
                print(f'No files found in {source_folder_or_file}')
                return
            assert len(prompt_files_json) == 0 or len(prompt_files_mask) == 0, \
                "Prompt folder must contain either json files or mask files, not both."
            assert len(input_files) == len(prompt_files_json) or len(input_files) == len(prompt_files_mask), \
                "Number of files in source folder and prompt folder must be the same."

            prompt_files = prompt_files_json if len(prompt_files_json) > 0 else prompt_files_mask

            # Check if the output folder exists
            if not os.path.isdir(output_folder_or_file):
                os.makedirs(output_folder_or_file)
            else:
                if not overwrite:
                    # Predicted files are named "<case_id>_lesion_<inst_id>.nii.gz"
                    # (see out_file = ofile + f'_lesion_{inst_id}' below), never
                    # "<case_id>.nii.gz" bare -- checking os.path.isfile(output_files[i])
                    # directly (the old check here) was always False, so every
                    # case was silently re-run regardless of --continue_prediction.
                    # Match by scanning the output folder for real lesion files
                    # instead, same pattern as lesionlocator_segment_and_track.py.
                    finished_files = subfiles(output_folder_or_file, suffix=_file_ending, join=True, sort=True)
                    finished_case_ids = {
                        os.path.basename(f)[:-len(_file_ending)].split('_lesion_')[0]
                        for f in finished_files if '_lesion_' in os.path.basename(f)
                    }
                    not_existing_indices = [i for i, c in enumerate(case_ids) if c not in finished_case_ids]
                    input_files = [input_files[i] for i in not_existing_indices]
                    prompt_files = [prompt_files[i] for i in not_existing_indices]
                    output_files = [output_files[i] for i in not_existing_indices]
                    inference_modalities = [inference_modalities[i] for i in not_existing_indices]
        else:
            assert not os.path.isdir(prompt_folder_or_file), \
                "If '-i' is a file then '-p' (prompt) must also be files not folders."
            if _num_modalities > 1:
                raise ValueError(
                    "Single-file prediction is not supported for '--modality petct'. "
                    "Provide a folder with paired _0000/_0001 channel files."
                )
            input_files = [[source_folder_or_file]]
            prompt_files = [prompt_folder_or_file]
            output_files = [join(output_folder_or_file, os.path.basename(source_folder_or_file))]
            inference_modalities = [None]

        # Truncate output files
        output_files = [i.replace(self.dataset_json['file_ending'], '') for i in output_files]
        data_iterator = preprocessing_iterator_fromfiles(input_files, prompt_files,
                                                output_files, prompt_type, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes_preprocessing, self.device.type == 'cuda',
                                                self.verbose_preprocessing, False, inference_modalities)
       
        return self.predict_from_data_iterator(data_iterator, prompt_type, output_folder_or_file, num_processes_segmentation_export)


    def predict_from_data_iterator(self,
                                   data_iterator,
                                   prompt_type: str,
                                   output_folder_or_file: str,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        This function takes a data iterator and makes predictions and saves each instance (lesion) as a separate file.
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            print(worker_list)
            r = []
            spacing_mm=(1.5, 1.5, 1.5)
            error_all={'dice': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}}, 
               'nsd': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}},
               'hausdorff': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}},
               'lesion_found':{'all':0, 'mean':0, 'TP0':{'all':0, 'mean':0}, 'TP1':{'all':0, 'mean':0}, 'TP2':{'all':0, 'mean':0}},
               'lesion_all': {'all':0, 'TP0':{'all':0}, 'TP1':{'all':0}, 'TP2':{'all':0}}}
            dice_score_all = []
            hausdorff_score_all = []
            nsd_score_all = []
            metrics = {
                'dice': 0.0,
                'hausdorff': 0.0,
                'nsd': 0.0
            }
            for preprocessed in data_iterator:                    
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                print('OFILE: ', ofile)
                print(f'\n === Predicting {os.path.basename(ofile)} === ')
                patient_tp = os.path.basename(ofile)
                timepoint = os.path.basename(ofile).split('_')[0]
                output_folder = ofile.split('/')[0]
                properties = preprocessed['data_properties']
                prompt = preprocessed['prompt']

                # Per-case single-modality inference: Phase 7 auto-detects a
                # missing file at grouping time and tags the case (see
                # _group_input_files_by_case / data_iterators.py's 'inference_modality'
                # item key); Phase 6's --force_inference_modality overrides every
                # case for measurement runs where both files are actually present.
                # A non-robust checkpoint has no code path for a dropped modality,
                # so a tagged case is a hard error rather than a silent full-fusion
                # fallback.
                case_inference_modality = preprocessed.get('inference_modality', None)
                if self.force_inference_modality is not None:
                    case_inference_modality = self.force_inference_modality
                if case_inference_modality is not None and not self.missing_modality_robust:
                    raise ValueError(
                        f"Case {os.path.basename(ofile)} requires single-modality "
                        f"inference (inference_modality={case_inference_modality}) "
                        "but the loaded checkpoint is not missing_modality_robust."
                    )
                if self.missing_modality_robust:
                    self._resolve_network_module().inference_modality = case_inference_modality

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                if len(prompt) == 0:
                    print(f" No prompt found for {os.path.basename(ofile)}")
                else:
                    for inst_id, p in enumerate(prompt):
                        inst_id += 1
                        print(f'\n Lesion ID {inst_id}: ')
                        p_sparse = p
                        print('P_SPARSE :', p_sparse)
                        p = sparse_to_dense_prompt(p, prompt_type, array=data)
                        if p is None:
                            print(f" Invalid prompt found for {os.path.basename(ofile)}")
                            continue
                        for k in error_all.keys():
                            if k == 'lesion_all' or k == 'lesion_found':
                                print('passing')
                                continue
                            if os.path.basename(ofile) not in error_all[k].keys():
                                error_all[k][patient_tp]={'mean':0, 'per_lesion':[]}
                        print('START PREDICTION')
                        # Predict the logits using the preprocessed data and the prompt
                        prediction = self.predict_logits_from_preprocessed_data(data, p).cpu()
                        seg = torch.softmax(prediction, 0).argmax(0)
                        pred = seg.detach().cpu().numpy().astype(np.uint8)
                        print('PREDICTION SHAPE: ', pred.shape)
                        print('GT SHAPE: ', p[0].detach().cpu().numpy().shape)
                        dice_score = compute_dice_coefficient(p[0].detach().cpu().numpy().astype(np.uint8), pred)
                        error_all['lesion_all']['all']+=1
                        error_all['lesion_all'][timepoint]['all']+=1
                        if dice_score >= 0.1:
                            error_all['lesion_found']['all']+=1
                            error_all['lesion_found'][timepoint]['all']+=1
                        print('DICE SCORE: ', dice_score)
                        surface_distances = compute_surface_distances(p[0].detach().cpu().numpy().astype(np.uint8), pred, spacing_mm)
                        dice_score_all.append(dice_score)
                        hausdorff_score = compute_robust_hausdorff(surface_distances, 95)
                        hausdorff_score_all.append(hausdorff_score)
                        nsd_score = compute_surface_dice_at_tolerance(surface_distances, 2)
                        nsd_score_all.append(nsd_score)
                        metrics = {
                            'dice': dice_score,
                            'hausdorff': hausdorff_score,
                            'nsd': nsd_score
                        }
                        # Update all metrics in a loop
                        for metric_name, score in metrics.items():
                            error_all[metric_name][timepoint]['all'].append(score)
                            error_all[metric_name][patient_tp]['per_lesion'].append(score)
                        print('Avg Mean Dice: ', np.mean(dice_score_all))
                        print('Avg Mean Hausdorff: ',  np.mean(hausdorff_score_all))
                        print('Avg Mean NSD: ', np.mean(nsd_score_all))
                        print('Avg Lesion Detection Score: {:.2f}%'.format((error_all['lesion_found']['all'] / error_all['lesion_all']['all']) * 100))
                        with open(os.path.join(output_folder_or_file, 'error_dict.json'), 'w') as fjson:
                            json.dump(error_all, fjson)
                        print('----------')
                        out_file = ofile + f'_lesion_{inst_id}'
                        # Visualize the prediction
                        if self.visualize:
                            mask_ones_gt_axial = np.where(p[0].detach().cpu().numpy().astype(np.uint8) == 1)
                            if len(mask_ones_gt_axial[0]) > 0:  # Check if mask is not empty
                                # Find the z-slice with most mask voxels
                                largest_mask_slice_id_axial = np.bincount(mask_ones_gt_axial[0]).argmax()
                            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                            # orginal img
                            ax1.imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            ax1.set_title('Image') 
                            ax1.axis('off')
                            # gt
                            ax2.imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            ax2.imshow(p[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy().astype(np.uint8), alpha=0.5)
                            ax2.set_title('Ground truth') 
                            ax2.axis('off')
                            # preds
                            ax3.imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            ax3.imshow(pred[largest_mask_slice_id_axial, :, :], alpha=0.5)
                            ax3.set_title('Prediction') 
                            ax3.axis('off')

                            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
                            plt.savefig(os.path.join(output_folder_or_file, f'{out_file}_dice_{dice_score:.2f}.png'), bbox_inches='tight')
                            plt.close()

                        
                        r.append(
                            export_pool.starmap_async(
                                export_prediction_from_logits,
                                ((prediction, properties, self.configuration_manager, self.plans_manager,
                                    self.dataset_json, out_file, False),)
                            )
                        )

                        # no multiprocessing
                        # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                        #     self.dataset_json, out_file, False)
                    for metric_name in metrics.keys():
                        error_all[metric_name][patient_tp]['mean'] = np.mean(error_all[metric_name][patient_tp]['per_lesion'])
                print(f'done with {os.path.basename(ofile)}')
            error_all['dice']['mean']= _safe_mean(dice_score_all)
            error_all['hausdorff']['mean'] = _safe_mean(hausdorff_score_all)
            error_all['nsd']['mean'] = _safe_mean(nsd_score_all)
            error_all['lesion_found']['mean'] = _safe_detection_percent(
                error_all['lesion_found']['all'], error_all['lesion_all']['all']
            )
            for tp in ['TP0','TP1','TP2']:
                error_all['dice'][tp]['mean']=_safe_mean(error_all['dice'][tp]['all'])
                error_all['hausdorff'][tp]['mean']=_safe_mean(error_all['hausdorff'][tp]['all'])
                error_all['nsd'][tp]['mean']=_safe_mean(error_all['nsd'][tp]['all'])
                error_all['lesion_found'][tp]['mean'] = _safe_detection_percent(
                    error_all['lesion_found'][tp]['all'], error_all['lesion_all'][tp]['all']
                )
            with open(os.path.join(output_folder_or_file, 'error_dict.json'), 'w') as fjson:
                json.dump(error_all, fjson)
            
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret


    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor, dense_prompt: torch.Tensor) -> torch.Tensor:
        """
        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None

        # Add the dense prompt to the data
        data = torch.cat([data, dense_prompt], dim=0)

        for params in self.list_of_parameters:

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)
        
            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data, dense_prompt).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data, dense_prompt).to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...], dense_prompt: torch.Tensor = None) -> List:
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            raise NotImplementedError('This predictor only supports 3D images')
        else:
            # No bbox will yield all slices
            if dense_prompt is None:
                steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                        self.tile_step_size)
                if self.verbose: print(
                    f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                    f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
                for sx in steps[0]:
                    for sy in steps[1]:
                        for sz in steps[2]:
                            slicers.append(
                                tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                    zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
            else:
                prompt_coords = torch.where(dense_prompt[0] > 0)
                prompt_coords = [int(prompt_coords[i].min().item()) for i in range(3)] + [int(prompt_coords[i].max().item()) for i in range(3)]
                prompt_coords = torch.tensor(prompt_coords)
                # Bbox focused
                if all(prompt_coords[3:] - prompt_coords[:3] < torch.tensor(self.configuration_manager.patch_size)):
                    # Return a slicer that covers the bbox in the middle of the patch
                    slicer = [slice(None)]
                    for i in range(3):
                        start = int((prompt_coords[i] + prompt_coords[i + 3] - self.configuration_manager.patch_size[i]) / 2)
                        end = start + self.configuration_manager.patch_size[i]
                        if start < 0:
                            start = 0
                            end = self.configuration_manager.patch_size[i]
                        elif end > image_size[i]:
                            end = image_size[i]
                            start = end - self.configuration_manager.patch_size[i]
                        slicer.append(slice(start, end))
                    slicers.append(slicer)
                else: # Non bbox focused, return all slices which overlap with the bbox
                    steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                            self.tile_step_size)
                    if self.verbose: print(
                        f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                        f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
                    for sx in steps[0]:
                        for sy in steps[1]:
                            for sz in steps[2]:
                                slc = tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                        zip((sx, sy, sz), self.configuration_manager.patch_size)]])
                                # Make sure the slicer has some overlap with the bbox
                                if (prompt_coords[0] < slc[1].stop and prompt_coords[3] > slc[1].start) and \
                                        (prompt_coords[1] < slc[2].stop and prompt_coords[4] > slc[2].start) and \
                                        (prompt_coords[2] < slc[3].stop and prompt_coords[5] > slc[3].start):
                                    slicers.append(slc)
        return slicers

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor, dense_prompt: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
       
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            # Make sure we get only the patches we need to predict, i.e. overlab with the prompt
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:], dense_prompt)

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits


def predict_seg_from_prompt():
    import argparse
    parser = argparse.ArgumentParser(description='This function handels the LesionLocator single timepoint segmentation'
                                     'inference using a point or 3D box prompt. Prompts can be the coordinates of a '
                                     'point or a 3D box as .json files or also (ground truth) intance segmentation maps.')
    parser.add_argument('-i', type=str, required=True,
                        help='Input image file or folder containing images to be predicted. File endings should be .nii.gz'
                        ' or specify another file_ending in the dataset.json file of the downloaded checkpoint.')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If the folder does not exist it will be created. Predicted segmentations'
                             'will have the same name as their source images with the lesion instance as suffix.')
    parser.add_argument('-p', type=str, required=True,
                        help='Prompt file or folder with prompts. Can contain .json files with a point or 3D box or instance'
                        'segmentation maps (.nii.gz). The file containing the prompt must have the same name as the image it belongs to.'
                        'If instance segmentation maps are used, they must be in the same shape as the input images. Binary masks '
                        'will be converted to instance segmentations.')
    parser.add_argument('-t', type=str, required=True, choices=['point', 'box'], default='box',
                        help='Specify the type of prompt. Options are "point" or "box". Default: box')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder of the LesionLocator model called "LesionLocatorCheckpoint"')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2).")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('--visualize', action='store_true', required=False, default=False,
                        help='Set this flag to visualize the prediction. This will open a napari viewer. You may need to '
                        ' run python -m pip install "napari[all]" first to use this feature.')

    parser.add_argument('--modality', type=str, required=True, choices=['ct', 'pet', 'petct'], default='ct', help="Use this to set the modality")
    parser.add_argument('--fusion_arch', type=str, required=False, default=None,
                        choices=['weighted', 'mcsa'],
                        help="Intermediate feature-level fusion variant for PET+CT. One of weighted/mcsa. "
                             "Only used when --modality petct. Omit for early fusion (default behaviour). "
                             "Auto-detected from the checkpoint if omitted.")
    parser.add_argument('--missing_modality_robust', action='store_true', required=False, default=False,
                        help="Only used with --modality petct and --fusion_arch weighted/mcsa. Set this if "
                             "the checkpoint was trained with the ShaSpec-inspired missing-modality "
                             "robustness add-on. Auto-detected from the checkpoint if omitted.")
    parser.add_argument('--force_inference_modality', type=str, required=False, default=None,
                        choices=['ct', 'pet'],
                        help="Measurement-only override: force EVERY case's forward pass to drop the "
                             "other modality, even if both _0000/_0001 files are present on disk. Use "
                             "this to quantify robustness on a paired dataset (compare Dice at "
                             "'ct'/'pet'/unset). Requires --missing_modality_robust. For real per-case "
                             "missing files, leave this unset — the loader auto-detects which modality "
                             "each case actually has.")
    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using LesionLocator:\n"
        "Rokuss, M., Kirchhoff, Y., Akbal, S., Kovacs, B., Roy, S., Ulrich, C., ... & Maier-Hein, K. (2025).\n"
        "LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging. "
        "CVPR.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        
        print(args.o)
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = LesionLocatorSegmenter(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose,
                                visualize=args.visualize)
    optimized_ckpt = "bbox_optimized" if args.t == 'box' else "point_optimized"
    checkpoint_folder = join(args.m, 'LesionLocatorSeg', optimized_ckpt)
    predictor.initialize_from_trained_model_folder(checkpoint_folder, args.f, args.modality, "checkpoint_final.pth",
                                                    fusion_arch=getattr(args, 'fusion_arch', None),
                                                    missing_modality_robust=getattr(args, 'missing_modality_robust', False),
                                                    force_inference_modality=getattr(args, 'force_inference_modality', None))
    predictor.predict_from_files(args.i, args.o, args.p, args.t,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 num_parts=1, part_id=0)

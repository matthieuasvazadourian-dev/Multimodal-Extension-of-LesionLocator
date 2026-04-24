import itertools
import multiprocessing
import os
import gc
import traceback
import json
import time
import glob
import re
import numpy as np
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import SimpleITK
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json, subfiles
from torch._dynamo import OptimizedModule
from tqdm import tqdm

import lesionlocator
from lesionlocator.preprocessing.resampling.default_resampling import compute_new_shape
from lesionlocator.configuration import default_num_processes
from lesionlocator.training.data_iterators import preprocessing_iterator_fromfiles
from lesionlocator.inference.export_prediction import export_prediction_from_logits
from lesionlocator.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from lesionlocator.utilities.file_path_utilities import check_workers_alive_and_busy
from lesionlocator.utilities.find_class_by_name import recursive_find_python_class
from lesionlocator.utilities.helpers import empty_cache, dummy_context
from lesionlocator.utilities.label_handling.label_handling import determine_num_input_channels
from lesionlocator.utilities.plans_handling.plans_handler import PlansManager
from lesionlocator.utilities.prompt_handling.prompt_handler import sparse_to_dense_prompt
from lesionlocator.utilities.surface_distance_based_measures import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient, compute_robust_hausdorff

from torch.cuda.amp import GradScaler
from torch.amp import autocast
import gc

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

def dice_loss(pred, target, epsilon=1e-6):
    """
    Dice loss function for segmentation training.
    
    Args:
        pred: Model predictions [B, C, H, W, D] (logits)
        target: Ground truth labels [B, H, W, D] (class indices)
        epsilon: Small value to avoid division by zero
        
    Returns:
        Dice loss value (1 - dice_coefficient)
    """
    pred_soft = torch.softmax(pred, dim=1)
    target_onehot = nn.functional.one_hot(target.long(), num_classes=pred.shape[1])
    target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float() if pred.dim() == 5 else target_onehot.permute(0, 3, 1, 2).float()
    
    dims = (0,) + tuple(range(2, pred.dim()))
    intersection = torch.sum(pred_soft * target_onehot, dims)
    union = torch.sum(pred_soft, dims) + torch.sum(target_onehot, dims)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    weighted_dice = 0.1 * dice[0] + 0.9 * dice[1]

    print(f"prediction is all zeros: {torch.all(pred_soft == 0)}")
    print(f"dice score {dice}")
    
    return 1 - weighted_dice


def _uses_cuda_device(device: torch.device) -> bool:
    return getattr(device, 'type', None) == 'cuda' and torch.cuda.is_available()


def _autocast_context(device: torch.device):
    return autocast('cuda') if _uses_cuda_device(device) else dummy_context()


def _maybe_empty_cache(device: torch.device):
    if _uses_cuda_device(device):
        torch.cuda.empty_cache()


def _cuda_memory_allocated_gb(device: torch.device) -> float:
    return torch.cuda.memory_allocated() / 1024**3 if _uses_cuda_device(device) else 0.0


def _cuda_memory_reserved_gb(device: torch.device) -> float:
    return torch.cuda.memory_reserved() / 1024**3 if _uses_cuda_device(device) else 0.0


def _cuda_max_memory_allocated_gb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated() / 1024**3 if _uses_cuda_device(device) else 0.0


def _cuda_total_memory_gb(device: torch.device):
    return torch.cuda.get_device_properties(0).total_memory / 1024**3 if _uses_cuda_device(device) else None

def unique_ids_to_indices(id_to_indices, unique_ids):
    indices = []
        
    #unique_ids_to_indices = {uid: [] for uid in unique_ids}
    for uni_id in unique_ids:
        indices.extend(id_to_indices.get(uni_id, []))
    return indices


def _representative_path(file_or_group):
    return file_or_group[0] if isinstance(file_or_group, list) else file_or_group


def _extract_case_id(file_or_group, file_ending=None):
    basename = os.path.basename(_representative_path(file_or_group))
    if file_ending is None:
        if basename.endswith('.nii.gz'):
            file_ending = '.nii.gz'
        else:
            _, file_ending = os.path.splitext(basename)
    stem = basename[:-len(file_ending)] if file_ending and basename.endswith(file_ending) else basename
    if re.search(r'_\d{4}$', stem):
        stem = stem[:-5]
    if re.match(r'^TP\d+_', stem):
        return stem.split('_', 1)[1]
    return stem


def _group_input_files_by_case(source_folder: str, file_ending: str, num_modalities: int):
    all_files = subfiles(source_folder, suffix=file_ending, join=True, sort=True)
    if num_modalities == 1:
        return all_files

    file_set = set(all_files)
    primary_files = [f for f in all_files if os.path.basename(f).endswith(f'_0000{file_ending}')]
    grouped = []
    for primary_file in primary_files:
        case_group = [primary_file]
        for channel_idx in range(1, num_modalities):
            channel_file = primary_file.replace(f'_0000{file_ending}', f'_{channel_idx:04d}{file_ending}')
            if channel_file not in file_set:
                raise FileNotFoundError(
                    f'Missing channel file for multimodal case: expected {channel_file}'
                )
            case_group.append(channel_file)
        grouped.append(case_group)
    return grouped

def create_cv_folds(input_files, prompt_files, output_files, n_folds=5, random_seed=42):
    """
    Create cross-validation folds for training data.
    
    Args:
        input_files: List of training input files
        prompt_files: List of training prompt files  
        output_files: List of training output files
        n_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducibility
        
    Returns:
        List of fold dictionaries, each containing train and val splits
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    unique_ids = sorted(list(set([_extract_case_id(i) for i in input_files])))
    
    id_to_indices = {}
    for i, f in enumerate(input_files):
        uid = _extract_case_id(f)
        if uid not in id_to_indices:
            id_to_indices[uid] = []
        id_to_indices[uid].append(i)

    if len(unique_ids) < 2:
        raise ValueError(
            f"Need at least 2 unique cases for cross-validation, but found {len(unique_ids)}."
        )

    effective_n_folds = min(n_folds, len(unique_ids))

    # Create KFold splitter
    kfold = KFold(n_splits=effective_n_folds, shuffle=True, random_state=random_seed)
    
    folds = []
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(unique_ids)):            
    #for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        train_indices = unique_ids_to_indices(id_to_indices, [unique_ids[i] for i in train_indices])
        val_indices = unique_ids_to_indices(id_to_indices, [unique_ids[i] for i in val_indices])

        fold = {
            'fold_idx': fold_idx,
            'train': {
                'input_files': [input_files[i] for i in train_indices],
                'prompt_files': [prompt_files[i] for i in train_indices],
                'output_files': [output_files[i] for i in train_indices]
            },
            'val': {
                'input_files': [input_files[i] for i in val_indices],
                'prompt_files': [prompt_files[i] for i in val_indices], 
                'output_files': [output_files[i] for i in val_indices]
            }
        }
        folds.append(fold)
    return folds


class LesionDatasetWrapper(IterableDataset):
    """
    PyTorch IterableDataset wrapper that preserves the existing multiprocessing pipeline
    while providing PyTorch DataLoader compatibility for training.
    
    This wrapper:
    1. Preserves all existing multiprocessing data loading
    2. Converts each lesion instance into a training sample
    3. Provides PyTorch DataLoader compatibility
    4. Maintains all preprocessing logic unchanged
    
    Example usage:
        # Create trainer instance
        trainer = LesionLocatorSegmenter(device=torch.device('cuda'))
        trainer.initialize_from_trained_model_folder(model_dir, track_dir, folds)
        
        # Create datasets
        train_dataset = trainer.create_training_dataset(
            input_files=['img1.nii.gz', 'img2.nii.gz'],
            prompt_files=['prompt1.nii.gz', 'prompt2.nii.gz'],
            output_files=['out1', 'out2'],
            prompt_type='box'
        )
        
        # Train
        trainer.train(train_dataset, epochs=100, lr=1e-4)
    """
    def __init__(self, input_files, prompt_files, output_files, prompt_type, 
                 plans_config, dataset_json, configuration_config, modality,
                 num_processes=8, pin_memory=False, verbose=False, track=False):
        self.input_files = input_files
        self.prompt_files = prompt_files
        self.output_files = output_files
        self.prompt_type = prompt_type
        self.plans_config = plans_config
        self.dataset_json = dataset_json
        self.configuration_config = configuration_config
        self.modality = modality
        self.num_processes = num_processes
        self.pin_memory = pin_memory
        self.verbose = verbose
        self.track = track
        
    def __len__(self):
        """
        Return an estimate of the dataset length for PyTorch DataLoader.
        This is an approximation since the actual number of lesions per file varies.
        """
        # Estimate: assume average of 2-3 lesions per file
        return len(self.input_files) 
        
    def __iter__(self):
        """
        Create the multiprocessing data iterator and yield training samples.
        This preserves the existing preprocessing pipeline completely.
        """

        if len(self.input_files) == len(self.prompt_files) == len(self.output_files):
            perm = np.random.permutation(len(self.input_files))
            input_files = [self.input_files[i] for i in perm]
            prompt_files = [self.prompt_files[i] for i in perm]
            output_files = [self.output_files[i] for i in perm]
        else:
            input_files = self.input_files
            prompt_files = self.prompt_files
            output_files = self.output_files

        data_iterator = preprocessing_iterator_fromfiles(
            input_files, prompt_files, output_files,
            self.prompt_type, self.plans_config, self.dataset_json,
            self.configuration_config, self.modality, self.num_processes, self.pin_memory,
            self.verbose, self.track, train=True
        )

        for preprocessed in data_iterator:
            data = preprocessed['data']
            prompt = preprocessed['prompt']
            seg_mask = preprocessed['seg']
            properties = preprocessed['data_properties']

            if self.track:
                bl_data = preprocessed['bl_data']

            if self.track and bl_data is None:
                # go to the next sample if no baseline data
                continue
                            
            # Convert each lesion instance into a training sample
            for inst_id, p in enumerate(prompt):
                # print(f'Processing instance {inst_id}', flush=True)
                if len(p) == 0:
                    continue

                # hard-coded for point prompts
                mask_id = inst_id + 1
                # mask_id = preprocessed['seg'][0, int(p[0]), int(p[1]), int(p[2])]
                if seg_mask is None:
                    raise ValueError(
                        "Training requires segmentation-mask prompts. JSON prompts are not supported "
                        "by the current supervised training path."
                    )
                gt_mask = (seg_mask == mask_id).astype(np.uint8)
                p_dense = sparse_to_dense_prompt(p, self.prompt_type, array=data)

                if self.track:
                    bl_seg = preprocessed['bl_data_properties'].get('seg', None)
                    bl_gt_mask = (bl_seg == mask_id).astype(np.uint8) if bl_seg is not None else None

                if self.track and (bl_gt_mask is None or (bl_gt_mask == 0).all()):
                    continue
                # if len(p) == 0:
                #     continue
                # mask_id += 1
                # gt_mask = (seg_mask == mask_id).astype(np.uint8)

                # # Convert sparse prompt to dense format
                # p_dense = sparse_to_dense_prompt(p, self.prompt_type, array=data)
                
                if p_dense is None:
                    continue
                
                # Yield training sample - convert to torch tensors for consistent batching
                # Handle both numpy arrays and already converted tensors
                if isinstance(data, torch.Tensor):
                    data_tensor = data.float()
                else:
                    data_tensor = torch.from_numpy(data).float()
                
                if isinstance(p_dense, torch.Tensor):
                    prompt_tensor = p_dense.float()
                else:
                    prompt_tensor = torch.from_numpy(p_dense).float()
                
                if isinstance(gt_mask[0], torch.Tensor):
                    target_tensor = gt_mask[0].long()
                else:
                    target_tensor = torch.from_numpy(gt_mask[0]).long()
                
                if self.track:
                    if isinstance(bl_data, torch.Tensor):
                        bl_data_tensor = bl_data.float()
                    else:
                        bl_data_tensor = torch.from_numpy(bl_data).float()
                    
                    #if bl_gt_mask is not None:
                    if isinstance(bl_gt_mask, torch.Tensor):
                        bl_target_tensor = bl_gt_mask.float()
                    else:
                        bl_target_tensor = torch.from_numpy(bl_gt_mask).float()
                    
                    yield {
                        'bl_data': bl_data_tensor, 
                        'fu_data': data_tensor,
                        'bl_prompt': bl_target_tensor,
                        'target': target_tensor,
                        'properties': properties,
                        'lesion_id': mask_id,                  # Lesion instance ID
                        'filename': preprocessed['ofile']      # Original filename
                    }
                else:    
                    yield {
                        'data': data_tensor,                    # Input image [C, H, W, D]
                        'prompt': prompt_tensor,               # Dense prompt [1, H, W, D]
                        'target': target_tensor,               # Ground truth [H, W, D]
                        'properties': properties,               # Metadata
                        'lesion_id': mask_id,                  # Lesion instance ID
                        'filename': preprocessed['ofile']      # Original filename
                    }


def training_collate_fn(batch):
    """
    Efficiently collate a list of dicts into a dict of stacked tensors/lists.
    Assumes all dicts have the same keys.
    """
    if len(batch) == 1:
        return batch[0]

    # Stack tensors for each key; keep lists for non-tensor values
    collated = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        # Stack if all values are tensors and have the same shape
        if isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values, dim=0)
            except Exception as e:
                print(f"Stacking failed for key '{key}': {e}")
                collated[key] = values  # fallback to list
        else:
            collated[key] = values
    return collated


def tracking_collate_fn(batch):
    """
    Custom collate function for tracking training samples.
    Handles baseline data, follow-up data, baseline prompt, and target.
    """
    if len(batch) == 1:
        return batch[0]
    
    # Stack all batch items into proper tensors
    batch_baseline = []
    batch_followup = []
    batch_prompts = []
    batch_targets = []
    batch_properties = []
    batch_lesion_ids = []
    batch_filenames = []

    for item in batch:
        batch_baseline.append(item['bl_data'])
        batch_followup.append(item['fu_data'])
        batch_prompts.append(item['bl_prompt'])
        batch_targets.append(item['target'])
        batch_properties.append(item['properties'])
        batch_lesion_ids.append(item['lesion_id'])
        batch_filenames.append(item['filename'])
    
    # Stack tensors - all should have same dimensions due to preprocessing
    try:
        stacked_baseline = torch.stack(batch_baseline, dim=0)       # [B, C, H, W, D]
        stacked_followup = torch.stack(batch_followup, dim=0)       # [B, C, H, W, D]
        stacked_prompts = torch.stack(batch_prompts, dim=0)         # [B, 1, H, W, D]
        stacked_targets = torch.stack(batch_targets, dim=0)         # [B, H, W, D]
        
        return {
            'bl_data': stacked_baseline,
            'fu_data': stacked_followup,
            'bl_prompt': stacked_prompts,
            'target': stacked_targets,
            'properties': batch_properties,
            'lesion_id': batch_lesion_ids,
            'filename': batch_filenames
        }

    except RuntimeError as e:
        # Print shapes for debugging
        print(f"Tracking batch stacking failed: {e}")
        print(f"Baseline shapes: {[d.shape for d in batch_baseline]}")
        print(f"Follow-up shapes: {[d.shape for d in batch_followup]}")
        print(f"Prompt shapes: {[p.shape for p in batch_prompts]}")
        print(f"Target shapes: {[t.shape for t in batch_targets]}")
        # Fallback: process as batch_size=1
        print(f"Warning: Could not stack tracking batch, falling back to single sample processing")
        return batch[0]



class LesionLocatorTrack(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 visualize: bool = False,
                 adaptive_mode: bool = False):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        # Training-specific attributes
        self.optimizer = None
        self.loss_function = None
        self.scheduler = None
        self.scaler = GradScaler(enabled=_uses_cuda_device(device))

        self.training_mode = False

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
        self.adaptive_mode = adaptive_mode
        
        print('Adaptive mode: ', self.adaptive_mode)

    @staticmethod
    def _find_first_conv_key(state_dict: dict) -> str:
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
    def _extend_first_conv_weights(state_dict: dict, first_conv_key: str,
                                   num_new_channels: int = 1) -> dict:
        ref_shape = tuple(state_dict[first_conv_key].shape)
        state_dict = dict(state_dict)
        extended = []
        for key in list(state_dict.keys()):
            tensor = state_dict[key]
            if not (hasattr(tensor, 'ndim') and tensor.ndim == 5 and tuple(tensor.shape) == ref_shape):
                continue
            old_w = tensor
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

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             model_track_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             modality: str = 'ct',
                                             checkpoint_name: str = 'checkpoint_final.pth',
                                             reinit: bool = False):
        """
        This is used when making predictions with a trained model
        """
        print("Loading tracking model")
        # print("Loading segmentation model.")
        if use_folds is None:
            use_folds = LesionLocatorTrack.auto_detect_available_folds(model_training_output_dir, checkpoint_name)
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        self.dataset_json = dataset_json
        
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        self.plans = plans
        self.modality = modality
        self.petct_mode = (modality == 'petct')
        self.first_conv_key = None
        
        plans_manager = PlansManager(plans)
        
        # Debug: Print plans structure for troubleshooting
        print("Plans structure debug:")
        print(f"  Plans type: {type(plans)}")
        print(f"  Plans keys: {list(plans.keys()) if isinstance(plans, dict) else 'Not a dict'}")
        if isinstance(plans, dict) and 'configurations' in plans:
            print(f"  Available configurations: {list(plans['configurations'].keys())}")
        else:
            print("  No 'configurations' key found in plans")

        if isinstance(use_folds, str):
             use_folds = [use_folds]

        # Tracker network
        dataset_json_tracker = load_json(join(model_track_training_output_dir, 'dataset.json'))
        plans_tracker = load_json(join(model_track_training_output_dir, 'plans.json'))
        plans_manager_tracker = PlansManager(plans_tracker)

        parameters_tracker = []
        for i, f in enumerate(use_folds):
            print(f'Loading fold {f}')
            f = int(f) if f != 'all' else f
            fold_dir = join(model_track_training_output_dir, f'fold_{f}')
            ckpt_candidates = ('best_tracking_model.pth', 'checkpoint_final.pth', 'final_tracking_checkpoint.pth')
            ckpt_path_tracker = next((join(fold_dir, name) for name in ckpt_candidates if isfile(join(fold_dir, name))), None)
            if ckpt_path_tracker is None:
                raise FileNotFoundError(
                    f'Could not find a tracker checkpoint in {fold_dir}. Tried: {ckpt_candidates}'
                )
            print(f"Loading tracker model checkpoint from: {ckpt_path_tracker}")
            checkpoint_tracker = torch.load(ckpt_path_tracker, map_location=torch.device('cpu'), weights_only=False)

            if i == 0:
                trainer_name_tracker = checkpoint_tracker['trainer_name']
                configuration_name_tracker = checkpoint_tracker.get('init_args', {}).get('configuration')
                if configuration_name_tracker is None:
                    available_configs = list(plans_tracker.get('configurations', {}).keys())
                    if '3d_fullres' in available_configs:
                        configuration_name_tracker = '3d_fullres'
                    elif len(available_configs) == 1:
                        configuration_name_tracker = available_configs[0]
                    else:
                        raise KeyError(
                            'Tracker checkpoint is missing init_args.configuration and no unique '
                            'fallback configuration could be inferred from plans.json'
                        )
                    print(f"[tracker] Checkpoint missing init_args.configuration, using '{configuration_name_tracker}' from plans.json")
                inference_allowed_mirroring_axes = checkpoint_tracker.get('inference_allowed_mirroring_axes')
            
            # load segmentation decoder for the tracker
            # for key in checkpoint_tracker['network_weights'].keys():
            #     if 'unet.decoder' in key:
            #         seg_key = key.replace('unet.', '')
            #         if seg_key in checkpoint['network_weights'].keys():
            #             checkpoint_tracker['network_weights'][key] = checkpoint['network_weights'][seg_key]
            #         else:
            #             print(f'Key {key} not in tracker network, skipping loading segmentation weights for it')

            parameters_tracker.append(checkpoint_tracker['network_weights'])

        if self.petct_mode:
            dataset_json_tracker = dict(dataset_json_tracker)
            dataset_json_tracker['channel_names'] = {'0': 'CT', '1': 'PET'}
            print('[petct] Patched tracker dataset_json channel_names for PET+CT early fusion.')

        configuration_manager_tracker = plans_manager_tracker.get_configuration(configuration_name_tracker, modality=modality)
        # set spacing
        # configuration_manager_tracker.set_spacing([3.3, 2.7, 2.7])
        self.configuration_name_tracker = configuration_name_tracker
        # restore networks
        num_input_channels = determine_num_input_channels(plans_manager_tracker, configuration_manager_tracker, dataset_json_tracker)
        trainer_class = recursive_find_python_class(join(lesionlocator.__path__[0], "training", "LesionLocatorTrainer"),
                                                    trainer_name_tracker, 'lesionlocator.training.LesionLocatorTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name_tracker} in lesionlocator.training.LesionLocatorTrainer. '
                               f'Please place it there (in any .py file)!')
        network_tracker = trainer_class.build_network_architecture(
            configuration_manager_tracker.network_arch_class_name,
            configuration_manager_tracker.network_arch_init_kwargs,
            configuration_manager_tracker.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager_tracker.get_label_manager(dataset_json_tracker).num_segmentation_heads,
            configuration_manager_tracker.patch_size,
            enable_deep_supervision=False
        )

        if self.petct_mode:
            tracker_first_conv_key = self._find_first_conv_key(parameters_tracker[0])
            expected_in_ch = num_input_channels + 1
            current_in_ch = parameters_tracker[0][tracker_first_conv_key].shape[1]
            if current_in_ch < expected_in_ch:
                parameters_tracker = [self._extend_first_conv_weights(
                                          p, tracker_first_conv_key,
                                          num_new_channels=expected_in_ch - current_in_ch)
                                      for p in parameters_tracker]
            else:
                print(f'[petct] Tracker checkpoint first conv already has {current_in_ch} input '
                      f'channels (expected {expected_in_ch}); skipping extension.')
       
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager_tracker
        self.plans_manager_tracker = plans_manager_tracker
        self.configuration_manager_tracker = configuration_manager_tracker
        self.list_of_parameters_tracker = parameters_tracker

        if not reinit:
            network_tracker.load_state_dict(parameters_tracker[0])

        self.network_tracker = network_tracker
        self.dataset_json_tracker = dataset_json_tracker
        self.trainer_name_tracker = trainer_name_tracker
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager_tracker.get_label_manager(dataset_json_tracker)
        # For LesionLocatorTrack, always use tracker spacing
        self.target_spacing = self.configuration_manager_tracker.spacing
        print('Using target spacing: ', self.target_spacing)
        print('Segmentation configuration: ', self.configuration_manager)
        print('Tracking configuration: ', self.configuration_manager_tracker)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

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
        print('NETWORK INPUT SHAPE: ', x.shape)
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


    def setup_tracking_training(self, learning_rate=1e-4, weight_decay=1e-5, use_scheduler=True, finetune_mode='all'):
        """
        Setup training components for tracking network: optimizer, loss function, and scheduler.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            use_scheduler: Whether to use learning rate scheduler
            finetune_mode: Which part to finetune ('reg_net', 'unet', 'all')
        """
        if self.network_tracker is None:
            raise RuntimeError("Tracking network not initialized. Call initialize_from_trained_model_folder first.")
        
        # Set tracking network to training mode
        self.network_tracker.train()
        self.training_mode = True
        
        # Freeze/unfreeze parameters based on finetune_mode for tracking network
        self._configure_tracking_trainable_parameters(finetune_mode)
        
        # Get trainable parameters for optimizer
        trainable_params = [p for p in self.network_tracker.parameters() if p.requires_grad]
        
        # Setup optimizer with only trainable parameters
        self.optimizer = optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function for tracking (segmentation + registration loss)
        self.loss_function = self._tracking_combined_loss
        
        # Setup learning rate scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = None
        
        # Count and print trainable parameters
        total_params = sum(p.numel() for p in self.network_tracker.parameters())
        trainable_params_count = sum(p.numel() for p in self.network_tracker.parameters() if p.requires_grad)
        frozen_params_count = total_params - trainable_params_count
        
        print(f"Tracking training setup complete. Mode: {finetune_mode}, LR: {learning_rate}, Device: {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params_count:,} ({trainable_params_count/1e6:.2f}M)")
        print(f"Frozen parameters: {frozen_params_count:,} ({frozen_params_count/1e6:.2f}M)")
        print(f"Trainable ratio: {100*trainable_params_count/total_params:.1f}%")

    def setup_training(self, learning_rate=1e-4, weight_decay=1e-5, use_scheduler=True, finetune_mode='all'):
        """
        Setup training components: optimizer, loss function, and scheduler.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            use_scheduler: Whether to use learning rate scheduler
            finetune_mode: Which part to finetune ('encoder', 'decoder', 'all')
        """
        if self.network is None:
            raise RuntimeError("Network not initialized. Call initialize_from_trained_model_folder first.")
        
        # Set network to training mode
        self.network.train()
        self.training_mode = True
        
        # Freeze/unfreeze parameters based on finetune_mode
        self._configure_trainable_parameters(finetune_mode)
        
        # Get trainable parameters for optimizer
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        
        # Setup optimizer with only trainable parameters
        self.optimizer = optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function (CrossEntropy + Dice loss)
        self.loss_function = self._combined_loss
        
        # Setup learning rate scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = None
        
        # Count and print trainable parameters
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params_count = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        frozen_params_count = total_params - trainable_params_count
        
        print(f"Training setup complete. Mode: {finetune_mode}, LR: {learning_rate}, Device: {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params_count:,} ({trainable_params_count/1e6:.2f}M)")
        print(f"Frozen parameters: {frozen_params_count:,} ({frozen_params_count/1e6:.2f}M)")
        print(f"Trainable ratio: {100*trainable_params_count/total_params:.1f}%")

    def _configure_trainable_parameters(self, finetune_mode='all'):
        """
        Configure which parameters are trainable based on finetune mode.
        
        Args:
            finetune_mode: 'encoder', 'decoder', or 'all'
        """
        print(f"Configuring trainable parameters for mode: {finetune_mode}")
        
        # Print network architecture for inspection
        print("\n" + "="*80)
        print("NETWORK ARCHITECTURE:")
        print("="*80)
        for name, param in self.network.named_parameters():
            print(f"{name}: {param.shape}")
        print("="*80 + "\n")
        
        if finetune_mode == 'all':
            # Enable gradients for all parameters
            for param in self.network.parameters():
                param.requires_grad = True
            print("All parameters enabled for training")
            
        elif finetune_mode == 'encoder':
            # Freeze all parameters first
            for param in self.network.parameters():
                param.requires_grad = False
            
            # Enable encoder parameters (encoder.stem.* and encoder.stages.*)
            enabled_count = 0
            
            for name, param in self.network.named_parameters():
                # Check if parameter belongs to encoder
                if name.startswith('encoder.'):
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  Enabled: {name}")
            
            print(f"Encoder mode: {enabled_count} parameter groups enabled")
            
        elif finetune_mode == 'decoder':
            # Freeze all parameters first
            for param in self.network.parameters():
                param.requires_grad = False
            
            # Enable decoder parameters (decoder.stages.*, decoder.transpconvs.*, decoder.seg_layers.*)
            enabled_count = 0
            
            for name, param in self.network.named_parameters():
                # Check if parameter belongs to decoder
                if name.startswith('decoder.'):
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  Enabled: {name}")
            
            print(f"Decoder mode: {enabled_count} parameter groups enabled")
            
        else:
            raise ValueError(f"Unknown finetune_mode: {finetune_mode}. Use 'encoder', 'decoder', or 'all'")
        
        # Print summary of enabled/disabled parameters
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    def _combined_loss(self, predictions, targets):
        """Combined CrossEntropy + Dice loss for segmentation training."""
        # CrossEntropy loss
        # apply focal loss
        # check if target or prediction
        print(f"targets all zero: {torch.all(targets==0)}, pred all zero: {torch.all(predictions==0)}")
        weights = torch.tensor([0.01, 0.99], device=predictions.device)  # Adjust weights for foreground and background
        ce_loss = nn.CrossEntropyLoss(weight=weights)(predictions, targets.long())

        #ce_loss = nn.CrossEntropyLoss()(predictions, targets.long())
        
        # Dice loss (use external function)
        dice_loss_val = dice_loss(predictions, targets)

        print(ce_loss)
        
        return ce_loss + dice_loss_val
    
    def _tracking_combined_loss(self, seg_output, reg_loss, targets, reg_loss_weight=1.0, scale_factor=1.0):
        """Combined segmentation and registration loss for tracking training."""
        # Segmentation loss (CrossEntropy + Dice)
        seg_loss = self._combined_loss(seg_output, targets)
        
        # Registration loss (if available)
        total_reg_loss = 0.0
        if reg_loss is not None:
            total_reg_loss = reg_loss.all_loss if hasattr(reg_loss, 'all_loss') else reg_loss
        
        # Combined loss
        total_loss = seg_loss + reg_loss_weight * total_reg_loss
        
        # Apply scaling for gradient accumulation
        # if scale_factor != 1.0:
        #     total_loss = total_loss * scale_factor
        
        return total_loss, seg_loss, total_reg_loss
    
    def _configure_tracking_trainable_parameters(self, finetune_mode='all'):
        """
        Configure which parameters are trainable for tracking network.
        
        Args:
            finetune_mode: 'reg_net', 'unet', or 'all'
        """
        print(f"Configuring tracking trainable parameters for mode: {finetune_mode}")
        
        if finetune_mode == 'all':
            # Enable gradients for all parameters
            for param in self.network_tracker.parameters():
                param.requires_grad = True
            print("All tracking parameters enabled for training")
            
        elif finetune_mode == 'reg_net':
            # Freeze all parameters first
            for param in self.network_tracker.parameters():
                param.requires_grad = False
            
            # Enable registration network parameters only
            enabled_count = 0
            for name, param in self.network_tracker.named_parameters():
                if name.startswith('reg_net.'):
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  Enabled: {name}")
            
            print(f"Registration network mode: {enabled_count} parameter groups enabled")
            
        elif finetune_mode == 'unet':
            # Freeze all parameters first
            for param in self.network_tracker.parameters():
                param.requires_grad = False
            
            # Enable UNet parameters only
            enabled_count = 0
            for name, param in self.network_tracker.named_parameters():
                if name.startswith('unet.decoder.'):
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  Enabled: {name}")
            
            print(f"UNet mode: {enabled_count} parameter groups enabled")
            
        else:
            raise ValueError(f"Unknown finetune_mode: {finetune_mode}. Use 'reg_net', 'unet', or 'all'")
        
        # Print summary of enabled/disabled parameters
        trainable_params = sum(p.numel() for p in self.network_tracker.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.network_tracker.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    def _visualize_validation_sample(self, data, target, prediction, filename, output_folder, epoch):
        """
        Visualize validation sample with overlaid masks in both axial and coronal views.
        Shows the cropped data that the network actually sees during training.
        
        Args:
            data: Input image [C, H, W, D] (cropped data seen by network)
            target: Ground truth mask [H, W, D] (cropped data seen by network)
            prediction: Predicted mask [H, W, D] (cropped data seen by network)
            filename: Sample filename
            output_folder: Output folder path
            epoch: Current epoch number
        """
        # Create epoch folder
        epoch_folder = os.path.join(output_folder, f'epoch_{epoch}')
        os.makedirs(epoch_folder, exist_ok=True)
        
        # Convert to numpy arrays
        if isinstance(data, torch.Tensor):
            data_np = data[0].cpu().numpy() if len(data.shape) > 3 else data.cpu().numpy()
        else:
            data_np = data[0] if len(data.shape) > 3 else data
            
        if isinstance(target, torch.Tensor):
            target_np = target.cpu().numpy()
        else:
            target_np = target
            
        if isinstance(prediction, torch.Tensor):
            pred_np = prediction.cpu().numpy()
        else:
            pred_np = prediction
        
        # Find the axial slice with the most target pixels (first dimension)
        target_sums_axial = np.sum(target_np, axis=(1, 2))  # Sum over each axial slice
        max_axial_slice = np.argmax(target_sums_axial) if np.max(target_sums_axial) > 0 else target_np.shape[0] // 2
        
        # Find the coronal slice with the most target pixels (second dimension)
        target_sums_coronal = np.sum(target_np, axis=(0, 2))  # Sum over each coronal slice
        max_coronal_slice = np.argmax(target_sums_coronal) if np.max(target_sums_coronal) > 0 else target_np.shape[1] // 2
        
        # Create visualization with 2 rows and 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # First row - Axial view
        # Original image
        axes[0, 0].imshow(data_np[max_axial_slice, :, :], cmap='gray')
        axes[0, 0].set_title('Original Image (Axial)')
        axes[0, 0].axis('off')
        
        # Ground truth overlay
        axes[0, 1].imshow(data_np[max_axial_slice, :, :], cmap='gray')
        axes[0, 1].imshow(target_np[max_axial_slice, :, :], alpha=0.5, cmap='Reds')
        axes[0, 1].set_title('Ground Truth (Axial)')
        axes[0, 1].axis('off')
        
        # Prediction overlay
        axes[0, 2].imshow(data_np[max_axial_slice, :, :], cmap='gray')
        axes[0, 2].imshow(pred_np[max_axial_slice, :, :], alpha=0.5, cmap='Blues')
        axes[0, 2].set_title('Prediction (Axial)')
        axes[0, 2].axis('off')
        
        # Second row - Coronal view
        # Original image
        axes[1, 0].imshow(data_np[:, max_coronal_slice, :], cmap='gray', origin='lower')
        axes[1, 0].set_title('Original Image (Coronal)')
        axes[1, 0].axis('off')
        
        # Ground truth overlay
        axes[1, 1].imshow(data_np[:, max_coronal_slice, :], cmap='gray', origin='lower')
        axes[1, 1].imshow(target_np[:, max_coronal_slice, :], alpha=0.5, cmap='Reds', origin='lower')
        axes[1, 1].set_title('Ground Truth (Coronal)')
        axes[1, 1].axis('off')
        
        # Prediction overlay
        axes[1, 2].imshow(data_np[:, max_coronal_slice, :], cmap='gray', origin='lower')
        axes[1, 2].imshow(pred_np[:, max_coronal_slice, :], alpha=0.5, cmap='Blues', origin='lower')
        axes[1, 2].set_title('Prediction (Coronal)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(epoch_folder, f'{filename}_axial_{max_axial_slice}_coronal_{max_coronal_slice}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def train_cross_validation(self, train_input_files, train_prompt_files, train_output_files,
                              test_dataset=None, epochs=10, batch_size=2, lr=1e-4, 
                              device=None, output_folder=None, n_folds=5, num_workers=4, prompt_type='box',
                              ckpt_path=None, finetune_mode='all', train_fold=None):
        """
        Perform 5-fold cross-validation training.
        
        Args:
            train_input_files: Training input files
            train_prompt_files: Training prompt files  
            train_output_files: Training output files
            test_dataset: Test dataset for evaluation and visualization
            epochs: Number of epochs per fold
            batch_size: Batch size
            lr: Learning rate
            device: Training device
            output_folder: Output folder for checkpoints
            n_folds: Number of cross-validation folds
            num_workers: Number of preprocessing workers
            prompt_type: Type of prompt ('box', 'point', etc.)
        """
        if device is None:
            device = self.device
        
        print(f"Starting {n_folds}-fold cross-validation...")
        print(f"Total training samples: {len(train_input_files)}")
        print(f"Total prompt samples: {len(train_prompt_files)}")


        # Create cross-validation folds
        folds = create_cv_folds(train_input_files, train_prompt_files, train_output_files, n_folds)
        if train_fold is not None and not 0 <= train_fold < len(folds):
            raise ValueError(
                f"train_fold must be between 0 and {len(folds) - 1} for the current dataset, got {train_fold}."
            )
        n_folds = len(folds)
        
        # Store results from all folds
        all_fold_results = []
        
        for fold_idx, fold_data in enumerate(folds):
            if fold_idx!=train_fold and train_fold is not None:
                continue
            print(f"\n{'='*50}")
            print(f"Starting: {fold_idx + 1}/{n_folds}")
            print(f"{'='*50}")
            
            # Reset network weights for each fold (reload from checkpoint).
            # Prefer fold-specific weights when an ensemble of checkpoints has been loaded;
            # fall back to index 0 when only a single pretrained snapshot is available.
            init_idx = fold_idx if fold_idx < len(self.list_of_parameters) else 0
            self.network.load_state_dict(self.list_of_parameters[init_idx])
            
            # Train this fold
            fold_results = self.train_cv_fold(
                fold_data=fold_data,
                test_dataset=test_dataset,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                output_folder=output_folder,
                fold_idx=fold_idx,
                num_workers=num_workers,
                prompt_type=prompt_type,
                ckpt_path=ckpt_path,
                finetune_mode=finetune_mode
            )
            
            all_fold_results.append(fold_results)
            gc.collect()
            if _uses_cuda_device(device):
                print(f"GPU memory allocated: {_cuda_memory_allocated_gb(device):.2f} GB")
                print(f"GPU memory reserved: {_cuda_memory_reserved_gb(device):.2f} GB")

            print(f"Fold {fold_idx+1} completed!")
            print(f"Best validation loss: {fold_results['best_val_loss']:.4f}")
            if fold_results['test_dice_scores']:
                print(f"Final test dice: {fold_results['test_dice_scores'][-1]:.4f}")

            _maybe_empty_cache(device)
        
        # Compute cross-validation statistics
        final_val_losses = [fold['val_losses'][-1] for fold in all_fold_results]
        best_val_losses = [fold['best_val_loss'] for fold in all_fold_results]
        
        if all_fold_results[0]['test_dice_scores']:
            final_test_dice = [fold['test_dice_scores'][-1] for fold in all_fold_results]
            print(f"\nCross-Validation Results:")
            print(f"Mean validation loss: {np.mean(final_val_losses):.4f} ± {np.std(final_val_losses):.4f}")
            print(f"Mean best validation loss: {np.mean(best_val_losses):.4f} ± {np.std(best_val_losses):.4f}")
            print(f"Mean test dice: {np.mean(final_test_dice):.4f} ± {np.std(final_test_dice):.4f}")
        
        # Save cross-validation summary
        if output_folder:
            cv_summary = {
                'n_folds': n_folds,
                'fold_results': all_fold_results,
                'mean_val_loss': np.mean(final_val_losses),
                'std_val_loss': np.std(final_val_losses),
                'mean_best_val_loss': np.mean(best_val_losses),
                'std_best_val_loss': np.std(best_val_losses)
            }
            
            if all_fold_results[0]['test_dice_scores']:
                cv_summary['mean_test_dice'] = np.mean(final_test_dice)
                cv_summary['std_test_dice'] = np.std(final_test_dice)
            
            with open(os.path.join(output_folder, 'cv_summary.json'), 'w') as f:
                json.dump(cv_summary, f, indent=2)
        
        return all_fold_results

    def train_cv_fold(self, fold_data, test_dataset=None, epochs=10, batch_size=2, lr=1e-4, 
                      device=None, output_folder=None, fold_idx=None, num_workers=4, prompt_type='box',
                      ckpt_path=None, finetune_mode='all'):
        """
        Training function for a single cross-validation fold.
        
        Args:
            fold_data: Dictionary containing train and val data for this fold
            test_dataset: Test dataset for dice score computation and visualization
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            device: Training device
            output_folder: Folder to save checkpoints
            fold_idx: Current fold index
            num_workers: Number of preprocessing workers
            prompt_type: Type of prompt ('box', 'point', etc.)
        """
        if device is None:
            device = self.device
            
        print(f"\n=== Training Fold {fold_idx} ===")
        
        # Create datasets for this fold
        train_dataset = self.create_training_dataset(
            input_files=fold_data['train']['input_files'],
            prompt_files=fold_data['train']['prompt_files'],
            output_files=fold_data['train']['output_files'],
            prompt_type=prompt_type,
            num_processes=num_workers,
            verbose=False,
            track=False
        )
        
        val_dataset = self.create_training_dataset(
            input_files=fold_data['val']['input_files'],
            prompt_files=fold_data['val']['prompt_files'],
            output_files=fold_data['val']['output_files'],
            prompt_type=prompt_type,
            num_processes=num_workers,
            verbose=False,
            track=False
        )
        
        # Setup training components
        self.setup_training(learning_rate=lr, finetune_mode=finetune_mode)
        self.network.to(device)
        
        # The IterableDataset starts its own preprocessing workers via
        # preprocessing_iterator_fromfiles. PyTorch DataLoader workers are daemonic,
        # so they cannot safely start that inner multiprocessing pipeline.
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=training_collate_fn,
            num_workers=0,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=training_collate_fn,
            num_workers=0,
        )

        test_dataloader = None
        if test_dataset is not None:
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                collate_fn=training_collate_fn,
                num_workers=0,
            )
        
        # Training history for this fold
        fold_train_losses = []
        fold_val_losses = []
        fold_test_dice_scores = []
        best_val_loss = float('inf')

        print(f"Fold {fold_idx} - Training samples: {len(fold_data['train']['input_files'])}, Prompt samples: {len(fold_data['train']['prompt_files'])}")
        print(f"Fold {fold_idx} - Validation samples: {len(fold_data['val']['input_files'])}, Prompt samples: {len(fold_data['val']['prompt_files'])}")
        if test_dataset is not None:
            print(f"Test samples: {len(test_dataset.input_files)}")
        
        # Check for existing checkpoint to resume training
        start_epoch = 0
        if output_folder and fold_idx is not None:
            fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
            checkpoint_path = os.path.join(fold_folder, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                print(f"Found existing checkpoint: {checkpoint_path}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    self.network.load_state_dict(checkpoint['network_weights'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                    if 'scheduler_state' in checkpoint and self.scheduler is not None:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                    start_epoch = checkpoint['epoch'] + 1
                    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    print(f"Resuming training from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    print("Starting fresh training...")
                    start_epoch = 0
            else:
                print("No existing checkpoint found, starting fresh training...")

        print(f"number of batches approximately: {len(train_dataloader)}")
        for epoch in range(start_epoch, epochs):
            # Training phase
            self.network.train()
            epoch_train_loss = 0.0
            num_train_batches = 0 
            
            print(f"\nFold {fold_idx}, Epoch {epoch+1}/{epochs}")
            print("Training...")
            file_names = set()

            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    data = batch['data'].to(device)
                    prompt = batch['prompt'].to(device)
                    target = batch['target'].to(device)
                    
                    if data.dim() == 4:
                        data = data.unsqueeze(0)
                        prompt = prompt.unsqueeze(0)
                        target = target.unsqueeze(0)
                    
                    combined_input = torch.cat([data, prompt], dim=1)

                    self.optimizer.zero_grad()

                    with _autocast_context(device):
                        outputs = self.network(combined_input)
                        loss = self.loss_function(outputs, target)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                                        
                    epoch_train_loss += loss.item()
                    num_train_batches += 1
                    
                    _maybe_empty_cache(device)
                    #if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
                    if _uses_cuda_device(device):
                        print(f"  GPU memory allocated: {_cuda_memory_allocated_gb(device):.2f} GB")
                        print(f"  GPU memory reserved: {_cuda_memory_reserved_gb(device):.2f} GB")
                        
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    _maybe_empty_cache(device)
                    continue

            avg_train_loss = epoch_train_loss / max(num_train_batches, 1)
            fold_train_losses.append(avg_train_loss)
            print(f"Training Loss: {avg_train_loss:.4f}")
            
            # Validation phase (for loss computation only)
            self.network.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            
            print("Validating (loss computation on CV fold)...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    try:
                        data = batch['data'].to(device)
                        prompt = batch['prompt'].to(device)
                        target = batch['target'].to(device)
                        
                        if data.dim() == 4:
                            data = data.unsqueeze(0)
                            prompt = prompt.unsqueeze(0)
                            target = target.unsqueeze(0)
                        
                        combined_input = torch.cat([data, prompt], dim=1)

                        with _autocast_context(device):
                            outputs = self.network(combined_input)
                            loss = self.loss_function(outputs, target)
                        
                        epoch_val_loss += loss.item()
                        num_val_batches += 1
                        
                        _maybe_empty_cache(device)
                    
                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        _maybe_empty_cache(device)
                        continue

            avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
            fold_val_losses.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Test phase (for dice computation and visualization)
            if test_dataset is not None:
                print("Testing (dice computation and visualization on test data)...")
                epoch_test_dice_scores = []
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_dataloader):
                        try:
                            data = batch['data'].to(device)
                            prompt = batch['prompt'].to(device)
                            target = batch['target'].to(device)
                            filenames = batch['filename']
                            
                            if data.dim() == 4:
                                data = data.unsqueeze(0)
                                prompt = prompt.unsqueeze(0)
                                target = target.unsqueeze(0)
                                filenames = [filenames]
                            
                            combined_input = torch.cat([data, prompt], dim=1)
    
                            with _autocast_context(device):
                                outputs = self.network(combined_input)
                            
                            # Process each sample for dice computation and visualization
                            for i in range(data.shape[0]):
                                filename = os.path.basename(filenames[i]).replace('.nii.gz', '')
                                
                                output_single = outputs[i:i+1]
                                pred_probs = torch.softmax(output_single, dim=1)
                                pred_classes = torch.argmax(pred_probs, dim=1).squeeze(0)
                                
                                data_single = data[i]
                                target_single = target[i]
                                
                                pred_cropped = pred_classes.cpu().numpy()
                                target_cropped = target_single.cpu().numpy()
                                
                                # Compute Dice score on test data
                                dice_score = compute_dice_coefficient(target_cropped, pred_cropped)
                                epoch_test_dice_scores.append(dice_score)
                                
                                if self.visualize:
                                    # Visualize test samples (first few batches only)
                                    if batch_idx < 1 and output_folder:
                                        test_viz_folder = os.path.join(output_folder, f'fold_{fold_idx}', 'test_visualizations')
                                        self._visualize_validation_sample(
                                            data_single, target_cropped, pred_cropped,
                                            f'{filename}_fold_{fold_idx}_epoch_{epoch}_batch_{batch_idx}_sample_{i}',
                                            test_viz_folder, epoch
                                        )
                            _maybe_empty_cache(device)
                        except Exception as e:
                            print(f"Error in test batch {batch_idx}: {e}")
                            _maybe_empty_cache(device)
                            continue

                avg_test_dice = np.mean(epoch_test_dice_scores) if epoch_test_dice_scores else 0.0
                fold_test_dice_scores.append(avg_test_dice)
                print(f"Test Dice Score: {avg_test_dice:.4f}")
            
            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step(avg_val_loss)
            
            # Save best model for this fold (based on validation loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if output_folder:
                    fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
                    self._save_checkpoint(fold_folder, 'best_model.pth', epoch, fold_idx=fold_idx, 
                                        ckpt_path=ckpt_path, prompt_type=prompt_type, best_val_loss=best_val_loss)
                    print(f"New best model saved for fold {fold_idx} (val_loss: {avg_val_loss:.4f})")
            
            # Save periodic checkpoint
            if output_folder and (epoch + 1) % 10 == 0:
                fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
                self._save_checkpoint(fold_folder, f'checkpoint_epoch_{epoch+1}.pth', epoch, fold_idx=fold_idx)
        
        # Save final checkpoint for this fold
        if output_folder:
            fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
            self._save_checkpoint(fold_folder, 'final_checkpoint.pth', epochs-1, fold_idx=fold_idx)
        
        return {
            'fold_idx': fold_idx,
            'train_losses': fold_train_losses,
            'val_losses': fold_val_losses,
            'test_dice_scores': fold_test_dice_scores,
            'best_val_loss': best_val_loss
        }

    def _save_checkpoint(self, output_folder, filename, epoch, 
                         fold_idx=None, ckpt_path=None, prompt_type='point', best_val_loss=None,
                         configuration=None):
        """Save model checkpoint and optionally save inference-compatible checkpoint."""
        os.makedirs(output_folder, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'network_weights': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'trainer_name': self.trainer_name,
        }
        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
        if best_val_loss is not None:
            checkpoint['best_val_loss'] = best_val_loss
            
        checkpoint_path = os.path.join(output_folder, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save inference-compatible checkpoint if ckpt_path is provided and this is a best model
        if ckpt_path and filename == 'best_model.pth' and fold_idx is not None:
            # Create inference-compatible directory structure
            optimized_folder = "point_optimized" if prompt_type == 'point' else "bbox_optimized"
            inference_dir = os.path.join(ckpt_path, 'LesionLocatorSeg', optimized_folder, f'fold_{fold_idx}')
            os.makedirs(inference_dir, exist_ok=True)
            
            # Get configuration name - this should match what's expected in the loading code
            # The loading code expects checkpoint['init_args']['configuration']
            config_name = getattr(self, 'configuration_name', None)
            if config_name is None:
                # Try to get it from the configuration manager or use a default
                config_name = 'default'  # This should be set to the actual configuration name used
                print(f"Warning: configuration_name not found, using default: {config_name}")
            
            # Create inference-compatible checkpoint with the same structure as the loaded checkpoints
            inference_checkpoint = {
                'network_weights': self.network.state_dict(),
                'trainer_name': self.trainer_name,
                'init_args': {
                    'configuration': config_name,
                },
                'inference_allowed_mirroring_axes': getattr(self, 'allowed_mirroring_axes', None),
            }
            
            inference_checkpoint_path = os.path.join(inference_dir, 'checkpoint_final.pth')
            torch.save(inference_checkpoint, inference_checkpoint_path)
            print(f"Inference-compatible checkpoint saved: {inference_checkpoint_path}")
            print(f"  - trainer_name: {self.trainer_name}")
            print(f"  - configuration: {config_name}")
            print(f"  - allowed_mirroring_axes: {getattr(self, 'allowed_mirroring_axes', None)}")

    def create_training_dataset(self, input_files, prompt_files, output_files, prompt_type,
                               num_processes=3, verbose=False, track=False):
        """
        Create a training dataset using the existing multiprocessing pipeline.
        
        Args:
            input_files: List of input image files
            prompt_files: List of prompt files (JSON or segmentation masks)
            output_files: List of output file paths
            prompt_type: Type of prompt ('point', 'box', etc.)
            num_processes: Number of preprocessing workers
            verbose: Verbose output
            track: Whether to include tracking data
            
        Returns:
            LesionDatasetWrapper instance
        """
        input_files = [i for i in input_files if 'TP0' not in os.path.basename(_representative_path(i))]
        prompt_files = [p for p in prompt_files if 'TP0' not in os.path.basename(p)]
        output_files = [o for o in output_files if 'TP0' not in os.path.basename(o)]

        return LesionDatasetWrapper(
            input_files=input_files,
            prompt_files=prompt_files,
            output_files=output_files,
            prompt_type=prompt_type,
            plans_config=self.plans_manager_tracker.plans,
            # plans_manager=self.plans_manager,
            dataset_json=self.dataset_json_tracker,
            configuration_config=self.configuration_name_tracker,
            modality=self.modality,
            # configuration_manager=self.configuration_manager,
            num_processes=num_processes,
            pin_memory=self.device.type == 'cuda',
            verbose=verbose,
            track=track
        )
    
    
    def train_tracking(self, train_dataset, val_dataset=None, test_dataset=None, epochs=10, batch_size=1, lr=1e-4, 
                      device=None, output_folder=None, num_workers=0, finetune_mode='all', gradient_accumulation_steps=1):
        """
        Training function for tracking model using paired baseline and follow-up data.
        
        Args:
            train_dataset: iterable tracking dataset for training data
            val_dataset: iterable tracking dataset for validation data (optional)
            test_dataset: Dataset for test evaluation (optional)
            epochs: Number of training epochs
            batch_size: Batch size (typically 1 for tracking due to memory constraints)
            lr: Learning rate
            device: Training device (uses self.device if None)
            output_folder: Folder to save checkpoints
            num_workers: Number of preprocessing workers
            finetune_mode: Which part to finetune ('reg_net', 'unet', 'all')
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating
        """
        if device is None:
            device = self.device
            
        # Setup tracking training components
        self.setup_tracking_training(learning_rate=lr, finetune_mode=finetune_mode)
        
        # Move tracking network to device
        self.network_tracker.to(device)
        
        # Create DataLoaders with tracking collate function. The IterableDataset
        # starts its own preprocessing workers, so DataLoader workers must stay 0.
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=tracking_collate_fn,
            num_workers=0
        )

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=1, # can only be 1 for validation
                collate_fn=tracking_collate_fn,
                num_workers=0
            )


        test_dataloader = None
        if test_dataset is not None:
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                collate_fn=tracking_collate_fn,
                num_workers=0
            )
        
        # Training history
        train_losses = []
        train_seg_losses = []
        train_reg_losses = []
        val_losses = []
        val_dice_scores = []
        test_dice_scores = []
        best_val_loss = float('inf')
        
        print(f"Starting tracking training for {epochs} epochs...")
        print(f"Device: {device}, Learning rate: {lr}")
        print(f"Batch size: {batch_size}, Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        
        # Monitor OOM events
        oom_count = 0
        max_oom_retries = 6  # Allow more retries
        
        # Clear any existing GPU memory
        _maybe_empty_cache(device)
        import gc
        gc.collect()
        
        # Print initial memory status
        if _uses_cuda_device(device):
            print(f"Initial GPU memory: {_cuda_memory_allocated_gb(device):.2f} GB allocated")
            print(f"Total GPU memory: {_cuda_total_memory_gb(device):.2f} GB")
        
        # Suggest memory-efficient settings if memory is limited
        total_memory = _cuda_total_memory_gb(device)
        if total_memory is not None and total_memory < 12:  # Less than 12GB
            print("WARNING: Limited GPU memory detected. Consider:")
            print("  - Using batch_size=1 and gradient_accumulation_steps=2-4")
            print("  - Setting finetune_mode='reg_net' to freeze UNet")
            print("  - Reducing input patch size in data preprocessing")
            
        # Memory optimization settings
        if _uses_cuda_device(device):
            torch.backends.cudnn.benchmark = False  # Disable for consistent memory usage
            torch.backends.cudnn.deterministic = True
        
        for epoch in range(epochs):
            # # Training phase
            self.network_tracker.train()
            epoch_train_loss = 0.0
            epoch_seg_loss = 0.0
            epoch_reg_loss = 0.0
            num_train_batches = 0
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("Training...")
            
            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    baseline_data = batch['bl_data'].to(device)      # [B, C, H, W, D]
                    followup_data = batch['fu_data'].to(device)      # [B, C, H, W, D]
                    baseline_prompt = batch['bl_prompt'].to(device)  # [B, 1, H, W, D]
                    
                    target = batch['target'].to(device)                    # [B, H, W, D]
                    filenames = batch['filename']

                    # if batch_idx % 50 == 0:
                    #     nonzero_indices = torch.sum(baseline_prompt, axis = (2,3)).nonzero(as_tuple=False)
                    #     nonzero_indices = nonzero_indices[:, 1]
                    #     nonzero_indices_target = torch.sum(target, axis = (1,2)).nonzero(as_tuple=False)
                    #     nonzero_indices_target = nonzero_indices_target[:, 0]
                    #     # # print(f"Non-zero slice indices in baseline prompt: {nonzero_indices.tolist()}")
                    #     # # print(f"Non-zero slice indices in target: {nonzero_indices_target.tolist()}")
                    #     indices = set(nonzero_indices.tolist()).intersection(set(nonzero_indices_target.tolist()))
                    #     overlap = baseline_prompt * target.unsqueeze(1)  # [B, 1, H, W, D]
                    #     overlap_sum = torch.sum(overlap, axis=(2,3)).nonzero(as_tuple=False).squeeze()
                    #     overlap_sum = overlap_sum[:, 1].squeeze()
                    #     # # print(f"Overlap slice indices between prompt and target: {overlap_sum}")

                    #     if len(indices) == 0:
                    #         slice_idx = 0
                    #     else:
                    #         slice_idx = sorted(indices)[len(indices)//2]
                    #     # slice_idx = nonzero_indices[sorted(indices)[len(indices)//2]].item()
                    #     plt.imshow(followup_data[0, slice_idx].detach().cpu(), cmap='gray')
                    #     plt.imshow(baseline_prompt[0, slice_idx].detach().cpu(), cmap='Reds', alpha=0.5)
                    #     plt.imshow(target[slice_idx].detach().cpu(), cmap='Greens', alpha=0.5)

                    #     plt.title(f'Baseline Prompt Slice {slice_idx}, overlap: {len(indices)}')
                    #     plt.show()
                    #     plt.savefig(f'/home/xiachen/scripts/warped_vis/baseline_prompt_epoch{epoch+1}_batch{batch_idx}.png')
                    #     plt.close()

                    # Remove channel dimension from prompt - tracknet expects [B, H, W, D]
                    if baseline_prompt.dim() == 5 and baseline_prompt.shape[1] == 1:
                        baseline_prompt = baseline_prompt.squeeze(1)  # [B, H, W, D]
                    
                    # Handle both batched and single sample data
                    if baseline_data.dim() == 4:  # Single sample [C, H, W, D]
                        baseline_data = baseline_data.unsqueeze(0)      # [1, C, H, W, D]
                        followup_data = followup_data.unsqueeze(0)      # [1, C, H, W, D]
                        baseline_prompt = baseline_prompt.unsqueeze(0)  # [1, H, W, D]
                        target = target.unsqueeze(0)                    # [1, H, W, D]
                    
                    # Clear gradients at start of accumulation cycle
                    #if batch_idx % gradient_accumulation_steps == 0:
                    
                    self.optimizer.zero_grad()
                    
                    with _autocast_context(device):
                        # Training mode with proper x1_mask parameter
                        try:
                            seg_output, reg_loss, x1_mask_cropped = self.network_tracker(baseline_data, followup_data, baseline_prompt, 
                                                                                         is_inference=False, x1_mask=target, visualize=False)

                            x1_mask_cropped = x1_mask_cropped.squeeze(1)  # [B, H, W, D]
                            # Calculate combined loss using the cropped target
                            # Calculate combined loss with scaling for gradient accumulation
                            total_loss, seg_loss, reg_loss_val = self._tracking_combined_loss(seg_output, reg_loss, x1_mask_cropped)

                            if batch_idx % 50 == 0:
                                # Compute and print dice on cropped target for monitoring
                                dice_loss_val = dice_loss(seg_output, x1_mask_cropped)
                                dice = 1.0 - dice_loss_val.item()
                                # seg = torch.softmax(seg_output, 1).argmax(1)
                                # pred = seg.detach().cpu().numpy().astype(np.uint8)
                                # dice = compute_dice_coefficient(pred, x1_mask_cropped.detach().cpu().numpy().astype(np.uint8))
                                # print(f"pred: {set(np.unique(pred))}, target: {set(np.unique(x1_mask_cropped.detach().cpu().numpy().astype(np.uint8)))}")
                                print(f"Batch {batch_idx}, Dice on cropped target: {dice:.4f}, reg_loss: {reg_loss_val:.4f}", flush=True)
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                oom_count += 1
                                # print(f"CUDA OOM in forward pass at batch {batch_idx} (OOM #{oom_count}). Clearing cache and skipping batch.")
                                # print(f"Current GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
                                
                                if oom_count >= max_oom_retries:
                                    print(f"Too many OOM errors ({oom_count}). Consider reducing batch size or gradient accumulation steps.")
                                    print("Suggested fixes:")
                                    print("  1. Use --batch_size 1 --gradient_accumulation_steps 1")
                                    print("  2. Use --finetune reg_net to only train registration network")
                                    print("  3. Check if your GPU has enough memory for this model")
                                    raise RuntimeError(f"Too many OOM errors ({oom_count}). Training stopped.")
                                
                                # Aggressive cleanup
                                _maybe_empty_cache(device)
                                # Clear intermediate variables
                                del baseline_data, followup_data, baseline_prompt, target
                                if 'seg_output' in locals():
                                    del seg_output
                                if 'reg_loss' in locals():
                                    del reg_loss
                                if 'x1_mask_cropped' in locals():
                                    del x1_mask_cropped
                                _maybe_empty_cache(device)
                                import gc
                                gc.collect()
                                continue
                            else:
                                raise e
                    
                    # Backward pass
                    try:
                        self.scaler.scale(total_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            oom_count += 1
                            print(f"CUDA OOM in backward pass at batch {batch_idx} (OOM #{oom_count}). Clearing cache and skipping batch.")
                            if _uses_cuda_device(device):
                                print(f"Current GPU memory: {_cuda_memory_allocated_gb(device):.2f} GB allocated, {_cuda_memory_reserved_gb(device):.2f} GB reserved")
                            
                            if oom_count >= max_oom_retries:
                                print(f"Too many OOM errors ({oom_count}). Consider reducing batch size or gradient accumulation steps.")
                                print("Suggested fixes:")
                                print("  1. Use --batch_size 1 --gradient_accumulation_steps 1") 
                                print("  2. Use --finetune reg_net to only train registration network")
                                print("  3. Check if your GPU has enough memory for this model")
                                raise RuntimeError(f"Too many OOM errors ({oom_count}). Training stopped.")
                            
                            # Aggressive cleanup
                            _maybe_empty_cache(device)
                            # Clear all tensors
                            del baseline_data, followup_data, baseline_prompt, target
                            del seg_output, reg_loss, x1_mask_cropped, total_loss, seg_loss
                            _maybe_empty_cache(device)
                            import gc
                            gc.collect()
                            continue
                        else:
                            raise e
                    
                    # Update metrics (scale back the loss for display/tracking)
                    epoch_train_loss += total_loss.item() # * gradient_accumulation_steps  # Unscale for correct average
                    epoch_seg_loss += seg_loss.item()
                    epoch_reg_loss += reg_loss_val if isinstance(reg_loss_val, (int, float)) else reg_loss_val.item()
                    num_train_batches += 1
                    
                    # Clear intermediate tensors to free memory
                    del baseline_data, followup_data, baseline_prompt, target
                    del seg_output, reg_loss, x1_mask_cropped, total_loss, seg_loss
                    
                    # Periodic memory cleanup - more aggressive for tracking
                    _maybe_empty_cache(device)
                    import gc
                    gc.collect()
                    
                    if batch_idx % 10 == 0:  # Less frequent reporting
                        print(f"  Batch {batch_idx}")
                        print(f"    Total Loss: {epoch_train_loss/num_train_batches:.4f}")
                        print(f"    Seg Loss: {epoch_seg_loss/num_train_batches:.4f}")
                        print(f"    Reg Loss: {epoch_reg_loss/num_train_batches:.4f}")
                        if _uses_cuda_device(device):
                            print(f"    GPU memory allocated: {_cuda_memory_allocated_gb(device):.2f} GB")
                            print(f"    GPU memory reserved: {_cuda_memory_reserved_gb(device):.2f} GB")
                            print(f"    Max GPU memory allocated: {_cuda_max_memory_allocated_gb(device):.2f} GB")
                        
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Compute average losses
            avg_train_loss = epoch_train_loss / max(num_train_batches, 1)
            avg_seg_loss = epoch_seg_loss / max(num_train_batches, 1)
            avg_reg_loss = epoch_reg_loss / max(num_train_batches, 1)
            
            train_losses.append(avg_train_loss)
            train_seg_losses.append(avg_seg_loss)
            train_reg_losses.append(avg_reg_loss)
            
            print(f"Training - Total Loss: {avg_train_loss:.4f}, Seg Loss: {avg_seg_loss:.4f}, Reg Loss: {avg_reg_loss:.4f}")
            
            # Validation phase
            if val_dataset is not None:
                self.network_tracker.eval()
                epoch_val_loss = 0.0
                epoch_val_dice_scores = []
                num_val_batches = 0
                
                print("Validating...")
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_dataloader):
                        try:
                            # Extract validation data from batch
                            baseline_data = batch['bl_data'].to(device)
                            followup_data = batch['fu_data'].to(device)
                            baseline_prompt = batch['bl_prompt'].to(device)
                            target = batch['target'].to(device)
                            filenames = batch['filename']
                            
                            # Remove channel dimension from prompt - tracknet expects [B, H, W, D]
                            baseline_prompt = baseline_prompt.squeeze(1)  # [B, H, W, D]
                            
                            # Handle both batched and single sample data
                            if baseline_data.dim() == 4:
                                baseline_data = baseline_data.unsqueeze(0)
                                followup_data = followup_data.unsqueeze(0)
                                baseline_prompt = baseline_prompt.unsqueeze(0)
                                target = target.unsqueeze(0)
                                filenames = [filenames]
                            
                            with _autocast_context(device):
                                # Training mode with proper x1_mask parameter
                                seg_output, reg_loss, x1_mask_cropped = self.network_tracker(baseline_data, followup_data, baseline_prompt, is_inference=False, x1_mask=target)
                                x1_mask_cropped = x1_mask_cropped.squeeze(1)  # [B, H, W, D]
                                total_loss, seg_loss, reg_loss_val = self._tracking_combined_loss(seg_output, reg_loss, x1_mask_cropped)
                            
                            # seg_loss = self._combined_loss(seg_output, target.to(seg_output.device))
                            epoch_val_loss += reg_loss.all_loss.item() + seg_loss.item()
                            num_val_batches += 1
                            
                            # Compute dice scores for each sample in the batch
                            # for i in range(baseline_data.shape[0]):
                            #     # Get predictions (convert to class predictions)
                            #     output_single = seg_output[i:i+1]
                            #     pred_probs = torch.softmax(output_single, dim=1)
                            #     pred_classes = torch.argmax(pred_probs, dim=1).squeeze(0)
                                
                            #     # Get target
                            #     target_single = target[i]
                                
                            #     # Convert to numpy for dice computation
                            #     pred_np = pred_classes.cpu().numpy()
                            #     target_np = target_single.cpu().numpy()
                                
                            #     # Compute Dice score
                            #     dice_score = compute_dice_coefficient(target_np, pred_np)
                            epoch_val_dice_scores.append(1-seg_loss.item())
                            
                        except Exception as e:
                            print(f"Error in validation batch {batch_idx}: {e}")
                            traceback.print_exc()
                            continue
                
                # Compute averages
                avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
                avg_val_dice = np.mean(epoch_val_dice_scores) if epoch_val_dice_scores else 0.0
                
                val_losses.append(avg_val_loss)
                val_dice_scores.append(avg_val_dice)
                print(f"Validation Loss: {avg_val_loss:.4f}")
                print(f"Validation Dice Score: {avg_val_dice:.4f}")
                
                # Update learning rate scheduler
                if self.scheduler:
                    self.scheduler.step(avg_val_loss)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if output_folder:
                        self._save_tracking_checkpoint(output_folder, 'best_tracking_model.pth', epoch)
            
            # Test evaluation phase
            # test_dice_score = 0.0
            # if test_dataset is not None:
            #     print("Testing...")
            #     test_dice_score = self._evaluate_test_dataset(test_dataset, test_dataloader, device, epoch, output_folder)
            #     test_dice_scores.append(test_dice_score)
            #     print(f"Test Dice Score: {test_dice_score:.4f}")
            
            # Save periodic checkpoint
            if output_folder and (epoch + 1) % 10 == 0:
                self._save_tracking_checkpoint(output_folder, f'tracking_checkpoint_epoch_{epoch+1}.pth', epoch)
        
        # Save final checkpoint
        if output_folder:
            self._save_tracking_checkpoint(output_folder, 'final_tracking_checkpoint.pth', epochs-1)
            
        print("Tracking training completed!")
        return {
            'train_losses': train_losses,
            'train_seg_losses': train_seg_losses,
            'train_reg_losses': train_reg_losses,
            'val_losses': val_losses,
            'val_dice_scores': val_dice_scores,
            'test_dice_scores': test_dice_scores,
            'best_val_loss': best_val_loss
        }
    
    
    def _save_tracking_checkpoint(self, output_folder, filename, epoch):
        """Save tracking model checkpoint."""
        os.makedirs(output_folder, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'network_weights': self.network_tracker.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'trainer_name': self.trainer_name_tracker,
            'init_args': {
                'configuration': self.configuration_name_tracker,
            },
            'inference_allowed_mirroring_axes': getattr(self, 'allowed_mirroring_axes', None),
        }
        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
            
        checkpoint_path = os.path.join(output_folder, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Tracking checkpoint saved: {checkpoint_path}")

        if filename == 'final_tracking_checkpoint.pth':
            inference_checkpoint_path = os.path.join(output_folder, 'checkpoint_final.pth')
            torch.save(checkpoint, inference_checkpoint_path)
            print(f"Tracking inference checkpoint saved: {inference_checkpoint_path}")

    def _evaluate_test_dataset(self, test_dataset, test_dataloader, device, epoch, output_folder=None):
        """
        Evaluate the tracking model on test dataset and return average dice score.
        
        Args:
            test_dataset: Test dataset
            test_dataloader: Test data loader  
            device: Device for evaluation
            epoch: Current epoch number (for visualization folder naming)
            output_folder: Output folder for visualizations (optional)
            
        Returns:
            Average dice score across test samples
        """
        self.network_tracker.eval()
        test_dice_scores = []
        reg_losses = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                # This is tracking test data
                baseline_data = batch['bl_data'].to(device)
                followup_data = batch['fu_data'].to(device)  
                baseline_prompt = batch['bl_prompt'].to(device)
                target = batch['target'].to(device)
                filenames = batch['filename']
                
                # Remove channel dimension from prompt
                baseline_prompt = baseline_prompt.squeeze(1)
                
                # Handle single sample case
                if baseline_data.dim() == 4:
                    baseline_data = baseline_data.unsqueeze(0)
                    followup_data = followup_data.unsqueeze(0)
                    baseline_prompt = baseline_prompt.unsqueeze(0)
                    target = target.unsqueeze(0)
                    filenames = [filenames]
                
                # Forward pass
                with _autocast_context(device):
                    # For inference, network_tracker returns only (seg_output, reg_loss)
                    network_output = self.network_tracker(
                        baseline_data, followup_data, baseline_prompt, is_inference=False
                    )
                    if len(network_output) == 3:
                        seg_output, reg_loss, x1_mask_cropped = network_output
                    else:
                        seg_output, reg_loss = network_output
                        x1_mask_cropped = None
                
                reg_losses.append(reg_loss.all_loss.item())
                # Compute dice scores for each sample in the batch
                for i in range(baseline_data.shape[0]):
                    # Get predictions (convert to class predictions)
                    output_single = seg_output[i:i+1]

                    # Compute Dice from the soft loss over this single sample for numerical stability
                    dice_loss_val = dice_loss(output_single, target[i:i+1].to(output_single.device))
                    dice_score = 1.0 - dice_loss_val.item()
                    test_dice_scores.append(dice_score)
 
                    # # Visualize some test samples (every 10th sample and first 3 samples)
                    # if output_folder and (len(test_dice_scores) <= 3 or len(test_dice_scores) % 10 == 0):
                    #     self._visualize_tracking_test_sample(
                    #         baseline_data[i], followup_data[i], baseline_prompt[i], 
                    #         target_single, pred_classes, filenames[i], 
                    #         output_folder, epoch, dice_score
                    #     )
        print(f'average registration loss on test set: {np.mean(reg_losses):.4f}')
        # Return average dice score
        return np.mean(test_dice_scores) if test_dice_scores else 0.0

    def _visualize_tracking_test_sample(self, baseline_data, followup_data, baseline_prompt, target, prediction, 
                                       filename, output_folder, epoch, dice_score):
        """
        Visualize tracking test sample with baseline, follow-up, prompt, target and prediction.
        """
        # Create test visualization folder
        test_vis_folder = os.path.join(output_folder, f'test_vis_epoch_{epoch}')
        os.makedirs(test_vis_folder, exist_ok=True)
        
        # Convert to numpy arrays
        baseline_np = baseline_data[0].cpu().numpy() if baseline_data.dim() > 3 else baseline_data.cpu().numpy()
        followup_np = followup_data[0].cpu().numpy() if followup_data.dim() > 3 else followup_data.cpu().numpy()
        prompt_np = baseline_prompt.cpu().numpy()
        target_np = target.cpu().numpy()
        pred_np = prediction.cpu().numpy()
        
        # Find the axial slice with the most target pixels
        target_sums = np.sum(target_np[0], axis=(1, 2))
        max_slice = np.argmax(target_sums) if np.max(target_sums) > 0 else target_np.shape[0] // 2

        ratio = max_slice / target_np.shape[1] 
        #max_slice = int(ratio * baseline_np.shape[0])
        
        # Ensure max_slice is within bounds for all arrays
        # min_depth = min(baseline_np.shape[0], prompt_np.shape[0], followup_np.shape[0], target_np.shape[0], pred_np.shape[0])
        max_slice = max(max_slice, 0)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # First row - baseline data
        bl_max_slice = int(ratio * baseline_np.shape[1])
        axes[0, 0].imshow(baseline_np[0, bl_max_slice, :, :], cmap='gray')
        axes[0, 0].set_title('Baseline Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(baseline_np[0, bl_max_slice, :, :], cmap='gray')  
        axes[0, 1].imshow(prompt_np[0, bl_max_slice, :, :], alpha=0.5, cmap='Greens')
        axes[0, 1].set_title('Baseline + Prompt')
        axes[0, 1].axis('off')
        
        fu_max_slice = int(ratio * followup_np.shape[1])
        axes[0, 2].imshow(followup_np[0, fu_max_slice, :, :], cmap='gray')
        axes[0, 2].set_title('Follow-up Image')
        axes[0, 2].axis('off')
        
        # Second row - targets and predictions
        axes[1, 0].imshow(followup_np[0, fu_max_slice, :, :], cmap='gray')
        axes[1, 0].imshow(target_np[0, max_slice, :, :], alpha=0.5, cmap='Reds') 
        axes[1, 0].set_title('Ground Truth')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(followup_np[0, fu_max_slice, :, :], cmap='gray')
        axes[1, 1].imshow(pred_np[0, max_slice, :, :], alpha=0.5, cmap='Blues')
        axes[1, 1].set_title('Prediction')
        axes[1, 1].axis('off')
        
        # Overlay comparison
        axes[1, 2].imshow(followup_np[0, fu_max_slice, :, :], cmap='gray')
        axes[1, 2].imshow(target_np[0, max_slice, :, :], alpha=0.3, cmap='Reds')
        axes[1, 2].imshow(pred_np[0, max_slice, :, :], alpha=0.3, cmap='Blues') 
        axes[1, 2].set_title('GT (Red) + Pred (Blue)')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Tracking Test Sample - Dice: {dice_score:.3f}')
        plt.tight_layout()
        
        # Save with descriptive filename
        safe_filename = filename.replace('/', '_').replace('\\', '_')
        save_path = os.path.join(test_vis_folder, f'{safe_filename}_dice_{dice_score:.3f}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_test_sample(self, data, target, prediction, filename, output_folder, epoch, dice_score):
        """
        Visualize segmentation test sample with image, target and prediction.
        """
        # Create test visualization folder  
        test_vis_folder = os.path.join(output_folder, f'test_vis_epoch_{epoch}')
        os.makedirs(test_vis_folder, exist_ok=True)
        
        # Convert to numpy arrays
        data_np = data[0].cpu().numpy() if data.dim() > 3 else data.cpu().numpy()
        target_np = target.cpu().numpy()
        pred_np = prediction.cpu().numpy()
        
        # Find the axial slice with the most target pixels
        target_sums = np.sum(target_np, axis=(1, 2))
        max_slice = np.argmax(target_sums) if np.max(target_sums) > 0 else target_np.shape[0] // 2
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(data_np[max_slice, :, :], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(data_np[max_slice, :, :], cmap='gray')
        axes[1].imshow(target_np[max_slice, :, :], alpha=0.5, cmap='Reds')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(data_np[max_slice, :, :], cmap='gray')
        axes[2].imshow(pred_np[max_slice, :, :], alpha=0.5, cmap='Blues')
        axes[2].set_title('Prediction') 
        axes[2].axis('off')
        
        # Overlay comparison
        axes[3].imshow(data_np[max_slice, :, :], cmap='gray')
        axes[3].imshow(target_np[max_slice, :, :], alpha=0.3, cmap='Reds')
        axes[3].imshow(pred_np[max_slice, :, :], alpha=0.3, cmap='Blues')
        axes[3].set_title('GT (Red) + Pred (Blue)')
        axes[3].axis('off')
        
        plt.suptitle(f'Segmentation Test Sample - Dice: {dice_score:.3f}')
        plt.tight_layout()
        
        # Save with descriptive filename
        safe_filename = filename.replace('/', '_').replace('\\', '_')
        save_path = os.path.join(test_vis_folder, f'{safe_filename}_dice_{dice_score:.3f}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()



def train_from_prompt():
    import argparse
    parser = argparse.ArgumentParser(description='This function handles the LesionLocator single timepoint segmentation'
                                     'training using a point or 3D box prompt with segmentation-mask supervision.')
     
    parser.add_argument('-i', type=str, required=True,
                        help='Input image file or folder containing images to be predicted. File endings should be .nii.gz'
                        ' or specify another file_ending in the dataset.json file of the downloaded checkpoint.')
    parser.add_argument('-iv', type=str, required=False,
                        help='TEST image files or folder. Used for dice computation and visualization after each epoch in cross-validation. File endings should be .nii.gz')
    parser.add_argument('-p', type=str, required=True,
                        help='TRAINING prompt file or folder with segmentation-mask prompts (.nii.gz). The file containing the prompt must have the same name as the image it belongs to.'
                        'If instance segmentation maps are used, they must be in the same shape as the input images. Binary masks '
                        'will be converted to instance segmentations.')
    parser.add_argument('-pv', type=str, required=False,
                        help='TEST prompt files or folder with test segmentation maps (.nii.gz). Used for dice computation and visualization after each epoch. The file containing the prompt must have the same name as the image it belongs to.'
                        'If instance segmentation maps are used, they must be in the same shape as the input images. Binary masks '
                        'will be converted to instance segmentations.')

    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If the folder does not exist it will be created. Training results and checkpoints'
                             'will be saved here.')
    parser.add_argument('-t', type=str, required=True, choices=['point', 'box'], default='box',
                        help='Specify the type of prompt. Options are "point" or "box". Default: box')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder of the LesionLocator model called "LesionLocatorCheckpoint"')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Folds of the pretrained LesionLocator checkpoint to load for the '
                             'tracker\'s initial weights (ensemble init). Pass a single fold (e.g. '
                             '"-f 0") for a fast smoke test. This is NOT the CV training fold — '
                             'use --train_fold for that. Default: (0, 1, 2, 3, 4)')
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
    # parser.add_argument('--track', action='store_true', required=False, default=False,
    #                     help='Set this flag to enable tracking. This will use the LesionLocatorTrack model to track lesions.')
    parser.add_argument('--modality', type=str, required=True, choices=['ct', 'pet', 'petct'], default='ct', help="Use this to set the modality")
    # parser.add_argument('--adaptive_mode', action='store_true', help='Enable selection between segmentation and tracking based on Dice/NSD scores.')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, required=False, default=1,
                        help='Number of training epochs. Default: 1')
    parser.add_argument('--lr', type=float, required=False, default=1e-4,
                        help='Learning rate for training. Default: 1e-4')
    parser.add_argument('--batch_size', type=int, required=False, default=3,
                        help='Batch size for training. Default: 3')
    parser.add_argument('--gradient_accumulation_steps', type=int, required=False, default=1,
                        help='Number of steps to accumulate gradients before updating. Effective batch size = batch_size * gradient_accumulation_steps. Default: 1')
    parser.add_argument('--num_workers', type=int, required=False, default=0,
                        help='Number of DataLoader workers. Kept at 0 by default because preprocessing is '
                             'already parallelised via -npp; non-zero values will spawn extra worker '
                             'processes on top of the preprocessing pool. Default: 0')
    parser.add_argument('--ckpt_path', type=str, required=False, default=None,
                        help='Path to save inference-compatible checkpoints. Will create LesionLocatorSeg/point_optimized/fold_X structure. Default: None (no inference checkpoints saved)')
    parser.add_argument('--finetune', type=str, required=False, default='all', choices=['reg_net', 'unet', 'all'],
                        help='Which part of the tracking model to finetune. Options: reg_net (registration network only), unet (segmentation network only), all (both networks). Default: all')
    parser.add_argument('--reinit', action = 'store_true', required=False, default=False,
                        help='Which part of the tracking model to reinitialize before training. Options: reg_net (registration network only), unet (segmentation network only), all (both networks), none (no reinitialization). Default: none')
    parser.add_argument('--train_fold', type=int, required=False, default=None,
                        help='Which fold configuration to use for training. Default: 0')

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


    # Initialize tracking trainer
    trainer = LesionLocatorTrack(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=args.verbose,
            allow_tqdm=True,
            verbose_preprocessing=args.verbose,
            visualize=args.visualize,
            adaptive_mode=False
        )
        
    # Load model checkpoints
    checkpoint_folder = join(args.m, 'LesionLocatorSeg', 'point_optimized')  # Use point optimized for tracking
    checkpoint_folder_track = join(args.m, 'LesionLocatorTrack')
    trainer.initialize_from_trained_model_folder(checkpoint_folder, checkpoint_folder_track, args.f, 
                                                 modality = args.modality, checkpoint_name="checkpoint_final.pth",
                                                 reinit = args.reinit)
        
    channel_dict = trainer.dataset_json_tracker.get(
        'channel_names', trainer.dataset_json_tracker.get('modality', {'0': 'CT'})
    )
    num_modalities = len(channel_dict)
    file_ending = trainer.dataset_json_tracker['file_ending']

    # Complete the tracking training function
    if os.path.isdir(args.i):
        train_input_files = _group_input_files_by_case(args.i, file_ending, num_modalities)
    else:
        if num_modalities > 1:
            raise ValueError(
                "Tracking training with '--modality petct' expects '-i' to point to a folder "
                "containing paired _0000/_0001 channel files."
            )
        train_input_files = [args.i]
    
    if os.path.isdir(args.p):
        train_prompt_files = subfiles(args.p, suffix=file_ending, join=True, sort=True)
        train_prompt_json = subfiles(args.p, suffix='.json', join=True, sort=True)
        if train_prompt_json:
            raise ValueError(
                "JSON prompts are not supported for tracking training. Provide segmentation-mask prompts instead."
            )
    else:
        if args.p.endswith('.json'):
            raise ValueError(
                "JSON prompts are not supported for tracking training. Provide segmentation-mask prompts instead."
            )
        train_prompt_files = [args.p]
        
    # Create output file names for training
    train_output_files = [
        join(args.o, 'train_' + os.path.basename(_representative_path(i)).replace(file_ending, ''))
        for i in train_input_files
    ]
    
    # Get TEST files (renamed from validation - these are your actual test data)
    test_input_files = None
    test_prompt_files = None
    test_output_files = None
    test_dataset = None
    
    if hasattr(args, 'iv') and args.iv:
        if os.path.isdir(args.iv):
            test_input_files = _group_input_files_by_case(args.iv, file_ending, num_modalities)
        else:
            if num_modalities > 1:
                raise ValueError(
                    "Tracking evaluation with '--modality petct' expects '-iv' to point to a folder "
                    "containing paired _0000/_0001 channel files."
                )
            test_input_files = [args.iv]

        test_input_files = [i for i in test_input_files if 'TP0' not in os.path.basename(_representative_path(i))]

    if hasattr(args, 'pv') and args.pv:
        if os.path.isdir(args.pv):
            test_prompt_files = subfiles(args.pv, suffix=file_ending, join=True, sort=True)
            test_prompt_json = subfiles(args.pv, suffix='.json', join=True, sort=True)
            if test_prompt_json:
                raise ValueError(
                    "JSON prompts are not supported for tracking training/evaluation. "
                    "Provide segmentation-mask prompts instead."
                )
        else:
            if args.pv.endswith('.json'):
                raise ValueError(
                    "JSON prompts are not supported for tracking training/evaluation. "
                    "Provide segmentation-mask prompts instead."
                )
            test_prompt_files = [args.pv]
        
        test_prompt_files = [i for i in test_prompt_files if 'TP0' not in os.path.basename(i)]
            
    if test_input_files and test_prompt_files:
        test_output_files = [
            join(args.o, 'test_' + os.path.basename(_representative_path(i)).replace(file_ending, ''))
            for i in test_input_files
        ]
        
        # Create test dataset for dice computation and visualization
        test_dataset = trainer.create_training_dataset(
            input_files=test_input_files,
            prompt_files=test_prompt_files,
            output_files=test_output_files,
            prompt_type=args.t,
            num_processes=args.npp,
            verbose=args.verbose,
            track=True
        )
        print(f"Test dataset created with {len(test_input_files)} samples")
        
    # Helper function to process file arguments (handles folders, individual files, and wildcard expansions)
    # def process_file_args(file_args, suffix):
    #     all_files = []
    #     for arg in file_args:
    #         if os.path.isdir(arg):
    #             # If it's a directory, get all files with the specified suffix
    #             from lesionlocator.utilities.file_path_utilities import subfiles
    #             all_files.extend(subfiles(arg, suffix=suffix, join=True, sort=True))
    #         else:
    #             # If it's a file (or expanded from wildcard), add it directly
    #             all_files.append(arg)
    #     return sorted(all_files)
    
    folds = create_cv_folds(train_input_files, train_prompt_files, train_output_files, n_folds=5)
    if args.train_fold is not None and not 0 <= args.train_fold < len(folds):
        raise ValueError(
            f"train_fold must be between 0 and {len(folds) - 1} for the current dataset, got {args.train_fold}."
        )

    fold_data = folds[args.train_fold] if args.train_fold is not None else folds[0]
    # Create training dataset
    train_dataset = trainer.create_training_dataset(
            input_files=fold_data['train']['input_files'],
            prompt_files=fold_data['train']['prompt_files'],
            output_files=fold_data['train']['output_files'],
            prompt_type=args.t,
            num_processes=args.npp,
            verbose=args.verbose,
            track=True
    )
    print(f"Training dataset created with {len(fold_data['train']['input_files'])} samples")

    val_dataset = trainer.create_training_dataset(
            input_files=fold_data['val']['input_files'],
            prompt_files=fold_data['val']['prompt_files'],
            output_files=fold_data['val']['output_files'],
            prompt_type=args.t,
            num_processes=args.npp,
            verbose=args.verbose,
            track=True
    ) 
    print(f"Validation dataset created with {len(fold_data['val']['input_files'])} samples")
        
    # Start tracking training
    training_results = trainer.train_tracking(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        output_folder=args.o,
        num_workers=args.num_workers,
        finetune_mode=args.finetune,
        # gradient_accumulation_steps=args.gradient_accumulation_steps
    )
        
    print("Tracking training completed!")
    print(f"Final training loss: {training_results['train_losses'][-1]:.4f}")
    if training_results['val_losses']:
        print(f"Final validation loss: {training_results['val_losses'][-1]:.4f}")
        print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    if training_results['test_dice_scores']:
        print(f"Final test dice score: {training_results['test_dice_scores'][-1]:.4f}")
        print(f"Best test dice score: {max(training_results['test_dice_scores']):.4f}")
    
    return  # Exit after tracking training
        

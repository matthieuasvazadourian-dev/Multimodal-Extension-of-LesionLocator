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
import SimpleITK
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json, subfiles
from torch._dynamo import OptimizedModule
from tqdm import tqdm

import lesionlocator
from lesionlocator.preprocessing.resampling.default_resampling import compute_new_shape
from lesionlocator.configuration import default_num_processes
from lesionlocator.inference.data_iterators import preprocessing_iterator_fromfiles
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


class EmbeddingExtractor:
    """Extract intermediate features from specific layers using forward hooks"""
    
    def __init__(self, model, layer_names=None):
        """
        Args:
            model: PyTorch model
            layer_names: List of layer name patterns to match (e.g., ['decoder.stages.0', 'encoder.stages.2'])
                        If None, extracts from all decoder and encoder stages
        """
        self.model = model
        self.embeddings = {}
        self.hooks = []
        self.hooked_layer_names = []
        self.layer_names = layer_names
        
        # Auto-detect decoder/encoder stages if no specific layers provided
        if layer_names is None:
            layer_names = self._auto_detect_layers()
        
        # Register hooks for matching layers
        self._register_hooks(layer_names)
    
    def _auto_detect_layers(self):
        """Auto-detect decoder and encoder stage layers"""
        detected = []
        for name, _ in self.model.named_modules():
            if 'decoder.stages' in name or 'encoder.stages' in name:
                # Only add the stage modules themselves, not sub-modules
                if name.count('.') == 2:  # e.g., 'decoder.stages.0'
                    detected.append(name)
        return detected
    
    def _register_hooks(self, layer_names):
        """Register forward hooks for specified layers"""
        for name, module in self.model.named_modules():
            # Check if this layer matches any of the patterns
            if any(pattern in name for pattern in layer_names):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
                self.hooked_layer_names.append(name)
                if self.layer_names is not None:  # Only print if explicitly set
                    print(f"Registered hook for layer: {name}")
    
    def _make_hook(self, layer_name):
        """Create a hook function that stores the output"""
        def hook(module, input, output):
            # Store detached copy on CPU to save memory
            if isinstance(output, tuple):
                # Some layers return tuples
                self.embeddings[layer_name] = output[0].detach().cpu().clone()
            else:
                self.embeddings[layer_name] = output.detach().cpu().clone()
        return hook
    
    def clear(self):
        """Clear stored embeddings to free memory"""
        self.embeddings = {}
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.hooked_layer_names = []
    
    def get_embeddings(self):
        """Get dictionary of extracted embeddings"""
        return dict(self.embeddings)
    
    def get_layer_names(self):
        """Get list of layers being extracted"""
        return list(self.hooked_layer_names)


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
                 visualize: bool = False,
                 track: bool = False,
                 adaptive_mode: bool = False,
                 embedding_output_folder: str = None,
                 lesion_focus: bool = False,
                 crop_size: int = 64):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm
        self.embedding_output_folder = embedding_output_folder
        self.lesion_focus = lesion_focus
        self.crop_size = crop_size

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        # Add embedding extraction attributes
        self.embedding_extractor = None
        self.embedding_extractor_tracker = None
        self.extract_embeddings = False
        self.embedding_layer_names = None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        # Optimize for limited GPU memory
        if device.type == 'cuda':
            # Enable memory optimization for limited GPU
            torch.backends.cudnn.benchmark = False  # Disable for memory conservation
            torch.backends.cudnn.deterministic = True
            
            # Set memory fraction if available
            try:
                # Reserve less memory for other processes
                torch.cuda.set_per_process_memory_fraction(0.9)
                torch.cuda.empty_cache()
            except:
                pass
            
            # Reduce tile step size for smaller patches
            if tile_step_size < 0.7:
                tile_step_size = 0.7  # Larger steps = fewer patches = less memory
            
            # Disable some memory-intensive features for limited GPU
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if total_mem < 12:  # Less than 12GB
                    print(f"Limited GPU memory detected ({total_mem:.1f}GB), optimizing settings...")
                    use_mirroring = False  # Disable mirroring to save memory
                    perform_everything_on_device = False  # Use CPU for results
            except:
                perform_everything_on_device = False
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device
        self.use_mirroring = use_mirroring
        self.tile_step_size = tile_step_size
        self.visualize = visualize
        self.track = track
        self.adaptive_mode = adaptive_mode
        
        print('Tracking: ', self.track)
        print('Adaptive mode: ', self.adaptive_mode)
        print('Lesion focus cropping: ', self.lesion_focus)
        if self.lesion_focus:
            print(f'Crop size: {self.crop_size}x{self.crop_size}x{self.crop_size}')

        self.petct_mode = False
        self.first_conv_key = None

    @staticmethod
    def _find_first_conv_key(state_dict: dict) -> str:
        # Find the 5-D weight with the smallest in_channels — the true input conv.
        # Alphabetical sort misidentifies a decoder weight in ResidualEncoderUNet.
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
        # shared-parameter aliases stay consistent under load_state_dict.
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

    @staticmethod
    def _group_input_files_by_case(source_folder: str, file_ending: str, num_modalities: int) -> list:
        """Return a list of per-case channel groups."""
        all_files = subfiles(source_folder, suffix=file_ending, join=True, sort=True)
        if num_modalities == 1:
            return [[f] for f in all_files]
        ct_files = sorted(f for f in all_files
                          if os.path.basename(f).endswith('_0000' + file_ending))
        groups = []
        for ct_file in ct_files:
            group = [ct_file]
            for c in range(1, num_modalities):
                other = ct_file.replace(f'_0000{file_ending}', f'_{c:04d}{file_ending}')
                group.append(other)
            groups.append(group)
        return groups

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             model_track_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             modality: str = 'ct',
                                             checkpoint_name: str = 'checkpoint_final.pth'):
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

            parameters.append(checkpoint['network_weights'])

        # PET+CT early-fusion: patch dataset_json so that num_input_channels = 2 and
        # extend the first conv layer of each checkpoint to accept the extra PET channel.
        self.petct_mode = (modality == 'petct')
        self.first_conv_key = None
        if self.petct_mode:
            dataset_json = dict(dataset_json)
            dataset_json['channel_names'] = {'0': 'CT', '1': 'PET'}
            print('[petct] Patched dataset_json channel_names for PET+CT early fusion.')

        configuration_manager = plans_manager.get_configuration(configuration_name, modality=modality)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(lesionlocator.__path__[0], "training", "LesionLocatorTrainer"),
                                                    trainer_name, 'lesionlocator.training.LesionLocatorTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in lesionlocator.training.LesionLocatorTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        # PET+CT early fusion: extend first conv from 2 -> 3 input channels before load_state_dict.
        if self.petct_mode:
            self.first_conv_key = self._find_first_conv_key(parameters[0])
            expected_in_ch = num_input_channels + 1
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

        need_tracker = self.track or self.adaptive_mode
        self.plans_manager_tracker = None
        self.configuration_manager_tracker = None
        self.list_of_parameters_tracker = None
        self.network_tracker = None
        self.dataset_json_tracker = None
        self.trainer_name_tracker = None

        if need_tracker:
            dataset_json_tracker = load_json(join(model_track_training_output_dir, 'dataset.json'))
            plans_tracker = load_json(join(model_track_training_output_dir, 'plans.json'))
            plans_manager_tracker = PlansManager(plans_tracker)

            parameters_tracker = []
            for i, f in enumerate(use_folds):
                f = int(f) if f != 'all' else f
                fold_dir = join(model_track_training_output_dir, f'fold_{f}')
                ckpt_candidates = ('best_tracking_model.pth', 'checkpoint_final.pth', 'final_tracking_checkpoint.pth')
                ckpt_path_tracker = next((join(fold_dir, name) for name in ckpt_candidates if isfile(join(fold_dir, name))), None)
                if ckpt_path_tracker is None:
                    raise FileNotFoundError(
                        f'Could not find a tracker checkpoint in {fold_dir}. Tried: {ckpt_candidates}'
                    )

                print(f"Loading tracker model checkpoint from: {ckpt_path_tracker}")
                checkpoint_tracker = torch.load(ckpt_path_tracker,
                                        map_location=torch.device('cpu'), weights_only=False)

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

                parameters_tracker.append(checkpoint_tracker['network_weights'])

            if self.petct_mode:
                dataset_json_tracker = dict(dataset_json_tracker)
                dataset_json_tracker['channel_names'] = {'0': 'CT', '1': 'PET'}
                print('[petct] Patched tracker dataset_json channel_names for PET+CT early fusion.')

            configuration_manager_tracker = plans_manager_tracker.get_configuration(configuration_name_tracker, modality=modality)
            parameters_tracker[0]['unet_patch_size'] = torch.tensor(configuration_manager_tracker.patch_size)
            num_input_channels_tracker = determine_num_input_channels(
                plans_manager_tracker, configuration_manager_tracker, dataset_json_tracker
            )
            trainer_class = recursive_find_python_class(join(lesionlocator.__path__[0], "training", "LesionLocatorTrainer"),
                                                        trainer_name_tracker, 'lesionlocator.training.LesionLocatorTrainer')
            if trainer_class is None:
                raise RuntimeError(f'Unable to locate trainer class {trainer_name_tracker} in lesionlocator.training.LesionLocatorTrainer. '
                                   f'Please place it there (in any .py file)!')
            
            network_tracker = trainer_class.build_network_architecture(
                configuration_manager_tracker.network_arch_class_name,
                configuration_manager_tracker.network_arch_init_kwargs,
                configuration_manager_tracker.network_arch_init_kwargs_req_import,
                num_input_channels_tracker,
                plans_manager_tracker.get_label_manager(dataset_json_tracker).num_segmentation_heads,
                configuration_manager_tracker.patch_size,
                enable_deep_supervision=False
            )
           
            if self.petct_mode:
                tracker_first_conv_key = self._find_first_conv_key(parameters_tracker[0])
                expected_in_ch_tracker = num_input_channels_tracker + 1
                current_in_ch_tracker = parameters_tracker[0][tracker_first_conv_key].shape[1]
                if current_in_ch_tracker < expected_in_ch_tracker:
                    parameters_tracker = [self._extend_first_conv_weights(
                                              p, tracker_first_conv_key,
                                              num_new_channels=expected_in_ch_tracker - current_in_ch_tracker)
                                          for p in parameters_tracker]
                else:
                    print(f'[petct] Tracker checkpoint first conv already has {current_in_ch_tracker} input '
                          f'channels (expected {expected_in_ch_tracker}); skipping extension.')

            self.plans_manager_tracker = plans_manager_tracker
            self.configuration_manager_tracker = configuration_manager_tracker
            self.list_of_parameters_tracker = parameters_tracker

            network_tracker.load_state_dict(parameters_tracker[0])
            self.network_tracker = network_tracker
            self.dataset_json_tracker = dataset_json_tracker
            self.trainer_name_tracker = trainer_name_tracker
            self.allowed_mirroring_axes = inference_allowed_mirroring_axes
            self.label_manager = plans_manager_tracker.get_label_manager(dataset_json_tracker)
        self.tile_step_size = 0.5
        self.use_gaussian = True
        self.use_mirroring = True
        self.target_spacing = self.configuration_manager.spacing
        if self.track and self.configuration_manager_tracker is not None:
            self.target_spacing = self.configuration_manager_tracker.spacing
        print('Using target spacing: ', self.target_spacing)
        print('Segmentation configuration: ', self.configuration_manager)
        print('Tracking configuration: ', self.configuration_manager_tracker)

    def enable_embedding_extraction(self, layer_names=None):
        """
        Enable embedding extraction from specified layers
        
        Args:
            layer_names: List of layer name patterns. Examples:
                         ['decoder.stages.0', 'encoder.stages.3'] - specific stages
                         ['decoder.stages', 'encoder.stages'] - all stages
                         None - auto-detect all decoder/encoder stages
        """
        self.extract_embeddings = True
        self.embedding_layer_names = layer_names
        
        print(f"\n=== Enabling embedding extraction ===")
        print(f"Layer patterns: {layer_names if layer_names else 'Auto-detect'}")
        
        # Create extractor for segmentation network (uses 'decoder.stages')
        seg_layer_names = layer_names if layer_names else ['decoder.stages.4']
        self.embedding_extractor = EmbeddingExtractor(self.network, seg_layer_names)
        print(f"Segmentation network - extracting from: {self.embedding_extractor.get_layer_names()}")
        
        # Create extractor for tracking network if it exists (uses 'unet.decoder.stages')
        if hasattr(self, 'network_tracker') and self.network_tracker is not None:
            # For tracker, convert 'decoder' to 'unet.decoder' in layer names
            if layer_names:
                track_layer_names = [name.replace('decoder', 'unet.decoder') if not name.startswith('unet.') else name 
                                    for name in layer_names]
            else:
                track_layer_names = ['unet.decoder.stages.4']
            
            self.embedding_extractor_tracker = EmbeddingExtractor(
                self.network_tracker, track_layer_names
            )
            print(f"Tracking network - extracting from: {self.embedding_extractor_tracker.get_layer_names()}")
        print("=====================================\n")

    def disable_embedding_extraction(self):
        """Disable embedding extraction and remove hooks"""
        self.extract_embeddings = False
        if self.embedding_extractor is not None:
            self.embedding_extractor.remove_hooks()
            self.embedding_extractor = None
        if self.embedding_extractor_tracker is not None:
            self.embedding_extractor_tracker.remove_hooks()
            self.embedding_extractor_tracker = None

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
            # Group input files by case (each element is a list of channel files)
            input_files = self._group_input_files_by_case(
                source_folder_or_file, _file_ending, _num_modalities)
            prompt_files_json = subfiles(prompt_folder_or_file, suffix='.json', join=True, sort=True)
            prompt_files_mask = subfiles(prompt_folder_or_file, suffix=_file_ending, join=True, sort=True)

            output_basenames = ['_'.join(os.path.basename(group[0]).split('.')[0].split('_')[:3]) for group in input_files]
            output_files = [join(output_folder_or_file, name) for name in output_basenames]

            if not os.path.isdir(output_folder_or_file):
                os.makedirs(output_folder_or_file)
            finished_files = subfiles(output_folder_or_file, suffix=_file_ending, join=True, sort=True)
            finished_output_files = ['_'.join(os.path.basename(i).split('.')[0].split('_')[:3]) for i in finished_files]
            finished_output_files = set(finished_output_files)

            # Assertions
            if len(input_files) == 0:
                print(f'No files found in {source_folder_or_file}')
                return
            assert len(prompt_files_json) == 0 or len(prompt_files_mask) == 0, \
                "Prompt folder must contain either json files or mask files, not both."
            assert len(input_files) == len(prompt_files_json) or len(input_files) == len(prompt_files_mask), \
                "Number of files in source folder and prompt folder must be the same."

            prompt_files = prompt_files_json if len(prompt_files_json) > 0 else prompt_files_mask

            if not overwrite:
                not_existing_indices = [i for i, name in enumerate(output_basenames) if name not in finished_output_files]
                input_files = [input_files[i] for i in not_existing_indices]
                prompt_files = [prompt_files[i] for i in not_existing_indices]
                output_files = [output_files[i] for i in not_existing_indices]


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

        # only evaluate tracking using TP1 and TP2
        indices = [j for j, i in enumerate(input_files) if 'TP0' not in os.path.basename(i[0])]
        input_files = [input_files[i] for i in indices]
        prompt_files = [prompt_files[i] for i in indices]
        output_files = [output_files[i] for i in indices]

        # Truncate output files
        print('Total number of input files: ', len(input_files))
        output_files = [i.replace(self.dataset_json['file_ending'], '') for i in output_files]
        
        print('Number of input files before part selection: ', len(input_files))
        data_iterator = preprocessing_iterator_fromfiles(input_files, prompt_files,
                                                output_files, prompt_type, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes_preprocessing, self.device.type == 'cuda',
                                                self.verbose_preprocessing, self.track)
        print('Number of input files after part selection: ', len(input_files))
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
            r = []
            error_all={'dice': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}},
               'nsd': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}},
               'hausdorff': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}},
               'lesion_found':{'all':0, 'mean':0, 'TP0':{'all':0, 'mean':0}, 'TP1':{'all':0, 'mean':0}, 'TP2':{'all':0, 'mean':0}},
               'lesion_all': {'all':0, 'TP0':{'all':0}, 'TP1':{'all':0}, 'TP2':{'all':0}},
               'tracked': {'all':0, 'mean':0, 'TP0':{'all':0, 'mean':0}, 'TP1':{'all':0, 'mean':0}, 'TP2':{'all':0, 'mean':0}},
               'track_targets': {'all':0, 'mean':0, 'TP0':{'all':0, 'mean':0}, 'TP1':{'all':0, 'mean':0}, 'TP2':{'all':0, 'mean':0}},
               'new_appears': {'all':0, 'mean':0, 'TP0':{'all':0, 'mean':0}, 'TP1':{'all':0, 'mean':0}, 'TP2':{'all':0, 'mean':0}}}

            dice_score_all = []
            hausdorff_score_all = []
            nsd_score_all = []
            metrics = {
                'dice': 0.0,
                'hausdorff': 0.0,
                'nsd': 0.0
            }

            # Initialize data_count variable that was missing
            data_count = 0
            
            # Add timing and progress tracking
            import time
            start_time = time.time()
            print("Starting data iterator processing...")
            
            for preprocessed in data_iterator:
                # if 'TP1_058' not in preprocessed['ofile']:
                #     continue
                print(f'Processing file: {preprocessed["ofile"]}', flush=True)

                data_count += 1
                iter_start_time = time.time()
                print(f"\n=== Starting item {data_count} ===", flush=True)
                
                # Clear cache every iteration for limited GPU
                if self.device.type == 'cuda':
                    empty_cache(self.device)
                    torch.cuda.synchronize()
                
                # Monitor memory usage
                self._check_gpu_memory(f"iteration {data_count}")
                
                # Add detailed progress tracking
                step_time = time.time()
                data = preprocessed['data']
                #baseline data, None for TP0 scans
                bl_data = preprocessed['bl_data']
                
                start_time = time.time()
                if isinstance(data, str):
                    # delfile = data
                    # data = torch.from_numpy(np.load(data))
                    # os.remove(delfile)
                    delfile = data
                    # Use memory mapping for faster loading
                    data = torch.from_numpy(np.load(data, mmap_mode='r')).clone()
                    os.remove(delfile)
                step_time = time.time()

                start_time = time.time()
                # Optimize memory transfers
                if not data.is_cuda and self.device.type == 'cuda':
                    data = data.to(self.device, non_blocking=True)
                
                # Clear cache periodically to prevent memory issues
                if data_count % 5 == 0:
                    empty_cache(self.device)
                
                iter_time = time.time() - iter_start_time
                print(f"Item {data_count} loaded and moved to device in {iter_time:.2f}s")
                
                ofile = preprocessed['ofile']
                print(f'\n === Predicting {os.path.basename(ofile)} === ')
                patient_tp = os.path.basename(ofile)
                timepoint = os.path.basename(ofile).split('_')[0]
                properties = preprocessed['data_properties']
                prompt = preprocessed['prompt']
                seg_mask = preprocessed['seg']
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
                        if seg_mask is None:
                            # JSON prompt path: no ground-truth segmentation available.
                            # Substitute a zero mask so metrics/visualization don't crash; Dice will be 0.
                            gt_mask = np.zeros((1, *data.shape[1:]), dtype=np.uint8)
                        else:
                            gt_mask = ((seg_mask == inst_id).astype(np.uint8))
                        if len(p) == 0:
                            print(f"--- No prompt found for Lesion ID {inst_id} ---")
                            continue
                        print(f'\n Lesion ID {inst_id}: ')
                        for k in error_all.keys():
                            if k == 'lesion_all' or k == 'lesion_found':
                                continue
                            if os.path.basename(ofile) not in error_all[k].keys():
                                error_all[k][patient_tp]={'mean':0, 'per_lesion':[]}
                        # p_sparse = p

                        p = sparse_to_dense_prompt(p, prompt_type, array=data)

                        if p is None:
                            print(f" Invalid prompt found for {os.path.basename(ofile)}")
                            continue
                        #Check if there is an existing segmentation mask for the baseline image
                        tp_order = ['TP2', 'TP1', 'TP0']
                        current_tp = None
                        for tp in tp_order:
                            if tp in os.path.basename(ofile):
                                current_tp = tp
                                break

                        prev_tp = None
                        use_prev_tp = False
                        low_score = False
                        if current_tp == 'TP2':
                            # Try TP1 first, then TP0
                            for candidate in ['TP1', 'TP0']:
                                prev_tp_candidate = os.path.basename(ofile).replace('TP2', candidate)
                                if os.path.exists(os.path.join(output_folder_or_file, prev_tp_candidate + '_lesion_' + str(inst_id) + '.nii.gz')):
                                    prev_tp = prev_tp_candidate
                                    break
                        elif current_tp == 'TP1':
                            prev_tp_candidate = os.path.basename(ofile).replace('TP1', 'TP0')
                            if os.path.exists(os.path.join(output_folder_or_file, prev_tp_candidate + '_lesion_' + str(inst_id) + '.nii.gz')):
                                prev_tp = prev_tp_candidate
                                                                                                    
                        if self.track and (prev_tp is not None):
                            use_prev_tp = True
                            prev_seg_sitk = SimpleITK.ReadImage(os.path.join(output_folder_or_file, prev_tp+'_lesion_'+str(inst_id)+'.nii.gz'))
                            original_spacing = prev_seg_sitk.GetSpacing()[::-1]
                            print('Reading segmentation mask with spacing: ', original_spacing, ', target spacing is: ', self.target_spacing)
                            # Convert to numpy and compute new shape
                            prev_seg_np = SimpleITK.GetArrayFromImage(prev_seg_sitk)
                            new_shape = compute_new_shape(prev_seg_np.shape, original_spacing, self.target_spacing)
                            bl_spacing = (prev_seg_np.shape[0]* original_spacing[0] /  bl_data.shape[1],
                                        prev_seg_np.shape[1] * original_spacing[1] /  bl_data.shape[2],
                                        prev_seg_np.shape[2] * original_spacing[2] /  bl_data.shape[3])
                            #print('BL SPACING: ', bl_spacing)
                            #print('New shape for resampling: ', new_shape)
                            #print('BL DATA SHAPE FOR RESAMPLING: ', bl_data.shape)
                            prev_seg_resampled = self.configuration_manager.resampling_fn_seg(
                                prev_seg_np[None], 
                                bl_data.shape[1:], 
                                original_spacing, 
                                bl_spacing
                            )[0]
                            print('Use previous timepoint prediction as prompt: ', prev_tp+'_lesion_'+str(inst_id))
                            prompt_bl = torch.from_numpy(prev_seg_resampled).unsqueeze(0).to(self.device).half()
                            print('Resampled prompt shape: ', prompt_bl.shape)
                            
                            # Apply lesion-focused cropping if enabled
                            if self.lesion_focus:
                                # Get center of the mask
                                # for bl data
                                bl_prompt_coords = torch.where(prompt_bl > 0)
                                
                                # Get center of the mask
                                bl_spacing = preprocessed['bl_data_properties']['spacing']
                                data_spacing = preprocessed['data_properties']['spacing']

                                # resample bl_prompt from bl_spacing to data_spacing if they are different
                                if bl_spacing != data_spacing:
                                    print(f'Resampling baseline prompt from spacing {bl_spacing} to {data_spacing}')
                                    bl_prompt_resampled = self.configuration_manager.resampling_fn_seg(
                                        prompt_bl.cpu().numpy(), 
                                        data.shape[1:], 
                                        bl_spacing, 
                                        data_spacing
                                    )[0]
                                    prompt_bl_resampled = torch.from_numpy(bl_prompt_resampled).unsqueeze(0).to(self.device).half()
                                    print('Resampled baseline prompt shape: ', prompt_bl_resampled.shape)
                                
                                    prompt_coords = torch.where(prompt_bl_resampled > 0)
                                else:
                                    prompt_coords = bl_prompt_coords

                                if len(prompt_coords[0]) > 0:
                                    if prompt_bl.dim() == 4:
                                        prompt_coords = prompt_coords[1:]  # Add batch dimension if missing
                                    
                                    center = [
                                        int((prompt_coords[0].min().item() + prompt_coords[0].max().item()) / 2),
                                        int((prompt_coords[1].min().item() + prompt_coords[1].max().item()) / 2),
                                        int((prompt_coords[2].min().item() + prompt_coords[2].max().item()) / 2)
                                    ]
                                    
                                    half_size = self.crop_size // 2
                                    if data.dim() == 4:
                                        data_shape = data.shape[1:]  # Add batch dimension if missing
                                    else:
                                        data_shape = data.shape

                                    bbox_centered = [
                                        max(0, center[0] - half_size),
                                        min(data_shape[0], center[0] + half_size),
                                        max(0, center[1] - half_size),
                                        min(data_shape[1], center[1] + half_size),
                                        max(0, center[2] - half_size),
                                        min(data_shape[2], center[2] + half_size)
                                    ]
                                
                            
                            # Clear embeddings before prediction
                            if self.extract_embeddings and self.embedding_extractor_tracker is not None:
                                self.embedding_extractor_tracker.clear()
                            
                            # Predict the logits using the preprocessed data and the prompt
                            prediction = self.track_single_lesion(torch.from_numpy(bl_data[np.newaxis,:]).to(self.device), data.unsqueeze(0).to(self.device), prompt_bl.unsqueeze(0)).cpu()
                                                        
                            prediction = prediction.cpu()
                            seg = torch.softmax(prediction, 0).argmax(0)
                            pred = seg.detach().cpu().numpy().astype(np.uint8)
                            print('Prediction shape: ', pred.shape)
                            print('Ground truth shape: ',  gt_mask[0].shape)
                            dice_score = compute_dice_coefficient(gt_mask[0], pred)
                            
                            # Save tracking embeddings if extraction is enabled
                            if self.extract_embeddings and self.embedding_extractor_tracker is not None:
                                track_embeddings = self.embedding_extractor_tracker.get_embeddings()

                                # plot 10 feature maps from the last layer
                                # if self.visualize:
                                #     # import matplotlib.pyplot as plt
                                #     for k, v in track_embeddings.items():
                                #         feat = v.numpy()
                                #         # num_feats = feat.shape[1]
                                #         #fig, axes = plt.subplots(1, min(5, num_feats), figsize=(20, 2))

                                #         # apply global average pooling to get a single 3D map
                                #         # [n, c, d, h, w] -> [n, d, h, w]
                                #         #feat = feat.mean(axis=1)
                                #         _feat = feat.max(axis=1)[0]
                                #         #for i in range(min(5, num_feats)):
                                #         plt.imshow(_feat[_feat.shape[0]//2, :, :], cmap='jet')
                                #         plt.axis('off')
                                #         plt.suptitle(f'Layer: {k} Feature Maps')
                                #         plt.savefig(f'{self.embedding_output_folder}/{os.path.basename(ofile)}_track_embeddings_layer_{k}_lesion_{inst_id}.png')
                                #         plt.close()
                                                                
                                if len(track_embeddings) > 0:
                                    track_emb_path = os.path.join(
                                        self.embedding_output_folder, 
                                        f'{os.path.basename(ofile)}_lesion_{inst_id}_track_embeddings.npz'
                                    )
                                    track_embeddings_np = {}
                                    for layer_name, feat in track_embeddings.items():
                                        if 'nonlin' in layer_name or 'norm' in layer_name or 'convs' in layer_name or 'all_modules' in layer_name:
                                            continue  # skip nonlin, norm, and conv layers
                                        key = layer_name.replace('.', '_')
                                        track_embeddings_np[key] = feat.numpy()
                                    track_embeddings_np['dice'] = dice_score

                                    if self.lesion_focus and bbox_centered is not None:
                                        track_embeddings_np['bbox'] = np.array(bbox_centered)
                                        # # convert pixel location to physical spacing?
                                        # center_physical = [
                                        #     center[0] * self.target_spacing[0],
                                        #     center[1] * self.target_spacing[1],
                                        #     center[2] * self.target_spacing[2]
                                        # ]
                                        # calculate the center of nonezeros in mask_gt[0]
                                        prompt_coords = torch.where(torch.from_numpy(gt_mask[0]) > 0)
                                        center = [
                                            int((prompt_coords[1].min().item() + prompt_coords[1].max().item()) / 2),
                                            int((prompt_coords[2].min().item() + prompt_coords[2].max().item()) / 2),
                                            int((prompt_coords[0].min().item() + prompt_coords[0].max().item()) / 2)
                                        ]
                                        data_spacing = preprocessed['data_properties']['spacing'][::-1]
                                        center_physical = [
                                            center[0] * data_spacing[0],
                                            center[1] * data_spacing[1],
                                            center[2] * data_spacing[2]
                                        ]
                                        track_embeddings_np['center'] = np.array(center)
                                        track_embeddings_np['center_physical'] = np.array(center_physical)
                                        track_embeddings_np['crop_size'] = self.crop_size

                                        data_physical_size = np.array(data.shape[1:]) * np.array(data_spacing)
                                        track_embeddings_np['data_physical_size'] = data_physical_size

                                    # check the byte size of track_embeddings_np
                                    total_bytes = sum(feat.nbytes for key, feat in track_embeddings_np.items() if 'decoder_stages_2' in key)
                                    print(f'Total bytes for tracking embeddings of lesion {inst_id}: {total_bytes} bytes')
                                    print(f'Embeddings shape and dtype for lesion {inst_id}:')
                                    for key, feat in track_embeddings_np.items():
                                        if isinstance(feat, np.ndarray) and 'decoder_stages_2' in key:
                                            print(f'  {key}: shape={feat.shape}, dtype={feat.dtype}')
                                    # print(f'Total size of tracking embeddings for lesion {inst_id}: {total_bytes / 1024**2:.2f} MB')
                                    np.savez_compressed(track_emb_path, **track_embeddings_np)
                                    print(f'Saved tracking embeddings to {track_emb_path}')

                            if dice_score < 0.1:
                                print(f'Low Dice score {dice_score:.2f} for lesion {inst_id} at timepoint {timepoint}. Disabling tracking...')
                                low_score = True
                            else:
                                error_all['tracked']['all']+=1
                                error_all['tracked'][timepoint]['all']+=1
                            
                            error_all['track_targets']['all']+=1
                            error_all['track_targets'][timepoint]['all']+=1

                        if (not self.track) or (prev_tp is None) or (low_score and self.adaptive_mode):
                            use_prev_tp = False
                            print('Use current timepoint ground truth as prompt: ', p.shape)
                            # Apply lesion-focused cropping if enabled
                            if self.lesion_focus:
                                prompt_coords = torch.where(p > 0)
                                if len(prompt_coords[0]) > 0:
                                    center = [
                                        int((prompt_coords[1].min().item() + prompt_coords[1].max().item()) / 2),
                                        int((prompt_coords[2].min().item() + prompt_coords[2].max().item()) / 2),
                                        int((prompt_coords[3].min().item() + prompt_coords[3].max().item()) / 2)
                                    ]
                                    # center_physical = [
                                    #     center[0] * self.target_spacing[0],
                                    #     center[1] * self.target_spacing[1],
                                    #     center[2] * self.target_spacing[2]
                                    # ]
                                    half_size = self.crop_size // 2
                                    bbox_centered = [ 
                                        max(0, center[0] - half_size),
                                        min(data.shape[1], center[0] + half_size),
                                        max(0, center[1] - half_size),
                                        min(data.shape[2], center[1] + half_size),
                                        max(0, center[2] - half_size),
                                        min(data.shape[3], center[2] + half_size)
                                    ]
                                    
                                    # Adjust if bbox goes out of bounds (ensure crop_size x crop_size x crop_size)
                                    for i in range(3):
                                        start_idx = i * 2
                                        end_idx = start_idx + 1
                                        current_size = bbox_centered[end_idx] - bbox_centered[start_idx]
                                        
                                        if current_size < self.crop_size:
                                            deficit = self.crop_size - current_size
                                            if bbox_centered[start_idx] == 0:
                                                bbox_centered[end_idx] = min(bbox_centered[end_idx] + deficit, data.shape[i+1])
                                            elif bbox_centered[end_idx] == data.shape[i+1]:
                                                bbox_centered[start_idx] = max(bbox_centered[start_idx] - deficit, 0)
                                    
                                    # Crop data
                                    data_cropped = data[:, 
                                                       bbox_centered[0]:bbox_centered[1],
                                                       bbox_centered[2]:bbox_centered[3],
                                                       bbox_centered[4]:bbox_centered[5]].clone()
                                    p_cropped = p[:,
                                                 bbox_centered[0]:bbox_centered[1],
                                                 bbox_centered[2]:bbox_centered[3],
                                                 bbox_centered[4]:bbox_centered[5]].clone()
                                    
                                    print(f'Lesion-focused cropping: center={center}, bbox={bbox_centered}')
                                    print(f'Cropped shapes: data={data_cropped.shape}, prompt={p_cropped.shape}')
                                else:
                                    print('Warning: Empty prompt mask, skipping cropping')
                                    data_cropped = data
                                    p_cropped = p
                                    bbox_centered = None
                            else:
                                data_cropped = data
                                p_cropped = p
                                bbox_centered = None
                            
                            # Clear embeddings before prediction
                            if self.extract_embeddings and self.embedding_extractor is not None:
                                self.embedding_extractor.clear()
                            
                            # Predict the logits using the preprocessed data and the prompt
                            prediction_cropped = self.predict_logits_from_preprocessed_data(data_cropped, p_cropped).cpu()
                            
                            # Reconstruct full-size prediction if cropping was applied
                            if self.lesion_focus and bbox_centered is not None:
                                prediction = torch.zeros((prediction_cropped.shape[0],) + data.shape[1:], 
                                                        dtype=prediction_cropped.dtype)
                                prediction[:,
                                          bbox_centered[0]:bbox_centered[1],
                                          bbox_centered[2]:bbox_centered[3],
                                          bbox_centered[4]:bbox_centered[5]] = prediction_cropped
                            else:
                                prediction = prediction_cropped
                            
                            prediction = prediction.cpu()
                            seg = torch.softmax(prediction, 0).argmax(0)
                            pred = seg.detach().cpu().numpy().astype(np.uint8)
                            print('Prediction shape: ', pred.shape)
                            dice_score = compute_dice_coefficient(gt_mask[0], pred)
                            
                            # Save segmentation embeddings if extraction is enabled
                            if self.extract_embeddings and self.embedding_extractor is not None:
                                seg_embeddings = self.embedding_extractor.get_embeddings()
                                
                                if len(seg_embeddings) > 0:
                                    seg_emb_path = os.path.join(
                                        self.embedding_output_folder, 
                                        f'{os.path.basename(ofile)}_lesion_{inst_id}_seg_embeddings.npz'
                                    )
                                    seg_embeddings_np = {}
                                    for layer_name, feat in seg_embeddings.items():
                                        if 'nonlin' in layer_name or 'norm' in layer_name or 'convs' in layer_name or 'all_modules' in layer_name:
                                            continue  # skip nonlin, norm, and conv layers

                                        key = layer_name.replace('.', '_')
                                        seg_embeddings_np[key] = feat.numpy()
                                    
                                    for key, feat in seg_embeddings_np.items():
                                        print(f'  {key}: shape={feat.shape}, dtype={feat.dtype}')

                                    seg_embeddings_np['dice'] = dice_score
                                    if self.lesion_focus and bbox_centered is not None:
                                        seg_embeddings_np['bbox'] = np.array(bbox_centered)
                                        prompt_coords = torch.where(torch.from_numpy(gt_mask[0]) > 0)
                                        center = [
                                            int((prompt_coords[1].min().item() + prompt_coords[1].max().item()) / 2),
                                            int((prompt_coords[2].min().item() + prompt_coords[2].max().item()) / 2),
                                            int((prompt_coords[0].min().item() + prompt_coords[0].max().item()) / 2)
                                        ]
                                        # convert center to physical space using spacing
                                        data_spacing = preprocessed['data_properties']['spacing']
                                        center_physical = [
                                            center[0] * data_spacing[0],
                                            center[1] * data_spacing[1],
                                            center[2] * data_spacing[2]
                                        ]
                                        seg_embeddings_np['center'] = center
                                        seg_embeddings_np['center_physical'] = np.array(center_physical)
                                        seg_embeddings_np['crop_size'] = self.crop_size

                                        data_physical_size = np.array(data.shape[1:]) * np.array(data_spacing)
                                        seg_embeddings_np['data_physical_size'] = data_physical_size
                                    print(f'Embeddings shape and dtype for lesion {inst_id}:')
                                    np.savez_compressed(seg_emb_path, **seg_embeddings_np)
                                    print(f'Saved segmentation embeddings to {seg_emb_path}')

                        if prev_tp is None:
                            error_all['new_appears']['all']+=1
                            error_all['new_appears'][timepoint]['all']+=1

                        # check if pred or dice_score exist
                        if 'prediction' not in locals() or 'dice_score' not in locals():
                            print(f" Prediction failed for Lesion ID {inst_id} ")
                            dice_score = 0.0
                            pred = np.zeros_like(gt_mask[0], dtype=np.uint8)
                            hausdorff_score = np.inf
                            nsd_score = 0.0
                        else:
                            surface_distances = compute_surface_distances(gt_mask[0], pred, self.target_spacing)
                            hausdorff_score = compute_robust_hausdorff(surface_distances, 95)
                            nsd_score = compute_surface_dice_at_tolerance(surface_distances, 2)
                            
                            dice_score_all.append(dice_score)
                            hausdorff_score_all.append(hausdorff_score)
                            nsd_score_all.append(nsd_score)
                            metrics = {
                                'dice': dice_score,
                                'hausdorff': hausdorff_score,
                                'nsd': nsd_score
                            }

                        error_all['lesion_all']['all']+=1
                        error_all['lesion_all'][timepoint]['all']+=1
                        if dice_score >= 0.1:
                            error_all['lesion_found']['all']+=1
                            error_all['lesion_found'][timepoint]['all']+=1
                        print('Dice Score: ', dice_score)

                        # surface_distances = compute_surface_distances(gt_mask[0], pred, self.target_spacing)
                        # hausdorff_score = compute_robust_hausdorff(surface_distances, 95)
                        # nsd_score = compute_surface_dice_at_tolerance(surface_distances, 2)
                        
                        # dice_score_all.append(dice_score)
                        # hausdorff_score_all.append(hausdorff_score)
                        # nsd_score_all.append(nsd_score)
                        # metrics = {
                        #     'dice': dice_score,
                        #     'hausdorff': hausdorff_score,
                        #     'nsd': nsd_score
                        # }
                        # Update all metrics in a loop
                        for metric_name, score in metrics.items():
                            error_all[metric_name][timepoint]['all'].append(score)
                            error_all[metric_name][patient_tp]['per_lesion'].append(score)
                        print('Avg Mean Dice: ', np.mean(dice_score_all))
                        print('Avg Mean Hausdorff: ',  np.mean(hausdorff_score_all))
                        print('Avg Mean NSD: ', np.mean(nsd_score_all))
                        print('Avg Lesion Detection Score: {:.2f}%'.format((error_all['lesion_found']['all'] / (1e-7+error_all['lesion_all']['all'])) * 100))
                        with open(os.path.join(output_folder_or_file, 'error_dict.json'), 'w') as fjson:
                            json.dump(error_all, fjson)
                        print('----------')
                        out_file = ofile + f'_lesion_{inst_id}'
                        # Visualize the prediction
                        if self.visualize:
                            subplot_count = 3
                            if use_prev_tp:
                                subplot_count = 4
                            
                            # Find axial slice with most lesion pixels
                            mask_sum_axial = np.sum(gt_mask[0], axis=(1,2))
                            largest_mask_slice_id_axial = np.argmax(mask_sum_axial)

                            mask_sum_coronal = np.sum(gt_mask[0], axis=(0,2))
                            largest_mask_slice_id_coronal = np.argmax(mask_sum_coronal)

                            # mask_ones_gt_axial = np.where(gt_mask[0] == 1)
                            # if len(mask_ones_gt_axial[0]) > 0:  # Check if mask is not empty
                            #     # Find the z-slice with most mask voxels for axial view
                            #     largest_mask_slice_id_axial = np.bincount(mask_ones_gt_axial[0]).argmax()
                            
                            # # Find coronal slice with most lesion pixels
                            # mask_ones_gt_coronal = np.where(gt_mask[0] == 1)
                            # if len(mask_ones_gt_coronal[1]) > 0:  # Check if mask is not empty
                            #     # Find the y-slice with most mask voxels for coronal view
                            #     largest_mask_slice_id_coronal = np.bincount(mask_ones_gt_coronal[1]).argmax()

                            # Create subplot with 2 rows
                            fig, axs = plt.subplots(2, subplot_count, figsize=(subplot_count * 4, 8))
                            
                            # First row - Axial View
                            # Original img
                            axs[0,0].imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            axs[0,0].set_title('Image (Axial)') 
                            axs[0,0].axis('off')
                            # Ground truth
                            axs[0,1].imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            axs[0,1].imshow(gt_mask[0][largest_mask_slice_id_axial, :, :]*255, alpha=0.5)
                            axs[0,1].set_title('Ground truth') 
                            axs[0,1].axis('off')
                            # Predictions
                            axs[0,2].imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            axs[0,2].imshow(pred[largest_mask_slice_id_axial, :, :], alpha=0.5)
                            axs[0,2].set_title('Prediction') 
                            axs[0,2].axis('off')

                            # Second row - Coronal View
                            # Original img
                            axs[1,0].imshow(data[0][:, largest_mask_slice_id_coronal, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                            axs[1,0].set_title('Image (Coronal)') 
                            axs[1,0].axis('off')
                            # Ground truth
                            axs[1,1].imshow(data[0][:, largest_mask_slice_id_coronal, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                            axs[1,1].imshow(gt_mask[0][:, largest_mask_slice_id_coronal, :]*255, alpha=0.5, origin='lower')
                            axs[1,1].set_title('Ground truth') 
                            axs[1,1].axis('off')
                            # Predictions
                            axs[1,2].imshow(data[0][:, largest_mask_slice_id_coronal, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                            axs[1,2].imshow(pred[:, largest_mask_slice_id_coronal, :], alpha=0.5, origin='lower')
                            axs[1,2].set_title('Prediction') 
                            axs[1,2].axis('off')
                            if use_prev_tp:
                                prompt_bl = prompt_bl[0].detach().cpu().numpy()
                                try:
                                    # Axial view for baseline
                                    mask_ones_gt_axial_bl = np.where(prev_seg_resampled == 1)
                                    largest_mask_slice_id_axial_bl = np.bincount(mask_ones_gt_axial_bl[0]).argmax()
                                    axs[0,3].imshow(bl_data[0][largest_mask_slice_id_axial_bl, :, :], cmap='gray')
                                    axs[0,3].imshow(prev_seg_resampled[largest_mask_slice_id_axial_bl, :, :], alpha=0.5)
                                    axs[0,3].set_title('Baseline prompt') 
                                    axs[0,3].axis('off')

                                    # Coronal view for baseline
                                    mask_ones_gt_coronal_bl = np.where(prev_seg_resampled == 1)
                                    largest_mask_slice_id_coronal_bl = np.bincount(mask_ones_gt_coronal_bl[1]).argmax()
                                    axs[1,3].imshow(bl_data[0][:, largest_mask_slice_id_coronal_bl, :], cmap='gray', origin='lower')
                                    axs[1,3].imshow(prev_seg_resampled[:, largest_mask_slice_id_coronal_bl, :], alpha=0.5, origin='lower')
                                    axs[1,3].set_title('Baseline prompt') 
                                    axs[1,3].axis('off')
                                except Exception as e:
                                    print(f'Error visualizing baseline prompt: {e}')

                            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.05, hspace=0.15)
                            plt.savefig(os.path.join(output_folder_or_file, f'{out_file}_dice_{dice_score:.2f}.png'), bbox_inches='tight')
                            plt.close()

                        # if dice_score < 0.1:
                        # # if low_score and not self.adaptive_mode:
                        #     out_file += '_tracked_failed'
                    
                        if 'prediction' in locals() and 'TP2' not in os.path.basename(ofile):
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

                if 'prediction' in locals():
                    del prediction
                if 'dice_score' in locals():
                    del dice_score
                empty_cache(self.device)

            error_all['dice']['mean']= np.mean(dice_score_all)
            error_all['hausdorff']['mean'] = np.mean(hausdorff_score_all)
            error_all['nsd']['mean'] = np.mean(nsd_score_all)
            error_all['lesion_found']['mean'] = (error_all['lesion_found']['all']/(1e-7+error_all['lesion_all']['all']))*100
            for tp in ['TP0','TP1','TP2']:
                error_all['dice'][tp]['mean']=np.mean(error_all['dice'][tp]['all'])
                error_all['hausdorff'][tp]['mean']=np.mean(error_all['hausdorff'][tp]['all'])
                error_all['nsd'][tp]['mean']=np.mean(error_all['nsd'][tp]['all'])
                if error_all['lesion_all'][tp]['all'] == 0:
                    error_all['lesion_found'][tp]['mean'] = 0
                else:
                    error_all['lesion_found'][tp]['mean'] = error_all['lesion_found'][tp]['all']/(1e-7+error_all['lesion_all'][tp]['all'])*100
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
    

    def mirror_and_predict(self, x0, x1, prompt):
        # Disable mirroring for limited GPU memory
        if self.device.type == 'cuda':
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if total_mem < 12:
                    print(f"Limited GPU memory ({total_mem:.1f}GB), disabling mirroring")
                    self.use_mirroring = False
            except:
                self.use_mirroring = False
        
        # Use mixed precision to save memory
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network_tracker(x0, x1, prompt, is_inference=True, visualize=False, lesion_focused=self.lesion_focus)
            prediction = output[0] if isinstance(output, tuple) else output
            
            # Clear intermediate outputs immediately
            if isinstance(output, tuple) and len(output) > 1:
                reg_loss = output[1]
                print('Registration Loss:', reg_loss.all_loss.item())
                del reg_loss
            del output
            
            # Simplified mirroring for memory conservation
            if self.use_mirroring:
                # Only use essential mirror axes to save memory
                essential_axes = [4]  # Only left-right mirroring
                
                for axis in essential_axes:
                    try:
                        # Clear cache before each mirroring
                        empty_cache(self.device)
                        
                        x0_flip = torch.flip(x0, [axis])
                        x1_flip = torch.flip(x1, [axis])
                        prompt_flip = torch.flip(prompt, [axis])
                        
                        mirror_output = self.network_tracker(x0_flip, x1_flip, prompt_flip, is_inference=True, visualize=False, lesion_focused=self.lesion_focus)
                        mirror_pred = mirror_output[0] if isinstance(mirror_output, tuple) else mirror_output
                        
                        # Immediate cleanup
                        del x0_flip, x1_flip, prompt_flip, mirror_output
                        
                        prediction += torch.flip(mirror_pred, [axis])
                        del mirror_pred
                        
                        empty_cache(self.device)
                        
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        print(f"Mirroring failed for axis {axis}: {e}, skipping...")
                        empty_cache(self.device)
                        continue
                
                prediction /= 2  # Original + 1 mirroring
        
        prediction = prediction[0]
        return prediction

    @torch.inference_mode()
    def track_single_lesion(self, bl: torch.Tensor, fu: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=True) if self.device.type == 'cuda' else dummy_context():
            prediction = None
            for params in self.list_of_parameters_tracker: # fold iteration
                # Clear cache before loading new model parameters
                empty_cache(self.device)
                
                self.network_tracker.load_state_dict(params)
                self.network_tracker = self.network_tracker.to(self.device)
                self.network_tracker.eval()
                print('BL shape', bl.shape, bl.dtype)
                print('FU shape', fu.shape, fu.dtype)
                print('PROMPT shape',prompt.shape, prompt.dtype)
                
                fold_prediction = self.mirror_and_predict(bl, fu, prompt).to('cpu')
                
                if prediction is None:
                    prediction = fold_prediction
                else:
                    prediction += fold_prediction
                
                # Clear intermediate results
                del fold_prediction
                empty_cache(self.device)

            if len(self.list_of_parameters_tracker) > 1:
                prediction /= len(self.list_of_parameters_tracker)
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

    def _check_gpu_memory(self, operation_name="operation"):
        """Monitor GPU memory and clear cache if needed."""
        if self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                cached = torch.cuda.memory_reserved() / 1024**3     # GB
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                print(f"GPU Memory - {operation_name}: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total:.1f}GB total")
                
                # Clear cache if using too much memory
                if allocated > total * 0.8 or cached > total * 0.9:
                    print(f"High memory usage detected, clearing cache...")
                    empty_cache(self.device)
                    torch.cuda.synchronize()
                    
            except Exception as e:
                print(f"Memory check failed: {e}")

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        # Force CPU results for limited GPU memory
        if self.device.type == 'cuda':
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if total_mem < 12:  # Less than 12GB, use CPU for results
                    do_on_device = False
                    print(f"Limited GPU memory ({total_mem:.1f}GB), using CPU for results")
            except:
                do_on_device = False
        
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for i, s in enumerate(slh):
                try:
                    # Convert to tuple to fix indexing warning
                    s_tuple = tuple(s) if not isinstance(s, tuple) else s
                    
                    # Clear cache every few iterations
                    if i % 5 == 0:
                        empty_cache(self.device)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                    
                    # Use smaller memory footprint
                    tensor_slice = d[s_tuple][None].clone(memory_format=torch.contiguous_format)
                    
                    # Move to device with memory check
                    if self.device.type == 'cuda':
                        try:
                            # Check available memory before transfer
                            if torch.cuda.is_available():
                                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                                tensor_size = tensor_slice.numel() * tensor_slice.element_size()
                                
                                if free_mem > tensor_size * 2:  # Keep some buffer
                                    tensor_slice = tensor_slice.to(self.device, non_blocking=False)
                                else:
                                    print(f"Insufficient GPU memory, keeping slice {i} on CPU")
                                    # Keep on CPU
                        except (RuntimeError, torch.cuda.OutOfMemoryError) as cuda_error:
                            print(f"CUDA transfer failed for slice {i}: {cuda_error}")
                            # Keep tensor on CPU
                    
                    q.put((tensor_slice, s))
                    
                except Exception as e:
                    print(f"Error processing slice {i}: {e}")
                    # Try CPU fallback
                    try:
                        tensor_slice = d[s_tuple][None].clone()
                        q.put((tensor_slice, s))
                    except Exception as fallback_e:
                        print(f"Complete failure for slice {s}: {fallback_e}")
                        continue
            
            q.put('end')
            
            # Validate CUDA context
            # if self.device.type == 'cuda' and not torch.cuda.is_available():
            #     print("CUDA is not available, falling back to CPU")
            #     device = torch.device('cpu')
            # else:
            #     device = self.device
                
            # for s in slh:
            #     try:
            #         q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(device), s))
            #     except RuntimeError as e:
            #         if "CUDA" in str(e):
            #             print(f"CUDA error, retrying with CPU: {e}")
            #             q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to('cpu'), s))
            #         else:
            #             raise e
            # q.put('end')

        try:
            self._check_gpu_memory("before sliding window")
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
                if self.lesion_focus:
                    # crop gaussian to crop size
                    start = [(gs - cs) // 2 for gs, cs in zip(gaussian.shape, [self.crop_size]*3)]
                    end = [start[i] + self.crop_size for i in range(3)]
                    gaussian = gaussian[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
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
                    # Convert slice objects to tuples to fix PyTorch indexing warning
                    sl_tuple = tuple(sl) if not isinstance(sl, tuple) else sl
                    sl_spatial = tuple(sl[1:]) if not isinstance(sl[1:], tuple) else sl[1:]
                    predicted_logits[sl_tuple] += prediction
                    n_predictions[sl_spatial] += gaussian
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
            print(f"Sliding window prediction failed: {e}")
            # Clear everything and retry with CPU
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            
            if do_on_device and "memory" in str(e).lower():
                print("Retrying with CPU results device...")
                return self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
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
            if not self.lesion_focus: 
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                        'constant', {'value': 0}, True,
                                                        None)
            else:
                data, slicer_revert_padding = pad_nd_image(input_image, [self.crop_size, self.crop_size, self.crop_size],
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
            # revert padding - Convert to tuple to fix PyTorch indexing warning
            slicer_tuple = tuple([slice(None)] + list(slicer_revert_padding[1:]))
            predicted_logits = predicted_logits[slicer_tuple]
        return predicted_logits


def segment_and_track():
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
    parser.add_argument('--track', action='store_true', required=False, default=False,
                        help='Set this flag to enable tracking. This will use the LesionLocatorTrack model to track lesions.')
    parser.add_argument('--modality', type=str, required=True, choices=['ct', 'pet', 'petct'], default='ct', help="Use this to set the modality")
    parser.add_argument('--adaptive_mode', action='store_true', help='Enable selection between segmentation and tracking based on Dice/NSD scores.')
    parser.add_argument('--lesion_focus', action='store_true', help='Enable lesion-focused inference by prioritizing patches overlapping with the prompt, rather than strict bbox-focused. This can improve performance for larger lesions that do not fit entirely within the patch size.')
    parser.add_argument('--crop_size', type=int, default=128, help='Crop size for lesion-focused inference. Only used if --lesion_focus is set. Default: 128')
    parser.add_argument('--extract_embeddings', action='store_true', required=False, default=False,
                        help='Extract and save intermediate layer embeddings from both segmentation and tracking networks') 
    parser.add_argument('--embedding_output_folder', type=str, required=False, default=None,
                        help='Folder to save extracted embeddings. Must be specified if --extract_embeddings is set.')
    parser.add_argument('--embedding_layers', type=str, nargs='+', required=False, default=None,
                        help='Specific layer names to extract embeddings from. Examples: '
                             'decoder.stages.0 encoder.stages.3. If not specified, auto-detects all stages.')
    
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

    if args.embedding_output_folder is not None and not isdir(args.embedding_output_folder):
        print(f"Embedding output folder {args.embedding_output_folder} does not exist. Creating it...")
        maybe_mkdir_p(args.embedding_output_folder)

    predictor = LesionLocatorSegmenter(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose,
                                visualize=args.visualize,
                                track=args.track,
                                adaptive_mode=args.adaptive_mode,
                                lesion_focus=args.lesion_focus,
                                crop_size=args.crop_size,
                                embedding_output_folder=args.embedding_output_folder)
    optimized_ckpt = "bbox_optimized" if args.t == 'box' else "point_optimized"
    checkpoint_folder = join(args.m, 'LesionLocatorSeg', optimized_ckpt)
    checkpoint_folder_track = join(args.m, 'LesionLocatorTrack')
    # checkpoint_folder_track = join(args.m, 'LesionLocatorSeg')
    predictor.initialize_from_trained_model_folder(checkpoint_folder, checkpoint_folder_track, args.f, args.modality, "checkpoint_final.pth")
    
    # Enable embedding extraction if requested
    if args.extract_embeddings:
        # Use specific layers if provided, otherwise use decoder.stages.4 for both tracker and segmentor
        if args.embedding_layers is None:
            # Default: decoder stage 4 (last decoder stage before segmentation head)
            embedding_layers = ['decoder.stages.4']
        else:
            embedding_layers = args.embedding_layers
        predictor.enable_embedding_extraction(layer_names=embedding_layers)
    
    predictor.predict_from_files(args.i, args.o, args.p, args.t,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 num_parts=1, part_id=0)
    
    # Clean up hooks when done
    if args.extract_embeddings:
        predictor.disable_embedding_extraction()

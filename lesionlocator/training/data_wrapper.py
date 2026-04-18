import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from lesionlocator.training.data_iterators import preprocessing_iterator_fromfiles


def _representative_path(file_or_group):
    return file_or_group[0] if isinstance(file_or_group, list) else file_or_group


class LesionTrackingDatasetWrapper(IterableDataset):
    """
    PyTorch IterableDataset wrapper for tracking training that handles paired baseline and follow-up data.
    
    This wrapper:
    1. Loads baseline and follow-up image pairs
    2. Handles baseline segmentation masks as tracking prompts
    3. Provides PyTorch DataLoader compatibility
    4. Maintains all preprocessing logic for tracking
    
    Example usage:
        trainer = LesionLocatorTrack(device=torch.device('cuda'))
        trainer.initialize_from_trained_model_folder(model_dir, track_dir, folds)
        
        train_dataset = trainer.create_tracking_dataset(
            baseline_files=['bl1.nii.gz', 'bl2.nii.gz'],
            followup_files=['fu1.nii.gz', 'fu2.nii.gz'],
            baseline_seg_files=['seg1.nii.gz', 'seg2.nii.gz'],
            followup_seg_files=['seg1_fu.nii.gz', 'seg2_fu.nii.gz'],
            output_files=['out1', 'out2']
        )
        
        trainer.train_tracking(train_dataset, epochs=100, lr=1e-4)
    """
    def __init__(self, baseline_files, followup_files, baseline_seg_files, followup_seg_files, 
                 output_files, plans_config, dataset_json, configuration_config, modality,
                 num_processes=3, pin_memory=False, verbose=False):
        self.baseline_files = baseline_files
        self.followup_files = followup_files
        self.baseline_seg_files = baseline_seg_files
        self.followup_seg_files = followup_seg_files
        self.output_files = output_files
        self.plans_config = plans_config
        self.dataset_json = dataset_json
        self.configuration_config = configuration_config
        self.modality = modality
        self.num_processes = num_processes
        self.pin_memory = pin_memory
        self.verbose = verbose
        
    def __len__(self):
        """
        Return an estimate of the dataset length for PyTorch DataLoader.
        This is an approximation since the actual number of lesions per file varies.
        """
        # Estimate: assume average of 2-3 lesions per file pair
        return len(self.baseline_files) 
        
    def __iter__(self):
        """
        Create the tracking data iterator and yield training samples.
        For tracking, we need baseline image, follow-up image, baseline segmentation as prompt,
        and follow-up segmentation as target.
        """
        # For simplicity, we'll process pairs of baseline and follow-up data
        # This creates separate iterators for baseline and follow-up data processing
        
        print('Creating tracking data iterator...')
        
        # Process each pair of files
        for i in range(len(self.baseline_files)):
            try:
                # Load and process baseline data with its segmentation
                baseline_iterator = preprocessing_iterator_fromfiles(
                    [self.baseline_files[i]], [self.baseline_seg_files[i]], 
                    [self.output_files[i] + '_baseline'], 'point',  # Use 'point' prompt type (centroid-based)
                    self.plans_config, self.dataset_json, self.configuration_config, 
                    self.modality, 1, self.pin_memory, self.verbose, track=False
                )
                
                # Load and process follow-up data with its segmentation  
                followup_iterator = preprocessing_iterator_fromfiles(
                    [self.followup_files[i]], [self.followup_seg_files[i]],
                    [self.output_files[i] + '_followup'], 'point',  # Use 'point' prompt type (centroid-based)
                    self.plans_config, self.dataset_json, self.configuration_config,
                    self.modality, 1, self.pin_memory, self.verbose, track=False
                )
                
                # Get preprocessed data from both iterators
                baseline_data_list = list(baseline_iterator)
                followup_data_list = list(followup_iterator)
                
                if len(baseline_data_list) == 0 or len(followup_data_list) == 0:
                    print(f"Skipping pair {i} - no data loaded")
                    continue
                
                baseline_preprocessed = baseline_data_list[0]  # Get first (and should be only) item
                followup_preprocessed = followup_data_list[0]
                
                baseline_data = baseline_preprocessed['data']           # [C, H, W, D] tensor
                baseline_seg = baseline_preprocessed['seg']             # [H, W, D] numpy array
                followup_data = followup_preprocessed['data']           # [C, H, W, D] tensor
                followup_seg = followup_preprocessed['seg']             # [H, W, D] numpy array
                properties = baseline_preprocessed['data_properties']   # Use baseline properties
                
                # Convert tensor to numpy if needed
                if isinstance(baseline_data, torch.Tensor):
                    baseline_data = baseline_data.numpy()
                if isinstance(followup_data, torch.Tensor):
                    followup_data = followup_data.numpy()
                
                # Process each lesion instance in the segmentation
                unique_ids = np.unique(baseline_seg)
                unique_ids = unique_ids[unique_ids > 0]  # Remove background
                
                for lesion_id in unique_ids:
                    # Create binary mask for this specific lesion in baseline
                    baseline_lesion_mask = (baseline_seg == lesion_id).astype(np.uint8)
                    followup_lesion_mask = (followup_seg == lesion_id).astype(np.uint8)
                    
                    # Skip if no corresponding lesion in follow-up
                    if np.sum(followup_lesion_mask) == 0:
                        # print(f"Skipping lesion {lesion_id} for {self.baseline_files[i]} - no corresponding lesion in follow-up")
                        continue
                    
                    # Convert to torch tensors
                    baseline_tensor = torch.from_numpy(baseline_data).float()
                    followup_tensor = torch.from_numpy(followup_data).float()
                    baseline_prompt_tensor = torch.from_numpy(baseline_lesion_mask).float().unsqueeze(0)  # Add channel dim
                    target_tensor = torch.from_numpy(followup_lesion_mask).long()
                    
                    yield {
                        'baseline_data': baseline_tensor,              # [C, H, W, D] - baseline image
                        'followup_data': followup_tensor,             # [C, H, W, D] - follow-up image
                        'baseline_prompt': baseline_prompt_tensor,     # [1, H, W, D] - baseline segmentation as prompt
                        'target': target_tensor,                       # [H, W, D] - follow-up segmentation target
                        'properties': properties,                      # Metadata
                        'lesion_id': lesion_id,                       # Lesion instance ID
                        'filename': (
                            f"{os.path.basename(_representative_path(self.baseline_files[i]))}"
                            f"_to_{os.path.basename(_representative_path(self.followup_files[i]))}"
                            f"_lesion_{lesion_id}"
                        )
                    }
                    
            except Exception as e:
                print(f"Error processing pair {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import math
import multiprocessing
import shutil
from time import sleep
from typing import Tuple, Union, List
import os
import SimpleITK
import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm

import lesionlocator
from lesionlocator.preprocessing.cropping.cropping import crop_to_nonzero
from lesionlocator.preprocessing.resampling.default_resampling import compute_new_shape
from lesionlocator.utilities.find_class_by_name import recursive_find_python_class
from lesionlocator.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from lesionlocator.utilities.utils import get_filenames_of_train_images_and_targets


class TrainingPreprocessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed
        if len(target_spacing) < len(data.shape[1:]):
            #target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        
        # median_size = configuration['median_image_size_in_voxels']
        # median_size = [395, 256, 256]
        # target_spacing = (seg.shape[1]* original_spacing[0] / median_size[0],
        #     seg.shape[2] * original_spacing[1] /  median_size[1],
        #     seg.shape[3] * original_spacing[2] /  median_size[2])

        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg, properties

    def all_exist(self, files):
        return all(os.path.exists(f) for f in files)

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str],
                 track: bool = False,
                 train: bool = False) -> Tuple[np.ndarray, Union[np.ndarray, None], dict, Union[np.ndarray, None], dict]:
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample -> crop_to_patch_size
        so when we export we need to run the following order: uncrop_from_patch -> resample -> crop -> transpose
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properties = rw.read_images(image_files)
        bl_files = None

        bl_seg_file = None
        if track:
            if 'TP0' in image_files[0]:
                candidate = [imf.replace('TP0', 'TP1') for imf in image_files]
                if self.all_exist(candidate):
                    bl_files = candidate
                    if train:
                        bl_seg_file = seg_file.replace('TP0', 'TP1') if seg_file is not None else None
                else:
                    candidate = [imf.replace('TP0', 'TP2') for imf in image_files]
                    if self.all_exist(candidate):
                        bl_files = candidate
                        if train:
                            bl_seg_file = seg_file.replace('TP0', 'TP2') if seg_file is not None else None
            elif 'TP1' in image_files[0]:
                candidate = [imf.replace('TP1', 'TP0') for imf in image_files]
                if self.all_exist(candidate):
                    bl_files = candidate
                    if train:
                        bl_seg_file = seg_file.replace('TP1', 'TP0') if seg_file is not None else None
                else:
                    candidate = [imf.replace('TP1', 'TP2') for imf in image_files]
                    if self.all_exist(candidate):
                        bl_files = candidate
                        if train:
                            bl_seg_file = seg_file.replace('TP1', 'TP2') if seg_file is not None else None
            elif 'TP2' in image_files[0]:
                candidate_tp1 = [imf.replace('TP2', 'TP1') for imf in image_files]
                if self.all_exist(candidate_tp1):
                    bl_files = candidate_tp1
                    if train:
                        bl_seg_file = seg_file.replace('TP2', 'TP1') if seg_file is not None else None
                else:
                    candidate_tp0 = [imf.replace('TP2', 'TP0') for imf in image_files]
                    if self.all_exist(candidate_tp0):
                        bl_files = candidate_tp0
                        if train:
                            bl_seg_file = seg_file.replace('TP2', 'TP0') if seg_file is not None else None
        # if track: load bl_file, bl_seg, resample to target spacing, crop bl_data, bl_seg, and crop data and seg using same crop info as bl_data, bl_seg. 
        # if not track: load data and seg, resample and crop
        # 
        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        if self.verbose:
            print(seg_file)

        # resample and normalize data and seg
        data, seg, data_properties = self.run_case_npy(data, seg, data_properties, plans_manager, configuration_manager,
                                      dataset_json)
        
        # NEW: Crop to patch size for training
        fu_shape_before_patch_crop = None
        if hasattr(configuration_manager, 'patch_size') and configuration_manager.patch_size is not None:
            # Find center point for cropping (use largest-lesion centroid if available).
            # Without this, whole-body PET/CT patches are cropped around the geometric
            # image center (typically abdomen) and often contain no foreground at all.
            center_point = self._largest_lesion_center(seg)
            if self.verbose and center_point is not None:
                print(f'Using largest-lesion centroid as crop center: {center_point}')

            fu_shape_before_patch_crop = data.shape[1:]
            data, seg, crop_info = self.crop_to_patch_size(data, seg, configuration_manager.patch_size, center_point)
            
            # Store crop information in properties
            data_properties['patch_crop_info'] = crop_info
            data_properties['shape_after_patch_cropping'] = data.shape[1:]
            
            if self.verbose:
                print(f'Final data shape after patch cropping: {data.shape}')
        
        # read the baseline image if tracking is enabled
        bl_data = None
        bl_data_properties = None
        if bl_files:
            bl_data, bl_data_properties = rw.read_images(bl_files)
            # The baseline data must always be normalized/resampled/cropped so that
            # the tracker receives it in the same space as the follow-up data.
            # Only the baseline segmentation is gated on `train` (only available
            # during training).
            if train:
                bl_seg = rw.read_seg(bl_seg_file)[0] if bl_seg_file is not None else None
            else:
                bl_seg = None

            bl_data, bl_seg, bl_data_properties = self.run_case_npy(
                bl_data, bl_seg, bl_data_properties, plans_manager,
                configuration_manager, dataset_json)

            # Apply same patch cropping to baseline data so it shares the
            # follow-up patch shape going into the tracker. Prefer the baseline
            # seg's own largest-lesion centroid when available; otherwise, fall
            # back to the follow-up's centroid mapped proportionally onto the
            # baseline voxel grid so both patches cover broadly corresponding anatomy.
            if hasattr(configuration_manager, 'patch_size') and configuration_manager.patch_size is not None:
                center_point_bl = self._largest_lesion_center(bl_seg)
                if center_point_bl is None and center_point is not None and fu_shape_before_patch_crop is not None:
                    bl_shape = bl_data.shape[1:]
                    if all(s > 0 for s in fu_shape_before_patch_crop):
                        center_point_bl = tuple(
                            int(c * (bl_s / fu_s))
                            for c, fu_s, bl_s in zip(center_point, fu_shape_before_patch_crop, bl_shape)
                        )
                if self.verbose and center_point_bl is not None:
                    print(f'Using baseline crop center: {center_point_bl}')

                bl_data, bl_seg, bl_crop_info = self.crop_to_patch_size(
                    bl_data, bl_seg, configuration_manager.patch_size, center_point_bl)
                bl_data_properties['patch_crop_info'] = bl_crop_info
                bl_data_properties['shape_after_patch_cropping'] = bl_data.shape[1:]

            if bl_seg is not None and train:
                bl_data_properties['seg'] = bl_seg
        # print(f"Final data shape: {data.shape}, bl_data shape: {bl_data.shape if bl_data is not None else None}", flush=True)
        return data, seg, data_properties, bl_data, bl_data_properties

    @staticmethod
    def _largest_lesion_center(seg: Union[np.ndarray, None]) -> Union[Tuple[int, ...], None]:
        """
        Return the (H, W, D) centroid of the largest positive-label connected region
        in `seg`, or None if `seg` is None / contains no foreground. Expects shape
        [1, H, W, D] (strips the channel dimension internally).
        """
        if seg is None:
            return None
        vol = seg[0] if seg.ndim == 4 else seg
        unique_labels = np.unique(vol)
        lesion_labels = unique_labels[unique_labels > 0]
        if len(lesion_labels) == 0:
            return None
        largest_lesion = None
        largest_size = 0
        for label in lesion_labels:
            size = int(np.sum(vol == label))
            if size > largest_size:
                largest_size = size
                largest_lesion = int(label)
        if largest_lesion is None:
            return None
        coords = np.where(vol == largest_lesion)
        if len(coords[0]) == 0:
            return None
        return tuple(int(np.mean(c)) for c in coords)

    def crop_to_patch_size(self, data: np.ndarray, seg: Union[np.ndarray, None],
                          patch_size: Tuple[int, ...], center_point: Union[Tuple[int, ...], None] = None) -> Tuple[np.ndarray, Union[np.ndarray, None], dict]:
        """
        Crop data and segmentation to a fixed patch size centered around a point.
        
        Args:
            data: Input image data with shape [C, H, W, D]
            seg: Segmentation mask with shape [1, H, W, D] or None
            patch_size: Target patch size (H, W, D)
            center_point: Center point for cropping (H, W, D). If None, use image center.
            
        Returns:
            Tuple of (cropped_data, cropped_seg, crop_info)
        """
        if len(patch_size) != 3:
            raise ValueError(f"patch_size must be 3D, got {len(patch_size)}D")
        
        original_shape = data.shape[1:]  # Exclude channel dimension
        
        # Use center of image if no center point provided
        if center_point is None:
            center_point = tuple(s // 2 for s in original_shape)
        
        # Calculate crop boundaries
        crop_start = []
        crop_end = []
        pad_before = []
        pad_after = []

        
        for i, (center, patch_dim, orig_dim) in enumerate(zip(center_point, patch_size, original_shape)):
            # Calculate start and end positions
            half_patch = patch_dim // 2

            # randomly shift center by 50% of patch size
            
            start = max(0, center - half_patch)
            end = min(orig_dim, center + half_patch + (patch_dim % 2))

            # Adjust if patch extends beyond image boundaries
            if end - start < patch_dim:
                if start == 0:
                    end = min(orig_dim, start + patch_dim)
                elif end == orig_dim:
                    start = max(0, end - patch_dim)
            
            crop_start.append(start)
            crop_end.append(end)
            
            # Calculate padding needed if image is smaller than patch
            actual_size = end - start
            pad_total = patch_dim - actual_size
            pad_before.append(pad_total // 2)
            pad_after.append(pad_total - pad_total // 2)

        # Create slice objects for cropping
        crop_slices = [slice(None)] + [slice(start, end) for start, end in zip(crop_start, crop_end)]
        
        # Crop the data
        cropped_data = data[tuple(crop_slices)]
        cropped_seg = seg[tuple(crop_slices)] if seg is not None else None
        
        # Pad if necessary to reach exact patch size
        if any(p > 0 for p in pad_before + pad_after):
            # Padding for data (all channels)
            pad_width_data = [(0, 0)] + [(pb, pa) for pb, pa in zip(pad_before, pad_after)]
            cropped_data = np.pad(cropped_data, pad_width_data, mode='constant', constant_values=0)
            
            # Padding for segmentation
            if cropped_seg is not None:
                pad_width_seg = [(0, 0)] + [(pb, pa) for pb, pa in zip(pad_before, pad_after)]
                cropped_seg = np.pad(cropped_seg, pad_width_seg, mode='constant', constant_values=0)
        
        # Store cropping information
        crop_info = {
            'original_shape': original_shape,
            'patch_size': patch_size,
            'center_point': center_point,
            'crop_start': crop_start,
            'crop_end': crop_end,
            'pad_before': pad_before,
            'pad_after': pad_after,
            'final_shape': cropped_data.shape[1:],
        }
        
        if self.verbose:
            print(f'Cropped from {original_shape} to {cropped_data.shape[1:]} with patch_size {patch_size}')
            print(f'Center point: {center_point}, Crop region: {crop_start} to {crop_end}')
            if any(p > 0 for p in pad_before + pad_after):
                print(f'Applied padding: before={pad_before}, after={pad_after}')
        
        return cropped_data, cropped_seg, crop_info


    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        foreground_mask = seg != 0
        foreground_coords = np.argwhere(foreground_mask)
        seg = seg[foreground_mask]
        del foreground_mask
        unique_labels = pd.unique(seg.ravel())

        # We don't need more than 1e7 foreground samples. That's insanity. Cap here
        if len(foreground_coords) > 1e7:
            take_every = math.floor(len(foreground_coords) / 1e7)
            # keep computation time reasonable
            if verbose:
                print(f'Subsampling foreground pixels 1:{take_every} for computational reasons')
            foreground_coords = foreground_coords[::take_every]
            seg = seg[::take_every]

        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)

            # check if any of the labels are in seg, if not skip c
            if isinstance(c, (tuple, list)):
                if not any([ci in unique_labels for ci in c]):
                    class_locs[k] = []
                    continue
            else:
                if c not in unique_labels:
                    class_locs[k] = []
                    continue

            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = foreground_coords[mask]
            else:
                mask = seg == c
                all_locs = foreground_coords[mask]
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
            seg = seg[~mask]
            foreground_coords = foreground_coords[~mask]
        return class_locs

    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(join(lesionlocator.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'lesionlocator.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError(f'Unable to locate class \'{scheme}\' for normalization')
            # Channel stats may be absent for newly-added modalities (e.g. PET in petct mode).
            # ZScoreNormalization computes its own per-scan statistics and doesn't use these
            # values, but still requires a dict to satisfy the base-class assertion.
            _default_props = {'mean': 0.0, 'std': 1.0,
                              'percentile_00_5': -1000.0, 'percentile_99_5': 1000.0}
            normalizer = normalizer_class(
                use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                intensityproperties=foreground_intensity_properties_per_channel.get(str(c), _default_props)
            )
            # For ZScoreNormalization (PET channel): build a body mask from signal
            # intensity rather than passing the prompt/seg mask.  The prompt mask has
            # all non-negative values so seg >= 0 would be True everywhere, causing
            # stats to be computed over background air — a train/inference mismatch.
            # Encoding follows nnUNet convention: 0 = body (included), -1 = outside.
            if scheme == 'ZScoreNormalization':
                mask = np.where(data[c] > 0.2, 0, -1).astype(np.int8)
            else:
                mask = seg[0] if seg is not None else None
            data[c] = normalizer.run(data[c], mask)
        return data


    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        # this function will be called at the end of self.run_case. Can be used to change the segmentation
        # after resampling. Useful for experimenting with sparse annotations: I can introduce sparsity after resampling
        # and don't have to create a new dataset each time I modify my experiments
        return seg


def example_test_case_preprocessing():
    # (paths to files may need adaptations)
    plans_file = '/home/isensee/drives/gpu_data/LesionLocator_preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans.json'
    dataset_json_file = '/home/isensee/drives/gpu_data/LesionLocator_preprocessed/Dataset219_AMOS2022_postChallenge_task2/dataset.json'
    input_images = ['/home/isensee/drives/e132-rohdaten/nnUNetv2/Dataset219_AMOS2022_postChallenge_task2/imagesTr/amos_0600_0000.nii.gz', ]  # if you only have one channel, you still need a list: ['case000_0000.nii.gz']

    configuration = '3d_fullres'
    pp = DefaultPreprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    plans_manager = PlansManager(plans_file)
    data, _, properties = pp.run_case(input_images, seg_file=None, plans_manager=plans_manager,
                                      configuration_manager=plans_manager.get_configuration(configuration),
                                      dataset_json=dataset_json_file)

    # voila. Now plug data into your prediction function of choice. We of course recommend nnU-Net's default (TODO)
    return data


if __name__ == '__main__':
    # example_test_case_preprocessing()
    # pp = DefaultPreprocessor()
    # pp.run(2, '2d', 'nnUNetPlans', 8)

    ###########################################################################################################
    # how to process a test cases? This is an example:
    # example_test_case_preprocessing()
    seg = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage('/home/isensee/temp/H-mito-val-v2.nii.gz'))[None]
    DefaultPreprocessor._sample_foreground_locations(seg, np.arange(1, np.max(seg) + 1))
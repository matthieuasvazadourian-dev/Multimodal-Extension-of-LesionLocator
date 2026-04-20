from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number
from scipy import ndimage as ndi

def get_pet_foreground(pet, thresh=None, pct=95):
    """
    Simple PET foreground mask.
    Foreground = voxels above the 95th percentile of the image (adaptive).
    """

    # threshold relative to PET intensity distribution
    if thresh is None:
        thr = np.percentile(pet[pet > 0], pct)
    else:
        thr = thresh

    # foreground is everything above threshold
    mask = pet > thr

    # largest connected component
    labels, n = ndi.label(mask)
    if n > 0:
        sizes = ndi.sum(mask, labels, index=range(1, n+1))
        largest = np.argmax(sizes) + 1
        mask = (labels == largest)

    # optional hole filling
    mask = ndi.binary_fill_holes(mask)
    
    return mask

def get_ct_foreground(ct, hu_threshold=-900):
    """
    Returns a simple foreground mask for CT.
    Foreground = everything > -500 HU, cleaned.
    """

    # threshold to remove air/background (air is around -1000 HU)
    # Use a more conservative threshold to avoid creating holes
    mask = ct > hu_threshold

    # morphological closing to fill small holes before connected components
    from scipy.ndimage import binary_closing, binary_dilation
    struct = np.ones((3, 3, 3))
    mask = binary_closing(mask, structure=struct, iterations=2)

    # keep largest connected component
    labels, n = ndi.label(mask)
    if n > 0:
        sizes = ndi.sum(mask, labels, index=range(1, n+1))
        largest = np.argmax(sizes) + 1
        mask = (labels == largest)

    # fill remaining holes in the largest component
    mask = ndi.binary_fill_holes(mask)
    
    # slight dilation to ensure we don't cut off edges
    mask = binary_dilation(mask, structure=struct, iterations=1)

    return mask

class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype, copy=False)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            if seg is not None:
                mask = seg >= 0
            else:
                mask = get_pet_foreground(image, thresh=0.03)
            # Fallback: the 0.03 threshold is tuned for SUV-scaled PET. On non-PET
            # modalities (or scans whose values all sit below 0.03) the mask can
            # collapse to empty, which would make mean()/std() return NaN and
            # propagate silently. Fall back to the full volume in that case.
            if mask.sum() < 100:
                mask = np.ones_like(mask, dtype=bool)
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
            image[~mask] = 0
        else:
            mean = image.mean()
            std = image.std()
            image -= mean
            image /= (max(std, 1e-8))

        
        return image


class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        # mask = get_ct_foreground(image)
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']
        # print(f"CT Normalization: mean={mean_intensity}, std={std_intensity}, "
        #       f"0.5th percentile={lower_bound}, 99.5th percentile={upper_bound}", flush=True)
        # print(f"image min={image.min()}, max={image.max()}", flush=True)
        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, lower_bound, upper_bound, out=image)
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        return image


class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype, copy=False)


class RescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        image -= image.min()
        image /= np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert image.min() >= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
        assert image.max() <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                   ". Your images do not seem to be RGB images"
        image = image.astype(self.target_dtype, copy=False)
        image /= 255.
        return image


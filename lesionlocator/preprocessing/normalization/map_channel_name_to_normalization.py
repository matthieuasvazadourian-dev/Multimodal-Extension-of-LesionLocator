from typing import Type

from lesionlocator.preprocessing.normalization.default_normalization_schemes import CTNormalization, NoNormalization, \
    ZScoreNormalization, RescaleTo01Normalization, RGBTo01Normalization, ImageNormalization

channel_name_to_normalization_mapping = {
    'ct': CTNormalization,
    'nonorm': NoNormalization,
    'zscore': ZScoreNormalization,
    'rescale_to_0_1': RescaleTo01Normalization,
    'rgb_to_0_1': RGBTo01Normalization
}


def get_ct_foreground(ct, hu_threshold=-500):
    """
    Returns a simple foreground mask for CT.
    Foreground = everything > -500 HU, cleaned.
    """

    # threshold to remove air/background
    mask = ct > hu_threshold

    # keep largest connected component
    labels, n = ndi.label(mask)
    if n > 0:
        sizes = ndi.sum(mask, labels, index=range(1, n+1))
        largest = np.argmax(sizes) + 1
        mask = (labels == largest)

    # fill holes
    mask = ndi.binary_fill_holes(mask)

    return mask

def get_pet_foreground(pet, pct=95):
    """
    Simple PET foreground mask.
    Foreground = voxels above the 95th percentile of the image (adaptive).
    """

    # threshold relative to PET intensity distribution
    thr = np.percentile(pet[pet > 0], pct)

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

def get_normalization_scheme(channel_name: str) -> Type[ImageNormalization]:
    """
    If we find the channel_name in channel_name_to_normalization_mapping return the corresponding normalization. If it is
    not found, use the default (ZScoreNormalization)
    """
    norm_scheme = channel_name_to_normalization_mapping.get(channel_name.casefold())
    if norm_scheme is None:
        norm_scheme = ZScoreNormalization
    return norm_scheme

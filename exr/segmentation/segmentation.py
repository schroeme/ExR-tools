import os
import tifffile
import h5py 
import numpy as np
from skimage import exposure
from skimage import morphology
from scipy import ndimage
from exr.utils import configure_logger
from exr.config import Config
from typing import Tuple

logger = configure_logger('ExR-Tools')

def synapses_roi_segmentation(config: Config, round: int, roi: int, channel: str, top_intensity_percentage: int = 3, size_thershold: int = 3500, dilate_erode_iteration: int = 2) -> None:
    r"""
    Segment synapses in a specific ROI (Region Of Interest) and round using various image processing steps.

    :param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param round: The round to segment.
    :type round: int
    :param roi: Region of interest to segment.
    :type roi: int
    :param channel: The channel to segment.
    :type channel: str
    :param top_intensity_percentage: The top intensity percentage for thresholding. Default is top 3%.
    :type top_intensity_percentage: int
    :param size_thershold: The size threshold for instance segmentation. Default is 3500.
    :type size_thershold: int
    :param dilate_erode_iteration: Number of dilate and erode iterations. Default is 2.
    :type dilate_erode_iteration: int

    :raises: Raises an exception if an error occurs during the segmentation process.

    :note: The results are saved to a TIFF file located at `<roi_analysis_path>/segmentation_masks`.
    """
    def adjust_brightness_contrast(image: np.ndarray) -> np.ndarray:
        return exposure.equalize_adapthist(image)

    def top_percent_threshold(image: np.ndarray, percent: int) -> np.ndarray:
        threshold_value = np.percentile(image, 100 - percent)
        return image > threshold_value

    def fill_holes(image: np.ndarray) -> np.ndarray:
        return ndimage.binary_fill_holes(image)

    def dilate_and_erode(image: np.ndarray, iterations: int) -> np.ndarray:
        for _ in range(iterations):
            image = morphology.dilation(image)
        for _ in range(iterations):
            image = morphology.erosion(image)
        return image

    def instance_mask_and_count(image: np.ndarray, size_threshold: int) -> Tuple[np.ndarray, int]:
        labeled_image, num_features = ndimage.label(image)
        cleaned_image = morphology.remove_small_objects(labeled_image, min_size=size_threshold)
        labeled_image, num_features = ndimage.label(cleaned_image)
        return labeled_image, num_features
    

    try:
        with h5py.File(config.h5_path.format(round, roi), 'r') as f:
            synapses_volume = f[channel][()]
        
        synapses_mask = adjust_brightness_contrast(synapses_volume)
        synapses_mask = top_percent_threshold(synapses_mask, top_intensity_percentage)
        synapses_mask = fill_holes(synapses_mask)
        synapses_mask = dilate_and_erode(synapses_mask, dilate_erode_iteration)
        synapses_mask = fill_holes(synapses_mask)
        synapses_mask, synapses_count = instance_mask_and_count(synapses_mask, size_thershold)
        
        tifffile.imwrite(os.path.join(config.roi_analysis_path, 'segmentation_masks',f'synapses_mask_ROI{roi}.tif'), synapses_mask)
        
        logger.info(f'Successfully segmented synapses from Round: {round}, ROI: {roi}. Results for {synapses_count} synapses saved.')
        
    except Exception as e:
        logger.error(f'Error during synapse segmentation from Round: {round}, ROI: {roi}, Error: {e}')
        raise
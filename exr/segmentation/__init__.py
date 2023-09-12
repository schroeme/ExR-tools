import os
import tifffile
import h5py 
import numpy as np
from skimage import exposure
from skimage import morphology
from scipy import ndimage


def synapses_roi_segmentation(config,round,roi,channel,top_intensity_percentage=3,size_thershold=3500,dilate_erode_iteration=2):

    def adjust_brightness_contrast(image):
        return exposure.equalize_adapthist(image)
    
    def top_percent_threshold(image, percent):
        threshold_value = np.percentile(image, 100-percent)
        return image > threshold_value
    
    def fill_holes(image):
        filled_image = ndimage.binary_fill_holes(image)
        return filled_image

    def dilate_and_erode(image, iterations):
        for _ in range(iterations):
            image = morphology.dilation(image)

        for _ in range(iterations):
            image = morphology.erosion(image)
        return image

    def instance_mask_and_count(image, size_threshold):
        labeled_image, num_features = ndimage.label(image)
        cleaned_image = morphology.remove_small_objects(labeled_image, min_size=size_threshold)
        labeled_image, num_features = ndimage.label(cleaned_image)
        return labeled_image, num_features
    

    with h5py.File(config.h5_path.format(round,roi),'r') as f:
        synapses_volume = f[channel][()] 
    
    synapses_mask = adjust_brightness_contrast(synapses_volume)
    synapses_mask = top_percent_threshold(synapses_mask,top_intensity_percentage)
    synapses_mask = fill_holes(synapses_mask)
    synapses_mask = dilate_and_erode(synapses_mask,dilate_erode_iteration)
    synapses_mask = fill_holes(synapses_mask)
    synapses_mask = instance_mask_and_count(synapses_mask,size_thershold)

    tifffile.imwrite(os.path.join(config.roi_analysis_path,'segmentation_masks'),synapses_mask)
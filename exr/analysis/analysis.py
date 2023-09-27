
import os
import json
import h5py
import tifffile
from typing import List, Tuple

import numpy as np
from skimage import measure, filters, morphology

from exr.utils.utils import calculate_volume_and_surface_area
from exr.io.io import load_json, save_json
from exr.config import Config
from exr.utils import configure_logger

logger = configure_logger('ExR-Tools')


def extract_synapse_coordinates(config: Config, roi: int) -> None:
    r"""
    Extracts bounding cube coordinates from the synapses segmenation masks in a given Region of Interest (ROI) and saves them in a JSON file.
    `segmentation_masks` from <exr.segmentation.synapses_roi_segmentation> are reqiured.

    param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param roi: The ROI to analyze.
    :type roi: int

    :raises: Raises an exception if an error occurs during the extraction of synapse coordinates.
    """

    try:
        # Initialize dictionaries for storing bounding cubes and their properties
        bounding_cube_properties = {}
        bounding_cube_dict = {}

        # Read the mask file for the given ROI
        mask_file_path = os.path.join(
            config.roi_analysis_path, 'segmentation_masks', f'synapses_mask_ROI{roi}.tif')
        mask_data = tifffile.imread(mask_file_path)

        # Extract region properties from the mask
        region_properties = measure.regionprops(mask_data)

        # Iterate through each region to find bounding cubes for synapses
        for i, prop in enumerate(region_properties):
            min_z, min_y, min_x, max_z, max_y, max_x = prop.bbox
            bounding_cube_properties[str(i)] = {"coordinate": [(
                min_z, min_y, min_x), (max_z, max_y, max_x)]}

        # Assign bounding cube properties to the respective ROI
        bounding_cube_dict[f"ROI_{roi}"] = bounding_cube_properties

        # Save the result as a JSON file
        output_file_path = os.path.join(
            config.roi_analysis_path, 'synapses_properties', f'synapses_coord_ROI{roi}.json')
        with open(output_file_path, 'w') as json_file:
            json.dump(bounding_cube_dict, json_file, indent=4)

        logger.info(
            f"Successfully extracted synapse coordinates for ROI: {roi}")

    except Exception as e:
        logger.error(
            f"Error during the extraction of synapse coordinates for ROI: {roi}. Error: {e}")
        raise


def measure_synapse_properties(config: Config, 
                               roi: int, 
                               round_channel_pairs: List[Tuple[int, int]], 
                               nonzero_threshold: float = 0.65) -> None:
    r"""
    Measures various properties of synapses for a given Region Of Interest (ROI) and round-channel pairs. The properties 
    include volume, surface area, and aspect ratios of synapses. The function then saves the computed properties 
    in a JSON file.

    param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param roi: The Region Of Interest (ROI) to measure properties for.
    :type roi: int
    :param round_channel_pairs: List of tuples where each tuple is a (round, channel) pair.
    :type round_channel_pairs: List[Tuple[int, int]]
    :param nonzero_threshold: Threshold for the fraction of non-zero pixels to determine synapse validity.
    :type nonzero_threshold: float, default is 0.65

    :raises: Logs an error message if there's an issue during processing.
    """

    synapse_coordinates_file_path = os.path.join(
        config.roi_analysis_path, 'synapses_properties', f'synapses_coord_ROI{roi}.json')
    
    synapse_coordinates = load_json(synapse_coordinates_file_path)

    if not synapse_coordinates:
        return

    pixel_spacing = [config.zstep, config.xystep, config.xystep]

    for round_num, channel in round_channel_pairs:

        for synapse_key, synapse_value in synapse_coordinates[f"ROI_{roi}"].items():
            
            round_properties = {}

            min_z, min_y, min_x = synapse_value["coordinate"][0]
            max_z, max_y, max_x = synapse_value["coordinate"][1]

            try:
                with h5py.File(config.h5_path.format(round_num, roi), 'r') as f:
                    synapses_volume = f[config.channel_names[channel]][min_z:max_z, min_y:max_y, min_x:max_x]
            except Exception as e:
                logger.error(f"Failed to load H5 data for round {round_num} and ROI {roi}: {e}")
                continue

            if np.all(synapses_volume == 0):
                synapse_value[f"{round_num}-{channel}"] = "No-overlap"
                continue

            if config.do_binarize:

                I = synapses_volume
                I = (I - np.min(I)) / (np.max(I) - np.min(I))

                # Thresholding
                if config.thresh_method == 'pct':
                    threshold = np.percentile(I, config.thresh_val)
                elif config.thresh_method == 'otsu':
                    threshold = filters.threshold_otsu(I)
                    threshold *= config.thresh_multiplier
                elif config.thresh_method == 'zscore':
                    meanint = np.mean(I)
                    stdint = np.std(I)
                    threshold = meanint + config.thresh_multiplier * stdint

                # Binarize the image
                I = (I > threshold).astype(int)

            else:
                mask_file_path = os.path.join(
                    config.roi_analysis_path, 'segmentation_masks', f'synapses_mask_ROI{roi}.tif')
                synapse_mask = tifffile.imread(mask_file_path)
                I = synapse_mask[min_z:max_z, min_y:max_y, min_x:max_x]

            # Median Filtering
            if config.do_med_filt:
                I_filt = filters.median(I, np.ones(config.filt_size))
            else:
                I_filt = I

            # Remove small objects
            I_filt = morphology.remove_small_objects(
                I_filt.astype(bool), min_size=config.minsize)

            # Find connected components
            labeled_image, nobjects_check = measure.label(
                I_filt, return_num=True, connectivity=3)

            # Calculate fraction of non-zero pixels
            fraction_nonzero = np.count_nonzero(I_filt) / np.size(I_filt)

            # Check conditions for connected components and fraction of non-zero pixels
            if nobjects_check > 0 and fraction_nonzero < nonzero_threshold:

                labeled_image, num_objects = measure.label(
                    I, return_num=True, connectivity=3)

                puncta_vols = []
                puncta_SAs = []
                puncta_physical_vols = []
                puncta_physical_SAs = []

                for label in np.unique(labeled_image):
                    if label != 0:
                        volume, surface_area, physical_volume, physical_surface_area = calculate_volume_and_surface_area(
                            labeled_image, label, pixel_spacing)
                        puncta_vols.append(volume)
                        puncta_SAs.append(surface_area)
                        puncta_physical_vols.append(physical_volume)
                        puncta_physical_SAs.append(physical_surface_area)

                # Calculate statistics
                round_properties["num_puncta"] = num_objects
                round_properties["mean_puncta_vol"] = np.mean(puncta_vols)
                round_properties["med_puncta_vol"] = np.median(puncta_vols)
                round_properties["std_puncta_vol"] = np.std(puncta_vols)

                round_properties["mean_puncta_SA"] = np.mean(puncta_SAs)
                round_properties["med_puncta_SA"] = np.median(puncta_SAs)
                round_properties["std_puncta_SA"] = np.std(puncta_SAs)

                round_properties["mean_physical_puncta_vol"] = np.mean(
                    puncta_physical_vols)
                round_properties["med_physical_puncta_vol"] = np.median(
                    puncta_physical_vols)
                round_properties["std_physical_puncta_vol"] = np.std(
                    puncta_physical_vols)

                round_properties["mean_physical_puncta_SA"] = np.mean(
                    puncta_physical_SAs)
                round_properties["med_physical_puncta_SA"] = np.median(
                    puncta_physical_SAs)
                round_properties["std_physical_puncta_SA"] = np.std(
                    puncta_physical_SAs)

                # Initialize lists to store different aspect ratios
                AR_Z_Y = []  # Aspect Ratio between first and second principal axes
                AR_Y_X = []  # Aspect Ratio between second and third principal axes
                AR_Z_X = []  # Aspect Ratio between first and third principal axes

                properties = measure.regionprops(
                    labeled_image, spacing=pixel_spacing)
                # Loop through properties to calculate aspect ratios
                for prop in properties:
                    l1, l2, l3 = measure.inertia_tensor_eigvals(
                        prop.image)

                    # Calculate aspect ratios, avoiding division by zero
                    AR_Z_Y.append(l1 / l2 if l2 != 0 else 0)
                    AR_Y_X.append(l2 / l3 if l3 != 0 else 0)
                    AR_Z_X.append(l1 / l3 if l3 != 0 else 0)

                # Calculate the maximum aspect ratio for each type
                round_properties["max_AR_Z_Y"] = max(AR_Z_Y)
                round_properties["max_AR_Y_X"] = max(AR_Y_X)
                round_properties["max_AR_Z_X"] = max(AR_Z_X)

                synapse_value[f"{round}-{channel}"] = round_properties

            else:
                synapse_value[f"{round}-{channel}"] = "Failed-condition"

    synapses_properties_path = os.path.join(
                config.roi_analysis_path, 'synapses_properties', f'synapse_properties_ROI{roi}.json')
    save_json(synapse_coordinates, synapses_properties_path)

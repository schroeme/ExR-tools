"""
The Puncta Analysis Module is designed to analyze expansion microscopy data, specifically to identify puncta—distinct fluorescent markers that indicate proteins at synapses. It offers tools for processing synapses either individually or in pairs across multiple rounds of analysis. This module focuses on mapping synapse coordinates to detect the presence of proteins and to explore the interactions between proteins and synapses.
"""
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

    :param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param roi: The ROI to analyze.
    :type roi: int

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

    :param config: Configuration options. This should be an instance of the Config class.
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
                    synapses_volume = f[config.channel_names[channel]
                                        ][min_z:max_z, min_y:max_y, min_x:max_x]
            except Exception as e:
                logger.error(
                    f"Failed to load H5 data for round {round_num} and ROI {roi}: {e}")
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

                synapse_value[f"{round_num}-{channel}"] = round_properties

            else:
                synapse_value[f"{round_num}-{channel}"] = "Failed-condition"

    synapses_properties_path = os.path.join(
        config.roi_analysis_path, 'synapses_properties', f'synapse_properties_ROI{roi}.json')
    save_json(synapse_coordinates, synapses_properties_path)


def measure_synapse_properties_pairwise(config: Config,
                                        roi: int,
                                        round_channel_pairs: List[Tuple[int, int]],
                                        nonzero_threshold: float = 0.65) -> None:
    r"""
    Computes pairwise statistics for synapse properties for a given Region Of Interest (ROI) based on the dataset configuration. 
    The function operates on different rounds and channels, processing each pair to determine properties such as correlation, overlap,
    and distances between puncta. The results are then saved in a JSON file.

    :param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param roi: The Region Of Interest (ROI) identifier for which the synapse properties are to be computed.
    :type roi: int
    :param round_channel_pairs: List of tuples where each tuple contains a round number and a channel identifier. 
        The function computes pairwise statistics for each combination of round-channel pairs.
    :type round_channel_pairs: List[Tuple[int, int]]
    :param nonzero_threshold: Threshold for non-zero pixel fraction. Used to determine the validity of the synapse based on the amount of non-zero pixels in the image, default is 0.65.
    :type nonzero_threshold: float

    :raises: Logs an error message if there's an issue during processing.
    """
    # Utility function to load the synapse volume from a given file.
    def load_synapse_volume(round_num, roi, channel):
        try:
            with h5py.File(config.h5_path.format(round_num, roi), 'r') as f:
                return f[config.channel_names[channel]][min_z:max_z, min_y:max_y, min_x:max_x]
        except Exception as e:
            logger.error(
                f"Failed to load H5 data for round {round_num} and ROI {roi}: {e}")
        return None

        # Utility function to binarize the image.
    def binarize_image(synapses_volume):
        I = (synapses_volume - np.min(synapses_volume)) / \
            (np.max(synapses_volume) - np.min(synapses_volume))
        if config.thresh_method == 'pct':
            threshold = np.percentile(I, config.thresh_val)
        elif config.thresh_method == 'otsu':
            threshold = filters.threshold_otsu(I)
            threshold *= config.thresh_multiplier
        elif config.thresh_method == 'zscore':
            meanint = np.mean(I)
            stdint = np.std(I)
            threshold = meanint + config.thresh_multiplier * stdint

        return (I > threshold).astype(int)

    def process_images_and_compute_properties(I):
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
            # Calculate statistics
            round_properties["image"] = I
            round_properties["num_puncta"] = num_objects

            properties = measure.regionprops(
                labeled_image, spacing=pixel_spacing)
            # # Loop through properties to calculate aspect ratios
            individual_prop = []
            for prop in properties:
                individual_prop.append((prop.coords, prop.centroid))
            synapse_value[f"{round_num}-{channel}"] = round_properties
            round_properties["puncta_prop"] = individual_prop
        else:
            round_properties["image"] = I
            round_properties["num_puncta"] = 0
            round_properties["puncta_prop"] = None

        return round_properties

    # Utility function to calculate pairwise statistics
    def compute_pairwise_statistics(data):
        results_for_key = {}

        for i, round_channel_value_1 in enumerate(round_channel_pairs):
            for round_channel_value_2 in round_channel_pairs[i:]:
                pair1 = f'{round_channel_value_1[0]}-{round_channel_value_1[1]}'
                pair2 = f'{round_channel_value_2[0]}-{round_channel_value_2[1]}'

                if data[pair1] == 'No-overlap' or data[pair2] == 'No-overlap':
                    results_for_key[f'{pair1},{pair2}'] = 'No-overlap'
                    continue
                else:
                    image1 = data[pair1]['image']
                    image2 = data[pair2]['image']

                    if np.std(image1) != 0 and np.std(image2) != 0:
                        correlation = np.corrcoef(
                            image1.flatten(), image2.flatten())[0, 1]

                        intersection = np.logical_and(image1, image2)
                        n_intersection = np.count_nonzero(intersection)

                        overlap = (n_intersection * 2) / \
                            (np.count_nonzero(image1) + np.count_nonzero(image2))

                    else:
                        correlation = np.nan
                        n_intersection = np.nan
                        overlap = np.nan

                frac_overlaps = []
                distances = []
                for puncta1 in range(data[pair1]['num_puncta']):
                    for puncta2 in range(data[pair2]['num_puncta']):
                        idx1, centroid1 = data[pair1]['puncta_prop'][puncta1]
                        idx2, centroid2 = data[pair2]['puncta_prop'][puncta2]

                        # Calculate overlap
                        puncta_overlap = len(np.intersect1d(idx1, idx2))
                        frac_overlap = (puncta_overlap * 2) / (len(idx1) + len(idx2))
                        frac_overlaps.append(frac_overlap)

                        # Calculate inter-puncta distance
                        distance = np.linalg.norm(
                            np.array(centroid1) - np.array(centroid2))
                        distances.append(distance)

                # Filter out negative values
                frac_overlaps = [x for x in frac_overlaps if x >= 0]
                distances = [x for x in distances if x >= 0]

                # Compute statistics on fractional overlap
                mean_frac_overlap = np.mean(frac_overlaps)
                med_frac_overlap = np.median(frac_overlaps)
                std_frac_overlap = np.std(frac_overlaps)

                # Compute statistics on distances
                mean_distance = np.mean(distances)
                med_distance = np.median(distances)
                max_distance = np.nan if not distances else np.max(distances)
                min_distance = np.nan if not distances else np.min(distances)
                std_distance = np.std(distances) if distances else np.nan

                results_for_key[f'{pair1},{pair2}'] = {
                    'volume_pairwise': {
                        'correlation': correlation,
                        'intersection': n_intersection,
                        'overlap': overlap
                    },
                    'puncta_pairwise': {
                        'mean_frac_overlap': mean_frac_overlap,
                        'med_frac_overlap': med_frac_overlap,
                        'std_frac_overlap': std_frac_overlap,
                        'mean_distance': mean_distance,
                        'med_distance': med_distance,
                        'max_distance': max_distance,
                        'min_distance': min_distance,
                        'std_distance': std_distance
                    }
                }

        return results_for_key

    # Load Synapse Coordinates
    synapse_coordinates_file_path = os.path.join(
        config.roi_analysis_path, 'synapses_properties', f'synapses_coord_ROI{roi}.json')
    synapse_coordinates = load_json(synapse_coordinates_file_path)
    if not synapse_coordinates:
        logger.error(f"Failed to load synapse coordinates file for ROI {roi}")
        return
    pixel_spacing = [config.zstep, config.xystep, config.xystep]

    # Loop through rounds and channels and analayze synapses
    for round_num, channel in round_channel_pairs:
        for synapse_key, synapse_value in synapse_coordinates[f"ROI_{roi}"].items():
            round_properties = {}
            min_z, min_y, min_x = synapse_value["coordinate"][0]
            max_z, max_y, max_x = synapse_value["coordinate"][1]

            # Load Synapse Volume
            synapses_volume = load_synapse_volume(round_num, roi, channel)

            if synapses_volume is None or np.all(synapses_volume == 0):
                synapse_value[f"{round_num}-{channel}"] = "No-overlap"
                continue

            # Binarize Image
            if config.do_binarize:
                I = binarize_image(synapses_volume)
            else:
                mask_file_path = os.path.join(
                    config.roi_analysis_path, 'segmentation_masks', f'synapses_mask_ROI{roi}.tif')
                synapse_mask = tifffile.imread(mask_file_path)
                I = synapse_mask[min_z:max_z, min_y:max_y, min_x:max_x]

            # Process Images and Compute Properties
            round_properties = process_images_and_compute_properties(I)
            synapse_value[f"{round_num}-{channel}"] = round_properties

    # Compute Pairwise Statistics
    pairwise_results = {f'ROI_{roi}': {}}
    for synapse_key in synapse_coordinates[f'ROI_{roi}'].keys():
        synapse_properties = synapse_coordinates[f'ROI_{roi}'][synapse_key]
        results_for_synapse = compute_pairwise_statistics(synapse_properties)
        pairwise_results[f'ROI_{roi}'][synapse_key] = results_for_synapse

    # Save Pairwise Results
    synapses__pairwise_properties_path = os.path.join(
        config.roi_analysis_path, 'synapses_properties', f'synapse_properties_ROI{roi}_pairwise.json')
    save_json(pairwise_results, synapses__pairwise_properties_path)

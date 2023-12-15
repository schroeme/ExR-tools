import os
import tifffile
import h5py 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from exr.io.io import load_json
from exr.utils import configure_logger
from exr.config import Config

from typing import Tuple,Dict, List, Optional


logger = configure_logger('ExR-Tools')

def inspect_roi_segmentation(config: Config, round_num: int, roi: int, channel: str) -> None:
    r"""
    Plots the mid-Z-slice of raw data and its corresponding synapse segmentation mask 
    for a given round, ROI, and channel. This visualization aids in assessing the quality 
    of synapse segmentation.

    :param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param round_num: The round of the dataset to inspect.
    :type round_num: int
    :param roi: Region of interest to inspect.
    :type roi: int
    :param channel: The channel of interest.
    :type channel: str

    :note: Plots are displayed and can be saved as PNG files in a specified directory.
    """

    try:
        # Read the mid-Z-slice of the raw data
        with h5py.File(config.h5_path.format(round_num, roi), 'r') as f:
            mid_z_slice = f[channel].shape[0] // 2
            synapses_slice = f[channel][mid_z_slice,:,:]

        # Read the corresponding mid-Z-slice of the mask
        mask_path = os.path.join(config.roi_analysis_path, 'segmentation_masks', f'synapses_mask_ROI{roi}.tif')
        synapses_mask = tifffile.imread(mask_path)
        synapses_mask_slice = synapses_mask[mid_z_slice,:,:]

        # Plotting the raw data and the mask side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(synapses_slice, cmap='gray')
        axes[0].set_title(f'Raw Data - Round {round_num}, ROI {roi}, Channel {channel}')
        axes[1].imshow(synapses_mask_slice, cmap='gray')
        axes[1].set_title(f'Segmentation Mask - ROI {roi}')
        
        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        # Save the figure as a PNG file
        save_path = os.path.join(config.roi_analysis_path,'inspect_synapses', f'ROI_{roi}_Round_{round_num}_Channel_{channel}.png')
        fig.savefig(save_path,dpi = 300)
        logger.info(f'Plot saved to {save_path}') 

    except Exception as e:
        logger.error(f'Error during visualization for Round: {round_num}, ROI: {roi}, Channel: {channel}. Error: {e}')
        raise



def inspect_pairwise_analysis(config: Config, roi: int, pairwise_type: str = 'volume',
                              pairwise_property: str = 'overlap', protein_names: Optional[List[str]] = None) -> None:
    r"""
    Visualizes and analyzes pairwise properties of synapses for a given ROI.

    The function loads JSON data containing pairwise synapse properties and generates a heatmap
    illustrating the specified pairwise property for each unique pair.

    :param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param roi: The Region Of Interest (ROI) identifier for which the synapse properties are to be computed.
    :type roi: int
    :param pairwise_type: The type of pairwise analysis to perform. Options include:
                          'volume_pairwise' and 'puncta_pairwise'.
    :type pairwise_type: str
    :param pairwise_property: The specific property within the pairwise analysis to visualize. Options include:
                              For 'volume_pairwise': 'correlation', 'intersection', 'overlap'.
                              For 'puncta_pairwise': 'mean_frac_overlap', 'med_frac_overlap', 'std_frac_overlap',
                              'mean_distance', 'med_distance', 'max_distance', 'min_distance', 'std_distance'.
    :type pairwise_property: str
    :param protein_names: Optional list of protein names for labeling the axes in the heatmap. Default is None.
    :type protein_names: Optional[List[str]]

    :raises ValueError: If an invalid pairwise type or property is provided.
    :note: The heatmap is displayed and can be saved as a PNG file in a specified directory.
    """
    try:

        json_data = load_json(os.path.join(config.roi_analysis_path, 'synapses_properties', f'synapse_properties_ROI{roi}_pairwise.json'))

        # Initialize a dictionary to store the sum of overlaps for each pair
        overlap_sums = {}

        # Initialize a dictionary to count occurrences of each pair
        pair_counts = {}

        for roi_key, roi_data in json_data.items():
        # Loop through each synapse in the ROI
            # Loop through each synapse
            for synapse_key, synapse_data in roi_data.items():
                # Loop through each pair within the synapse
                for pair_key, pair_data in synapse_data.items():
                    if pair_data != "No-overlap" and f"{pairwise_type}_pairwise" in pair_data:
                        # Extract the overlap value
                        overlap = pair_data[f"{pairwise_type}_pairwise"].get(pairwise_property, 0)
                        # Replace NaN with zero
                        if overlap != overlap:  # This checks for NaN (since NaN != NaN)
                            continue
                        # Add the overlap to the sum for this pair
                        if pair_key not in overlap_sums:
                            overlap_sums[pair_key] = 0
                            pair_counts[pair_key] = 0
                        overlap_sums[pair_key] += overlap
                        pair_counts[pair_key] += 1

        # Calculate the average overlap for each pair
        average_overlaps = {pair: (overlap_sums[pair] / pair_counts[pair]) for pair in overlap_sums}

        # Extract unique pairs, sorting by the round and then then channel number in each pair
        unique_pairs = sorted(set(pair.split(',')[0] for pair in average_overlaps.keys()),
                            key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))

        # Initialize a matrix
        matrix_size = len(unique_pairs)
        overlap_matrix = np.zeros((matrix_size, matrix_size))

        # Mapping of pairs to indices
        pair_to_index = {pair: idx for idx, pair in enumerate(unique_pairs)}

        # Fill the matrix with average overlap values
        for pair, overlap in average_overlaps.items():
            pair1, pair2 = pair.split(',')
            idx1, idx2 = pair_to_index[pair1], pair_to_index[pair2]
            overlap_matrix[idx1][idx2] = overlap
            overlap_matrix[idx2][idx1] = overlap  # Ensure the matrix is symmetric
            
        max_value = np.nanmax(overlap_matrix[np.nonzero(overlap_matrix < 1)])

        if protein_names:
            unique_pairs = [protein_names[pair] for pair in unique_pairs]

        # Plot the heatmap
        plt.figure(figsize=(14, 14))
        sns.heatmap(overlap_matrix, annot=True, fmt=".3f", xticklabels=unique_pairs, yticklabels=unique_pairs, cmap="crest",vmin=0, vmax=max_value)
        plt.title(f'ROI {roi} {pairwise_type} {pairwise_property} Heatmap')
        plt.show()

        # Save the figure as a PNG file
        save_path = os.path.join(config.roi_analysis_path,'inspect_synapses', f'ROI_{roi}_{pairwise_type}_pairwise_{pairwise_property}.png')
        plt.savefig(save_path,dpi = 300)
        logger.info(f'Plot saved to {save_path}') 

    except ValueError as ve:
        logger.error(f'ValueError occurred: {ve}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error during pairwise analysis: {e}')
        raise


def plot_violin_for_property(config: Config, roi: int, selected_proteins: Dict[str, str], selected_property: str) -> None:
    r"""
    Generates a violin plot for a specified property across selected proteins for a given ROI.

    This function loads synapse properties data and creates a violin plot showing the distribution 
    of the selected property for each specified protein within the ROI.

    :param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param roi: Region of Interest identifier for which the properties are to be plotted.
    :type roi: int
    :param selected_proteins: A dictionary mapping protein codes to protein names.
    :type selected_proteins: Dict[str, str]
    :param selected_property: The property of the synapses to be visualized.
    :type selected_property: str

    :note: The plot is displayed and can be saved as a PNG file in a specified directory.
    """

    try:
        # Initialize a list to store data
        data = []
        synapse_data = load_json(os.path.join(
            config.roi_analysis_path, 'synapses_properties', f'synapse_properties_ROI{roi}.json'))
        
            # Loop through each synapse in the JSON data
        for _roi, synapses in synapse_data.items():
            for synapse, proteins in synapses.items():
                for protein_code, properties in proteins.items():
                    # Check if the protein code is one of the selected proteins
                    if protein_code in selected_proteins:
                        # Check if the property exists and is not a special case
                        if isinstance(properties, dict) and selected_property in properties:
                            value = properties[selected_property]
                            if value != "No-overlap" and value != "Failed-condition":
                                data.append({'Protein': selected_proteins[protein_code], 'Value': value})
        
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Plotting the violin plot
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(x='Protein', y='Value', data=df)
        ax.set_ylabel(selected_property)
        ax.set_title(f'{selected_property} - ROI {roi}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()

        # Save the figure as a PNG file
        save_path = os.path.join(config.roi_analysis_path, 'inspect_synapses', f'ROI_{roi}_Violin_{selected_property}.png')
        plt.savefig(save_path, dpi=150)
        logger.info(f'Violin plot saved to {save_path}')

    except Exception as e:
        logger.error(f'Error in plot_violin_for_property: {e}')
        raise
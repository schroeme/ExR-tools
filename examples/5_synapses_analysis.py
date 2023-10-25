from exr.config.config import Config
from exr.analysis.analysis import extract_synapse_coordinates, measure_synapse_properties,measure_synapse_properties_pairwise

# Step 1: Load Configuration Settings
# ====================================

# Create a new Config object instance.
config = Config()

# Provide the path to the configuration file.
config_file_path = '/path/to/your/configuration/file.json'

# Load the configuration settings from the specified file.
config.load_config(config_file_path)


# Step 2: Extract Synapse Coordinates from Segmentation Masks
# ===========================================================

# Define the list of ROI number for which coordinates will be extracted.
rois_to_analyze = config.rois  # Adjust this based on your dataset for example [1,3].

# Extract the coordinates.
for roi in rois_to_analyze:
    extract_synapse_coordinates(config=config, roi=roi)

# Note: Ensure that segmentation masks have been generated before this step.


# Step 3: Define Round-Channel Pairs for Analysis
# ===============================================

# `round_channel_pairs`: List of round-channel pairs to be analyzed. 
# Each pair is represented as [round, channel_index], where `channel_index` 
# corresponds to the index in the `config.channel_names` list. 
# For instance, if config.channel_names = ['488', '561', '640'], 
# then a channel_index of 1 refers to '488', 2 refers to '561', and so on.

round_channel_pairs_to_analyze = [[1,1],[1,2],[2,1],[2,2],
                       [3,1],[3,2],
                       [4,1],[4,2],
                       [5,1],[5,2],
                       [6,2],
                       [7,2],
                       [8,1],[8,2],
                       [9,1],
                       [10,1],[10,2],
                       [11,1],[11,2],
                       [12,1]]

# Step 4: Measure Synapse Properties
# ==================================

# Adjust analysis specific configuration settings

# Binarization Settings:
# `do_binarize`: If set to True, the function will binarize the image using the specified thresholding method.
config.do_binarize = True

# Thresholding Settings:
# `thresh_method`: The method to use for image thresholding. Can be "pct", "otsu", or "zscore".
config.thresh_method = "zscore"
# `thresh_val`: If using the "pct" method, this specifies the percentile for thresholding. 
# For "zscore", it represents the base threshold value.
config.thresh_val = 98
# `thresh_multiplier`: Used to multiply the threshold for the "zscore" method.
config.thresh_multiplier = 3

# Median Filtering Settings:
# `do_med_filt`: If set to True, the function will apply median filtering to the image.
config.do_med_filt = True
# `filt_size`: The size of the filter to use for median filtering. Specified as (Z, Y, X).
config.filt_size = (3, 3, 3)

# Morphological Processing Settings:
# `minsize`: Minimum size (in pixels) of structures to retain after morphological processing.
config.minsize = 50

# Pixel Spacing Settings:
# `zstep` and `xystep`: Define the spacing between pixels in Z and XY directions, respectively.
# Used for measuring physical properties of the structures.
config.zstep = 0.25/40
config.xystep = 0.1625/40

# Additional Settings for Synapse Property Measurement
# =====================================================

# `nonzero_pixel_threshold`: This parameter sets the threshold for the fraction of 
# non-zero pixels within a segmented synapse required to consider the synapse as valid. 
# It helps in filtering out spurious segmentations where only a small fraction of 
# the segmented volume corresponds to the actual synapse. For instance, if set to 
# 0.65, it means that at least 65% of the pixels in the segmented volume must be 
# non-zero (or part of the synapse) for it to be considered a valid synapse segmentation.
# Adjust this threshold based on experimental requirements and quality of segmentations.

nonzero_pixel_threshold = 0.65

# Execute the measurement of synapse properties.
for roi in rois_to_analyze:
    measure_synapse_properties(
        config=config,
        roi=roi,
        round_channel_pairs=round_channel_pairs_to_analyze,
        nonzero_threshold=nonzero_pixel_threshold
    )

# Execute the measurement of synapse pairwise properties.
for roi in rois_to_analyze:
    measure_synapse_properties_pairwise(
        config=config,
        roi=roi,
        round_channel_pairs=round_channel_pairs_to_analyze,
        nonzero_threshold=nonzero_pixel_threshold
    )

# Note: Always monitor the logs or console output to ensure that synapse property 
# measurement is proceeding without errors.
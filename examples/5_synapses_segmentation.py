from exr.config.config import Config
from exr.segmentation.segmentation import synapses_roi_segmentation

# 1: Load Configuration Settings
# ====================================

# Create a new Config object instance.
config = Config()

# Provide the path to the configuration file. Typically, this file is generated 
# during the pipeline configuration step and has a '.json' extension.
config_file_path = '/path/to/your/configuration/file.json'

# Load the configuration settings from the specified file.
config.load_config(config_file_path)

# Step 2: Synapses ROI Segmentation
# ==================================

# Define the parameters required for the segmentation.

# The round to segment. This could be any round number from your experiment.
segment_round = 1  # Replace with your desired round number

# Region of interest to segment. This could be any ROI number from your experiment.
segment_roi = 1  # Replace with your desired ROI number

# Important notice: Choose round/roi that has a good synaptic regions contrast for effective segmentation

# The channel to segment. Choose from the channels defined in your config.
segment_channel = '561'  # Replace with your desired channel

# These parameters fine-tune the segmentation process. Adjust them based 
# on the specific requirements and characteristics of your dataset.

# `intensity_percentage`:
# -----------------------
# Determines the top intensity percentage for thresholding. It uses the 
# top X% of intensities in the image to define the threshold for 
# binarization. A smaller percentage will result in capturing only the 
# most intense regions, potentially reducing noise but might miss 
# fainter structures.
# Default: 3% (top 3% intensities)
intensity_percentage = 3

# `size_threshold`:
# -----------------
# Specifies the minimum size (in pixels) of connected components to be 
# retained after segmentation. Smaller structures will be removed. This 
# helps in eliminating small noise particles from the segmented image.
# Default: 3500 pixels
size_threshold = 3500

# `iterations`:
# -------------
# The number of dilate and erode iterations to be applied. This process 
# can help in refining the segmented structures, by expanding them and 
# then contracting, potentially filling small gaps and holes.
# Default: 2 iterations
iterations = 2

# Execute the synapses ROI segmentation.
synapses_roi_segmentation(
    config=config,
    round=segment_round,
    roi=segment_roi,
    channel=segment_channel,
    top_intensity_percentage=intensity_percentage,
    size_thershold=size_threshold,
    dilate_erode_iteration=iterations
)

# Note: Always monitor the logs or console output to ensure the segmentation process 
# is proceeding without errors.
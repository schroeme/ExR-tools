from exr.config.config import Config
from exr.align.align_eval import measure_round_alignment_NCC,plot_alignment_evaluation,calculate_alignment_evaluation_ci

# Step 1: Load Configuration Settings
# ====================================

# Create a new Config object instance.
config = Config()

# Provide the path to the configuration file.
config_file_path = '/path/to/your/configuration/file.json'

# Load the configuration settings from the specified file.
config.load_config(config_file_path)

# Step 2: Additional Configuration for Alignment Evaluation
# ================================================

# `nonzero_thresh`: This parameter specifies the threshold for the number of non-zero 
# pixels in a given volume. If the number of non-zero pixels in the aligned volume 
# is below this threshold, then the alignment evaluation for that volume is skipped.
config.nonzero_thresh = .2 * 2048 * 2048 * 80

# `N`: Number of random sub-volumes to sample for the NCC calculation.
config.N = 1000

# `subvol_dim`: Dimension (in pixels) of the cubic sub-volumes to sample for NCC calculation.
config.subvol_dim = 100

# `xystep`: Pixel size in the XY plane. Adjust based on microscope and imaging settings.
config.xystep = 0.1625/40 # check value

# `zstep`: Pixel size in the Z dimension. Adjust based on microscope and imaging settings.
config.zstep = 0.25/40 # check value

# `pct_thresh`: Percentile threshold value used for alignment evaluation. 
config.pct_thresh = 99


# Step 3: Alignment Measurement
# ===========================================================

# Define the list of Round, ROI number for which alignment will be evaluated.
round_to_analyze = config.rounds
roi_to_analyze = config.rois  # Adjust this based on your dataset for example [1,3].

# Extract the coordinates.
for roi in roi_to_analyze:
    for round in round_to_analyze:
        measure_round_alignment_NCC(config=config,round=round, roi=roi)


# 4: Alignment Evaluation and Confidence Interval Calculation
# ==========================================================

# Define CI and percentile parameters.
ci_percentage = 95
percentile_filter_value = 95

for roi in roi_to_analyze:
    # Plot alignment evaluation
    plot_alignment_evaluation(config, roi, percentile=percentile_filter_value, save_fig=True)
    
    # Calculate alignment evaluation confidence interval (CI)
    calculate_alignment_evaluation_ci(config, roi, ci=ci_percentage, percentile_filter=percentile_filter_value)
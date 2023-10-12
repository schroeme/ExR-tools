from exr.config.config import Config
from exr.align.align import volumetric_alignment 

# 1: Load the Configuration
# =================================

# Initialize the configuration object.
config = Config()

# Provide the path to the configuration file. Typically, this file is generated 
# during the pipeline configuration step and has a '.json' extension.
config_file_path = '/path/to/your/configuration/file.json'

# Load the configuration settings from the specified file.
config.load_config(config_file_path)

# Note: Ensure the provided path points to the correct configuration file 
# to avoid any inconsistencies during subsequent processing steps.

# 2: Execute Volumetric Alignment
# ====================================

# Specify additional parameters for alignment
parallelization = 4  # Number of parallel processes
alignment_method = 'bigstream'  # or None for SimpleITK
background_subtraction = ''  # 'top_hat' or 'rolling_ball' , or '' for no background subtraction  

# If you have specific round and ROI pairs you want to align, specify them here.
# Otherwise, the function will use all rounds and ROIs from the config.

specific_round_roi_pairs = [(round_val, roi_val) for round_val in config.rounds for roi_val in config.rois]  # or [(1,2), (2,3), ...] for spesifc rounds/rois alignment

volumetric_alignment(
    config=config,
    round_roi_pairs=specific_round_roi_pairs,
    parallel_processes=parallelization,
    method=alignment_method,
    bg_sub=background_subtraction
)

# Note: Always monitor the logs or console output to ensure the alignment process 
# is proceeding without errors.






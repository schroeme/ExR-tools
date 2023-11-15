1- Pipeline Configuration
=================================

This tutorial covers the initial setup and configuration for the ExR dataset. It is an essential first step to correctly prepare the environment and data paths for subsequent processing.

Step 1: Import Libraries and Initialize Configuration
-----------------------------------------------------

Start by importing necessary libraries and initializing the configuration object.

.. code-block:: python

    import numpy as np
    from exr.config.config import Config

    # Initialize the configuration object.
    config = Config()

Step 2: Set Mandatory Configuration
-----------------------------------

Define the mandatory settings, particularly the path to your raw data directory.

.. code-block:: python

    raw_data_directory = '/Path/to/your/raw_data/'

Step 3: Define Raw Data Directory Structure
-------------------------------------------

Ensure your raw data adheres to the required directory structure.

.. code-block:: text

    # RawDataDir/
    # ├── R1/
    # │   ├── ROI1.nd2
    # │   ├── ROI2.nd2
    # │   └── ...
    # ├── R2/
    # │   ├── ROI1.nd2
    # │   ├── ROI2.nd2
    # │   └── ...
    # ├── R3/
    # │   └── ...
    # └── ...

Step 4: Specify Optional Configuration
--------------------------------------

Adjust optional settings such as the number of rounds, ROIs, and pixel spacing.

.. code-block:: python

    # Define the number of rounds in the ExR dataset.
    # By default, we set it to 12 rounds, but you can adjust this as needed.
    num_rounds = 12

    # Define the number of regions of interest (ROIs).
    # This might vary depending on the dataset.
    num_rois = 4

    # The absolute path to the processed data directory. 
    # By default, a 'processed_data' subdirectory inside the raw_data_path is used.
    processed_data_directory = '/path/to/processed/data/'

    # Spacing between pixels in the format [Z,Y,X].
    # Adjust these values according to your experimental setup.
    pixel_spacing = [0.250, 0.1625, 0.1625]

    # Set the names of channels in the ND2 file.
    # Adjust these names according to your experimental setup.
    channel_names_list = ['640', '561', '488']

    # Reference round number. Adjust based on your experiment.
    reference_round = 1

    # Reference channel name. Adjust based on your experiment.
    reference_channel = '640'

    # Provide the name for the configuration file.
    # It's recommended to provide a meaningful name related to the dataset or experiment.
    config_file = "your_config_name"

    # If set to True, changes permission of the processed data to allow other users to read/write.
    permissions_flag = False

    # If set to True, creates the directory structure in the specified project path.
    create_directory_structure_flag = True

Step 5: Apply Configuration Settings
------------------------------------

Apply the configuration settings using the `set_config` method.

.. code-block:: python

    config.set_config(
    raw_data_path=raw_data_directory,
    processed_data_path=processed_data_directory,
    rounds=list(range(1, num_rounds+1)),
    rois=num_rois,
    spacing=pixel_spacing,
    channel_names=channel_names_list,
    ref_round=reference_round,
    ref_channel=reference_channel,
    permission=permissions_flag,
    create_directory_structure=create_directory_structure_flag,
    config_file_name=config_file
    )

    # Note: Always verify the correctness of paths and parameters before running.

Next Steps
----------

After completing the initial configuration, the next step is to proceed to *Volume Alignment*. For details on performing volume alignment, refer to the `Volume Alignment <volume_alignment.html>`_ section of this guide.

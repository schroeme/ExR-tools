4- Synapses ROI Segmentation
=============================

This tutorial focuses on the segmentation of regions of interest (ROIs) for synapses in the ExR-Tools, an essential step for analyzing synaptic regions in your dataset.

Step 1: Load Configuration Settings
------------------------------------

Initialize and load the configuration settings from a file.

.. code-block:: python

    from exr.config.config import Config

    # Create a new Config object instance.
    config = Config()

    # Provide the path to the configuration file.
    config_file_path = '/path/to/your/configuration/file.json'

    # Load the configuration settings from the specified file.
    config.load_config(config_file_path)

Step 2: Synapses ROI Segmentation
---------------------------------

Execute synapses ROI segmentation with specific parameters.

.. code-block:: python

    from exr.segmentation.segmentation import synapses_roi_segmentation

    # Define parameters for segmentation
    segment_round = 1  # Replace with your desired round number
    segment_roi = 1    # Replace with your desired ROI number
    segment_channel = '561'  # Replace with your desired channel

    intensity_percentage = 3
    size_threshold = 3500
    iterations = 2

    # Execute segmentation
    synapses_roi_segmentation(
        config=config,
        round=segment_round,
        roi=segment_roi,
        channel=segment_channel,
        top_intensity_percentage=intensity_percentage,
        size_thershold=size_threshold,
        dilate_erode_iteration=iterations
    )

    # Note: Monitor the logs or console output for segmentation process.

Next Steps
----------

With the completion of the Synapses ROI Segmentation, the final step in the ExSeq-Toolbox pipeline is *Synapses Analysis*. This step is crucial for detailed analysis and interpretation of synaptic structures. For instructions on how to proceed with Synapses Analysis, refer to the `Synapses Analysis <synapses_analysis.html>`_ section of this guide.

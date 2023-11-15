5- Synapses Analysis
=====================

This tutorial details the steps for analyzing synapses in the ExR-Tools, including extracting synapse coordinates and measuring their properties.

Step 1: Load Configuration Settings
------------------------------------

Begin by initializing and loading the configuration settings.

.. code-block:: python

    from exr.config.config import Config

    # Create a new Config object instance.
    config = Config()

    # Provide the path to the configuration file.
    config_file_path = '/path/to/your/configuration/file.json'

    # Load the configuration settings from the specified file.
    config.load_config(config_file_path)

Step 2: Extract Synapse Coordinates from Segmentation Masks
-----------------------------------------------------------

Extract synapse coordinates for each region of interest (ROI).

.. code-block:: python

    from exr.analysis.analysis import extract_synapse_coordinates

    rois_to_analyze = config.rois  # Adjust based on your dataset.

    for roi in rois_to_analyze:
        extract_synapse_coordinates(config=config, roi=roi)

    # Note: Ensure segmentation masks are generated prior to this step.

Step 3: Define Round-Channel Pairs for Analysis
-----------------------------------------------

Specify the round-channel pairs to be analyzed for synapse properties.

.. code-block:: python

    round_channel_pairs_to_analyze = [[1,1], [1,2], ... [12,1]]

Step 4: Measure Synapse Properties
----------------------------------

Configure and execute the measurement of synapse properties.

.. code-block:: python

    from exr.analysis.analysis import measure_synapse_properties, measure_synapse_properties_pairwise

    # Configuration for synapse property measurement
    config.do_binarize = True
    config.thresh_method = "zscore"
    config.thresh_val = 98
    config.thresh_multiplier = 3
    config.do_med_filt = True
    config.filt_size = (3, 3, 3)
    config.minsize = 50
    config.zstep = 0.25/40
    config.xystep = 0.1625/40
    nonzero_pixel_threshold = 0.65

    for roi in rois_to_analyze:
        measure_synapse_properties(
            config=config,
            roi=roi,
            round_channel_pairs=round_channel_pairs_to_analyze,
            nonzero_threshold=nonzero_pixel_threshold
        )

        measure_synapse_properties_pairwise(
            config=config,
            roi=roi,
            round_channel_pairs=round_channel_pairs_to_analyze,
            nonzero_threshold=nonzero_pixel_threshold
        )

    # Note: Monitor logs or console output for the analysis process.

Conclusion
----------

With the completion of Synapses Analysis, the ExR-Tools data processing pipeline is concluded. This comprehensive analysis provides detailed insights into the synaptic structures within your dataset, ready for further interpretation and research applications.

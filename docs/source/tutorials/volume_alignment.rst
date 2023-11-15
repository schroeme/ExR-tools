2- Volumetric Alignment
========================

This tutorial outlines the process of executing volumetric alignment in the ExR-Tools, an essential step for aligning data across different imaging rounds and fields of view (FOVs).

Step 1: Load the Configuration
------------------------------

Begin by initializing and loading the configuration settings.

.. code-block:: python

    from exr.config.config import Config

    # Initialize the configuration object.
    config = Config()

    # Provide the path to the configuration file.
    config_file_path = '/path/to/your/configuration/file.json'

    # Load the configuration settings from the specified file.
    config.load_config(config_file_path)

    # Note: Ensure the provided path points to the correct configuration file.

Step 2: Execute Volumetric Alignment
------------------------------------

Perform volumetric alignment using the specified parameters and configuration.

.. code-block:: python

    from exr.align.align import volumetric_alignment

    # Specify additional parameters for alignment
    parallelization = 4  # Number of parallel processes
    alignment_method = 'bigstream'  # or None for SimpleITK
    background_subtraction = ''  # 'top_hat' or 'rolling_ball'

    # Specific round and ROI pairs for alignment
    specific_round_roi_pairs = [(round_val, roi_val) for round_val in config.rounds for roi_val in config.rois]

    volumetric_alignment(
        config=config,
        round_roi_pairs=specific_round_roi_pairs,
        parallel_processes=parallelization,
        method=alignment_method,
        bg_sub=background_subtraction
    )

    # Note: Monitor the logs or console output for the alignment process.

Next Steps
----------

After completing the volumetric alignment, the next step in the pipeline is *Alignment Evaluation*. This involves assessing the quality and accuracy of the alignment to ensure the data is correctly aligned across different imaging rounds and FOVs. For detailed instructions on how to perform alignment evaluation, refer to the `Alignment Evaluation <alignment_evaluation.html>`_ section of this guide.
